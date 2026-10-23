"""Fine-tune a causal LM on one domain and track representation-geometry drift + forgetting at checkpoints.

At each evaluation step we record (relative to the step-0 = pretrained model):
  * contextual representations of fixed probe texts (mean-pooled hidden states, mid + final layer)
    -> saved to <out>/reps/step_XXXXX.npz (fp16) for offline metrics (neighbourhood preservation,
       isotropy, CKA) computed in src/metrics.py
  * general-knowledge accuracy: MMLU (570 q, 0-shot, letter log-probs) and ARC-Easy (500 q, length-normalised
    choice log-likelihood)
  * WikiText perplexity (general LM ability) and held-out in-domain loss (did fine-tuning learn?)

Usage: python src/train_eval.py --model Qwen/Qwen2.5-0.5B --domain medical --method lora --seed 1 --out results/runs/<tag>
"""
import argparse, json, math, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

P = "datasets/prepared"
LORA_TARGETS = {"qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]}
EVAL_STEPS = [0, 10, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--domain", required=True, choices=["medical", "code", "general"])
    ap.add_argument("--method", default="lora", choices=["lora", "full"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--probe_len", type=int, default=128)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def token_stream(tok, domain, seed, n_tokens):
    """Shuffle training docs (seeded), tokenize, join with EOS, and chunk into fixed-length sequences."""
    docs = [json.loads(l) for l in open(f"{P}/train_docs_{domain}.jsonl")]
    random.Random(seed).shuffle(docs)
    ids, out = [], []
    for d in docs:
        ids.extend(tok(d, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
        if len(ids) >= n_tokens:
            break
    assert len(ids) >= n_tokens, f"not enough tokens: {len(ids)} < {n_tokens}"
    return torch.tensor(ids[:n_tokens])


def chunked_lm_loss(lm_head, h, labels, chunk=256):
    """Next-token CE computed in token chunks with recomputation, so full-vocab fp32 logits never coexist."""
    h = h[:, :-1].reshape(-1, h.size(-1))
    y = labels[:, 1:].reshape(-1)
    def f(hc, yc):
        return F.cross_entropy(lm_head(hc).float(), yc, reduction="sum")
    tot = 0.0
    for i in range(0, h.size(0), chunk):
        tot = tot + checkpoint(f, h[i:i + chunk], y[i:i + chunk], use_reentrant=False)
    return tot / y.numel()


def backbone_and_head(model):
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return getattr(m, m.base_model_prefix), m.get_output_embeddings()


@torch.no_grad()
def probe_reps(model, tok, texts, layers, max_len, bs):
    """Mean-pooled hidden states (excluding position 0, an attention-sink token) for each probe text."""
    backbone, _ = backbone_and_head(model)
    outs = {l: [] for l in layers}
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_len,
                  add_special_tokens=False).to("cuda")
        hs = backbone(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0  # right padding -> position 0 is the first real token
        mask = mask.unsqueeze(-1).float()
        for l in layers:
            outs[l].append(((hs[l].float() * mask).sum(1) / mask.sum(1)).cpu())
    return {l: torch.cat(v).numpy() for l, v in outs.items()}


@torch.no_grad()
def seq_logprobs(model, tok, ctxs, conts, bs):
    """Sum log p(cont | ctx) and continuation char length, batched with right padding."""
    _, head = backbone_and_head(model)
    backbone, _ = backbone_and_head(model)
    res = []
    for i in range(0, len(ctxs), bs):
        c_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in ctxs[i:i + bs]]
        x_ids = [tok(x, add_special_tokens=False)["input_ids"] for x in conts[i:i + bs]]
        seqs = [(c + x)[-1024:] for c, x in zip(c_ids, x_ids)]
        L = max(map(len, seqs))
        inp = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(inp)
        for j, s in enumerate(seqs):
            inp[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
        inp, att = inp.cuda(), att.cuda()
        h = backbone(input_ids=inp, attention_mask=att).last_hidden_state
        for j, (s, x) in enumerate(zip(seqs, x_ids)):
            n = len(x); s_len = len(s)
            logits = head(h[j, s_len - n - 1:s_len - 1]).float()
            lp = torch.log_softmax(logits, -1).gather(1, inp[j, s_len - n:s_len, None]).sum().item()
            res.append(lp)
    return res


@torch.no_grad()
def eval_mmlu(model, tok, items, bs):
    """0-shot MMLU: compare next-token log-probs of ' A'..' D' after 'Answer:'."""
    _, head = backbone_and_head(model)
    backbone, _ = backbone_and_head(model)
    letter_ids = [tok(" " + L, add_special_tokens=False)["input_ids"][-1] for L in "ABCD"]
    correct = []
    for i in range(0, len(items), bs):
        batch = items[i:i + bs]
        prompts = []
        for it in batch:
            subj = it["subject"].replace("_", " ")
            p = f"The following is a multiple choice question about {subj}.\n\n{it['q']}\n"
            p += "".join(f"{L}. {c}\n" for L, c in zip("ABCD", it["choices"])) + "Answer:"
            prompts.append(p)
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        h = backbone(**enc).last_hidden_state
        last = enc["attention_mask"].sum(1) - 1
        logits = head(h[torch.arange(len(batch)), last]).float()[:, letter_ids]
        pred = logits.argmax(-1).cpu().tolist()
        correct += [int(p == it["answer"]) for p, it in zip(pred, batch)]
    return correct


def eval_arc(model, tok, items, bs):
    """ARC-Easy, acc_norm: argmax of choice log-likelihood / choice char length."""
    ctxs, conts, owner = [], [], []
    for qi, it in enumerate(items):
        for c in it["choices"]:
            ctxs.append(f"Question: {it['q']}\nAnswer:"); conts.append(" " + c); owner.append(qi)
    lps = seq_logprobs(model, tok, ctxs, conts, bs)
    scores = [[] for _ in items]
    for qi, lp, x in zip(owner, lps, conts):
        scores[qi].append(lp / len(x))
    return [int(int(np.argmax(s)) == it["answer"]) for s, it in zip(scores, items)]


@torch.no_grad()
def eval_ppl(model, tok, texts, seqlen, bs):
    """Token-level mean NLL over held-out documents chunked to seqlen."""
    backbone, head = backbone_and_head(model)
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
    n = len(ids) // seqlen
    x = torch.tensor(ids[:n * seqlen]).view(n, seqlen)[:64]
    tot, cnt = 0.0, 0
    for i in range(0, x.size(0), bs):
        b = x[i:i + bs].cuda()
        h = backbone(input_ids=b).last_hidden_state
        for j in range(b.size(0)):
            lg = head(h[j, :-1]).float()
            tot += F.cross_entropy(lg, b[j, 1:], reduction="sum").item(); cnt += seqlen - 1
    return tot / cnt


def main():
    a = parse()
    os.makedirs(f"{a.out}/reps", exist_ok=True)
    set_seed(a.seed)
    lr = a.lr if a.lr is not None else (2e-4 if a.method == "lora" else 1e-5)
    cfg = vars(a) | {"lr": lr, "eval_steps": [s for s in EVAL_STEPS if s <= a.steps], "start": time.ctime()}
    json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=1)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dtype = torch.bfloat16 if a.method == "lora" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype)
    if os.environ.get("RELEASE_FILE"):  # hand over memory reserved by src/gpu_hold.py right before we need it
        open(os.environ["RELEASE_FILE"], "w").close(); time.sleep(3)
    model = model.cuda()
    model.config.use_cache = False
    n_layers = model.config.num_hidden_layers
    layers = [n_layers // 2, n_layers]  # hidden_states index: mid layer, final (post-norm) layer

    if a.method == "lora":
        from peft import LoraConfig, get_peft_model
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0,
                                                 target_modules=LORA_TARGETS[model.config.model_type]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    warm = 50
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))  # constant after warmup

    probes = json.load(open(f"{P}/probes.json"))
    mmlu = json.load(open(f"{P}/mmlu.json")); arc = json.load(open(f"{P}/arc_easy.json"))
    held_gen = json.load(open(f"{P}/heldout_general.json")); held_dom = json.load(open(f"{P}/heldout_{a.domain}.json"))
    stream = token_stream(tok, a.domain, a.seed, a.steps * a.bs * a.seqlen).view(-1, a.seqlen)
    print(f"model={a.model} trainable={n_train/1e6:.1f}M seqs={stream.size(0)} lr={lr}", flush=True)

    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.method == "full"))
    backbone, head = backbone_and_head(model)
    log = open(f"{a.out}/metrics.jsonl", "a")
    t0, run_loss = time.time(), []

    def evaluate(step):
        model.eval()
        te = time.time()
        with autocast:
            reps = {}
            for name in ["general", "medical", "code"]:
                r = probe_reps(model, tok, probes[name], layers, a.probe_len, a.eval_bs)
                for l, v in r.items():
                    reps[f"{name}_L{l}"] = v.astype(np.float16)
            np.savez_compressed(f"{a.out}/reps/step_{step:05d}.npz", **reps)
            mm = eval_mmlu(model, tok, mmlu, a.eval_bs)
            ar = eval_arc(model, tok, arc, a.eval_bs * 2)
            nll_gen = eval_ppl(model, tok, held_gen, 512, a.eval_bs // 2)
            nll_dom = eval_ppl(model, tok, held_dom, 512, a.eval_bs // 2)
        rec = {"step": step, "mmlu": float(np.mean(mm)), "arc_easy": float(np.mean(ar)),
               "nll_general": nll_gen, "nll_domain": nll_dom,
               "train_loss": float(np.mean(run_loss)) if run_loss else None,
               "elapsed": time.time() - t0, "eval_secs": time.time() - te,
               "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        json.dump({"step": step, "mmlu": mm, "arc": ar}, open(f"{a.out}/reps/items_{step:05d}.json", "w"))
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
        model.train()

    evaluate(0)
    model.train()
    for step in range(1, a.steps + 1):
        x = stream[(step - 1) * a.bs: step * a.bs].cuda()
        with autocast:
            h = backbone(input_ids=x).last_hidden_state
            loss = chunked_lm_loss(head, h, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run_loss.append(loss.item())
        if step in EVAL_STEPS:
            evaluate(step)
            run_loss = []
    open(f"{a.out}/DONE", "w").write(time.ctime())


if __name__ == "__main__":
    main()
