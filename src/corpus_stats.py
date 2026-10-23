"""Vocabulary overlap of each fine-tuning corpus with general (pretraining-like) text.

General reference = WikiText-103 training paragraphs from the END of the general pool (a proxy: Qwen2.5's real
pretraining corpus is not public). The "general" fine-tuning corpus is sampled from the START of the pool, so its
row gives the within-distribution noise floor. Measures (~2M Qwen2.5 tokens per corpus):
  * Jensen-Shannon divergence (bits) between token unigram distributions (corpus vs reference)
  * token mass covered by the 10k most frequent reference tokens
  * word-level: fraction of alphabetic word tokens whose lower-cased type occurs in the reference sample
Writes results/corpus_stats.json.
"""
import collections, json, re
import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
P = "datasets/prepared"
N_TOK = 2_000_000


def counts(docs):
    tc, wc, n = collections.Counter(), collections.Counter(), 0
    for d in docs:
        ids = tok(d, add_special_tokens=False)["input_ids"]
        tc.update(ids); n += len(ids)
        wc.update(re.findall(r"[A-Za-z]+", d.lower()))
        if n >= N_TOK:
            break
    return tc, wc


def jsd(p, q):
    keys = list(set(p) | set(q))
    a = np.array([p.get(k, 0) for k in keys], float); b = np.array([q.get(k, 0) for k in keys], float)
    a /= a.sum(); b /= b.sum(); m = (a + b) / 2
    kl = lambda x, y: np.sum(x[x > 0] * np.log2(x[x > 0] / y[x > 0]))
    return float((kl(a, m) + kl(b, m)) / 2)


load = lambda dom: [json.loads(l) for l in open(f"{P}/train_docs_{dom}.jsonl")]
gen_docs = load("general")
ref_t, ref_w = counts(gen_docs[::-1])
top10k = {t for t, _ in ref_t.most_common(10000)}
out = {}
for dom in ["medical", "code", "general"]:
    t, w = counts(gen_docs if dom == "general" else load(dom))
    tot, wtot = sum(t.values()), sum(w.values())
    out[dom] = {"jsd_vs_wikitext": jsd(t, ref_t),
                "top10k_token_mass": sum(v for k, v in t.items() if k in top10k) / tot,
                "word_coverage": sum(v for k, v in w.items() if k in ref_w) / wtot,
                "n_tokens": tot, "n_token_types": len(t)}
    print(dom, out[dom], flush=True)
json.dump(out, open("results/corpus_stats.json", "w"), indent=1)
