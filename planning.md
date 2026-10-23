# Planning: Embedding-Space Drift Under Fine-Tuning

## Motivation & Novelty Assessment

### Why This Research Matters
Domain fine-tuning is the standard way to specialise LLMs, but it causes catastrophic forgetting of general
capabilities. Benchmarking general ability after every checkpoint is expensive, while measuring the geometry of a
model's representations on a fixed probe set is cheap and label-free. If geometric drift (e.g. loss of
nearest-neighbour structure) reliably precedes or tracks forgetting, it could serve as an early-warning monitor
during fine-tuning, and it would tell practitioners how model scale and target domain affect how fast a model
"forgets its pretraining geometry".

### Gap in Existing Work
No literature review or papers were pre-gathered for this workspace (the resource-finder stage was skipped), so
this assessment rests on background knowledge rather than a systematic review. Related work: representation
similarity across fine-tuning (CKA/SVCCA studies of BERT fine-tuning, e.g. Merchant et al. 2020 "What happens to
BERT embeddings during fine-tuning?"), anisotropy of contextual embeddings (Ethayarajh 2019; Mu & Viswanath 2018;
IsoScore, Rudman et al. 2022), and forgetting in LLM continual fine-tuning (Luo et al. 2023; Kalajdzievski 2024
"Scaling laws for forgetting"; Biderman et al. 2024 "LoRA learns less and forgets less"). Gaps: (i) those studies
mostly measure similarity at the end of training, not as a *step-resolved trajectory*; (ii) they rarely relate
local neighbourhood structure (kNN preservation) quantitatively to forgetting of general knowledge; (iii) scale
comparisons within one model family with an identical recipe are rare; (iv) the claim that domain vocabulary
overlap with pretraining data predicts geometric preservation is untested.

### Our Novel Contribution
A step-resolved study (17 checkpoints, 0–3000 steps) of four same-family base models (Qwen2.5 0.5B/1.5B/3B/7B)
fine-tuned with an identical LoRA recipe on medical, code and general (control) text, tracking kNN neighbourhood
preservation, isotropy, CKA, and general-knowledge accuracy (MMLU, ARC-Easy) + general perplexity at every
checkpoint. We test four specific, pre-registered quantitative claims (below) rather than just reporting trends.

### Experiment Justification
- **Exp 1 – Scale × steps (core):** LoRA fine-tuning of 0.5B/1.5B/3B/7B on medical and code text with probe
  evaluation at 17 checkpoints. Needed to test "<50% preservation within 500 steps for sub-1B, 2000+ for 7B".
- **Exp 2 – Isotropy trajectories:** same checkpoints, three isotropy measures (mean pairwise cosine, IsoScore,
  effective rank). Needed to test "isotropy decreases monotonically".
- **Exp 3 – Geometry vs forgetting:** MMLU/ARC-Easy/WikiText NLL per checkpoint, aligned to preservation. Needed to
  test the "60% preservation threshold" early-warning claim (does accuracy start to drop at ~60%? does
  preservation lead accuracy?).
- **Exp 4 – Domain comparison + vocabulary overlap:** medical vs code vs general-text control, plus corpus-level
  measures of overlap with general text (unigram JS divergence vs WikiText, base-model NLL). Needed to test
  "medical preserves more structure than code because of vocabulary overlap".
- **Robustness:** 3 seeds for 0.5B/1.5B (medical, code), multiple k for kNN, two layers (mid, final); if time,
  learning-rate and full-fine-tuning ablations for the small model (method/lr are confounds of "steps").

## Research Question
How fast, in optimisation steps, does fine-tuning destroy the neighbourhood structure and isotropy of a pretrained
LM's representation space, how does that depend on model scale and fine-tuning domain, and does the loss of
neighbourhood structure predict forgetting of general knowledge?

## Hypothesis Decomposition
- **H1 (scale/speed):** kNN@10 neighbourhood preservation (general-text probes, final layer) falls below 0.50
  by step 500 for the sub-1B model (0.5B) and does not fall below 0.50 before step 2000 for 7B.
- **H2 (isotropy):** isotropy decreases monotonically with steps (mean pairwise cosine rises; IsoScore/effective
  rank fall); tested by Spearman ρ(step, metric) and the fraction of monotonicity violations.
- **H3 (early warning):** general-knowledge accuracy (MMLU, ARC-E) is stable while preservation ≥ 0.60 and starts
  declining once it drops below 0.60; i.e. a 0.60 threshold separates "no forgetting" from "forgetting"
  checkpoints, and preservation changes precede accuracy changes.
- **H4 (domain):** medical fine-tuning preserves more structure (higher preservation at matched steps) than code
  fine-tuning; medical text is closer to general text in vocabulary.

Independent variables: model size (0.5/1.5/3/7B), domain (medical/code/general), training step, seed.
Dependent variables: kNN preservation (k=5/10/20), CKA, isotropy metrics, MMLU, ARC-E, NLL on WikiText, domain NLL.

## Proposed Methodology

### Approach
Same base-model family (Qwen2.5 base, shared tokenizer/pretraining data) to isolate scale. LoRA (r=64, α=128, all
linear layers, lr 2e-4, constant after 50 warm-up steps, AdamW β=(0.9,0.95), batch 4×512 tokens, 3000 steps =
6.1M tokens) is used for every size because full fine-tuning of 7B is infeasible on the shared GPUs (only one
GPU slice ≥ 30 GB available); identical recipe across sizes makes "steps" comparable. Representations: mean-pooled
hidden states (positions ≥1, 128-token truncation) of 1000 held-out WikiText-103 paragraphs (general probes) plus
300 medical and 300 code probes, at the middle and final layer.

### Experimental Steps
1. Prepare fixed probe sets, held-out eval texts, MMLU (570 q, 10/subject) and ARC-Easy (500 q) subsets.
2. For each run, evaluate at steps {0,10,25,50,100,150,200,300,400,500,750,1000,1250,1500,2000,2500,3000};
   store representations, item-level accuracy and NLLs.
3. Offline: compute kNN preservation vs step 0, CKA, isotropy; threshold-crossing steps (linear interpolation in
   log-step); relate to accuracy.
4. Corpus analysis: unigram token distribution JS divergence domain↔WikiText, OOV-ish rates, base-model NLL.

### Baselines
- Step 0 (pretrained model) is the reference for all geometric metrics.
- General-text (WikiText) fine-tuning as a control domain: isolates the effect of *being fine-tuned* from the
  effect of *domain shift*.
- Noise floor: preservation between two seeds' runs / bootstrap over probes.

### Evaluation Metrics
- kNN preservation@k: mean over probes of |N_k^base(i) ∩ N_k^t(i)| / k (cosine similarity).
- Linear CKA (global similarity).
- Isotropy: mean pairwise cosine (anisotropy), IsoScore, effective rank (exp of entropy of normalised
  covariance eigenvalues).
- Forgetting: Δ MMLU, Δ ARC-E accuracy vs step 0 (McNemar / paired bootstrap on items), Δ WikiText NLL.

### Statistical Analysis Plan
- Crossing steps with seed variability (3 seeds for 0.5B/1.5B) and bootstrap CIs over probes.
- Paired item-level tests (McNemar exact) for accuracy change vs step 0; bootstrap 95% CIs.
- H3: compare accuracy deltas for checkpoints with preservation ≥0.6 vs <0.6 (Mann–Whitney), plus
  Spearman/Kendall correlation between preservation and Δaccuracy across checkpoints, and a lead/lag check.
- H2: Spearman ρ(step, metric) per run; count monotonicity violations beyond bootstrap noise.
- α = 0.05; Holm correction across the four hypotheses' primary tests.

## Expected Outcomes
H1 supported if 0.5B crosses 0.5 by ≤500 steps and 7B only after ≥2000. H2 supported if all isotropy metrics move
monotonically toward anisotropy. H3 supported if accuracy drops coincide with preservation <0.6 and not before.
H4 supported if medical preservation > code preservation at matched steps, consistently across sizes.

## Timeline and Milestones
Setup+data 30 min; runs ~3–5 h (7B LoRA dominant; shared GPUs); analysis 1 h; report 45 min.

## Potential Challenges
- Shared, contended GPUs (5 GB slices; one ~30 GB slice): queue with retries, small loss chunks, prioritise 7B.
- LoRA vs full FT and fixed lr are confounds for "steps"; documented and partially ablated.
- Probe pooling/layer choice affects metrics: report two layers and several k.

## Success Criteria
All four hypotheses receive a clear supported / refuted / partially supported verdict backed by step-resolved data
for ≥3 model sizes and ≥2 domains, with variance estimates.
