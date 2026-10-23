# Embedding-Space Drift Under Fine-Tuning

Step-resolved study of how quickly LoRA / full fine-tuning on medical, code or general text changes the
representation geometry of pretrained LMs. Five base models from 160M to 3B (Qwen2.5 and Pythia), 22 analysed runs,
17 checkpoints each. Geometry is measured as kNN neighbourhood preservation, CKA and isotropy on 1,000 fixed
general-text probes. We relate it to forgetting on MMLU + ARC-Easy and to WikiText perplexity.
Full write-up: **[REPORT.md](REPORT.md)**.

## Key findings
- **Larger models drift less, but the training recipe matters more than size.** At step 500, preservation@10
  was 0.42 (Pythia-160M), 0.74 (410M), 0.79 (Qwen-0.5B), 0.82 (1.5B) and 0.85 (3B). Qwen-0.5B never went below
  0.71 in 3,000 steps at lr 2e-4, but ended at 0.57 with lr 5e-4 and crossed 0.5 at ~420–500 steps with lr 1e-3.
- **Preservation < 0.5 occurred only in runs where training was unstable.** In those runs the held-out
  in-domain loss went *up*: Pythia-160M LoRA and Qwen-0.5B at lr 1e-3. Runs that actually learned the domain
  ended at 0.57–0.82. 7B could not be run on the shared GPUs.
- **Isotropy does not fall monotonically** (0/22 runs). The net drift toward anisotropy is small (median
  IsoScore −8 %).
- **There is no 60 % early-warning threshold.** Qwen models lose general knowledge (significant McNemar
  declines, up to −10 pp) while preservation is still 0.77–0.90. Preservation does track WikiText-NLL damage
  closely (ρ = −0.90).
- **Code, not medical, preserves more general geometry** in most model groups (144 matched checkpoints, p = 3e-12), even though clinical
  text overlaps lexically far more with general text. Caveat: the code fine-tuning did not reduce code loss
  (the base models already model code well).

## Reproduce
```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch transformers peft datasets accelerate numpy pandas scipy matplotlib scikit-learn statsmodels
python src/download_models.py            # Qwen2.5-0.5B/3B/7B base (others assumed in HF cache)
# code corpus + ARC-Easy parquet: see datasets/README.md
python src/prep_data.py                  # probes, eval sets, training pools -> datasets/prepared/
python src/corpus_stats.py               # vocabulary-overlap statistics
python src/train_eval.py --model Qwen/Qwen2.5-0.5B --domain medical --seed 1 --out results/runs/q0.5b_lora_medical_s1
#   (job lists used: logs/jobs_*.txt; queue runners: src/worker*.sh)
python src/metrics.py                    # geometry metrics for all runs -> results/all_metrics.csv
python src/analyze.py                    # hypothesis tests, tables (results/analysis/) and figures (figures/)
```
A Qwen-0.5B run takes ~20 min on a free 5 GB GPU slice, and Pythia-160M ~7 min.

## Layout
- `src/prep_data.py`: fixed probe/eval/training sets.
- `src/train_eval.py`: fine-tuning with checkpoint evaluation (stores representations and item-level accuracy).
- `src/metrics.py`: kNN preservation, CKA, isotropy.
- `src/analyze.py`: tests for H1–H4 and the figures.
- `src/corpus_stats.py`: vocabulary overlap.
- `src/worker*.sh`, `src/gpu_hold.py`: GPU queueing on the shared machine.
- `results/runs/<run>/`: config, `metrics.jsonl`, per-checkpoint representations and item-level results.
- `results/analysis/`: summary tables and JSON.
- `figures/`: fig1–fig6.
- `planning.md`: pre-registered plan.
- `REPORT.md`: full results.
