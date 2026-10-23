# Datasets (not committed; large files ignored)

- `code/code.parquet` — codeparrot/codeparrot-clean-valid, train shard 0:
  `curl -L -o datasets/code/code.parquet "https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"`
- `arc/arc_easy_test.parquet` — allenai/ai2_arc ARC-Easy test:
  `curl -L -o datasets/arc/arc_easy_test.parquet "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Easy/test-00000-of-00001.parquet"`
- From the local HF cache (`/data/transformers/hub`): Salesforce/wikitext (wikitext-103-raw-v1),
  starmpcc/Asclepius-Synthetic-Clinical-Notes (synthetic.csv), cais/mmlu (all/test).
- `prepared/` — built by `python src/prep_data.py` (seeded, deterministic).
