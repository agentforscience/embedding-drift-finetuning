#!/bin/bash
# one-off chain on GPU0: wait for 410M medical, then run 410M code (lock-claimed like the workers)
cd /data/summereunann/neurico-workspaces/embedding_drift_finetuning_20261022_120002_uv12wx34
source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_HOME=/data/transformers PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "pythia-410m --domain medical" >/dev/null; do sleep 20; done
mkdir results/locks/p410m_lora_code_s1 || exit 0
rm -rf results/runs/p410m_lora_code_s1
CUDA_VISIBLE_DEVICES=0 python src/train_eval.py --model EleutherAI/pythia-410m --domain code --seed 1 --eval_bs 8 --out results/runs/p410m_lora_code_s1 > logs/runs/p410m_lora_code_s1.log 2>&1
