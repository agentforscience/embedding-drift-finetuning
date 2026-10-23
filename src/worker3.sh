#!/bin/bash
# Work-stealing worker (v2): like worker.sh but only claims jobs whose estimated memory fits the GPU's free memory.
# Usage: src/worker2.sh <gpu> <jobs_file>
GPU=$1; JOBS=$2
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_HOME=/data/transformers PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/runs results/locks logs/runs
need_mb() { case "$1" in q0.5b*) echo 2700;; q1.5b*) echo 6600;; q3b*) echo 9500;; *) echo 20000;; esac; }
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
  claimed=""; remaining=0
  while read -r tag args; do
    [ -z "$tag" ] && continue
    [ -f results/runs/$tag/DONE ] && continue
    remaining=1
    [ "$free" -lt "$(need_mb $tag)" ] && continue
    if mkdir results/locks/$tag 2>/dev/null; then claimed=$tag; jargs=$args; break; fi
  done < "$JOBS"
  [ "$remaining" = 0 ] && { echo "[$GPU] no jobs left"; exit 0; }
  [ -z "$claimed" ] && { sleep 30; continue; }
  echo "[$GPU] $(date +%T) start $claimed (free ${free}MB)"
  rm -rf results/runs/$claimed
  CUDA_VISIBLE_DEVICES=$GPU python src/train_eval.py $jargs --out results/runs/$claimed > logs/runs/$claimed.log 2>&1
  if [ -f results/runs/$claimed/DONE ]; then echo "[$GPU] $(date +%T) done $claimed"
  else echo "[$GPU] $(date +%T) FAILED $claimed"; rmdir results/locks/$claimed; sleep 60; fi
done
