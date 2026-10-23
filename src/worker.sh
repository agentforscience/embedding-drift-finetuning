#!/bin/bash
# Work-stealing worker: claims jobs from a jobs file (one "<tag> <args...>" per line) via atomic mkdir locks.
# Usage: src/worker.sh <gpu> <jobs_file> ; failed jobs are released and retried after a pause.
GPU=$1; JOBS=$2
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_HOME=/data/transformers PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/runs results/locks logs/runs
while true; do
  claimed=""
  while read -r tag args; do
    [ -z "$tag" ] && continue
    [ -f results/runs/$tag/DONE ] && continue
    if mkdir results/locks/$tag 2>/dev/null; then claimed=$tag; jargs=$args; break; fi
  done < "$JOBS"
  [ -z "$claimed" ] && { echo "[$GPU] no jobs left"; exit 0; }
  echo "[$GPU] $(date +%T) start $claimed"
  rm -rf results/runs/$claimed
  CUDA_VISIBLE_DEVICES=$GPU python src/train_eval.py $jargs --out results/runs/$claimed > logs/runs/$claimed.log 2>&1
  if [ -f results/runs/$claimed/DONE ]; then echo "[$GPU] $(date +%T) done $claimed"
  else echo "[$GPU] $(date +%T) FAILED $claimed"; rmdir results/locks/$claimed; sleep 90; fi
done
