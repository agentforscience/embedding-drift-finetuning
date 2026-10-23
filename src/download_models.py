from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]:
    p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"])
    print(m, p, flush=True)
