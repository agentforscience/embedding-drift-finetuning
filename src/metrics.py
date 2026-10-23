"""Offline geometry metrics for every run: kNN neighbourhood preservation, CKA and isotropy per checkpoint.

Reads results/runs/<tag>/reps/step_*.npz (+ metrics.jsonl) and writes results/all_metrics.csv
(one row per run × step × probe-set × layer, with the run's accuracy / NLL metrics joined in).
"""
import glob, json, os
import numpy as np
import pandas as pd
import torch

KS = (5, 10, 20)
dev = "cuda" if torch.cuda.is_available() and os.environ.get("METRICS_GPU") else "cpu"


def knn_idx(X, k):
    """Indices of k nearest neighbours by cosine similarity (self excluded)."""
    X = torch.nn.functional.normalize(torch.as_tensor(X, dtype=torch.float64, device=dev), dim=1)
    S = X @ X.T
    S.fill_diagonal_(-np.inf)
    return S.topk(k, dim=1).indices.cpu().numpy()


def preservation(nb0, nb1):
    """Mean fraction of base-model neighbours retained."""
    k = nb0.shape[1]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(nb0, nb1)]))


def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    X = torch.as_tensor(X, dtype=torch.float64, device=dev); Y = torch.as_tensor(Y, dtype=torch.float64, device=dev)
    hsic = (X.T @ Y).norm() ** 2
    return float(hsic / ((X.T @ X).norm() * (Y.T @ Y).norm()))


def isotropy(X):
    """Return (mean pairwise cosine, IsoScore [Rudman et al. 2022], effective rank of covariance)."""
    X = torch.as_tensor(X, dtype=torch.float64, device=dev)
    Xn = torch.nn.functional.normalize(X, dim=1)
    n = X.shape[0]
    S = Xn @ Xn.T
    avg_cos = float((S.sum() - n) / (n * (n - 1)))
    Xc = X - X.mean(0)
    # eigenvalues of covariance via Gram matrix (n << d is typical)
    ev = torch.linalg.eigvalsh(Xc @ Xc.T / (n - 1)).clamp(min=0).flip(0)
    p = ev / ev.sum()
    eff_rank = float(torch.exp(-(p[p > 0] * torch.log(p[p > 0])).sum()))
    # IsoScore: variance vector in PCA basis, padded with zeros to ambient dimension d
    d = X.shape[1]
    var = torch.zeros(d, dtype=torch.float64, device=dev); var[:len(ev)] = ev[:d]
    vhat = np.sqrt(d) * var / var.norm()
    delta = (vhat - 1).norm() / np.sqrt(2 * (d - np.sqrt(d)))
    phi = (d - delta ** 2 * (d - np.sqrt(d))) ** 2 / d ** 2
    iso = float((d * phi - 1) / (d - 1))
    return avg_cos, iso, eff_rank


def parse_tag(tag):
    # e.g. q0.5b_lora_medical_s1, q0.5b_full_code_s1, q0.5b_lora_medical_s1_lr5e-4
    parts = tag.split("_")
    return {"model": parts[0][1:].upper(), "method": parts[1], "domain": parts[2], "seed": int(parts[3][1:]),
            "variant": "_".join(parts[4:]) or "base"}


def process_run(rdir, cache):
    tag = os.path.basename(rdir)
    files = sorted(glob.glob(f"{rdir}/reps/step_*.npz"))
    if not files:
        return []
    m = {json.loads(l)["step"]: json.loads(l) for l in open(f"{rdir}/metrics.jsonl")}
    # only checkpoints whose evaluation finished (reps are written before the accuracy evals)
    files = [f for f in files if int(f.split("_")[-1].split(".")[0]) in m]
    if tag in cache and cache[tag]["n"] == len(files):
        return cache[tag]["rows"]
    base = dict(np.load(files[0]))
    keys = list(base.keys())
    nb0 = {(key, k): knn_idx(base[key], k) for key in keys for k in KS}
    # centred variant: remove each space's own mean (the dominant common direction) before cosine kNN
    nb0c = {key: knn_idx(base[key] - base[key].astype(np.float64).mean(0), 10) for key in keys}
    rows = []
    for f in files:
        step = int(f.split("_")[-1].split(".")[0])
        cur = base if step == 0 else dict(np.load(f))
        for key in keys:
            probe, layer = key.split("_L")
            X0, X1 = base[key].astype(np.float64), cur[key].astype(np.float64)
            r = {"run": tag, **parse_tag(tag), "step": step, "probe": probe, "layer": int(layer)}
            for k in KS:
                r[f"pres{k}"] = preservation(nb0[(key, k)], knn_idx(X1, k))
            r["pres10_centered"] = preservation(nb0c[key], knn_idx(X1 - X1.mean(0), 10))
            r["cka"] = linear_cka(X0, X1)
            r["avg_cos"], r["isoscore"], r["eff_rank"] = isotropy(X1)
            r.update({kk: m.get(step, {}).get(kk) for kk in ["mmlu", "arc_easy", "nll_general", "nll_domain", "train_loss"]})
            rows.append(r)
    cache[tag] = {"n": len(files), "rows": rows}
    return rows


def main():
    cache_f = "results/geometry_cache.json"
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    rows = []
    for rdir in sorted(glob.glob("results/runs/*")):
        rows += process_run(rdir, cache)
        print(rdir, len(rows), flush=True)
    json.dump(cache, open(cache_f, "w"))
    df = pd.DataFrame(rows)
    df.to_csv("results/all_metrics.csv", index=False)
    print(df.groupby(["run"]).step.max())


if __name__ == "__main__":
    main()
