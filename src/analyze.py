"""Hypothesis tests + figures from results/all_metrics.csv (produced by src/metrics.py).

H1  crossing steps of kNN@10 preservation below 0.5 / 0.6 by model size
H2  monotonicity of isotropy metrics over steps
H3  preservation vs general-knowledge forgetting (0.6 threshold, lead/lag)
H4  medical vs code preservation; vocabulary overlap
Outputs: results/analysis/*.csv|json, figures/*.png
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

os.makedirs("results/analysis", exist_ok=True); os.makedirs("figures", exist_ok=True)
df = pd.read_csv("results/all_metrics.csv")
SIZE = {"160M": 0.16, "410M": 0.41, "0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0}
df["size_b"] = df.model.map(SIZE)
df["family"] = np.where(df.run.str.startswith("p"), "Pythia", "Qwen2.5")
df["final"] = df.groupby("run").layer.transform("max") == df.layer
df["gk"] = (df.mmlu * 570 + df.arc_easy * 500) / 1070  # pooled general-knowledge accuracy (1070 items)
base = df[df.step == 0].set_index(["run", "probe", "layer"])
for c in ["mmlu", "arc_easy", "gk", "nll_general"]:
    df["d_" + c] = df[c].values - base.loc[list(zip(df.run, df.probe, df.layer)), c].values
G = df[(df.probe == "general") & df.final].copy()  # primary view: general probes, final layer
G.to_csv("results/analysis/primary_view.csv", index=False)
out = {}
COL = {"medical": "#1f77b4", "code": "#d62728", "general": "#7f7f7f"}
MCOL = {"160M": "#e6a700", "410M": "#e66100", "0.5B": "#5e3c99", "1.5B": "#1b9e77", "3B": "#2166ac", "7B": "#000000"}


def crossing(steps, vals, thr):
    """First step where the curve goes below thr (linear interpolation in log(1+step)); None if never."""
    steps, vals = np.asarray(steps, float), np.asarray(vals, float)
    for i in range(1, len(steps)):
        if vals[i] < thr <= vals[i - 1]:
            x0, x1 = np.log1p(steps[i - 1]), np.log1p(steps[i])
            return float(np.expm1(x0 + (vals[i - 1] - thr) / (vals[i - 1] - vals[i]) * (x1 - x0)))
    return None


# ---------------- H1: crossing steps
rows = []
for (run, layer), g in df[(df.probe == "general")].groupby(["run", "layer"]):
    g = g.sort_values("step")
    r = {"run": run, "model": g.model.iloc[0], "family": g.family.iloc[0], "method": g.method.iloc[0],
         "domain": g.domain.iloc[0], "seed": g.seed.iloc[0], "variant": g.variant.iloc[0],
         "layer": layer, "final_layer": bool(g.final.iloc[0]), "max_step": int(g.step.max())}
    for m in ["pres10", "pres10_centered"]:
        for thr in [0.5, 0.6]:
            r[f"{m}_cross{thr}"] = crossing(g.step, g[m], thr)
        r[f"{m}_at500"] = float(g[g.step == 500][m].iloc[0]) if (g.step == 500).any() else None
        r[f"{m}_at_max"] = float(g[m].iloc[-1]); r[f"{m}_min"] = float(g[m].min())
    rows.append(r)
cross = pd.DataFrame(rows).sort_values(["family", "model", "domain", "method", "variant", "seed", "layer"])
cross.to_csv("results/analysis/h1_crossings.csv", index=False)
print(cross[cross.final_layer][["run", "max_step", "pres10_at500", "pres10_at_max", "pres10_cross0.5",
                                "pres10_cross0.6", "pres10_centered_at_max"]].to_string())

# Fig 1: preservation trajectories
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, dom in zip(axes, ["medical", "code", "general"]):
    for run, g in G[(G.domain == dom) & (G.variant == "base")].groupby("run"):
        g = g.sort_values("step")
        ls = "-" if g.method.iloc[0] == "lora" else "--"
        alpha = 1.0 if g.seed.iloc[0] == 1 else 0.35
        ax.plot(g.step + 1, g.pres10, ls, marker="o", ms=3, alpha=alpha, color=MCOL[g.model.iloc[0]],
                label=f"{g.family.iloc[0]}-{g.model.iloc[0]} {g.method.iloc[0]}" if g.seed.iloc[0] == 1 else None)
    ax.axhline(0.5, color="k", lw=0.8, ls=":"); ax.axhline(0.6, color="gray", lw=0.8, ls=":")
    ax.axvline(501, color="k", lw=0.6, alpha=0.4); ax.axvline(2001, color="k", lw=0.6, alpha=0.4)
    ax.set_xscale("log"); ax.set_title(f"{dom} fine-tuning"); ax.set_xlabel("training step + 1 (log)")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
axes[0].set_ylabel("kNN@10 neighbourhood preservation\n(general probes, final layer)")
axes[0].legend(fontsize=7, loc="lower left"); axes[1].text(1.2, 0.03, "faint = seeds 2,3; dashed = full FT; lr 2e-4 (LoRA) / 1e-5 (full)", fontsize=7)
plt.tight_layout(); plt.savefig("figures/fig1_preservation_trajectories.png", dpi=150); plt.close()

# ---------------- H2: isotropy monotonicity
rows = []
for (run, probe, layer), g in df[df.probe == "general"].groupby(["run", "probe", "layer"]):
    g = g.sort_values("step")
    r = {"run": run, "layer": layer, "final_layer": bool(g.final.iloc[0]), "domain": g.domain.iloc[0],
         "model": g.model.iloc[0], "variant": g.variant.iloc[0], "method": g.method.iloc[0]}
    for m, aniso_sign in [("avg_cos", +1), ("isoscore", -1), ("eff_rank", -1)]:
        v = g[m].values
        rho, p = stats.spearmanr(g.step, v)
        d = np.diff(v) * aniso_sign  # >0 means "more anisotropic" at the next checkpoint
        r[f"{m}_rho"] = rho; r[f"{m}_p"] = p
        r[f"{m}_frac_steps_more_aniso"] = float(np.mean(d > 0))
        r[f"{m}_rel_change"] = float((v[-1] - v[0]) / abs(v[0]))
        r[f"{m}_monotone_aniso"] = bool(np.all(d >= 0))
    rows.append(r)
iso = pd.DataFrame(rows)
iso.to_csv("results/analysis/h2_isotropy.csv", index=False)
isof = iso[iso.final_layer]
out["h2"] = {"n_runs": int(len(isof)),
             "n_monotone_more_anisotropic": {m: int(isof[f"{m}_monotone_aniso"].sum()) for m in ["avg_cos", "isoscore", "eff_rank"]},
             "n_net_more_anisotropic": {"avg_cos": int((isof.avg_cos_rel_change > 0).sum()),
                                        "isoscore": int((isof.isoscore_rel_change < 0).sum()),
                                        "eff_rank": int((isof.eff_rank_rel_change < 0).sum())},
             "median_rel_change": {m: float(isof[f"{m}_rel_change"].median()) for m in ["avg_cos", "isoscore", "eff_rank"]}}
print(json.dumps(out["h2"], indent=1))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, m, lab in zip(axes, ["avg_cos", "isoscore", "eff_rank"],
                      ["mean pairwise cosine (↑ = more anisotropic)", "IsoScore (↓ = less isotropic)",
                       "effective rank (↓ = less isotropic)"]):
    for run, g in G[(G.variant == "base") & (G.seed == 1)].groupby("run"):
        g = g.sort_values("step")
        ax.plot(g.step + 1, g[m] / g[m].iloc[0], marker="o", ms=2, color=COL[g.domain.iloc[0]],
                ls="-" if g.method.iloc[0] == "lora" else "--", alpha=0.4 + 0.1 * np.log1p(g.size_b.iloc[0] * 4))
    ax.set_xscale("log"); ax.set_title(lab, fontsize=10); ax.set_xlabel("step + 1 (log)"); ax.grid(alpha=0.3)
    ax.set_ylabel("relative to step 0")
from matplotlib.lines import Line2D
axes[0].legend([Line2D([], [], color=c) for c in COL.values()], list(COL), fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig2_isotropy.png", dpi=150); plt.close()

# ---------------- H3: preservation vs forgetting
# item-level McNemar test of each checkpoint vs step 0 on the 1070 MMLU+ARC items
def items(run, step):
    f = f"results/runs/{run}/reps/items_{step:05d}.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f)); return np.array(d["mmlu"] + d["arc"])

mc = []
for run, g in G.groupby("run"):
    i0 = items(run, 0)
    for s in sorted(g.step):
        i1 = items(run, s)
        if i0 is None or i1 is None:
            continue
        b, c = int(((i0 == 1) & (i1 == 0)).sum()), int(((i0 == 0) & (i1 == 1)).sum())
        p = stats.binomtest(b, b + c).pvalue if b + c > 0 else 1.0
        mc.append({"run": run, "step": s, "lost": b, "gained": c, "mcnemar_p": p,
                   "sig_decline": bool(p < 0.05 and b > c)})
mc = pd.DataFrame(mc)
G = G.merge(mc, on=["run", "step"], how="left")
G.to_csv("results/analysis/primary_view.csv", index=False)

h3 = G[(G.step > 0)].dropna(subset=["pres10", "d_gk"]).copy()
hi, lo = h3[h3.pres10 >= 0.6], h3[h3.pres10 < 0.6]
out["h3"] = {"n_ckpt_pres_ge_0.6": int(len(hi)), "n_ckpt_pres_lt_0.6": int(len(lo)),
             "mean_d_gk_pres_ge_0.6": float(hi.d_gk.mean()) if len(hi) else None,
             "mean_d_gk_pres_lt_0.6": float(lo.d_gk.mean()) if len(lo) else None,
             "frac_sig_decline_pres_ge_0.6": float(hi.sig_decline.mean()) if len(hi) else None,
             "frac_sig_decline_pres_lt_0.6": float(lo.sig_decline.mean()) if len(lo) else None}
if len(hi) and len(lo):
    out["h3"]["mwu_p"] = float(stats.mannwhitneyu(hi.d_gk, lo.d_gk).pvalue)
rho, p = stats.spearmanr(h3.pres10, h3.d_gk); out["h3"]["spearman_pres_vs_dgk_all"] = [rho, p]
rho, p = stats.spearmanr(h3.pres10, h3.d_nll_general); out["h3"]["spearman_pres_vs_dnllgen_all"] = [rho, p]
# per-run: first significant decline vs preservation at that moment
per = []
for run, g in G.groupby("run"):
    g = g.sort_values("step")
    sig = g[g.sig_decline == True]
    first = int(sig.step.iloc[0]) if len(sig) else None
    # sustained decline: first step after which every later checkpoint is below step-0 accuracy
    below = (g.d_gk < 0).values
    sustained = next((int(s) for k, s in enumerate(g.step) if s > 0 and below[k:].all()), None)
    rr, pp = stats.spearmanr(g.pres10[g.step > 0], g.d_gk[g.step > 0])
    per.append({"run": run, "first_sig_decline_step": first,
                "pres10_at_first_sig_decline": float(g[g.step == first].pres10.iloc[0]) if first is not None else None,
                "sustained_decline_from_step": sustained,
                "pres10_at_sustained": float(g[g.step == sustained].pres10.iloc[0]) if sustained is not None else None,
                "d_gk_at_max_step": float(g.d_gk.iloc[-1]), "d_mmlu_at_max": float(g.d_mmlu.iloc[-1]),
                "d_arc_at_max": float(g.d_arc_easy.iloc[-1]), "d_nll_general_at_max": float(g.d_nll_general.iloc[-1]),
                "pres10_cross0.6": crossing(g.step, g.pres10, 0.6), "within_run_spearman": rr, "within_run_p": pp,
                "max_step": int(g.step.max())})
per = pd.DataFrame(per); per.to_csv("results/analysis/h3_per_run.csv", index=False)
print(per.to_string())
print(json.dumps(out["h3"], indent=1))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for run, g in G[G.variant == "base"].groupby("run"):
    g = g.sort_values("step")
    for ax, y in zip(axes, ["d_gk", "d_nll_general"]):
        ax.plot(g.pres10, g[y] * (100 if y == "d_gk" else 1), marker={"medical": "o", "code": "s", "general": "^"}[g.domain.iloc[0]],
                ms=3, alpha=0.55, lw=0.9, color=MCOL[g.model.iloc[0]], ls="-" if g.method.iloc[0] == "lora" else "--")
for ax, lab in zip(axes, ["Δ general-knowledge acc (pp, MMLU+ARC-E)", "Δ WikiText NLL (nats/token)"]):
    ax.axvline(0.6, color="k", ls=":", lw=1); ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("kNN@10 preservation (general probes, final layer)"); ax.set_ylabel(lab); ax.invert_xaxis()
    ax.grid(alpha=0.3)
hs = [Line2D([], [], color=c) for m, c in MCOL.items() if m in set(G.model)]
axes[0].legend(hs + [Line2D([], [], marker=k, ls="", color="k") for k in "os^"],
               [m for m in MCOL if m in set(G.model)] + ["medical", "code", "general"], fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig("figures/fig3_preservation_vs_forgetting.png", dpi=150); plt.close()

# ---------------- H4: domain comparison (matched model/seed/method/variant/step)
piv = G[G.step > 0].pivot_table(index=["model", "method", "variant", "seed", "step"], columns="domain", values="pres10")
if {"medical", "code"} <= set(piv.columns):
    pv = piv.dropna(subset=["medical", "code"])
    diff = pv.medical - pv.code
    out["h4"] = {"n_matched_ckpts": int(len(pv)), "mean_diff_med_minus_code": float(diff.mean()),
                 "frac_med_gt_code": float((diff > 0).mean()),
                 "wilcoxon_p": float(stats.wilcoxon(diff).pvalue) if len(pv) > 5 else None}
    # run-level (last common step) comparison, one value per model/method/variant/seed
    last = pv.reset_index().sort_values("step").groupby(["model", "method", "variant", "seed"]).tail(1)
    last["diff"] = last.medical - last.code
    last.to_csv("results/analysis/h4_last_common_step.csv", index=False)
    out["h4"]["per_run_last_common_step"] = last.round(4).to_dict("records")
    for m in ["pres10_centered", "cka", "d_gk", "d_nll_general"]:
        pm = G[G.step > 0].pivot_table(index=["model", "method", "variant", "seed", "step"], columns="domain",
                                       values=m).dropna(subset=["medical", "code"])
        out["h4"][f"mean_diff_{m}"] = float((pm.medical - pm.code).mean())
        out["h4"][f"wilcoxon_p_{m}"] = float(stats.wilcoxon(pm.medical - pm.code).pvalue) if len(pm) > 5 else None
out["corpus_stats"] = json.load(open("results/corpus_stats.json"))
# in-domain probes: how much do medical/code probe neighbourhoods move under each fine-tuning domain?
dom_probe = df[df.final & (df.step > 0)].groupby(["run", "probe"]).apply(
    lambda g: g.sort_values("step").pres10.iloc[-1], include_groups=False).unstack()
dom_probe.to_csv("results/analysis/probe_set_preservation_at_max_step.csv")
print(dom_probe.round(3).to_string())
print(json.dumps({k: v for k, v in out.get("h4", {}).items() if k != "per_run_last_common_step"}, indent=1))

# Fig 4: domain comparison at matched steps
fig, ax = plt.subplots(figsize=(7, 4.6))
for (model, method, variant), g in G[(G.seed == 1) & (G.variant == "base")].groupby(["model", "method", "variant"]):
    for dom in ["medical", "code", "general"]:
        h = g[g.domain == dom].sort_values("step")
        if len(h):
            ax.plot(h.step + 1, h.pres10, color=COL[dom], ls="-" if method == "lora" else "--", alpha=0.6)
ax.set_xscale("log"); ax.set_xlabel("step + 1 (log)"); ax.set_ylabel("kNN@10 preservation"); ax.grid(alpha=0.3)
ax.legend([Line2D([], [], color=c) for c in COL.values()], list(COL), fontsize=8)
ax.set_title("Domain comparison (all models, seed 1)")
plt.tight_layout(); plt.savefig("figures/fig4_domain_comparison.png", dpi=150); plt.close()

# Fig 5: preservation vs model size at fixed steps (medical, standard recipe: LoRA lr 2e-4), both families
q = G[(G.method == "lora") & (G.variant == "base") & (G.domain == "medical")]
fig, ax = plt.subplots(figsize=(7, 4.4))
for s_, c in zip([100, 500, 1000, 2000, 3000], plt.cm.viridis(np.linspace(0, 0.9, 5))):
    for fam, mk, ls in [("Qwen2.5", "o", "-"), ("Pythia", "s", "--")]:
        h = q[(q.family == fam) & (q.step == s_)].groupby("size_b").pres10.agg(["mean", "std"]).reset_index()
        if len(h):
            ax.errorbar(h.size_b, h["mean"], yerr=h["std"].fillna(0), marker=mk, ls=ls, color=c, capsize=3,
                        label=f"step {s_}" if fam == "Qwen2.5" else None)
ax.set_xscale("log"); ax.set_xlabel("model size (B params, log)"); ax.grid(alpha=0.3)
ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.set_ylabel("kNN@10 preservation (medical FT)")
ax.set_title("Preservation vs size (circles/solid: Qwen2.5, squares/dashed: Pythia)", fontsize=9)
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig5_preservation_vs_size.png", dpi=150); plt.close()

# Fig 6: recipe ablations (update magnitude): Qwen-0.5B LoRA lr 2e-4 vs 1e-3; Pythia-160M LoRA (2e-4) vs full FT (1e-5)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, model in zip(axes, ["0.5B", "160M"]):
    for (dom, method, variant, seed), g in G[(G.model == model) & (G.domain != "general")].groupby(["domain", "method", "variant", "seed"]):
        if seed != 1:
            continue
        g = g.sort_values("step")
        lr = variant.replace("lr", "") if variant.startswith("lr") else ("2e-4" if method == "lora" else "1e-5")
        ls = {"2e-4": "-", "5e-4": "-.", "1e-3": ":", "1e-5": "--"}[lr]
        ax.plot(g.step + 1, g.pres10, marker="o", ms=3, color=COL[dom], ls=ls, label=f"{dom} {method} lr {lr}")
    ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    ax.set_title(f"{'Qwen2.5' if model == '0.5B' else 'Pythia'}-{model}: training recipe"); ax.set_xlabel("step + 1 (log)")
    ax.legend(fontsize=8)
axes[0].set_ylabel("kNN@10 preservation")
plt.tight_layout(); plt.savefig("figures/fig6_recipe_ablation.png", dpi=150); plt.close()

# Table: preservation mean ± sd over seeds at selected steps (primary view)
tab = G[G.step.isin([10, 100, 500, 1000, 2000, 3000])].groupby(["family", "model", "method", "variant", "domain", "step"]).agg(
    pres10_mean=("pres10", "mean"), pres10_sd=("pres10", "std"), n_seeds=("pres10", "size"),
    d_gk_mean=("d_gk", "mean"), d_mmlu_mean=("d_mmlu", "mean"), d_arc_mean=("d_arc_easy", "mean"),
    d_nll_gen_mean=("d_nll_general", "mean")).reset_index()
tab.to_csv("results/analysis/table_by_step.csv", index=False)
# mid-layer robustness: crossings / values at max step for mid layer
mid = cross[~cross.final_layer][["run", "max_step", "pres10_at500", "pres10_at_max", "pres10_cross0.5", "pres10_cross0.6"]]
mid.to_csv("results/analysis/h1_mid_layer.csv", index=False)
print(mid.to_string())
# base-model familiarity with each domain (step-0 NLL on held-out domain text)
fam = df[(df.step == 0) & (df.probe == "general") & df.final].groupby(["model", "domain"]).nll_domain.first().unstack()
fam.to_csv("results/analysis/base_domain_nll.csv"); print(fam.round(3))

# ---------------- Training health: did fine-tuning actually reduce held-out in-domain loss?
# "learned": final in-domain NLL < base - 0.05; "no gain": within ±0.05..+0.25; "diverged": > base + 0.25
hl = []
for run, g in G.groupby("run"):
    g = g.sort_values("step")
    d_dom = float(g.nll_domain.iloc[-1] - g.nll_domain.iloc[0])
    status = "learned" if d_dom < -0.05 else ("diverged" if d_dom > 0.25 else "no gain")
    hl.append({"run": run, "model": g.model.iloc[0], "domain": g.domain.iloc[0], "method": g.method.iloc[0],
               "variant": g.variant.iloc[0], "max_step": int(g.step.max()), "nll_domain_step0": float(g.nll_domain.iloc[0]),
               "nll_domain_final": float(g.nll_domain.iloc[-1]), "d_nll_domain": d_dom, "status": status,
               "pres10_final": float(g.pres10.iloc[-1]), "pres10_min": float(g.pres10.min()),
               "d_gk_final_pp": float(g.d_gk.iloc[-1] * 100)})
hl = pd.DataFrame(hl).sort_values(["status", "pres10_final"]); hl.to_csv("results/analysis/training_health.csv", index=False)
print(hl.round(3).to_string())
print(hl.groupby("status").agg(n=("run", "size"), pres10_final_min=("pres10_final", "min"),
                               pres10_final_max=("pres10_final", "max"), d_gk_mean=("d_gk_final_pp", "mean")).round(3))
