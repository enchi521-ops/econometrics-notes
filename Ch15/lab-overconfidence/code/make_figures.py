"""Generate the figures used in ../README.md.

    pip install -r ../requirements.txt
    python make_figures.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)
d = pd.read_csv(HERE.parent / "data" / "instruments.csv")

BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 10.5, "axes.edgecolor": "#b9b8b2", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "legend.frameon": False, "lines.linewidth": 2.0,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)


GROUPS = [(0, 0), (1, 0), (0, 1), (1, 1)]
LABELS = ["neither\ngreen", "only me\ngreen", "only partner\ngreen", "both\ngreen"]


def group_means(col):
    m, se, k = [], [], []
    for b, ob in GROUPS:
        s = d.loc[(d.ball == b) & (d.other_ball == ob), col]
        m.append(s.mean()); se.append(s.std(ddof=1) / np.sqrt(len(s))); k.append(len(s))
    return np.array(m), np.array(se), k


fs = smf.ols("sum_belief ~ ball + other_ball + score_p2 + other_score2", d).fit()
iv = IV2SLS.from_formula(
    "final_pot_perc ~ 1 + score_p2 + other_score2 + [sum_belief ~ ball + other_ball]", d
).fit(cov_type="unadjusted")
ols = smf.ols("final_pot_perc ~ sum_belief + score_p2 + other_score2", d).fit()
d["u1hat"] = iv.resids
aux = smf.ols("u1hat ~ ball + other_ball + score_p2 + other_score2", d).fit()

# 1 -- relevance: the signals move confidence --------------------------------------------
m, se, k = group_means("sum_belief")
fig, ax = plt.subplots(figsize=(6.6, 3.9))
x = np.arange(4)
ax.bar(x, m, color=[MUTED, BLUE, BLUE, BLUE], alpha=0.85, width=0.6)
ax.errorbar(x, m, yerr=1.96 * se, fmt="none", ecolor=INK, capsize=4, lw=1.2)
for xi, mi, si, ki in zip(x, m, se, k):
    ax.text(xi, mi + 1.96 * si + 3, f"{mi:.0f}\n(n={ki})", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(x, LABELS)
ax.set_ylim(0, 180)
ax.set_ylabel("sum of beliefs (0–200)")
ax.set_title("Stage 1 — each green ball adds about 30 points of confidence")
ax.text(-0.4, 168, f"first stage: ball {fs.params['ball']:.1f} (t = {fs.tvalues['ball']:.1f})\n"
                  f"other_ball {fs.params['other_ball']:.1f} (t = {fs.tvalues['other_ball']:.1f})\n"
                  f"joint F(2,144) = {float(fs.f_test('ball = 0, other_ball = 0').fvalue):.1f}",
        ha="left", va="top", fontsize=9, color=MUTED)
save(fig, "01_first_stage_relevance.png")

# 2 -- OLS vs 2SLS -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 2.9))
rows = [("OLS", ols.params["sum_belief"], ols.conf_int().loc["sum_belief"], MUTED),
        ("2SLS", iv.params["sum_belief"], iv.conf_int().loc["sum_belief"], ORANGE)]
for i, (lab, b, ci, c) in enumerate(rows):
    y = 1 - i
    ax.plot([ci.iloc[0], ci.iloc[1]], [y, y], color=c, lw=3, solid_capstyle="round")
    ax.plot(b, y, "o", color=c, ms=9)
    ax.text(b, y + 0.22, f"{b:.3f}  [{ci.iloc[0]:.3f}, {ci.iloc[1]:.3f}]", ha="center", fontsize=9.5, color=INK)
ax.axvline(0, color=INK, lw=0.9)
ax.set_yticks([1, 0], ["OLS", "2SLS"])
ax.set_ylim(-0.6, 1.6)
ax.set_xlim(-0.85, 0.15)
ax.set_xlabel("effect of +1 point of pair confidence on % of prize awarded (95% CI)")
ax.set_title("Instrumenting doubles the (negative) effect of confidence")
ax.grid(axis="y", visible=False)
save(fig, "02_ols_vs_2sls.png")

# 3 -- overidentification intuition: instruments should move y2, not the 2SLS residual ----
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), sharex=True)
m1, se1, _ = group_means("sum_belief")
m1 = m1 - d.sum_belief.mean()
m2, se2, _ = group_means("u1hat")
for ax, mm, ss, c, title, note in [
    (axes[0], m1, se1, BLUE, "Instruments vs confidence (y2)",
     f"partial R² of the IVs = {iv.first_stage.diagnostics.loc['sum_belief', 'partial.rsquared']:.2f}\n→ relevance"),
    (axes[1], m2, se2, ORANGE, "Instruments vs 2SLS residual (û1)",
     f"R² of û1 on all exogenous = {aux.rsquared:.3f}\nnR² = {len(d) * aux.rsquared:.2f}, p = {iv.sargan.pval:.2f}\n→ no sign the IVs are endogenous"),
]:
    ax.bar(np.arange(4), mm, color=c, alpha=0.85, width=0.6)
    ax.errorbar(np.arange(4), mm, yerr=1.96 * ss, fmt="none", ecolor=INK, capsize=4, lw=1.2)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_title(title)
    ax.set_xticks(np.arange(4), LABELS, fontsize=8.5)
    ax.set_ylim(-60, 75)
    ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", fontsize=8.8, color=MUTED)
axes[0].set_ylabel("group mean minus overall mean")
fig.suptitle("Overidentification test: a valid instrument shifts y2 but cannot predict the structural error",
             x=0.01, ha="left", fontsize=10.5, color=INK)
save(fig, "03_overid_intuition.png")

print("figures written to", OUT)
