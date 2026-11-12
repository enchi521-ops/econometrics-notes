"""Generate the figures used in README.md.

Run:  pip install wooldridge statsmodels matplotlib  &&  python code/make_figures.py
Simulated figures use a fixed seed; the others use the `wooldridge` data sets.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import wooldridge as woo

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 10.5, "axes.edgecolor": "#b9b8b2", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "legend.frameon": False, "lines.linewidth": 2.0,
})
rng = np.random.default_rng(7)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=170)
    plt.close(fig)


# 1 -- R2 vs adjusted R2 when pure-noise regressors are added ---------------------------
n = 60
x = rng.normal(size=n)
y = 1 + 0.6 * x + rng.normal(size=n)
junk = rng.normal(size=(n, 40))
ks, r2, ar2 = [], [], []
for k in range(0, 41):
    X = sm.add_constant(np.column_stack([x, junk[:, :k]]))
    res = sm.OLS(y, X).fit()
    ks.append(k); r2.append(res.rsquared); ar2.append(res.rsquared_adj)
fig, ax = plt.subplots(figsize=(6.6, 3.9))
ax.plot(ks, r2, color=BLUE, label="R²")
ax.plot(ks, ar2, color=ORANGE, label="adjusted R²")
ax.text(ks[-1] + 0.6, r2[-1], "R²", color=INK, va="center")
ax.text(ks[-1] + 0.6, ar2[-1], "adjusted R²", color=INK, va="center")
ax.set_xlim(0, 49); ax.set_xlabel("number of pure-noise regressors added (n = 60)"); ax.set_ylabel("goodness of fit")
ax.set_title("R² can only rise; adjusted R² charges for every regressor")
ax.legend(loc="upper left")
save(fig, "01_r2_vs_adjusted_r2.png")

# 2 -- retransformation bias when predicting y from a log(y) model ----------------------
c2 = woo.data("ceosal2")
lm = smf.ols("np.log(salary) ~ np.log(sales) + np.log(mktval) + ceoten", c2).fit()
alpha0 = float(np.exp(lm.resid).mean())
naive = np.exp(lm.fittedvalues)
fig, ax = plt.subplots(figsize=(6.6, 3.9))
order = np.argsort(naive.values)
ax.scatter(naive, c2.salary, s=14, color="#9a9992", alpha=0.55, linewidths=0, label="CEOs (actual salary)")
ax.plot(naive.values[order], naive.values[order], color=BLUE, label="naive: exp(fitted log salary)")
ax.plot(naive.values[order], alpha0 * naive.values[order], color=ORANGE, label=f"corrected: {alpha0:.3f} × exp(fitted)")
ax.set_ylim(0, 3200); ax.set_xlabel("exp(fitted log salary), $1,000s"); ax.set_ylabel("salary, $1,000s")
ax.set_title("Exponentiating a log prediction under-predicts the mean of y")
ax.legend(loc="upper left")
save(fig, "02_log_retransformation.png")

# 3 -- intercept shift (single dummy) ---------------------------------------------------
w = woo.data("wage1")
m = smf.ols("wage ~ female + educ", w).fit()
e = np.linspace(0, 18, 50)
fig, ax = plt.subplots(figsize=(6.6, 3.9))
ax.scatter(w.educ[w.female == 0], w.wage[w.female == 0], s=10, color=BLUE, alpha=0.25, linewidths=0)
ax.scatter(w.educ[w.female == 1], w.wage[w.female == 1], s=10, color=ORANGE, alpha=0.25, linewidths=0)
ax.plot(e, m.params["Intercept"] + m.params["educ"] * e, color=BLUE, label="men (base group)")
ax.plot(e, m.params["Intercept"] + m.params["female"] + m.params["educ"] * e, color=ORANGE, label="women")
ax.annotate("", xy=(16.5, m.params["Intercept"] + m.params["female"] + m.params["educ"] * 16.5),
            xytext=(16.5, m.params["Intercept"] + m.params["educ"] * 16.5),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
ax.text(16.7, m.params["Intercept"] + m.params["female"] / 2 + m.params["educ"] * 16.5,
        f"δ₀ = {m.params['female']:.2f}", va="center", color=INK)
ax.set_ylim(-1, 16); ax.set_xlim(0, 19.5); ax.set_xlabel("years of education"); ax.set_ylabel("hourly wage ($, 1976)")
ax.set_title("A dummy shifts the intercept: parallel lines, constant gap")
ax.legend(loc="upper left")
save(fig, "03_intercept_shift.png")

# 4 -- different slopes: the two textbook cases -----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), sharey=True)
e = np.linspace(0, 18, 50)
for ax, (d0, d1, ttl) in zip(axes, [(-0.20, -0.02, "(a) δ₀ < 0, δ₁ < 0: gap widens"), (-0.45, 0.04, "(b) δ₀ < 0, δ₁ > 0: lines cross")]):
    ax.plot(e, 0.6 + 0.08 * e, color=BLUE, label="men")
    ax.plot(e, 0.6 + d0 + (0.08 + d1) * e, color=ORANGE, label="women")
    ax.set_title(ttl); ax.set_xlabel("years of education")
    if d1 > 0:
        xs = -d0 / d1
        ax.axvline(xs, color=MUTED, lw=0.9, ls=":")
        ax.text(xs + 0.3, 0.45, f"educ* = −δ₀/δ₁ = {xs:.1f}", color=INK, fontsize=9.5)
axes[0].set_ylabel("log(wage)"); axes[0].legend(loc="upper left")
save(fig, "04_different_slopes.png")

# 5 -- Chow test: pooled fit vs separate fits (simulated) -------------------------------
n1 = n2 = 80
x1, x2 = rng.uniform(0, 10, n1), rng.uniform(0, 10, n2)
y1 = 1.0 + 0.90 * x1 + rng.normal(0, 1.2, n1)
y2 = 4.0 + 0.35 * x2 + rng.normal(0, 1.2, n2)
xp, yp = np.r_[x1, x2], np.r_[y1, y2]
fp = sm.OLS(yp, sm.add_constant(xp)).fit(); f1 = sm.OLS(y1, sm.add_constant(x1)).fit(); f2 = sm.OLS(y2, sm.add_constant(x2)).fit()
k = 1
F = ((fp.ssr - (f1.ssr + f2.ssr)) / (k + 1)) / ((f1.ssr + f2.ssr) / (n1 + n2 - 2 * (k + 1)))
g = np.linspace(0, 10, 20)
fig, ax = plt.subplots(figsize=(6.6, 3.9))
ax.scatter(x1, y1, s=13, color=BLUE, alpha=0.45, linewidths=0); ax.scatter(x2, y2, s=13, color=ORANGE, alpha=0.45, linewidths=0)
ax.plot(g, f1.params[0] + f1.params[1] * g, color=BLUE, label=f"group 1 fit (SSR₁ = {f1.ssr:.0f})")
ax.plot(g, f2.params[0] + f2.params[1] * g, color=ORANGE, label=f"group 2 fit (SSR₂ = {f2.ssr:.0f})")
ax.plot(g, fp.params[0] + fp.params[1] * g, color=INK, ls="--", lw=1.6, label=f"pooled fit, restricted (SSR_P = {fp.ssr:.0f})")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(f"Chow test compares SSR_P with SSR₁ + SSR₂  (here F = {F:.1f})")
ax.legend(loc="upper left", fontsize=9.5)
save(fig, "05_chow_test.png")

# 6 -- LPM: fitted probability line leaves [0, 1] ---------------------------------------
mz = woo.data("mroz")
lpm = smf.ols("inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6", mz).fit()
b = lpm.params
base = b["Intercept"] + b["nwifeinc"] * 50 + b["exper"] * 5 + b["I(exper ** 2)"] * 25 + b["age"] * 30 + b["kidslt6"] * 1
e = np.linspace(0, 20, 50)
zero = -base / b["educ"]
fig, ax = plt.subplots(figsize=(6.6, 3.9))
ax.axhspan(-0.3, 0, color="#f1d9d2", alpha=0.6, lw=0)
ax.plot(e, base + b["educ"] * e, color=BLUE)
ax.axhline(0, color=MUTED, lw=0.9); ax.axvline(5, color=MUTED, lw=0.9, ls=":"); ax.axvline(17, color=MUTED, lw=0.9, ls=":")
ax.text(0.3, -0.24, "impossible: P < 0", color=INK, fontsize=9.5)
ax.text(5.2, 0.52, "sample range of educ (5 to 17)", color=INK, fontsize=9.5)
ax.text(zero + 0.5, -0.07, f"crosses 0 at educ = {zero:.2f}", color=INK, fontsize=9.5)
ax.text(12.2, base + b["educ"] * 12 - 0.09, f"slope = {b['educ']:.3f}", color=INK, fontsize=9.5)
ax.set_ylim(-0.3, 0.7); ax.set_xlabel("years of education"); ax.set_ylabel("predicted P(in labor force)")
ax.set_title("Linear probability model: a line does not respect 0 ≤ P ≤ 1")
save(fig, "06_lpm_fitted_line.png")

# 7 -- LPM heteroskedasticity: Var(y|x) = p(1-p) ----------------------------------------
p = np.linspace(0, 1, 200)
fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.plot(p, p * (1 - p), color=BLUE)
for pv in (0.1, 0.5):
    ax.scatter([pv], [pv * (1 - pv)], s=45, color=ORANGE, zorder=3, edgecolors=SURF, linewidths=1.5)
    ax.text(pv + (0.04 if pv < 0.5 else 0), pv * (1 - pv) + (-0.012 if pv < 0.5 else 0.012), f"p = {pv}: u ∈ {{{1 - pv:.1f}, −{pv:.1f}}},  Var = {pv * (1 - pv):.2f}",
            color=INK, fontsize=9.5, va="top" if pv < 0.5 else "bottom", ha="left" if pv < 0.5 else "center")
ax.set_xlabel("p(x) = P(y = 1 | x)"); ax.set_ylabel("Var(y | x) = p(1 − p)"); ax.set_ylim(0, 0.3)
ax.set_title("In the LPM the error variance depends on x by construction")
save(fig, "07_lpm_variance.png")

AQUA = "#1baf7a"
R = 2000   # Monte Carlo replications


def hist(ax, vals, color, label, bins):
    ax.hist(vals, bins=bins, histtype="stepfilled", color=color, alpha=0.18, lw=0)
    ax.hist(vals, bins=bins, histtype="step", color=color, lw=2.0, label=label)


# 8 -- adding a regressor: precision gain vs multicollinearity cost (simulated) ----------
n = 200
est = {"base": [], "z": [], "w": []}
for _ in range(R):
    x = rng.normal(size=n); z = rng.normal(size=n); w = 0.95 * x + 0.31 * rng.normal(size=n)
    y = 1 + 0.5 * x + 1.5 * z + rng.normal(size=n)
    est["base"].append(sm.OLS(y, sm.add_constant(x)).fit().params[1])
    est["z"].append(sm.OLS(y, sm.add_constant(np.column_stack([x, z]))).fit().params[1])
    est["w"].append(sm.OLS(y, sm.add_constant(np.column_stack([x, w]))).fit().params[1])
bins = np.linspace(-0.6, 1.6, 70)
fig, ax = plt.subplots(figsize=(6.6, 3.9))
hist(ax, est["z"], BLUE, f"+ z: explains y, uncorrelated with x   (sd {np.std(est['z']):.3f})", bins)
hist(ax, est["base"], ORANGE, f"y on x only   (sd {np.std(est['base']):.3f})", bins)
hist(ax, est["w"], AQUA, f"+ w: irrelevant, corr(x, w) = .95   (sd {np.std(est['w']):.3f})", bins)
ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
ax.axvline(0.5, color=INK, lw=1, ls=":"); ax.text(0.53, ax.get_ylim()[1] * 0.50, "true β₁ = 0.5", color=INK, fontsize=9.5, bbox=dict(fc=SURF, ec="none", alpha=0.85, pad=1.5))
ax.set_xlabel("estimate of β₁ across 2,000 samples (n = 200)"); ax.set_ylabel("count"); ax.set_yticks([])
ax.set_title("All three are unbiased; only the extra regressor's correlations change precision")
ax.legend(loc="upper left", fontsize=9)
save(fig, "08_adding_regressors_precision.png")

# 9 -- over-controlling: conditioning on a mediator (simulated) -------------------------
total, direct = [], []
for _ in range(R):
    tax = rng.normal(size=n); cons = -0.8 * tax + rng.normal(size=n)
    fat = 0.6 * cons + rng.normal(size=n)            # tax affects fatalities only through consumption
    total.append(sm.OLS(fat, sm.add_constant(tax)).fit().params[1])
    direct.append(sm.OLS(fat, sm.add_constant(np.column_stack([tax, cons]))).fit().params[1])
bins = np.linspace(-0.85, 0.3, 70)
fig, ax = plt.subplots(figsize=(6.6, 3.9))
hist(ax, total, BLUE, "fatalities on tax", bins)
hist(ax, direct, ORANGE, "fatalities on tax + beer consumption", bins)
ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
for v, lab in ((-0.48, "total effect\n−0.8 × 0.6 = −0.48"), (0.0, "effect left after\ncontrolling: 0")):
    ax.axvline(v, color=INK, lw=1, ls=":"); ax.text(v + 0.02, ax.get_ylim()[1] * 0.60, lab, color=INK, fontsize=9.5, bbox=dict(fc=SURF, ec="none", alpha=0.85, pad=1.5))
ax.set_xlabel("coefficient on beer tax across 2,000 samples"); ax.set_ylabel("count"); ax.set_yticks([])
ax.set_title("Controlling for the channel (consumption) erases the policy effect")
ax.legend(loc="upper left", fontsize=9.5)
save(fig, "09_over_controlling.png")

# 10 -- log(y) with a dummy: approximate vs exact percentage ----------------------------
bgrid = np.linspace(-0.6, 0.6, 200)
fig, ax = plt.subplots(figsize=(6.6, 3.9))
ax.plot(100 * bgrid, 100 * bgrid, color=BLUE, label="approximation: 100·β")
ax.plot(100 * bgrid, 100 * (np.exp(bgrid) - 1), color=ORANGE, label="exact: 100·[exp(β) − 1]")
for b0, lab in ((0.054, "colonial .054"), (-0.297, "female −.297")):
    ax.scatter([100 * b0] * 2, [100 * b0, 100 * (np.exp(b0) - 1)], s=40, color=[BLUE, ORANGE], zorder=3, edgecolors=SURF, linewidths=1.5)
    ax.text(100 * b0 + 2, 100 * (np.exp(b0) - 1) - (7 if b0 < 0 else -3),
            f"{lab}: {100 * b0:.1f}% vs {100 * (np.exp(b0) - 1):.1f}%", color=INK, fontsize=9.5, bbox=dict(fc=SURF, ec="none", alpha=0.85, pad=1.5))
ax.set_xlabel("coefficient on the dummy × 100"); ax.set_ylabel("implied % difference in y")
ax.set_title("The ×100 shortcut is fine near zero and drifts for large coefficients")
ax.legend(loc="upper left")
save(fig, "10_log_dummy_exact_percent.png")

# 11 -- four groups: coefficients relative to the base group ----------------------------
w = woo.data("wage1")
w["marrmale"] = ((w.female == 0) & (w.married == 1)).astype(int)
w["marrfem"] = ((w.female == 1) & (w.married == 1)).astype(int)
w["singfem"] = ((w.female == 1) & (w.married == 0)).astype(int)
ctrl = "educ + exper + I(exper**2) + tenure + I(tenure**2)"
m4 = smf.ols(f"np.log(wage) ~ marrmale + marrfem + singfem + {ctrl}", w).fit()
labs = [("marrmale", "married men"), ("singfem", "single women"), ("marrfem", "married women")]
fig, ax = plt.subplots(figsize=(6.6, 3.2))
ax.axvline(0, color=INK, lw=1)
ax.text(0.006, 3.35, "base group: single men (= 0)", color=INK, fontsize=9.5)
for i, (k, lab) in enumerate(labs):
    b, se = m4.params[k], m4.bse[k]
    ax.plot([b - 1.96 * se, b + 1.96 * se], [i, i], color=BLUE, lw=2)
    ax.scatter([b], [i], s=55, color=BLUE, zorder=3, edgecolors=SURF, linewidths=1.5)
    ax.text(b, i + 0.22, f"{b:+.3f}", color=INK, fontsize=9.5, ha="center")
ax.set_yticks(range(3)); ax.set_yticklabels([l for _, l in labs]); ax.set_ylim(-0.6, 3.7)
ax.set_xlabel("difference in log(wage) from single men, with 95% CI"); ax.grid(axis="y", visible=False)
ax.set_title("Each dummy coefficient is a gap relative to the omitted group")
save(fig, "11_four_groups.png")

# 12 -- re-centering: the female gap as a function of education -------------------------
m18 = smf.ols(f"np.log(wage) ~ female*educ + exper + I(exper**2) + tenure + I(tenure**2)", w).fit()
V = m18.cov_params()
e = np.linspace(0, 18, 100)
gap = m18.params["female"] + m18.params["female:educ"] * e
se = np.sqrt(V.loc["female", "female"] + e ** 2 * V.loc["female:educ", "female:educ"] + 2 * e * V.loc["female", "female:educ"])
fig, ax = plt.subplots(figsize=(6.6, 3.9))
ax.fill_between(e, gap - 1.96 * se, gap + 1.96 * se, color=BLUE, alpha=0.16, lw=0)
ax.plot(e, gap, color=BLUE, label="estimated gap δ₀ + δ₁·educ, with 95% band")
ax.axhline(0, color=INK, lw=0.9)
for e0 in (0, 12.5):
    g0 = m18.params["female"] + m18.params["female:educ"] * e0
    s0 = np.sqrt(V.loc["female", "female"] + e0 ** 2 * V.loc["female:educ", "female:educ"] + 2 * e0 * V.loc["female", "female:educ"])
    ax.scatter([e0], [g0], s=45, color=ORANGE, zorder=3, edgecolors=SURF, linewidths=1.5)
    ax.text(e0 + 0.4, g0 + 0.07, f"educ = {e0:g}: {g0:.3f} (se {s0:.3f})", color=INK, fontsize=9.5)
ax.plot(w.educ + rng.uniform(-0.15, 0.15, len(w)), np.full(len(w), -0.62), "|", color="#9a9992", ms=7, alpha=0.4)
ax.text(0.2, -0.57, "sample values of educ", color=MUTED, fontsize=9)
ax.set_ylim(-0.68, 0.3); ax.set_xlabel("years of education"); ax.set_ylabel("female − male, log(wage)")
ax.set_title("The coefficient on female is the gap at educ = 0, where there is no data")
ax.legend(loc="upper right", fontsize=9.5)
save(fig, "12_recentering.png")

# 13 -- ordinal variable: one linear term vs a dummy per category (simulated) -----------
true = np.array([0.0, -0.15, -0.90, -1.05, -1.15])
cr = rng.integers(0, 5, 400)
mbr = 6 + true[cr] + rng.normal(0, 0.45, 400)
dfo = __import__("pandas").DataFrame({"mbr": mbr, "cr": cr})
md = smf.ols("mbr ~ C(cr)", dfo).fit(); ml = smf.ols("mbr ~ cr", dfo).fit()
Ft = md.compare_f_test(ml)
fig, ax = plt.subplots(figsize=(6.6, 3.9))
pred = md.get_prediction(__import__("pandas").DataFrame({"cr": range(5)})).summary_frame()
ax.errorbar(range(5), pred["mean"], yerr=1.96 * pred["mean_se"], fmt="o", color=BLUE, ms=7, capsize=0, lw=2, label="dummy for each rating (base: CR = 0)")
ax.plot(np.arange(5), ml.params["Intercept"] + ml.params["cr"] * np.arange(5), color=ORANGE, label="single linear term in CR")
ax.text(2.6, 5.75, f"F test of equal steps: F = {Ft[0]:.1f}, p < .001", color=INK, fontsize=9.5)
ax.set_xticks(range(5)); ax.set_xlabel("credit rating CR (0 = worst, 4 = best)"); ax.set_ylabel("municipal bond rate (%)")
ax.set_title("A linear term forces every one-step improvement to have the same effect")
ax.legend(loc="lower left")
save(fig, "13_ordinal_dummies.png")

# 14 -- self-selection into a grant: naive vs adjusted estimates (simulated) -------------
naive, adj = [], []
for _ in range(R):
    trouble = rng.normal(size=n)
    grant = (0.9 * trouble + rng.normal(size=n) > 0).astype(float)    # troubled firms apply more eagerly
    lscrap = 1 - 0.30 * grant + 0.8 * trouble + rng.normal(size=n)
    naive.append(sm.OLS(lscrap, sm.add_constant(grant)).fit().params[1])
    adj.append(sm.OLS(lscrap, sm.add_constant(np.column_stack([grant, trouble]))).fit().params[1])
bins = np.linspace(-0.9, 1.3, 70)
fig, ax = plt.subplots(figsize=(6.6, 3.9))
hist(ax, adj, BLUE, "log(scrap) on grant + firm trouble", bins)
hist(ax, naive, ORANGE, "log(scrap) on grant only", bins)
ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
ax.axvline(-0.30, color=INK, lw=1, ls=":"); ax.text(-0.33, ax.get_ylim()[1] * 0.60, "true effect\nτ = −0.30", color=INK, fontsize=9.5, ha="right", bbox=dict(fc=SURF, ec="none", alpha=0.85, pad=1.5))
ax.axvline(np.mean(naive), color=ORANGE, lw=1, ls="--")
ax.text(np.mean(naive) + 0.03, ax.get_ylim()[1] * 0.60, f"naive mean {np.mean(naive):+.2f}", color=INK, fontsize=9.5, bbox=dict(fc=SURF, ec="none", alpha=0.85, pad=1.5))
ax.set_xlabel("estimated grant effect across 2,000 samples (n = 200)"); ax.set_ylabel("count"); ax.set_yticks([])
ax.set_title("Troubled firms select into grants: the naive estimate even flips sign")
ax.legend(loc="upper left", fontsize=9.5)
save(fig, "14_self_selection.png")
print("figures written to", os.path.abspath(OUT))
