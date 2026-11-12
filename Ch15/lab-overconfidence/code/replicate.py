"""
Python replication of the IV lab (Soldà et al., 2021, Table 2, Model 1).

Re-estimates every number quoted in ../README.md and checks it against the
paper and against the R output in ../output/iv_lab_R_output.txt.

    pip install -r ../requirements.txt
    python replicate.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from scipy import stats

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE.parent / "data" / "instruments.csv")
n = len(d)

FORMULA = "final_pot_perc ~ 1 + score_p2 + other_score2 + [sum_belief ~ ball + other_ball]"

# 1. OLS ----------------------------------------------------------------------
ols = smf.ols("final_pot_perc ~ sum_belief + score_p2 + other_score2", d).fit()

# 2. Manual 2SLS --------------------------------------------------------------
fs = smf.ols("sum_belief ~ ball + other_ball + score_p2 + other_score2", d).fit()
excl = fs.f_test("ball = 0, other_ball = 0")
d["sum_belief_hat"] = fs.fittedvalues
manual = smf.ols("final_pot_perc ~ sum_belief_hat + score_p2 + other_score2", d).fit()

# 3. Proper 2SLS --------------------------------------------------------------
iv_large = IV2SLS.from_formula(FORMULA, d).fit(cov_type="unadjusted")               # ivreg2 default
iv_small = IV2SLS.from_formula(FORMULA, d).fit(cov_type="unadjusted", debiased=True)  # ivreg / small

# 4. Diagnostics --------------------------------------------------------------
sargan = iv_large.sargan
wu = iv_large.wu_hausman()

# regression-based versions (Wooldridge 15.5)
d["v2hat"] = fs.resid
cf = smf.ols("final_pot_perc ~ sum_belief + score_p2 + other_score2 + v2hat", d).fit()
b = iv_large.params
d["u1hat"] = d.final_pot_perc - (b["Intercept"] + b["sum_belief"] * d.sum_belief
                                 + b["score_p2"] * d.score_p2 + b["other_score2"] * d.other_score2)
aux = smf.ols("u1hat ~ ball + other_ball + score_p2 + other_score2", d).fit()
nR2 = n * aux.rsquared
p_nR2 = stats.chi2.sf(nR2, df=1)

# 5. Report -------------------------------------------------------------------
ci = iv_large.conf_int().loc["sum_belief"]
rows = [
    # label, ours, reference, source
    ("OLS: sum_belief",                       ols.params["sum_belief"],         None,    "R"),
    ("First stage: ball",                     fs.params["ball"],                31.67,   "R"),
    ("First stage: other_ball",               fs.params["other_ball"],          29.95,   "R"),
    ("First stage F, all regressors (4,144)", fs.fvalue,                        17.66,   "paper"),
    ("First stage F, excluded IVs (2,144)",   float(excl.fvalue),               31.38,   "R"),
    ("2SLS: sum_belief",                      b["sum_belief"],                  -0.401,  "paper"),
    ("2SLS: constant",                        b["Intercept"],                   111.20,  "paper"),
    ("2SLS 95% CI lower",                     ci.iloc[0],                       -0.744,  "paper"),
    ("2SLS 95% CI upper",                     ci.iloc[1],                       -0.0587, "paper"),
    ("2SLS p-value",                          iv_large.pvalues["sum_belief"],   0.022,   "paper"),
    ("Manual stage-2 coefficient",            manual.params["sum_belief_hat"],  -0.401,  "paper"),
    ("SE, manual stage 2 (wrong)",            manual.bse["sum_belief_hat"],     0.1741,  "R"),
    ("SE, 2SLS (ivreg2 default)",             iv_large.std_errors["sum_belief"], 0.1749, "R"),
    ("SE, 2SLS (small / ivreg)",              iv_small.std_errors["sum_belief"], 0.1773, "R"),
    ("Sargan chi2(1)",                        sargan.stat,                      0.4868,  "R"),
    ("Sargan p-value",                        sargan.pval,                      0.4854,  "R"),
    ("Sargan by hand: n*R2",                  nR2,                              0.4868,  "R"),
    ("Wu-Hausman F(1,144)",                   wu.stat,                          1.905,   "R"),
    ("Wu-Hausman p-value",                    wu.pval,                          0.1697,  "R"),
    ("Residual test: delta1 t-stat",          cf.tvalues["v2hat"],              1.380,   "R"),
]

print(f"n = {n}\n")
print(f"{'quantity':42s}{'python':>11s}{'reference':>11s}  source  check")
ok_all = True
for label, ours, ref, src in rows:
    if ref is None:
        print(f"{label:42s}{ours:11.4f}{'':>11s}")
        continue
    tol = max(abs(ref) * 0.005, 0.0006)            # rounding in the paper / R print-out
    ok = abs(ours - ref) <= tol
    ok_all &= ok
    print(f"{label:42s}{ours:11.4f}{ref:11.4f}  {src:6s}  {'OK' if ok else 'MISMATCH'}")

print("\nall checks passed" if ok_all else "\nSOME CHECKS FAILED")
print(f"\nIV vs OLS: {b['sum_belief']:.3f} vs {ols.params['sum_belief']:.3f} "
      f"-> IV effect is {b['sum_belief'] / ols.params['sum_belief']:.1f}x the OLS effect")
print(f"Interpretation: +10 points of pair confidence -> {10 * b['sum_belief']:.1f} pp of the prize")
