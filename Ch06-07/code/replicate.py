"""Replicate the textbook examples used in the Ch.6-7 review notes.

Data: `wooldridge` Python package (datasets from Wooldridge, Introductory Econometrics).
Run:  pip install wooldridge statsmodels  &&  python code/replicate.py
"""
import numpy as np
import statsmodels.formula.api as smf
import wooldridge as woo


def show(title, res, keys):
    print(f"\n== {title}  (n={int(res.nobs)}, R2={res.rsquared:.3f}, adjR2={res.rsquared_adj:.3f})")
    for k in keys:
        print(f"   {k:<22} coef={res.params[k]: .4f}  se={res.bse[k]:.4f}  t={res.tvalues[k]: .2f}")


# ---- Ch.6: adjusted R2 for nonnested models (eq. 6.23 vs 6.24) -------------------------
rd = woo.data("rdchem")
m1 = smf.ols("rdintens ~ np.log(sales)", rd).fit()
m2 = smf.ols("rdintens ~ sales + I(sales**2)", rd).fit()
print("== Nonnested models, R&D intensity")
print(f"   log(sales) model : R2={m1.rsquared:.3f}  adjR2={m1.rsquared_adj:.3f}")
print(f"   quadratic model  : R2={m2.rsquared:.3f}  adjR2={m2.rsquared_adj:.3f}")

# ---- Ch.6: different dependent variables (Example 6.4) ---------------------------------
ceo = woo.data("ceosal1")
lv = smf.ols("salary ~ sales + roe", ceo).fit()
lg = smf.ols("np.log(salary) ~ np.log(sales) + roe", ceo).fit()
sst = lambda y: float(((y - y.mean()) ** 2).sum())
print("== CEO salary: R2 is not comparable across dependent variables")
print(f"   level: R2={lv.rsquared:.3f}, SST={sst(ceo.salary):,.0f} | log: R2={lg.rsquared:.3f}, SST={sst(np.log(ceo.salary)):.2f}")

# ---- Ch.6: predicting y from a log(y) model (Example 6.7) ------------------------------
c2 = woo.data("ceosal2")
lm = smf.ols("np.log(salary) ~ np.log(sales) + np.log(mktval) + ceoten", c2).fit()
alpha0 = np.exp(lm.resid).mean()                      # Duan smearing estimate, eq. 6.43
print(f"== Smearing estimate alpha0_hat = {alpha0:.3f}  (naive exp() under-predicts by this factor)")

# ---- Ch.7: single dummy (eq. 7.4, 7.5) -------------------------------------------------
w = woo.data("wage1")
show("Wage with intercept shift (7.4)", smf.ols("wage ~ female + educ + exper + tenure", w).fit(), ["female", "educ"])
show("Difference in means (7.5)", smf.ols("wage ~ female", w).fit(), ["Intercept", "female"])

# ---- Ch.7: multiple categories (eq. 7.11), base group = single men ---------------------
w["marrmale"] = ((w.female == 0) & (w.married == 1)).astype(int)
w["marrfem"] = ((w.female == 1) & (w.married == 1)).astype(int)
w["singfem"] = ((w.female == 1) & (w.married == 0)).astype(int)
ctrl = "educ + exper + I(exper**2) + tenure + I(tenure**2)"
show("Four groups (7.11)", smf.ols(f"np.log(wage) ~ marrmale + marrfem + singfem + {ctrl}", w).fit(),
     ["marrmale", "marrfem", "singfem"])

# ---- Ch.7: comparing two non-base groups by re-basing (single vs married women) --------
w["singmale"] = ((w.female == 0) & (w.married == 0)).astype(int)
rb = smf.ols(f"np.log(wage) ~ marrmale + singmale + singfem + {ctrl}", w).fit()
show("Re-based on married women", rb, ["singfem"])

# ---- Ch.7: dummy in a log(y) equation (eq. 7.8) and the exact percentage ---------------
hp = woo.data("hprice1")
r78 = smf.ols("np.log(price) ~ np.log(lotsize) + np.log(sqrft) + bdrms + colonial", hp).fit()
show("Housing price (7.8)", r78, ["colonial"])
b = r78.params["colonial"]
print(f"   approximate {100*b:.1f}%, exact {100*(np.exp(b)-1):.2f}%;  female -.297 -> exact {100*(np.exp(-.297)-1):.1f}%")

# ---- Ch.7: dummy x continuous interaction (eq. 7.18) and re-centering ------------------
r18 = smf.ols(f"np.log(wage) ~ female*educ + exper + I(exper**2) + tenure + I(tenure**2)", w).fit()
show("Different slopes (7.18)", r18, ["female", "female:educ"])
f = r18.f_test("female = 0, female:educ = 0")
print(f"   joint F(female, female:educ) = {float(f.fvalue):.2f}, p = {float(f.pvalue):.4f}")
w["educ_c"] = w.educ - 12.5
show("Re-centered at educ = 12.5", smf.ols(f"np.log(wage) ~ female + educ + female:educ_c + exper + I(exper**2) + tenure + I(tenure**2)", w).fit(), ["female"])

# ---- Ch.7: Chow test (eq. 7.20-7.24), spring term of GPA3 ------------------------------
g = woo.data("gpa3")
g = g[g.term == 2].dropna(subset=["cumgpa", "sat", "hsperc", "tothrs"])
rhs = "sat + hsperc + tothrs"
ssr_p = smf.ols(f"cumgpa ~ {rhs}", g).fit().ssr
ssr_1 = smf.ols(f"cumgpa ~ {rhs}", g[g.female == 1]).fit().ssr
ssr_2 = smf.ols(f"cumgpa ~ {rhs}", g[g.female == 0]).fit().ssr
n, k = len(g), 3
chow = ((ssr_p - (ssr_1 + ssr_2)) / (k + 1)) / ((ssr_1 + ssr_2) / (n - 2 * (k + 1)))
print(f"\n== Chow test (n={n}): SSR_P={ssr_p:.3f}, SSR_1={ssr_1:.3f}, SSR_2={ssr_2:.3f}, F={chow:.2f}")
ur = smf.ols(f"cumgpa ~ female*({rhs})", g).fit()
print(f"   same F from the interaction model: {float(ur.f_test('female = 0, female:sat = 0, female:hsperc = 0, female:tothrs = 0').fvalue):.2f}")
print(f"   slopes only (intercept may differ): p = {float(ur.f_test('female:sat = 0, female:hsperc = 0, female:tothrs = 0').pvalue):.3f}")

# ---- Ch.7: linear probability model (eq. 7.29) with robust SEs -------------------------
mz = woo.data("mroz")
lpm = smf.ols("inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6", mz).fit()
rob = lpm.get_robustcov_results("HC1")
show("LPM, labor force participation (7.29)", lpm, ["educ", "kidslt6"])
print(f"   fitted values < 0: {(lpm.fittedvalues < 0).sum()},  > 1: {(lpm.fittedvalues > 1).sum()}")
i = list(lpm.params.index).index("kidslt6")
print(f"   kidslt6 se: usual={lpm.bse['kidslt6']:.4f}, HC1 robust={rob.bse[i]:.4f}")

# ---- Ch.7: program evaluation (eq. 7.7, 7.33) ------------------------------------------
jt = woo.data("jtrain")
j88 = jt[jt.year == 1988]
show("Training hours on grant (7.7)", smf.ols("hrsemp ~ grant + np.log(sales) + np.log(employ)", j88).fit(), ["grant"])
show("log(scrap) on grant (7.33)", smf.ols("np.log(scrap) ~ grant + np.log(sales) + np.log(employ)", j88).fit(), ["grant"])
