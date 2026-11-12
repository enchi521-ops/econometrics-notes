# Reading notes — Soldà, Ke, von Hippel & Page (2021)

**Absolute versus relative success: Why overconfidence creates an inefficient equilibrium.** *Psychological Science, 32*(10), 1662–1674. [doi:10.1177/09567976211007414](https://doi.org/10.1177/09567976211007414) · open access (CC BY) · preregistered · open data ([OSF](https://osf.io/km8c9/))

Read for the econometrics course as an applied example of Chapter 15 (IV and 2SLS). The coding lab built on it is in [`README.md`](README.md).

---

## 1. The puzzle

Overconfidence is common, yet it is costly: overconfident negotiators hold out for better deals and fail to reach agreements. If it hurts, why does it persist?

**The paper's answer:** confidence pays in **relative** terms (you get more than your partner) even when it costs in **absolute** terms (the pair ends up with less). If people care about relative standing, high confidence is individually rational, and the outcome is an inefficient equilibrium.

---

## 2. Design

298 university students (149 pairs), lab experiment in France, programmed in oTree.

| Phase | What happens | Purpose |
|---|---|---|
| Individual | 10 quiz questions alone | Rank participants to form pairs of similar ability |
| Partner | 30 questions each; correct answers fill a **joint prize** (scaled by a random productivity draw) | Create a jointly earned prize |
| Manipulation | Beliefs elicited → **noisy signal** (ball from urn) → beliefs elicited again | Shift confidence exogenously |
| Negotiation | Prize split 70/30; the pair must agree who takes the large share. Stage 1: claim. Stage 2: 3 min chat. Stage 3: prize **shrinks to zero over 30 s** | Make delay costly to both |

**The signal.** If you did better than your partner, the ball comes from an urn with 15 green and 5 red balls; if worse, from an urn with 5 green and 15 red. A green ball is right 75% of the time. Conditional on true performance, the colour is random.

**Why manipulate rather than measure?** Measured confidence is correlated with traits such as dominance or extraversion. Manipulated confidence is not, *except* through the signal.

---

## 3. Identification (the Chapter 15 part)

| IV ingredient | In this paper |
|---|---|
| Endogenous regressor | Posterior belief that one out-performed the partner (summed over the pair in Table 2) |
| Instruments | Own signal (`ball`) and partner's signal (`other_ball`) |
| Relevance | Beliefs before the signal do not differ by signal; beliefs after it differ strongly (Mann–Whitney $z \approx -6.4$, $p<.001$, Fig. 2) |
| Exogeneity argument | Signal is random given who performed better, and the regressions control for performance |
| Exclusion support | Participants were **not allowed to reveal their signal** to the partner, so it can only act through private beliefs |
| Estimators | 2SLS for the pair-level outcome (Table 2); 3SLS for individual absolute and relative payoffs (Fig. 3) |

Footnote 2 is a nice detail: once the signal is received, being the better performer no longer predicts confidence. The signal "absorbs" the information.

---

## 4. Results

**Raw payoffs (Table 1).** When both partners got a good signal, each walked away with 26.96% of the initial prize; when both got a bad signal, 39.18%. With opposite signals, the green-ball participant earned more in both absolute and relative terms.

**Individual payoffs (Fig. 3, 3SLS).**

| Outcome | Finding |
|---|---|
| Absolute payoff | Own confidence helps when the partner is *not* confident and hurts when the partner *is*. Interaction $\beta=-0.0148$, $p=.008$ |
| Relative payoff | Own confidence always helps ($\beta=0.212$, $p=.026$); being more confident than the partner helps ($\beta=0.216$, $p<.001$) |

**Pair-level efficiency (Table 2, 2SLS).** Outcome: % of the initial prize still awarded at the end.

| Model | Sample | Controls | $\hat\beta$ on belief$_i$ + belief$_j$ | 95% CI | $n$ |
|---|---|---|---|---|---|
| **1** | All pairs | Performance | **−0.401** | [−0.744, −0.059] | 149 |
| 2 | All pairs | + demographics | −0.412 | [−0.793, −0.031] | 149 |
| 3 | Pairs that agreed | Performance | −0.340 | [−0.562, −0.118] | 127 |
| 4 | Pairs that agreed | + demographics | −0.362 | [−0.616, −0.108] | 127 |

+10 points of pair confidence → about −4 percentage points of the prize. Models 3–4 show the loss is not only from failed agreements but also from **delay**.

**Agreement timing (Fig. 4).** Only 6% of pairs agreed immediately; 14.8% never agreed. Agreements spike right when the prize starts shrinking (180 s).

**Communication.** Positive signals → more claims of *entitlement* ("I performed better") and *assertiveness* ("I won't back down"). Controlling for these does not remove the confidence effect.

---

## 5. Authors' caveats

- The shrinking pie makes delay costly by design; with more time, mutual confidence might cost less.
- The split is forced to be unequal, which raises the chance of disagreement.
- Purely distributive negotiation, with no room to expand the pie.

---

## 6. Critique from class (Nov 12)

| Point | Argument | Consequence |
|---|---|---|
| Endogeneity concern is real | Confidence ↔ assertiveness, extraversion, success | Justifies IV |
| Relevance | Convincing in theory and data (first-stage $t\approx6$) | ✓ |
| **Fairness concern** | A green ball may create a sense of entitlement that changes bargaining directly, not only via belief | Exclusion restriction may fail |
| **Summed beliefs** | One $\beta$ for own + partner belief; if their effects differ, the leftover goes into the error, which the signals move | Instruments correlated with the error |
| Reporting | First stage reported as a bare "$F(4,144)=17.66$" without saying what it tests | Report what a statistic tests and its $p$-value, or leave it out |
| Sargan $p=.49$ | Fails to reject | Not proof of exogeneity |
| Wu-Hausman $p=.17$ | Fails to reject | Not proof that OLS is fine |

My own additions:

- The **summed-belief** specification is a modelling choice the 3SLS analysis in Fig. 3 avoids. Estimating Table 2 with own and partner beliefs as two endogenous regressors, each instrumented, would test the restriction directly. That model would be exactly identified, though, so the Sargan test would no longer be available.
- Weak-instrument-robust inference (Anderson–Rubin, $p=.059$) is a little weaker than the reported Wald $p=.022$. The finding survives, but the margin is thin.
- The sample is lab students; effects in high-stakes or repeated negotiations could differ.

---

## 7. Why this matters beyond the lab

- **Policy and negotiation practice.** Inter-state or labour negotiations with deadlines and shrinking surpluses are "shrinking-pie" games. If both sides receive signals that make them confident (domestic polls, intelligence briefings), the joint loss can grow even as each side feels it is winning in relative terms.
- **Method.** The paper shows how an experiment can create its own instrument: randomise an informative signal instead of randomising the treatment itself. The same trick, "encouragement" or noisy information designs, appears in development economics field experiments.

---

## Links

- Paper: [doi:10.1177/09567976211007414](https://doi.org/10.1177/09567976211007414)
- Data and analysis code: [osf.io/km8c9](https://osf.io/km8c9/)
- Materials and supplementary material: [osf.io/2j6ne](https://osf.io/2j6ne/)
- Preregistration: [aspredicted.org/bj9er.pdf](https://aspredicted.org/bj9er.pdf)
