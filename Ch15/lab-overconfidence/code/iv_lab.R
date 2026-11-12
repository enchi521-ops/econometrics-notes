# =============================================================================
# IV / 2SLS lab — Does confidence change what negotiators take home?
# Data: Soldà, Ke, von Hippel & Page (2021), Psychological Science 32(10).
# Target: Table 2, Model 1 of the paper (beta = -0.401, constant = 111.20).
#
# Run from this folder:   Rscript iv_lab.R
# Packages (first run only):
#   install.packages(c("ivreg", "ivreg2r"))
# =============================================================================

library(ivreg)     # ivreg(): 2SLS with Wu-Hausman, weak-instrument and Sargan diagnostics
library(ivreg2r)   # ivreg2(): R port of Stata's ivreg2 (Sargan, first-stage F, etc.)

data <- read.csv("../data/instruments.csv")

# Variables used -------------------------------------------------------------
# final_pot_perc : % of the initial prize the participant was awarded  (y1)
# sum_belief     : participant's + partner's belief (0-100 each) that they
#                  out-performed the other  -> pair-level confidence     (y2, endogenous)
# ball           : 1 if the participant saw a green ball ("you probably did better")
# other_ball     : same signal for the partner                          (instruments)
# score_p2       : participant's quiz score in the partner phase
# other_score2   : partner's quiz score                                 (exogenous controls)
vars <- c("final_pot_perc", "sum_belief", "ball", "other_ball", "score_p2", "other_score2")
print(summary(data[, vars]))


# 1. OLS (biased if confidence is endogenous) ---------------------------------
ols <- lm(final_pot_perc ~ sum_belief + score_p2 + other_score2, data = data)
summary(ols)$coefficients


# 2. 2SLS by hand -------------------------------------------------------------
# Stage 1: regress the endogenous variable on ALL exogenous variables.
firstModel <- lm(sum_belief ~ ball + other_ball + score_p2 + other_score2, data = data)
summary(firstModel)
# Relevance: both instruments are strongly significant (t > 6).
# Joint test of the excluded instruments (the F that matters for weak IVs):
anova(lm(sum_belief ~ score_p2 + other_score2, data = data), firstModel)

# Stage 2: replace sum_belief with its fitted value.
secondModel <- lm(final_pot_perc ~ fitted(firstModel) + score_p2 + other_score2, data = data)
summary(secondModel)$coefficients
# The coefficient is the 2SLS estimate, but the standard error is NOT valid:
# the stage-2 error contains beta1 * v2, which lm() ignores.


# 3. 2SLS with a dedicated routine -------------------------------------------
# Formula: y ~ exogenous | endogenous | instruments
simpleModel <- ivreg2(
  final_pot_perc ~ score_p2 + other_score2 | sum_belief | ball + other_ball,
  data = data,
  first_stage = TRUE          # keep the first-stage regression for inspection
)
summary(simpleModel)          # includes the Sargan overidentification test
coef(simpleModel)             # same point estimates as the manual version
first_stage(simpleModel)      # stored first stage


# 4. Endogeneity (Wu-Hausman) test -------------------------------------------
# ivreg() reports it in its diagnostics. ivreg and ivreg2 differ slightly in
# small-sample conventions, so do not mix their numbers in a paper.
ivModel <- ivreg(final_pot_perc ~ score_p2 + other_score2 | sum_belief | ball + other_ball,
                 data = data)
summary(ivModel)$diagnostics


# 5. Regression-based tests by hand (Wooldridge 15.5) ------------------------
# (a) Endogeneity: add the first-stage residual to the structural equation.
data$v2hat <- resid(firstModel)
cf <- lm(final_pot_perc ~ sum_belief + score_p2 + other_score2 + v2hat, data = data)
summary(cf)$coefficients["v2hat", ]      # H0: delta1 = 0  (sum_belief is exogenous)

# (b) Overidentification: regress the 2SLS residuals on all exogenous variables.
data$u1hat <- data$final_pot_perc - model.matrix(~ sum_belief + score_p2 + other_score2, data) %*%
  coef(ivModel)[c("(Intercept)", "sum_belief", "score_p2", "other_score2")]
aux <- lm(u1hat ~ ball + other_ball + score_p2 + other_score2, data = data)
nR2 <- nrow(data) * summary(aux)$r.squared
q   <- 2 - 1                                 # instruments - endogenous regressors
c(nR2 = nR2, df = q, p.value = pchisq(nR2, df = q, lower.tail = FALSE))
