#!/usr/bin/env python3
r"""
analysis.py -- test hypotheses A, B and C on one ISO band.

    python analysis.py --band alpha
    python analysis.py --band sigma

Reads ISO_table_<band>.csv, whose columns are <channel group>_iso_<body position>:

    channel groups   frontal    = mean of F3, F4
                     occipital  = mean of O1, O2
                     left       = mean of O1, C3, F3   (left hemisphere)
                     right      = mean of O2, C4, F4   (right hemisphere)
    body positions   supine, left, right

Each cell is one subject's ISO strength for that channel group while in that body
position, already summarised across eligible N2 windows upstream. This script does
the between-subject statistics only; it never touches the EEG.

================================================================================
THE THREE HYPOTHESES, AS CONTRASTS

Every hypothesis is a WITHIN-PERSON paired contrast: both sides of each comparison
come from the same subject in the same body position, so between-subject variation
in overall ISO strength cancels.

  A  supine: occipital > frontal        D_A = occipital_iso_supine - frontal_iso_supine
             predicted sign  D_A > 0
  B  left:   left hemi > right hemi     D_B = left_iso_left      - right_iso_left
             predicted sign  D_B > 0
  C  right:  left hemi < right hemi     D_C = left_iso_right     - right_iso_right
             predicted sign  D_C < 0

Plan rev5 lines 98-108 define exactly these contrasts, and warns at line 109 that
"the order of average, median and difference is not fixed and it changes the sign
of hypothesis C in alpha. It has to be stated." The order used here is fixed by the
input table: channels were averaged first, then summarised across windows upstream,
and the difference is taken last, here.

================================================================================
THE TEST PLAN, AND WHY

1. PRIMARY, per plan rev5 lines 98-104: "regress D ~ age + sex + BMI. The quantity
   of interest is the intercept. Use clustered bootstrapping for a 95 percent
   confidence interval."

   Covariates come from a separate file keyed on SiteID + BDSPPatientID +
   SessionID, joined here to BDSPID = SiteID concatenated with BDSPPatientID.
   MEASURED on the supplied file: 2,836 rows of which 116 are ENTIRELY BLANK, so
   2,720 join 1:1 with no duplicates, and 2,719 carry a complete Age, Sex and BMI.

   ONE FACT ABOUT THIS MODEL HAS TO BE STATED, because it bounds what the
   covariates can possibly do:

     The raw intercept of D ~ age + sex + BMI is the fitted value at age = 0 and
     BMI = 0, which is meaningless extrapolation. The covariates are therefore
     MEAN-CENTRED, so the intercept is the adjusted mean of D at the sample's own
     average covariate profile -- the only reading under which "the intercept" is
     the quantity the hypotheses are about. Ledger row Q-B3, intercept centering,
     is still open; this is Claude's resolution of it, flagged per Part 6.

     What follows from the centring depends on WHICH statistic is primary, and
     the two cases are genuinely different:

       mean (--statistic mean).  In ordinary least squares with every predictor
         mean-centred, the intercept is EXACTLY the sample mean of the outcome,
         as a matter of algebra. Adjustment therefore CANNOT move the point
         estimate. It can only (a) change which subjects are analysed, because
         the model is complete-case, and (b) narrow the interval slightly by
         removing covariate-explained variance.

       median (--statistic median, THE DEFAULT, ledger D12).  The model becomes
         median (quantile) regression at q = 0.5. NO such identity holds: the
         centred intercept is only approximately the median of D, so adjustment
         CAN move the point estimate. The amount it moved is printed rather than
         assumed to be zero.

     Either way the check is NUMERICAL, not asserted: the centred intercept is
     compared against the unadjusted statistic on the same rows, and the
     difference is reported.

   So both estimates are reported side by side, and the covariate SLOPES are
   reported too, since they are the only part of this model carrying information
   the unadjusted analysis does not already have.

1b. SUMMARY STATISTIC. Ledger D12 settles this as the MEDIAN (Dr. Sun 27AUG,
   ">> Yes, median"), confirmed by Jacob 2026-09-18. The median is primary: it
   supplies the point estimate, the interval, the p-value that decides each
   verdict, and the values BH corrects. The mean is computed and printed beside
   it, and any contrast where the two disagree -- in sign or in significance --
   is flagged explicitly, because on these heavy-tailed contrasts they do.
   MEASURED example: on the remove-contralateral tables, hypothesis C has a
   non-significant mean (p = 0.097 alpha, 0.30 sigma) and a significant median
   with a CI excluding zero in both bands. The choice of statistic decides that
   verdict, so it is stated wherever the verdict appears.

2. Clustered bootstrap, resampling whole subjects with replacement. MEASURED on
   both supplied tables: 2,836 rows and 2,836 distinct BDSPIDs, so every cluster is
   a singleton and this is arithmetically an ordinary bootstrap. The cluster
   machinery is kept because it is what the plan specifies and because it becomes
   load-bearing the moment a subject contributes a second session.

3. SECONDARY, non-parametric: Wilcoxon signed-rank on each contrast. MEASURED, the
   contrasts have excess kurtosis of 16 to 239 and skew up to |6.8|, so they are
   very far from normal and a paired t-test's tail behaviour cannot be trusted at
   face value. The t-test is reported too, but only as a third line of evidence.

4. Both a two-sided and a one-sided p-value are reported for every test. The
   hypotheses are directional, so the one-sided value is the one they predict; the
   two-sided value is what the multiplicity correction uses, because it does not
   reward a result that lands opposite to the prediction. A contrast that is
   significant in the WRONG direction is reported as CONTRADICTED, never as
   "not significant", which would hide a real finding.

5. MULTIPLICITY. Plan rev5 line 118: Benjamini-Hochberg at q = 0.05 across the
   declared family, and line 119 leaves open "whether alpha and sigma count as two
   independent tests". This script runs one band at a time, so it cannot settle
   that. It reports BOTH: BH over the 3 tests in this band, and BH over a family of
   6 assuming the other band is counted too. Raw p-values are written out so a
   single 6-test correction can be done across both runs afterwards.

6. SEQUENTIAL TESTING. The three hypotheses are also run as a fixed-sequence
   gatekeeping procedure in the stated order A -> B -> C, each at alpha = 0.05,
   stopping at the first failure. Fixed-sequence testing controls the family-wise
   error rate without any p-value penalty, but only while the sequence holds. It is
   reported beside BH, not instead of it.

================================================================================
ONE ADDITION THAT IS CLAUDE'S AND NOT THE PLAN'S, flagged per Part 6 of the
project instructions.

  B AND C SHARE A CONFOUND, AND NEITHER ONE ALONE CAN SEPARATE IT.

  B asks whether left-hemisphere ISO exceeds right-hemisphere ISO while lying on
  the left. C asks whether it falls below while lying on the right. A subject with
  a FIXED left-right asymmetry -- unequal electrode impedance, a skull or cortical
  asymmetry, a montage error -- produces the same sign of D in BOTH positions. Such
  a subject satisfies B and contradicts C, with no position effect whatsoever.

  The quantity that isolates the position effect is the difference of the two
  differences, computed on subjects who contributed both positions:

      D_interaction = D_B - D_C = (left-right | left) - (left-right | right)

  Any constant per-subject asymmetry cancels exactly. The hypotheses as written
  predict D_B > 0 and D_C < 0, so they jointly predict D_interaction > 0. This is
  reported as a secondary paired contrast. It is NOT a substitute for B and C, and
  it is not in the plan; it is here because B and C cannot be interpreted without
  it.

================================================================================
MISSING DATA. Each hypothesis uses the subjects who have both of its own columns
(pairwise-complete). MEASURED: the supine columns are missing for 2.4% of rows,
the left and right position columns for about 30%, because many recordings never
record a left or a right position at all. The n behind every number is reported.
Subjects are NOT imputed and NOT carried across hypotheses, so A, B and C rest on
different, overlapping subject sets, which is stated in the output.

Outputs land in <outdir>/<band>/ : result CSVs, plots, and a readable summary.
"""

import argparse
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SCRIPT_VERSION = "analysis.py v1.0 (2026-09-11)"

DEFAULT_SRC = r"C:\Users\wuouj\OneDrive\Desktop\Research-Sun"
DEFAULT_OUT = r"D:\HSP\ISO_analysis_results"
DEFAULT_COV = r"C:\Users\wuouj\OneDrive\Desktop\Research-Sun\covariates.csv"

ALPHA = 0.05
N_BOOT = 10000

COVARIATES = ["Age", "Sex_male", "BMI"]     # plan rev5: age, sex, BMI

# Ledger D12, Dr. Sun 27AUG ">> Yes, median". Confirmed by Jacob 2026-09-18,
# "We use median". The median is the primary summary statistic: it sets the point
# estimate, the interval, the p-value that drives every verdict, and the family
# that Benjamini-Hochberg corrects. The mean is still computed and reported beside
# it, because the two disagree materially on these contrasts -- see below.
STATISTIC = "median"

# name -> (left column, right column, predicted sign, plain statement)
HYPOTHESES = {
    "A": ("occipital_iso_supine", "frontal_iso_supine", +1,
          "Supine: occipital (O1,O2) ISO > frontal (F3,F4) ISO"),
    "B": ("left_iso_left", "right_iso_left", +1,
          "Left position: left-hemisphere (O1,C3,F3) ISO > right-hemisphere (O2,C4,F4) ISO"),
    "C": ("left_iso_right", "right_iso_right", -1,
          "Right position: left-hemisphere (O1,C3,F3) ISO < right-hemisphere (O2,C4,F4) ISO"),
}
ORDER = ["A", "B", "C"]


# ---------------------------------------------------------------- covariates
def load_covariates(path):
    """Covariates keyed to BDSPID = SiteID + BDSPPatientID, as the ISO tables key them.

    BDSPPatientID is stored as a float, so it is cast through int64 to recover the
    identifier. Rows missing SiteID or BDSPPatientID cannot be joined at all and
    are dropped here; on the supplied file those are the 116 entirely blank rows.
    """
    cv = pd.read_csv(path)
    need = {"SiteID", "BDSPPatientID", "SessionID", "Age", "Sex", "BMI"}
    missing = need - set(cv.columns)
    if missing:
        sys.exit("covariate file is missing columns: %s" % sorted(missing))
    n_raw = len(cv)
    joinable = cv["SiteID"].notna() & cv["BDSPPatientID"].notna()
    cv = cv[joinable].copy()
    frac = cv["BDSPPatientID"] % 1
    if (frac != 0).any():
        sys.exit("BDSPPatientID has %d non-integral values and cannot be used as a key"
                 % int((frac != 0).sum()))
    cv["BDSPID"] = (cv["SiteID"].astype(str)
                    + cv["BDSPPatientID"].astype("int64").astype(str))
    dup = cv.duplicated(["BDSPID", "SessionID"]).sum()
    if dup:
        sys.exit("covariate file has %d duplicated (BDSPID, SessionID) keys" % dup)
    # Sex as a 0/1 indicator; anything not recognised becomes missing rather than 0
    cv["Sex_male"] = cv["Sex"].map({"Male": 1.0, "Female": 0.0})
    keep = ["BDSPID", "SessionID", "Age", "Sex", "Sex_male", "BMI"]
    if "Race" in cv.columns:
        keep.append("Race")
    return cv[keep], n_raw


def fit_centred(diff, X, statistic="mean"):
    """Regress diff on mean-centred X with an intercept. Returns the coefficients.

    statistic="mean"   ordinary least squares. With X mean-centred the intercept
                       is ALGEBRAICALLY the mean of diff, so adjustment cannot
                       move the point estimate at all.
    statistic="median" median (quantile) regression at q = 0.5, the analogue of
                       the plan's model when the summary statistic is the median
                       (ledger D12). Here the intercept is only APPROXIMATELY the
                       median of diff -- no algebraic identity holds -- so
                       adjustment CAN move the point estimate. That is a real
                       difference from the mean version and is reported, not
                       hidden: the discrepancy is printed at run time.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    A = np.column_stack([np.ones(Xc.shape[0]), Xc])
    if statistic == "median":
        from statsmodels.regression.quantile_regression import QuantReg
        return np.asarray(QuantReg(diff, A).fit(q=0.5).params, dtype=float)
    beta, *_ = np.linalg.lstsq(A, diff, rcond=None)
    return beta


def adjusted_model(diff, X, clusters, n_boot, seed, predicted_sign):
    """Plan rev5's model: intercept of D ~ mean-centred covariates, clustered bootstrap.

    Slopes are bootstrapped in the same resamples, so their intervals share the
    clustering. Returns a dict of estimates or None if the design is rank-deficient.
    """
    if diff.size <= X.shape[1] + 2:
        return None
    beta = fit_centred(diff, X, STATISTIC)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(clusters, return_inverse=True)
    by_cluster = [np.where(inv == i)[0] for i in range(uniq.size)]
    n = uniq.size
    boots = np.empty((n_boot, X.shape[1] + 1))
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        idx = np.concatenate([by_cluster[j] for j in pick])
        try:
            boots[b] = fit_centred(diff[idx], X[idx], STATISTIC)
        except Exception:            # singular design, or QuantReg failing to converge
            boots[b] = np.nan
    boots = boots[~np.isnan(boots).any(axis=1)]
    if boots.shape[0] < max(100, n_boot // 10):
        return None
    lo, hi = np.percentile(boots, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)], axis=0)

    d0 = boots[:, 0]
    p_lo, p_hi = float(np.mean(d0 <= 0)), float(np.mean(d0 >= 0))
    p_two = max(min(1.0, 2 * min(p_lo, p_hi)), 1.0 / (len(boots) + 1))
    p_one = p_lo if predicted_sign > 0 else p_hi
    p_one = max(min(p_one, 1.0), 1.0 / (len(boots) + 1))

    out = {
        "n_adjusted": int(diff.size),
        "adj_intercept": float(beta[0]),
        "adj_ci_lo": float(lo[0]),
        "adj_ci_hi": float(hi[0]),
        "p_adjusted_two_sided": p_two,
        "p_adjusted_one_sided_predicted": p_one,
        "adj_statistic": STATISTIC,
        # Under the mean this is an algebraic identity and must be 0. Under the
        # median it is only an approximation, so a small nonzero value is
        # expected and is the amount the adjustment actually moved the estimate.
        "intercept_minus_unadjusted": float(
            beta[0] - (np.median(diff) if STATISTIC == "median" else np.mean(diff))),
    }
    for j, name in enumerate(COVARIATES, start=1):
        sl = boots[:, j]
        pl, ph = float(np.mean(sl <= 0)), float(np.mean(sl >= 0))
        out["slope_%s" % name] = float(beta[j])
        out["slope_%s_ci_lo" % name] = float(lo[j])
        out["slope_%s_ci_hi" % name] = float(hi[j])
        out["p_slope_%s" % name] = max(min(1.0, 2 * min(pl, ph)),
                                       1.0 / (len(boots) + 1))
    return out


# ----------------------------------------------------------------- bootstrap
def cluster_bootstrap(values, clusters, statistic, n_boot, seed):
    """Resample whole clusters with replacement. Returns the bootstrap distribution.

    With one row per subject each cluster is a singleton and this is an ordinary
    bootstrap; the machinery is kept so repeated sessions stay handled correctly.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(clusters, return_inverse=True)
    by_cluster = [np.where(inv == i)[0] for i in range(uniq.size)]
    n = uniq.size
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        idx = np.concatenate([by_cluster[j] for j in pick])
        out[b] = statistic(values[idx])
    return out


def boot_summary(values, clusters, statistic, n_boot, seed, predicted_sign):
    """Point estimate, percentile CI, and a two-sided bootstrap p-value against 0."""
    est = float(statistic(values))
    dist = cluster_bootstrap(values, clusters, statistic, n_boot, seed)
    lo, hi = np.percentile(dist, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    # two-sided: twice the smaller tail mass on the far side of zero
    p_lo = float(np.mean(dist <= 0))
    p_hi = float(np.mean(dist >= 0))
    p_two = min(1.0, 2 * min(p_lo, p_hi))
    p_two = max(p_two, 1.0 / (n_boot + 1))          # never report exactly zero
    # one-sided in the direction the hypothesis predicts
    p_one = p_lo if predicted_sign > 0 else p_hi
    p_one = max(min(p_one, 1.0), 1.0 / (n_boot + 1))
    return est, float(lo), float(hi), p_two, p_one, dist


# ----------------------------------------------------------------- one test
def run_contrast(name, d, col_hi, col_lo, predicted_sign, statement, n_boot, seed,
                 use_covariates=False):
    """All statistics for one paired contrast. Returns a result dict."""
    sub = d[["BDSPID", col_hi, col_lo]].dropna()
    x = sub[col_hi].to_numpy(float)
    y = sub[col_lo].to_numpy(float)
    diff = x - y
    n = diff.size
    if n < 10:
        raise SystemExit("hypothesis %s has only %d complete pairs; refusing to test"
                         % (name, n))
    clusters = sub["BDSPID"].to_numpy()

    # Both are computed every time. Which one is PRIMARY -- i.e. which supplies
    # the p-value that decides the verdict and enters the BH family -- is set by
    # STATISTIC, and is the median by default (ledger D12).
    mean_est, mean_lo, mean_hi, p_mean2, p_mean1, dist_mean = boot_summary(
        diff, clusters, np.mean, n_boot, seed, predicted_sign)
    med_est, med_lo, med_hi, p_med2, p_med1, dist_med = boot_summary(
        diff, clusters, np.median, n_boot, seed + 1, predicted_sign)

    if STATISTIC == "median":
        prim_est, prim_lo, prim_hi = med_est, med_lo, med_hi
        p_boot2, p_boot1, dist = p_med2, p_med1, dist_med
    else:
        prim_est, prim_lo, prim_hi = mean_est, mean_lo, mean_hi
        p_boot2, p_boot1, dist = p_mean2, p_mean1, dist_mean

    # Wilcoxon signed-rank, two-sided plus the predicted one-sided direction
    w_stat, w_p2 = stats.wilcoxon(diff, alternative="two-sided")
    w_p1 = stats.wilcoxon(
        diff, alternative="greater" if predicted_sign > 0 else "less").pvalue
    # rank-biserial correlation, the effect size that goes with the signed-rank test
    nz = diff[diff != 0]
    pos = float(np.sum(stats.rankdata(np.abs(nz))[nz > 0]))
    tot = float(np.sum(stats.rankdata(np.abs(nz))))
    rbc = 2 * pos / tot - 1 if tot else np.nan

    t_stat, t_p2 = stats.ttest_rel(x, y)
    dz = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) else np.nan

    observed_sign = int(np.sign(prim_est)) or 1
    matches = observed_sign == predicted_sign
    # Flag where the two statistics would give different answers, because on
    # these heavy-tailed contrasts they genuinely can.
    mean_sig = p_mean2 < ALPHA
    med_sig = p_med2 < ALPHA
    disagree = (mean_sig != med_sig) or (np.sign(mean_est) != np.sign(med_est))

    # ---- plan rev5's adjusted model, on the complete-covariate subset ------
    adj = None
    if use_covariates:
        cs = d[["BDSPID", col_hi, col_lo] + COVARIATES].dropna()
        if len(cs) >= 30:
            adj = adjusted_model(
                (cs[col_hi] - cs[col_lo]).to_numpy(float),
                cs[COVARIATES].to_numpy(float),
                cs["BDSPID"].to_numpy(), n_boot, seed + 50, predicted_sign)

    return {
        "hypothesis": name,
        "statement": statement,
        "column_high": col_hi,
        "column_low": col_lo,
        "predicted_sign": "positive" if predicted_sign > 0 else "negative",
        "n_pairs": n,
        "primary_statistic": STATISTIC,
        "primary_diff": prim_est,
        "primary_ci_lo": prim_lo,
        "primary_ci_hi": prim_hi,
        "mean_diff": mean_est,
        "mean_ci_lo": mean_lo,
        "mean_ci_hi": mean_hi,
        "p_mean_two_sided": p_mean2,
        "median_diff": med_est,
        "median_ci_lo": med_lo,
        "median_ci_hi": med_hi,
        "p_median_two_sided": p_med2,
        "mean_median_disagree": bool(disagree),
        "sd_diff": float(np.std(diff, ddof=1)),
        "skew": float(stats.skew(diff)),
        "excess_kurtosis": float(stats.kurtosis(diff)),
        "p_bootstrap_two_sided": p_boot2,
        "p_bootstrap_one_sided_predicted": p_boot1,
        "wilcoxon_stat": float(w_stat),
        "p_wilcoxon_two_sided": float(w_p2),
        "p_wilcoxon_one_sided_predicted": float(w_p1),
        "rank_biserial_r": float(rbc),
        "t_stat": float(t_stat),
        "p_ttest_two_sided": float(t_p2),
        "cohens_dz": dz,
        "direction_matches_prediction": bool(matches),
        "_adjusted": adj,
        "_diff": diff,
        "_x": x,
        "_y": y,
        "_boot": dist,
        "_ids": sub["BDSPID"].to_numpy(),
    }


def interaction_contrast(d, n_boot, seed):
    """D_B - D_C on subjects with BOTH positions. Claude's addition; see header."""
    cols = ["BDSPID", "left_iso_left", "right_iso_left",
            "left_iso_right", "right_iso_right"]
    sub = d[cols].dropna()
    db = sub["left_iso_left"].to_numpy(float) - sub["right_iso_left"].to_numpy(float)
    dc = sub["left_iso_right"].to_numpy(float) - sub["right_iso_right"].to_numpy(float)
    diff = db - dc
    if diff.size < 10:
        return None
    clusters = sub["BDSPID"].to_numpy()
    stat = np.median if STATISTIC == "median" else np.mean
    est, lo, hi, p2, p1, dist = boot_summary(
        diff, clusters, stat, n_boot, seed, +1)
    _, w_p2 = stats.wilcoxon(diff, alternative="two-sided")
    return {
        "hypothesis": "B-C interaction",
        "statement": "Position effect on left-right asymmetry, (L-R | left) - (L-R | right); "
                     "cancels any fixed per-subject asymmetry. Predicted positive if B and C both hold.",
        "n_pairs": int(diff.size),
        "primary_statistic": STATISTIC,
        "mean_diff": est, "mean_ci_lo": lo, "mean_ci_hi": hi,
        "p_bootstrap_two_sided": p2,
        "p_bootstrap_one_sided_predicted": p1,
        "p_wilcoxon_two_sided": float(w_p2),
        "_diff": diff, "_db": db, "_dc": dc, "_boot": dist,
    }


# ------------------------------------------------------------- multiplicity
def benjamini_hochberg(pvals, q=ALPHA, m=None):
    """BH adjusted p-values. `m` allows a family larger than the vector supplied."""
    p = np.asarray(pvals, float)
    k = p.size
    m = k if m is None else m
    order = np.argsort(p)
    adj = np.empty(k)
    running = 1.0
    for rank, i in enumerate(order[::-1]):
        j = k - rank                      # 1-based rank within the supplied vector
        running = min(running, p[i] * m / j)
        adj[i] = running
    return np.minimum(adj, 1.0)


def fixed_sequence(results, order, alpha=ALPHA):
    """Gatekeeping in the stated order; stops at the first non-significant test."""
    out, gate_open = {}, True
    for name in order:
        r = results[name]
        if not gate_open:
            out[name] = "not tested (sequence stopped earlier)"
            continue
        sig = r["p_bootstrap_two_sided"] < alpha
        if sig and r["direction_matches_prediction"]:
            out[name] = "PASS"
        elif sig:
            out[name] = "STOP: significant in the OPPOSITE direction"
            gate_open = False
        else:
            out[name] = "STOP: not significant"
            gate_open = False
    return out


def verdict(r):
    sig = r["p_bootstrap_two_sided"] < ALPHA
    if not sig:
        return "NOT SUPPORTED (no difference detected)"
    return ("SUPPORTED" if r["direction_matches_prediction"]
            else "CONTRADICTED (significant in the opposite direction)")


# -------------------------------------------------------------------- plots
def plot_forest(results, inter, band, path):
    rows = [(n, results[n]["primary_diff"], results[n]["primary_ci_lo"],
             results[n]["primary_ci_hi"], results[n]["predicted_sign"]) for n in ORDER]
    if inter:
        rows.append(("B-C", inter["mean_diff"], inter["mean_ci_lo"],
                     inter["mean_ci_hi"], "positive"))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ys = np.arange(len(rows))[::-1]
    for y, (name, est, lo, hi, psign) in zip(ys, rows):
        ok = (est > 0) if psign == "positive" else (est < 0)
        colour = "#1b7837" if ok else "#b2182b"
        ax.plot([lo, hi], [y, y], color=colour, lw=2.4, solid_capstyle="round")
        ax.plot([est], [y], "o", color=colour, ms=7, zorder=3)
    ax.axvline(0, color="0.35", lw=1, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("paired %s difference in ISO strength (95%% clustered bootstrap CI)"
                  % STATISTIC)
    ax.set_title("%s band: contrasts A, B, C%s\ngreen = in predicted direction, "
                 "red = against it" % (band, " and the B-C interaction" if inter else ""))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distributions(results, band, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, name in zip(axes, ORDER):
        r = results[name]
        d = r["_diff"]
        lim = np.percentile(np.abs(d), 99.0)
        ax.hist(np.clip(d, -lim, lim), bins=70, color="#4393c3",
                edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="0.3", ls="--", lw=1)
        ax.axvline(r["mean_diff"], color="#b2182b", lw=2,
                   label="mean %+.4f" % r["mean_diff"])
        ax.axvline(r["median_diff"], color="#1b7837", lw=2,
                   label="median %+.4f" % r["median_diff"])
        ax.set_title("%s  (n=%d)\npredicted %s" %
                     (name, r["n_pairs"], r["predicted_sign"]), fontsize=10)
        ax.set_xlabel("difference")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("subjects")
    fig.suptitle("%s band: within-person paired differences "
                 "(x clipped at the 99th percentile of |d| for display)" % band,
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_paired_scatter(results, band, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, name in zip(axes, ORDER):
        r = results[name]
        x, y = r["_y"], r["_x"]          # x axis = the "low" column
        hi = np.percentile(np.concatenate([x, y]), 99.5)
        ax.scatter(x, y, s=5, alpha=0.25, color="#2166ac", edgecolors="none")
        ax.plot([0, hi], [0, hi], color="0.35", ls="--", lw=1)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel(r["column_low"], fontsize=8)
        ax.set_ylabel(r["column_high"], fontsize=8)
        frac = float(np.mean(r["_diff"] > 0))
        ax.set_title("%s  %.1f%% of subjects above the line" % (name, 100 * frac),
                     fontsize=10)
    fig.suptitle("%s band: paired values, identity line dashed" % band, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bootstrap(results, band, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, name in zip(axes, ORDER):
        r = results[name]
        ax.hist(r["_boot"], bins=60, color="#92c5de", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="0.3", ls="--", lw=1)
        ax.axvline(r["mean_ci_lo"], color="#b2182b", lw=1.4)
        ax.axvline(r["mean_ci_hi"], color="#b2182b", lw=1.4)
        ax.set_title("%s  bootstrap mean\n95%% CI [%+.4f, %+.4f]"
                     % (name, r["mean_ci_lo"], r["mean_ci_hi"]), fontsize=10)
        ax.set_xlabel("bootstrap mean difference")
    fig.suptitle("%s band: clustered bootstrap distributions of the %s"
                 % (band, STATISTIC), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_interaction(inter, band, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    ax.scatter(inter["_dc"], inter["_db"], s=6, alpha=0.3,
               color="#762a83", edgecolors="none")
    lim = np.percentile(np.abs(np.concatenate([inter["_db"], inter["_dc"]])), 99)
    ax.plot([-lim, lim], [-lim, lim], color="0.35", ls="--", lw=1)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("D_C  =  (left - right) while lying RIGHT")
    ax.set_ylabel("D_B  =  (left - right) while lying LEFT")
    ax.set_title("Points on the identity line have a FIXED asymmetry\n"
                 "and no position effect", fontsize=9.5)
    ax = axes[1]
    d = inter["_diff"]
    lim2 = np.percentile(np.abs(d), 99)
    ax.hist(np.clip(d, -lim2, lim2), bins=70, color="#9970ab",
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="0.3", ls="--", lw=1)
    ax.axvline(inter["mean_diff"], color="#b2182b", lw=2,
               label="%s %+.4f" % (STATISTIC, inter["mean_diff"]))
    ax.set_xlabel("D_B - D_C")
    ax.set_title("Position effect, fixed asymmetry removed (n=%d)" % inter["n_pairs"],
                 fontsize=9.5)
    ax.legend(fontsize=8)
    fig.suptitle("%s band: separating a position effect from a fixed left-right "
                 "asymmetry" % band, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    global STATISTIC
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--band", required=True, choices=["alpha", "sigma"],
                    help="which ISO table to analyse; the two are analysed separately")
    ap.add_argument("--csv", default=None,
                    help="path to ISO_table_<band>.csv (default: the Research-Sun folder)")
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    ap.add_argument("--covariates", default=DEFAULT_COV,
                    help="covariate csv keyed on SiteID + BDSPPatientID + SessionID")
    ap.add_argument("--no-covariates", action="store_true",
                    help="skip the adjusted model and report the unadjusted analysis only")
    ap.add_argument("--statistic", default=STATISTIC, choices=["median", "mean"],
                    help="primary summary statistic; median is ledger D12 and the "
                         "default. Sets the point estimate, the interval, the "
                         "p-value behind every verdict, and the BH family.")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=20260911)
    a = ap.parse_args()
    STATISTIC = a.statistic

    src = a.csv or os.path.join(DEFAULT_SRC, "ISO_table_%s.csv" % a.band)
    if not os.path.isfile(src):
        sys.exit("no such file: %s" % src)
    outdir = os.path.join(a.outdir, a.band)
    os.makedirs(outdir, exist_ok=True)

    d = pd.read_csv(src)
    need = {c for h in HYPOTHESES.values() for c in h[:2]}
    missing = need - set(d.columns)
    if missing:
        sys.exit("input is missing required columns: %s" % sorted(missing))

    lines = []

    def say(m=""):
        lines.append(m)
        print(m)

    say(SCRIPT_VERSION)
    say("band              : %s" % a.band)
    say("input             : %s" % src)
    say("rows              : %d,  distinct BDSPID: %d" % (len(d), d.BDSPID.nunique()))
    say("bootstrap         : %d clustered resamples, seed %d" % (a.n_boot, a.seed))
    say("summary statistic : %s  (ledger D12, Dr. Sun 27AUG; confirmed 2026-09-18)"
        % STATISTIC)
    say("                    the other statistic is reported beside it, and any")
    say("                    contrast where the two disagree is flagged")

    # ---- covariates --------------------------------------------------------
    use_cov = False
    if not a.no_covariates and os.path.isfile(a.covariates):
        cv, n_raw = load_covariates(a.covariates)
        before = len(d)
        d = d.merge(cv, on=["BDSPID", "SessionID"], how="left")
        n_match = d["Age"].notna().sum()
        n_full = d[COVARIATES].notna().all(axis=1).sum()
        use_cov = n_full >= 30
        say("covariates        : %s" % a.covariates)
        say("                    %d rows in file, %d joinable, %d of %d ISO rows"
            % (n_raw, len(cv), n_match, before))
        say("                    matched, %d with a complete Age, Sex and BMI" % n_full)
        say("                    model: D ~ Age + Sex + BMI, covariates MEAN-CENTRED")
        say("                    so the intercept is the adjusted mean of D")
        say("                    (ledger Q-B3 is open; this is Claude's resolution)")
        merged = os.path.join(outdir, "ISO_table_%s_with_covariates.csv" % a.band)
        d.to_csv(merged, index=False)
        say("                    merged table written to %s" % os.path.basename(merged))
    elif a.no_covariates:
        say("covariates        : skipped on request; results are UNADJUSTED")
    else:
        say("covariates        : NOT FOUND at %s; results are UNADJUSTED" % a.covariates)
    if d.BDSPID.nunique() == len(d):
        say("clustering        : every subject appears once, so the clustered")
        say("                    bootstrap is arithmetically an ordinary bootstrap")
    say("=" * 78)

    # ---- the three hypotheses, in order -----------------------------------
    results = {}
    for i, name in enumerate(ORDER):
        col_hi, col_lo, sign, statement = HYPOTHESES[name]
        results[name] = run_contrast(name, d, col_hi, col_lo, sign, statement,
                                     a.n_boot, a.seed + 100 * i,
                                     use_covariates=use_cov)

    p_two = [results[n]["p_bootstrap_two_sided"] for n in ORDER]
    bh3 = benjamini_hochberg(p_two, m=3)
    bh6 = benjamini_hochberg(p_two, m=6)
    for n, q3, q6 in zip(ORDER, bh3, bh6):
        results[n]["p_bh_within_band_m3"] = float(q3)
        results[n]["p_bh_both_bands_m6"] = float(q6)
        results[n]["verdict"] = verdict(results[n])

    seq = fixed_sequence(results, ORDER)

    for name in ORDER:
        r = results[name]
        say("")
        say("HYPOTHESIS %s  --  %s" % (name, r["statement"]))
        say("  contrast          : %s  minus  %s" % (r["column_high"], r["column_low"]))
        say("  predicted sign    : %s" % r["predicted_sign"])
        say("  subjects (pairs)  : %d" % r["n_pairs"])
        star = lambda nm: " <-- PRIMARY" if nm == STATISTIC else ""
        say("  mean difference   : %+.6f   95%% CI [%+.6f, %+.6f]  p=%.4g%s"
            % (r["mean_diff"], r["mean_ci_lo"], r["mean_ci_hi"],
               r["p_mean_two_sided"], star("mean")))
        say("  median difference : %+.6f   95%% CI [%+.6f, %+.6f]  p=%.4g%s"
            % (r["median_diff"], r["median_ci_lo"], r["median_ci_hi"],
               r["p_median_two_sided"], star("median")))
        if r["mean_median_disagree"]:
            say("  ** the mean and the median DISAGREE on this contrast, either in")
            say("     sign or in significance. The verdict follows the %s (D12)."
                % STATISTIC)
        say("  distribution      : sd %.5f, skew %+.2f, excess kurtosis %+.1f"
            % (r["sd_diff"], r["skew"], r["excess_kurtosis"]))
        say("  bootstrap p       : %.4g two-sided | %.4g one-sided (predicted dir.)"
            % (r["p_bootstrap_two_sided"], r["p_bootstrap_one_sided_predicted"]))
        say("  Wilcoxon p        : %.4g two-sided | %.4g one-sided (predicted dir.)"
            % (r["p_wilcoxon_two_sided"], r["p_wilcoxon_one_sided_predicted"]))
        say("  paired t p        : %.4g two-sided   (reported, but see kurtosis)"
            % r["p_ttest_two_sided"])
        say("  effect size       : rank-biserial %+.3f, Cohen's dz %+.3f"
            % (r["rank_biserial_r"], r["cohens_dz"]))
        say("  BH adjusted p     : %.4g (family of 3, this band) | %.4g (family of 6)"
            % (r["p_bh_within_band_m3"], r["p_bh_both_bands_m6"]))
        say("  direction         : %s"
            % ("as predicted" if r["direction_matches_prediction"]
               else "OPPOSITE to the prediction"))
        adj = r["_adjusted"]
        if adj:
            say("  ADJUSTED (plan rev5: intercept of D ~ Age + Sex + BMI, centred)")
            say("    n complete      : %d of %d  (%d dropped for missing covariates)"
                % (adj["n_adjusted"], r["n_pairs"], r["n_pairs"] - adj["n_adjusted"]))
            say("    intercept       : %+.6f   95%% CI [%+.6f, %+.6f]"
                % (adj["adj_intercept"], adj["adj_ci_lo"], adj["adj_ci_hi"]))
            say("    p               : %.4g two-sided | %.4g one-sided (predicted dir.)"
                % (adj["p_adjusted_two_sided"], adj["p_adjusted_one_sided_predicted"]))
            delta = adj["intercept_minus_unadjusted"]
            if adj["adj_statistic"] == "mean":
                say("    identity check  : intercept minus mean(D) = %+.3e  %s"
                    % (delta, "as algebra requires" if abs(delta) < 1e-9
                       else "*** NOT ZERO, the centring is wrong ***"))
            else:
                say("    adjustment moved the estimate by %+.3e (median regression"
                    % delta)
                say("                      has no exact centring identity, unlike OLS)")
            say("    covariate slopes (per unit; Sex_male is Female->Male):")
            for cname in COVARIATES:
                say("      %-9s %+.3e  95%% CI [%+.3e, %+.3e]  p=%.4g"
                    % (cname, adj["slope_%s" % cname],
                       adj["slope_%s_ci_lo" % cname], adj["slope_%s_ci_hi" % cname],
                       adj["p_slope_%s" % cname]))
        say("  VERDICT           : %s" % r["verdict"])

    # ---- sequential gatekeeping -------------------------------------------
    say("")
    say("=" * 78)
    say("SEQUENTIAL (fixed-sequence) TESTING, order A -> B -> C, alpha = %.2f" % ALPHA)
    say("  Controls the family-wise error rate with no p-value penalty, but only")
    say("  for as long as the sequence keeps passing.")
    for name in ORDER:
        say("    %s : %s" % (name, seq[name]))

    # ---- interaction -------------------------------------------------------
    inter = interaction_contrast(d, a.n_boot, a.seed + 999)
    say("")
    say("=" * 78)
    say("SECONDARY, NOT IN THE PLAN (Claude's addition, flagged per Part 6)")
    say("  B and C cannot separate a position effect from a FIXED left-right")
    say("  asymmetry: a subject with unequal impedance or anatomy shows the same")
    say("  sign in both positions, satisfying B and contradicting C with no")
    say("  position effect at all. The difference of the two differences cancels")
    say("  any such constant.")
    if inter:
        say("  D_B - D_C         : %+.6f   95%% CI [%+.6f, %+.6f]  (n=%d)"
            % (inter["mean_diff"], inter["mean_ci_lo"], inter["mean_ci_hi"],
               inter["n_pairs"]))
        say("  bootstrap p       : %.4g two-sided" % inter["p_bootstrap_two_sided"])
        say("  Wilcoxon p        : %.4g two-sided" % inter["p_wilcoxon_two_sided"])
        say("  reading           : %s" % (
            "a real position effect survives once a fixed asymmetry is removed"
            if inter["p_bootstrap_two_sided"] < ALPHA and inter["mean_diff"] > 0
            else "no position effect survives once a fixed asymmetry is removed"))
    else:
        say("  too few subjects contributed both positions to compute it")

    # ---- write outputs -----------------------------------------------------
    def drop(r):
        out = {k: v for k, v in r.items() if not k.startswith("_")}
        if r.get("_adjusted"):
            out.update(r["_adjusted"])
        return out

    res_df = pd.DataFrame([drop(results[n]) for n in ORDER])
    res_df.insert(0, "band", a.band)
    res_df["sequential_result"] = [seq[n] for n in ORDER]
    res_path = os.path.join(outdir, "hypothesis_results_%s.csv" % a.band)
    res_df.to_csv(res_path, index=False)

    if inter:
        pd.DataFrame([drop(inter)]).assign(band=a.band).to_csv(
            os.path.join(outdir, "interaction_B_minus_C_%s.csv" % a.band), index=False)

    per = pd.DataFrame({"BDSPID": d["BDSPID"], "SessionID": d["SessionID"]})
    for name in ORDER:
        col_hi, col_lo, _, _ = HYPOTHESES[name]
        per["D_%s" % name] = d[col_hi] - d[col_lo]
    per["D_B_minus_D_C"] = per["D_B"] - per["D_C"]
    per_path = os.path.join(outdir, "per_subject_contrasts_%s.csv" % a.band)
    per.to_csv(per_path, index=False)

    plot_forest(results, inter, a.band, os.path.join(outdir, "forest_%s.png" % a.band))
    plot_distributions(results, a.band,
                       os.path.join(outdir, "difference_distributions_%s.png" % a.band))
    plot_paired_scatter(results, a.band,
                        os.path.join(outdir, "paired_scatter_%s.png" % a.band))
    plot_bootstrap(results, a.band,
                   os.path.join(outdir, "bootstrap_means_%s.png" % a.band))
    if inter:
        plot_interaction(inter, a.band,
                         os.path.join(outdir, "interaction_%s.png" % a.band))

    say("")
    say("=" * 78)
    say("SUBJECT SETS DIFFER between hypotheses (pairwise-complete):")
    for name in ORDER:
        say("   %s : n = %d" % (name, results[name]["n_pairs"]))
    say("   A, B and C therefore rest on different, overlapping subject sets.")
    say("")
    say("written to %s" % outdir)

    with open(os.path.join(outdir, "summary_%s.txt" % a.band), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
