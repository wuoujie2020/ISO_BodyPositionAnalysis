#!/usr/bin/env python3
r"""
subset_analysis.py -- the prespecified subset analyses for hypotheses A, B and C.

    python subset_analysis.py                 # both bands
    python subset_analysis.py --band sigma

Plan rev5 lines 121-126: "Repeat hypothesis A in people with a supine apnea index
above 15 per hour, repeat hypotheses B and C in people with a left or right index
above 15, and split the analysis by age band and sex. ... These subsets condition
on a post-exposure variable, so they are prespecified and reported but cannot be
read causally."

The contrasts, the bootstrap, the covariate model and the verdict rule are imported
from analysis.py so the two scripts cannot drift apart. Only the row selection and
the reporting differ.

================================================================================
TWO THINGS ABOUT THE SUPPLIED AHI FILE THAT CHANGE WHAT THESE SUBSETS CAN SHOW.
Both are measured, not assumed, and both are reported again at run time.

1. THERE IS NO LEFT/RIGHT APNEA INDEX. AHI_body_position.csv carries AHI_supine and
   AHI_lateral only. The plan asks for hypotheses B and C to be repeated "in people
   with a left or right index above 15"; that cannot be done, because lateral is a
   single number covering both sides. B and C are therefore both subset on
   AHI_lateral > 15, which means THE TWO SUBSETS ARE DEFINED BY THE SAME VARIABLE
   and neither is specific to the side the subject is lying on. Plan rev5 line 124
   already flags this as unconfirmed; it is now confirmed absent.

2. THE SUPINE THRESHOLD BARELY SUBSETS THIS COHORT. Measured on the supplied file:
   median AHI_supine is 54 per hour and 98.7 percent of subjects with a value
   exceed 15. Restricting hypothesis A to AHI_supine > 15 removes about 1 percent
   of the analysable subjects, so that subset is very nearly the whole-sample
   analysis and should not be read as evidence about severe disease specifically.
   AHI_lateral > 15 is more selective at 80.4 percent, but still keeps most people.
   This is a severe-OSA referral cohort, not a general population.

Missingness in the index itself: AHI_supine is absent for 201 rows and AHI_lateral
for 433 of 3,046. Subjects with no index CANNOT enter a subset defined by it and
are dropped, which is a further, separate loss from the missing-covariate loss.

================================================================================
AGE BANDS. The plan says "age band and sex" without giving cut points; the request
gives "<50y / 50-70y / >70y". Read literally that leaves 70 itself unassigned, so
the bands used are  Age < 50,  50 <= Age <= 70,  Age > 70,  which is the reading
that keeps every subject. This is Claude's resolution of an unstated boundary and
is flagged per Part 6.

COVARIATE ADJUSTMENT INSIDE STRATA. In the AHI subsets the full plan model
D ~ Age + Sex + BMI is fitted. Inside an age-by-sex cell, Sex is constant and Age
varies only within the band, so Sex is dropped and the model becomes D ~ Age + BMI.
Fitting a constant would make the design rank-deficient.

MULTIPLICITY. Each family is corrected separately with Benjamini-Hochberg at
q = 0.05: the AHI family is 3 tests per band, the age-by-sex family is 18 per band
(3 hypotheses x 6 cells). These are secondary analyses that condition on a
post-exposure variable, so significance here is descriptive, never causal.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A_                     # contrasts, bootstrap, BH, covariates

SCRIPT_VERSION = "subset_analysis.py v1.0 (2026-09-15)"

DEFAULT_SRC = A_.DEFAULT_SRC
DEFAULT_COV = A_.DEFAULT_COV
DEFAULT_AHI = r"D:\HSP\AHI_body_position.csv"
DEFAULT_OUT = r"D:\HSP\ISO_subset_analysis_results"

AHI_CUT = 15.0
# hypothesis -> the index its subset is defined on
AHI_COLUMN = {"A": "AHI_supine", "B": "AHI_lateral", "C": "AHI_lateral"}

AGE_BANDS = [("<50", lambda s: s < 50),
             ("50-70", lambda s: (s >= 50) & (s <= 70)),
             (">70", lambda s: s > 70)]
SEXES = ["Female", "Male"]


def load_ahi(path):
    """Position-specific apnea index, keyed the way the ISO tables are keyed."""
    ah = pd.read_csv(path)
    need = {"SiteID", "BDSPPatientID", "SessionID", "AHI_supine", "AHI_lateral"}
    missing = need - set(ah.columns)
    if missing:
        sys.exit("AHI file is missing columns: %s" % sorted(missing))
    ah = ah[ah["SiteID"].notna() & ah["BDSPPatientID"].notna()].copy()
    ah["BDSPID"] = (ah["SiteID"].astype(str)
                    + ah["BDSPPatientID"].astype("int64").astype(str))
    dup = ah.duplicated(["BDSPID", "SessionID"]).sum()
    if dup:
        sys.exit("AHI file has %d duplicated (BDSPID, SessionID) keys" % dup)
    return ah[["BDSPID", "SessionID", "AHI_supine", "AHI_lateral"]]


def adjusted(diff, X, clusters, names, n_boot, seed, sign):
    """Plan model on an arbitrary covariate list; mean-centred, clustered bootstrap."""
    if diff.size <= X.shape[1] + 10 or np.linalg.matrix_rank(
            X - X.mean(axis=0, keepdims=True)) < X.shape[1]:
        return None
    beta = A_.fit_centred(diff, X)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(clusters, return_inverse=True)
    by = [np.where(inv == i)[0] for i in range(uniq.size)]
    n = uniq.size
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate([by[j] for j in rng.integers(0, n, n)])
        try:
            rows.append(A_.fit_centred(diff[idx], X[idx]))
        except np.linalg.LinAlgError:
            pass
    if len(rows) < n_boot // 2:
        return None
    boots = np.asarray(rows)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    d0 = boots[:, 0]
    p2 = max(min(1.0, 2 * min(np.mean(d0 <= 0), np.mean(d0 >= 0))),
             1.0 / (len(boots) + 1))
    out = {"adj_intercept": float(beta[0]), "adj_ci_lo": float(lo[0]),
           "adj_ci_hi": float(hi[0]), "p_adjusted_two_sided": float(p2),
           "adj_model": "D ~ " + " + ".join(names),
           "intercept_minus_mean": float(beta[0] - np.mean(diff))}
    for j, nm in enumerate(names, start=1):
        out["slope_%s" % nm] = float(beta[j])
        out["slope_%s_ci_lo" % nm] = float(lo[j])
        out["slope_%s_ci_hi" % nm] = float(hi[j])
    return out


def run_one(dsub, hyp, n_boot, seed, cov_names):
    """One hypothesis on one subset of rows. Returns a flat result dict or None."""
    col_hi, col_lo, sign, statement = A_.HYPOTHESES[hyp]
    sub = dsub[["BDSPID", col_hi, col_lo]].dropna()
    n = len(sub)
    if n < 30:
        return {"hypothesis": hyp, "n_pairs": int(n), "insufficient": True,
                "verdict": "NOT TESTED (n < 30)"}
    diff = (sub[col_hi] - sub[col_lo]).to_numpy(float)
    clusters = sub["BDSPID"].to_numpy()
    est, lo, hi, p2, p1, _ = A_.boot_summary(diff, clusters, np.mean,
                                             n_boot, seed, sign)
    med, mlo, mhi, _, _, _ = A_.boot_summary(diff, clusters, np.median,
                                             n_boot, seed + 1, sign)
    w_p2 = float(stats.wilcoxon(diff, alternative="two-sided").pvalue)
    matches = (int(np.sign(est)) or 1) == sign

    res = {"hypothesis": hyp, "insufficient": False, "n_pairs": int(n),
           "predicted_sign": "positive" if sign > 0 else "negative",
           "mean_diff": est, "mean_ci_lo": lo, "mean_ci_hi": hi,
           "median_diff": med, "median_ci_lo": mlo, "median_ci_hi": mhi,
           "p_bootstrap_two_sided": p2, "p_bootstrap_one_sided_predicted": p1,
           "p_wilcoxon_two_sided": w_p2,
           "direction_matches_prediction": bool(matches)}

    if cov_names:
        cs = dsub[["BDSPID", col_hi, col_lo] + cov_names].dropna()
        if len(cs) >= 40:
            adj = adjusted((cs[col_hi] - cs[col_lo]).to_numpy(float),
                           cs[cov_names].to_numpy(float),
                           cs["BDSPID"].to_numpy(), cov_names,
                           n_boot, seed + 7, sign)
            if adj:
                adj["n_adjusted"] = int(len(cs))
                res.update(adj)
    return res


def finish(rows, alpha=A_.ALPHA):
    """Attach BH across the tested rows of one family, plus a verdict."""
    tested = [r for r in rows if not r.get("insufficient")]
    if tested:
        q = A_.benjamini_hochberg([r["p_bootstrap_two_sided"] for r in tested])
        for r, qq in zip(tested, q):
            r["p_bh_within_family"] = float(qq)
            sig = r["p_bootstrap_two_sided"] < alpha
            r["verdict"] = ("NOT SUPPORTED (no difference detected)" if not sig
                            else "SUPPORTED" if r["direction_matches_prediction"]
                            else "CONTRADICTED (significant in the opposite direction)")
    return rows


# --------------------------------------------------------------------- plots
def plot_ahi(rows, full, band, path):
    """Whole sample against the AHI subset, one row per hypothesis."""
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ys, labels = [], []
    y = 0
    for hyp in A_.ORDER:
        r = next((x for x in rows if x["hypothesis"] == hyp), None)
        f = full.get(hyp)
        for tag, src, colour in (("all", f, "#8c8c8c"), ("AHI>15", r, "#2166ac")):
            if not src or src.get("insufficient"):
                continue
            ok = ((src["mean_diff"] > 0) if src["predicted_sign"] == "positive"
                  else (src["mean_diff"] < 0))
            ax.plot([src["mean_ci_lo"], src["mean_ci_hi"]], [y, y],
                    color=colour, lw=2.4, solid_capstyle="round")
            ax.plot([src["mean_diff"]], [y], "o", color=colour, ms=6,
                    markeredgecolor="#1b7837" if ok else "#b2182b", markeredgewidth=1.6)
            ys.append(y)
            labels.append("%s  %s (n=%d)" % (hyp, tag, src["n_pairs"]))
            y -= 1
        y -= 0.5
    ax.axvline(0, color="0.35", lw=1, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("paired difference (95% clustered bootstrap CI)")
    ax.set_title("%s band: whole sample (grey) vs position-specific AHI > 15 (blue)\n"
                 "marker edge green = predicted direction, red = against it" % band,
                 fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_agesex(rows, band, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharex=False)
    for ax, hyp in zip(axes, A_.ORDER):
        sel = [r for r in rows if r["hypothesis"] == hyp]
        ys, labels = [], []
        for i, r in enumerate(sel):
            y = -i
            lab = "%s %s (n=%d)" % (r["age_band"], r["sex"], r["n_pairs"])
            if r.get("insufficient"):
                labels.append(lab + " -")
                ys.append(y)
                continue
            ok = ((r["mean_diff"] > 0) if r["predicted_sign"] == "positive"
                  else (r["mean_diff"] < 0))
            colour = "#1b7837" if ok else "#b2182b"
            ax.plot([r["mean_ci_lo"], r["mean_ci_hi"]], [y, y], color=colour, lw=2.2)
            ax.plot([r["mean_diff"]], [y], "o", color=colour, ms=6)
            ys.append(y)
            labels.append(lab)
        ax.axvline(0, color="0.35", lw=1, ls="--")
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title("Hypothesis %s (predicted %s)"
                     % (hyp, A_.HYPOTHESES[hyp][2] > 0 and "positive" or "negative"),
                     fontsize=10)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_xlabel("paired difference (95% CI)")
    fig.suptitle("%s band: by age band and sex   "
                 "(green = predicted direction, red = against it)" % band, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def run_band(band, args, outdir, say):
    src = os.path.join(DEFAULT_SRC, "ISO_table_%s.csv" % band)
    if not os.path.isfile(src):
        sys.exit("no such file: %s" % src)
    d = pd.read_csv(src)

    cv, _ = A_.load_covariates(args.covariates)
    d = d.merge(cv, on=["BDSPID", "SessionID"], how="left")
    ah = load_ahi(args.ahi)
    d = d.merge(ah, on=["BDSPID", "SessionID"], how="left")

    say("")
    say("=" * 78)
    say("BAND %s" % band.upper())
    say("  rows %d | covariates matched %d | AHI matched %d"
        % (len(d), int(d["Age"].notna().sum()), int(d["AHI_supine"].notna().sum())))
    say("  AHI_supine  present %d, above %g: %d"
        % (int(d.AHI_supine.notna().sum()), AHI_CUT, int((d.AHI_supine > AHI_CUT).sum())))
    say("  AHI_lateral present %d, above %g: %d"
        % (int(d.AHI_lateral.notna().sum()), AHI_CUT, int((d.AHI_lateral > AHI_CUT).sum())))
    say("  NOTE: no left/right index exists in the file, so B and C are both")
    say("        subset on AHI_lateral and neither is side-specific.")

    merged = os.path.join(outdir, "ISO_table_%s_with_covariates_and_AHI.csv" % band)
    d.to_csv(merged, index=False)

    # ---- whole-sample reference, for the comparison plot -------------------
    full = {}
    for i, hyp in enumerate(A_.ORDER):
        full[hyp] = run_one(d, hyp, args.n_boot, args.seed + 10 * i, None)

    # ---- family 1: position-specific AHI > 15 ------------------------------
    say("")
    say("-" * 78)
    say("FAMILY 1  position-specific AHI > %g" % AHI_CUT)
    ahi_rows = []
    for i, hyp in enumerate(A_.ORDER):
        col = AHI_COLUMN[hyp]
        dsub = d[d[col] > AHI_CUT]
        r = run_one(dsub, hyp, args.n_boot, args.seed + 1000 + 10 * i, A_.COVARIATES)
        r.update({"band": band, "family": "AHI>15", "subset_on": col,
                  "n_in_subset": int(len(dsub))})
        ahi_rows.append(r)
    finish(ahi_rows)
    for r in ahi_rows:
        f = full[r["hypothesis"]]
        if r.get("insufficient"):
            say("  %s  %s" % (r["hypothesis"], r["verdict"]))
            continue
        say("  %s on %-12s n=%4d (whole sample %4d)" %
            (r["hypothesis"], r["subset_on"], r["n_pairs"], f["n_pairs"]))
        say("      subset  %+.6f  [%+.6f, %+.6f]  p=%.4g  BH=%.4g"
            % (r["mean_diff"], r["mean_ci_lo"], r["mean_ci_hi"],
               r["p_bootstrap_two_sided"], r["p_bh_within_family"]))
        say("      all     %+.6f  [%+.6f, %+.6f]"
            % (f["mean_diff"], f["mean_ci_lo"], f["mean_ci_hi"]))
        if "adj_intercept" in r:
            say("      adjusted intercept %+.6f  [%+.6f, %+.6f]  (%s, n=%d)"
                % (r["adj_intercept"], r["adj_ci_lo"], r["adj_ci_hi"],
                   r["adj_model"], r["n_adjusted"]))
        say("      %s" % r["verdict"])

    # ---- family 2: age band x sex ------------------------------------------
    say("")
    say("-" * 78)
    say("FAMILY 2  age band x sex")
    cells = []
    k = 0
    for hyp in A_.ORDER:
        for aname, atest in AGE_BANDS:
            for sex in SEXES:
                dsub = d[atest(d["Age"]) & (d["Sex"] == sex)]
                r = run_one(dsub, hyp, args.n_boot, args.seed + 2000 + 10 * k,
                            ["Age", "BMI"])       # Sex is constant inside the cell
                r.update({"band": band, "family": "age_x_sex", "age_band": aname,
                          "sex": sex, "n_in_subset": int(len(dsub))})
                cells.append(r)
                k += 1
    finish(cells)
    for hyp in A_.ORDER:
        say("  hypothesis %s (predicted %s)"
            % (hyp, "positive" if A_.HYPOTHESES[hyp][2] > 0 else "negative"))
        for r in [x for x in cells if x["hypothesis"] == hyp]:
            if r.get("insufficient"):
                say("     %-6s %-6s  n=%4d   %s"
                    % (r["age_band"], r["sex"], r["n_pairs"], r["verdict"]))
                continue
            say("     %-6s %-6s  n=%4d  %+.6f  [%+.6f, %+.6f]  p=%.4g  BH=%.4g  %s"
                % (r["age_band"], r["sex"], r["n_pairs"], r["mean_diff"],
                   r["mean_ci_lo"], r["mean_ci_hi"], r["p_bootstrap_two_sided"],
                   r["p_bh_within_family"], r["verdict"].split(" (")[0]))

    # ---- write -------------------------------------------------------------
    pd.DataFrame(ahi_rows).to_csv(
        os.path.join(outdir, "subset_AHI_%s.csv" % band), index=False)
    pd.DataFrame(cells).to_csv(
        os.path.join(outdir, "subset_age_sex_%s.csv" % band), index=False)
    plot_ahi(ahi_rows, full, band, os.path.join(outdir, "subset_AHI_%s.png" % band))
    plot_agesex(cells, band, os.path.join(outdir, "subset_age_sex_%s.png" % band))
    return ahi_rows, cells


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--band", default="both", choices=["alpha", "sigma", "both"])
    ap.add_argument("--covariates", default=DEFAULT_COV)
    ap.add_argument("--ahi", default=DEFAULT_AHI)
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    ap.add_argument("--n-boot", type=int, default=A_.N_BOOT)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    lines = []

    def say(m=""):
        lines.append(m)
        print(m)

    say(SCRIPT_VERSION)
    say("contrasts, bootstrap and covariate model imported from analysis.py")
    say("AHI file  : %s" % args.ahi)
    say("age bands : <50 | 50-70 inclusive | >70   (70 itself falls in 50-70)")
    say("bootstrap : %d clustered resamples" % args.n_boot)
    say("CAUTION   : these subsets condition on a post-exposure variable. Plan")
    say("            rev5 line 126 -- prespecified and reported, but they cannot")
    say("            be read causally.")

    bands = ["alpha", "sigma"] if args.band == "both" else [args.band]
    for b in bands:
        run_band(b, args, args.outdir, say)

    say("")
    say("written to %s" % args.outdir)
    # Named by the bands actually run: a single-band run must not clobber the
    # summary a previous run wrote for the other band.
    stem = "summary_subsets_%s.txt" % "_".join(bands)
    with open(os.path.join(args.outdir, stem), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
