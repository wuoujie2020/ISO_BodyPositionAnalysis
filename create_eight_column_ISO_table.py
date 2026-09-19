#!/usr/bin/env python3
r"""
create_eight_column_ISO_table.py

One row per subject-session, eight columns:

  1 BDSPID                    site+subject, e.g. S0001113993796
  2 SessionID                 integer, from ses-<n>
  3 frontal_iso_supine        mean of (F3M2, F4M1) ISO power, eligible SUPINE windows
  4 occipital_iso_supine      mean of (O1M2, O2M1) ISO power, eligible SUPINE windows
  5 left_iso_left             mean of (F3M2, C3M2, O1M2) ISO power, eligible LEFT windows
  6 right_iso_left            mean of (F4M1, C4M1, O2M1) ISO power, eligible LEFT windows
  7 left_iso_right            mean of (F3M2, C3M2, O1M2) ISO power, eligible RIGHT windows
  8 right_iso_right           mean of (F4M1, C4M1, O2M1) ISO power, eligible RIGHT windows

Signals come from D:\HSP\batch*\sub-<site><subject>_ses-<n>.h5.
Body position comes from the companion _task-psg_annotations.csv in the same folder.

--------------------------------------------------------------------------------
WHERE THE NUMBERS COME FROM.  Every parameter below is the value already settled
in this project, and is cited to the decisions ledger so this script cannot drift
from 06_pipeline/code/. It is a re-implementation for HDF5 input, not a new method.

  D2   sigma band 11-15 Hz; total-power band 0.5-35 Hz
  D9   multitaper NW = 2, K = 3, 4 s windows, no overlap, tiled from sample 0
  Q5   each window is labeled by the epoch holding its MIDPOINT
  D22  what gets band-passed is NOT raw band power. The sigma series is divided
       by the within-person mean across NREM, logged, and linearly detrended
       first (Dr. Sun 2026-08-29)
  D7   Butterworth order 4, cutoffs 0.008-0.037 Hz, zero-phase (sosfiltfilt), so
       that 0.01-0.03 Hz is a real passband rather than the 3 dB points
  F    the ISO quantity is the SQUARED HILBERT ENVELOPE of that filtered series
       (Dr. Sun 2026-08-27)
  D13  eligibility is N2 only
  I1   5-minute washout after each position change
  Q7   windows containing a position change are excluded
  D14  artifact windows excluded at eligibility: |robust z| > 5 on log10 total
       power on ANY of the six derivations
  C1   flat-window floor at 1e-9 of the median, interpolated before the log

TWO DELIBERATE DEPARTURES, both requested by Jacob 2026-09-05:

  1. SUMMARY STATISTIC IS THE MEAN, NOT THE MEDIAN.
     Ledger D12 records Dr. Sun settling this as the median ("Yes, median",
     27AUG). This script was asked for the mean and computes the mean. The
     ledger notes a 3.3-9.2x mean/median ratio on this quantity, so the two are
     NOT interchangeable and nothing here is comparable to a median-based figure.

  2. POSITION VOCABULARY IS WIDER THAN step1_position_stage.py's.
     That script matches ^Body_Position:_(.+)$ only, and raises CheckFailure
     when it finds none. Measured across all 3,064 annotation files in D:\HSP,
     that pattern appears in 706 rows while "Position - <X>" appears in 25,894.
     Restricted to Body_Position:_ , only 74 of 3,064 recordings (2.4%) carry
     any position at all. Both spellings are accepted here, plus the trailing
     "- <n>" counter. This widening is Claude's and is flagged per Part 6 of the
     project instructions; it does not change step1, which still needs fixing.

UNITS.  Because of D22 the filtered series is a log ratio, so the ISO columns are
DIMENSIONLESS (squared envelope of a band-passed log-ratio), not uV^4.

Usage
    python create_eight_column_ISO_table.py
    python create_eight_column_ISO_table.py --workers 6
    python create_eight_column_ISO_table.py --limit 20 --out test.csv
"""

import argparse
import csv
import glob
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, detrend
from scipy.signal import windows as spwin
from tqdm import tqdm

SCRIPT_VERSION = "create_eight_column_ISO_table v1.0 (2026-09-05)"

HSP_ROOT = r"D:\HSP"
BATCHES = ["batch%d" % i for i in range(1, 12)]

WIN_S, EPOCH_S = 4, 30                  # D9, Q5
SIGMA = (11.0, 15.0)                    # D2
TOTAL_BAND = (0.5, 35.0)                # D2
NW, KTAP = 2.0, 3                       # D9
FS_WIN = 1.0 / WIN_S                    # 0.25 Hz, the band-power series rate
DESIGN_LO, DESIGN_HI = 0.008, 0.037     # D7
FILT_ORDER = 4                          # D7
WASHOUT_S = 300                         # I1
ARTIFACT_K = 5.0                        # D14, provisional threshold
STAGES_ELIGIBLE = ("N2",)               # D13
NORM_STAGES = ("N1", "N2", "N3")        # D22, "across NREM"
NONPOS_REL_FLOOR = 1e-9                 # C1

# table name -> dataset name inside the h5
DERIV = {"F3M2": "f3-m2", "F4M1": "f4-m1", "C3M2": "c3-m2",
         "C4M1": "c4-m1", "O1M2": "o1-m2", "O2M1": "o2-m1"}

FRONTAL = ("F3M2", "F4M1")
OCCIPITAL = ("O1M2", "O2M1")
LEFT_CH = ("F3M2", "C3M2", "O1M2")
RIGHT_CH = ("F4M1", "C4M1", "O2M1")

COLUMNS = ["BDSPID", "SessionID",
           "frontal_iso_supine", "occipital_iso_supine",
           "left_iso_left", "right_iso_left",
           "left_iso_right", "right_iso_right"]

# Both spellings, plus the trailing counter HSP writes on repeat marks.
POSITION_RE = re.compile(
    r"^(?:Body_Position:_|Position\s*-\s*)"
    r"(Supine|Left|Right|Prone|Upright|Sitting|Disconnect)"
    r"(?:\s*-\s*\d+)?\s*$", re.I)
# Only the first three are asked for. The rest are real marks that END the run of
# a scored position and must therefore be parsed, but are never averaged over.
WANTED_POSITIONS = ("Supine", "Left", "Right")

STEM_RE = re.compile(r"^sub-(S\d{4})(\d+)_ses-(\d+)\.h5$")


class Skip(Exception):
    """This recording cannot be processed. The reason is carried to the log."""


# ----------------------------------------------------------------- annotations
def hms_to_seconds(s):
    p = s.strip().split(":")
    if len(p) != 3:
        raise Skip("unparsable clock value %r in the annotations" % s)
    return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])


def read_position_marks(csv_path):
    """Position marks as (elapsed_s, position), plus the recording's epoch-1 clock.

    The `time` column is H:MM:SS with no date, so it wraps at midnight and is
    unwrapped here by watching for the clock going backwards. Elapsed time is
    measured from epoch 1, which is the first row of the file.
    """
    rows = []
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            t = (r.get("time") or "").strip()
            e = (r.get("event") or "").strip()
            if not t:
                continue
            rows.append((hms_to_seconds(t), e))
    if not rows:
        raise Skip("annotations file has no readable rows")

    t0_sod = rows[0][0]
    day, prev, marks = 0, rows[0][0], []
    for sod, event in rows:
        if sod < prev - 1:           # a genuine backwards step = midnight crossed
            day += 1
        prev = sod
        m = POSITION_RE.match(event)
        if m:
            marks.append((sod + 86400 * day - t0_sod, m.group(1).capitalize()))

    marks.sort(key=lambda x: x[0])
    # A repeat of the same position is not a change; collapse it so the washout
    # is not restarted by a technician re-marking a position already held.
    collapsed = []
    for t, p in marks:
        if collapsed and collapsed[-1][1] == p:
            continue
        collapsed.append((t, p))
    return collapsed, t0_sod


def position_per_window(marks, w_mid, total_s, nwin):
    """Carry each mark forward until the next one (the file states position nowhere else).

    Returns the position at each window midpoint, a flag for windows containing a
    change (Q7), and seconds since the last change (for the washout, I1).
    Before the first mark the position is UNDEFINED.
    """
    pos = np.full(nwin, "UNDEFINED", dtype=object)
    contains = np.zeros(nwin, bool)
    since = np.full(nwin, -1.0)
    if not marks:
        return pos, contains, since

    times = np.array([t for t, _ in marks], dtype=float)
    names = [p for _, p in marks]
    prev_i = np.searchsorted(times, w_mid, side="right") - 1
    has = prev_i >= 0
    for i in np.where(has)[0]:
        pos[i] = names[prev_i[i]]
    since[has] = w_mid[has] - times[prev_i[has]]

    for t, _ in marks:
        if 0 <= t < total_s:
            k = int(t) // WIN_S
            if k < nwin:
                contains[k] = True
    return pos, contains, since


# ---------------------------------------------------------------------- signals
def to_microvolts(dset):
    """int16 digital -> uV, using this dataset's own calibration attrs."""
    a = dset.attrs
    for k in ("dig_min", "dig_max", "phys_min", "phys_max"):
        if k not in a:
            raise Skip("dataset %s has no %s attribute, so it cannot be scaled"
                       % (dset.name, k))
    dmin, dmax = float(a["dig_min"]), float(a["dig_max"])
    pmin, pmax = float(a["phys_min"]), float(a["phys_max"])
    if dmax == dmin:
        raise Skip("dataset %s has a degenerate digital range" % dset.name)
    raw = dset[:].astype(np.float32)
    return (raw - dmin) * np.float32((pmax - pmin) / (dmax - dmin)) + np.float32(pmin)


def resolve_fs(f):
    """The sampling rate of the six derivations. Never assumed, never defaulted."""
    rates = set()
    for ch, key in DERIV.items():
        d = f["signals"].get(key)
        if d is None:
            raise Skip("h5 has no signals/%s" % key)
        if "fs" not in d.attrs:
            raise Skip("signals/%s carries no fs attribute" % key)
        rates.add(float(d.attrs["fs"]))
    if len(rates) != 1:
        raise Skip("the six derivations are stored at different rates %s" % sorted(rates))
    fs = rates.pop()
    if fs <= 0 or fs != int(fs):
        raise Skip("sampling rate %r is not a whole number of samples per second" % fs)
    return int(fs)


def multitaper_psd(x2d, nsamp, fs):
    """One-sided multitaper PSD in uV^2/Hz. Mirrors compute_bandpower.multitaper_psd."""
    tapers = spwin.dpss(nsamp, NW, KTAP).astype(np.float32)
    psd = np.zeros((x2d.shape[0], nsamp // 2 + 1), dtype=np.float32)
    for j in range(KTAP):
        F = np.fft.rfft(x2d * tapers[j], axis=1)
        psd += (F.real ** 2 + F.imag ** 2).astype(np.float32)
        del F
    psd /= np.float32(KTAP * fs)
    psd[:, 1:-1] *= np.float32(2.0)
    return psd


def band_area(psd, freqs, lo, hi):
    m = (freqs >= lo) & (freqs <= hi)
    return np.trapezoid(psd[:, m], freqs[m], axis=1)


def stage_per_epoch(f, n_epochs):
    """Stage name per 30 s epoch, from the h5 event table and its own event_map."""
    g = f.get("annotations/expert_1/stage")
    if g is None:
        raise Skip("h5 has no annotations/expert_1/stage")
    if "event_map" not in g.attrs:
        raise Skip("stage table carries no event_map, so codes cannot be named")
    emap = {int(k): v for k, v in json.loads(g.attrs["event_map"]).items()}
    starts = np.asarray(g["starts"][:], dtype=float)
    codes = np.asarray(g["codes"][:], dtype=int)
    out = np.full(n_epochs, "OUT_OF_GRID", dtype=object)
    ep = (starts // EPOCH_S).astype(int)
    ok = (ep >= 0) & (ep < n_epochs)
    rename = {"Wake": "W", "REM": "R", "Unknown/Unscored": "?"}
    for e, c in zip(ep[ok], codes[ok]):
        name = emap.get(int(c), "Unknown/Unscored")
        out[e] = rename.get(name, name)
    return out


# ------------------------------------------------------------------ D22 + ISO
def fix_nonpositive(y):
    """C1. Floor relative to the median, interpolated, so the log is defined."""
    fp = y[np.isfinite(y) & (y > 0)]
    floor = NONPOS_REL_FLOOR * float(np.median(fp)) if fp.size else 0.0
    bad = ~(np.isfinite(y) & (y > floor))
    if bad.any():
        if bad.all():
            raise Skip("band power is degenerate on every window")
        i = np.arange(y.size)
        y = y.copy()
        y[bad] = np.interp(i[bad], i[~bad], y[~bad])
    return y


def iso_series(sigma_abs, norm_mask, sos):
    """D22 then D7 then F: normalize, log, detrend, band-pass, squared envelope."""
    y = fix_nonpositive(np.asarray(sigma_abs, dtype=float))
    mu = float(np.mean(y[norm_mask]))
    if not np.isfinite(mu) or mu <= 0:
        raise Skip("the NREM normalizing mean is not finite and positive")
    lg = detrend(np.log(y / mu), type="linear")
    return np.abs(hilbert(sosfiltfilt(sos, lg))) ** 2


# --------------------------------------------------------------- one recording
def process(job):
    """Return (row_dict_or_None, note). Failures come back as note strings."""
    h5_path, csv_path, site, subject, session = job
    try:
        marks, t0_sod = read_position_marks(csv_path)
        if not marks:
            raise Skip("no position marks of any recognised spelling")

        with h5py.File(h5_path, "r") as f:
            fs = resolve_fs(f)
            nsamp_win = fs * WIN_S

            meas = f.attrs.get("meas_date")
            if meas and " " in str(meas):
                clock = str(meas).split(" ")[1]
                if abs(hms_to_seconds(clock) - t0_sod) > 1:
                    raise Skip("annotations epoch 1 is at %d s past midnight but the "
                               "h5 starts at %s, so position cannot be aligned to "
                               "the signals" % (t0_sod, clock))

            n_total = f["signals"][DERIV["F3M2"]].shape[0]
            nwin = n_total // nsamp_win
            if nwin < 100:
                raise Skip("only %d complete windows in the recording" % nwin)

            freqs = np.fft.rfftfreq(nsamp_win, 1.0 / fs)
            sigma_abs, total_abs = {}, {}
            for ch, key in DERIV.items():
                d = f["signals"][key]
                if d.shape[0] // nsamp_win != nwin:
                    raise Skip("derivation %s has a different length from F3M2" % ch)
                x = to_microvolts(d)[:nwin * nsamp_win].reshape(nwin, nsamp_win)
                x -= x.mean(axis=1, keepdims=True)
                psd = multitaper_psd(x, nsamp_win, fs)
                sigma_abs[ch] = band_area(psd, freqs, *SIGMA)
                total_abs[ch] = band_area(psd, freqs, *TOTAL_BAND)
                del x, psd

            stage_ep = stage_per_epoch(f, nwin * WIN_S // EPOCH_S + 2)

        # ---- window labels ---------------------------------------------------
        w_mid = np.arange(nwin) * WIN_S + WIN_S // 2
        epoch_ix = w_mid // EPOCH_S
        stage = np.array([stage_ep[e] if e < stage_ep.size else "OUT_OF_GRID"
                          for e in epoch_ix], dtype=object)
        pos, contains_tr, since_tr = position_per_window(
            marks, w_mid.astype(float), nwin * WIN_S, nwin)
        in_washout = (since_tr >= 0) & (since_tr < WASHOUT_S)

        stage_s = stage.astype(str)
        is_n2 = np.isin(stage_s, STAGES_ELIGIBLE)
        norm_mask = np.isin(stage_s, NORM_STAGES)
        if not norm_mask.any():
            raise Skip("no NREM windows, so the D22 normalization is undefined")

        # ---- D14 artifact rule -----------------------------------------------
        artifact = np.zeros(nwin, bool)
        for ch in DERIV:
            x = np.log10(np.maximum(np.asarray(total_abs[ch], dtype=float), 1e-12))
            med = np.median(x)
            mad = np.median(np.abs(x - med))
            if mad <= 0:
                raise Skip("total power on %s has zero MAD, so the artifact rule "
                           "cannot be formed" % ch)
            artifact |= np.abs((x - med) / (1.4826 * mad)) > ARTIFACT_K

        eligible = (is_n2 & (pos != "UNDEFINED") & ~contains_tr
                    & ~in_washout & ~artifact)

        # ---- ISO --------------------------------------------------------------
        sos = butter(FILT_ORDER, [DESIGN_LO, DESIGN_HI], btype="bandpass",
                     fs=FS_WIN, output="sos")
        iso = {ch: iso_series(sigma_abs[ch], norm_mask, sos) for ch in DERIV}

        def cell(channels, position):
            m = eligible & (pos == position)
            if not m.any():
                return "", 0
            v = np.mean([iso[c][m] for c in channels], axis=0)
            return float(np.mean(v)), int(m.sum())   # MEAN, not median: see header

        c3, n_sup = cell(FRONTAL, "Supine")
        c4, _ = cell(OCCIPITAL, "Supine")
        c5, n_left = cell(LEFT_CH, "Left")
        c6, _ = cell(RIGHT_CH, "Left")
        c7, n_right = cell(LEFT_CH, "Right")
        c8, _ = cell(RIGHT_CH, "Right")

        row = {"BDSPID": "%s%s" % (site, subject), "SessionID": int(session),
               "frontal_iso_supine": c3, "occipital_iso_supine": c4,
               "left_iso_left": c5, "right_iso_left": c6,
               "left_iso_right": c7, "right_iso_right": c8}
        note = "ok eligible supine=%d left=%d right=%d" % (n_sup, n_left, n_right)
        return row, note

    except Skip as e:
        return None, "SKIP %s" % e
    except Exception as e:                                  # noqa: BLE001
        return None, "ERROR %s: %s" % (type(e).__name__, e)


# ------------------------------------------------------------------------ main
def find_jobs(only_batches):
    jobs = []
    for batch in (only_batches or BATCHES):
        d = os.path.join(HSP_ROOT, batch)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            m = STEM_RE.match(name)
            if not m:
                continue
            site, subject, session = m.groups()
            stem = name[:-3]
            hits = [p for p in glob.glob(os.path.join(d, stem + "*annotations.csv"))
                    if "caisr" not in os.path.basename(p).lower()]
            jobs.append((os.path.join(d, name), hits[0] if hits else None,
                         site, subject, session))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HSP_ROOT, "eight_column_ISO_table.csv"))
    ap.add_argument("--log", default=os.path.join(HSP_ROOT, "eight_column_ISO_table_log.txt"))
    ap.add_argument("--batches", nargs="+", metavar="BATCH")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel recordings (default 1, a plain serial loop)")
    ap.add_argument("--restart", action="store_true",
                    help="ignore an existing output file and start over")
    a = ap.parse_args()

    jobs = find_jobs(a.batches)
    if a.limit:
        jobs = jobs[:a.limit]
    print("%s\nFound %d subject-sessions." % (SCRIPT_VERSION, len(jobs)))

    done = set()
    if os.path.exists(a.out) and not a.restart:
        with open(a.out, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r["BDSPID"], str(r["SessionID"])))
        print("resuming: %d rows already in %s" % (len(done), a.out))
    todo = [j for j in jobs if ("%s%s" % (j[2], j[3]), j[4]) not in done]

    no_csv = [j for j in todo if j[1] is None]
    todo = [j for j in todo if j[1] is not None]
    print("%d to process (%d skipped: no annotations file)" % (len(todo), len(no_csv)))
    if not todo and not no_csv:
        print("nothing to do.")
        return

    new = not os.path.exists(a.out) or a.restart
    fout = open(a.out, "w" if new else "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fout, fieldnames=COLUMNS)
    if new:
        writer.writeheader()
    flog = open(a.log, "w" if new else "a", encoding="utf-8")
    flog.write("# %s  started %s\n"
               % (SCRIPT_VERSION, time.strftime("%Y-%m-%dT%H:%M:%S")))
    flog.write("# statistic = MEAN (departs from ledger D12, which settled median)\n")
    for j in no_csv:
        flog.write("%s%s\tses-%s\tSKIP no annotations csv beside the h5\n"
                   % (j[2], j[3], j[4]))

    counts = {"ok": 0, "skip": 0}
    bar = tqdm(total=len(todo), desc="recordings", unit="rec")

    def handle(job, row, note):
        flog.write("%s%s\tses-%s\t%s\n" % (job[2], job[3], job[4], note))
        if row is None:
            counts["skip"] += 1
        else:
            writer.writerow(row)
            counts["ok"] += 1
        if (counts["ok"] + counts["skip"]) % 25 == 0:
            fout.flush()
            flog.flush()
        bar.update(1)

    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futs = {pool.submit(process, j): j for j in todo}
            for fut in as_completed(futs):
                row, note = fut.result()
                handle(futs[fut], row, note)
    else:
        for job in todo:
            row, note = process(job)
            handle(job, row, note)

    bar.close()
    fout.close()
    flog.write("# done: %d rows written, %d skipped\n"
               % (counts["ok"], counts["skip"] + len(no_csv)))
    flog.close()
    print("")
    print("rows written: %d   skipped: %d"
          % (counts["ok"], counts["skip"] + len(no_csv)))
    print("table: %s" % a.out)
    print("log:   %s" % a.log)


if __name__ == "__main__":
    main()
