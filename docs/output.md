# What a run writes

## Folder layout

A `dyco detect-remove` run fills `--output-dir` like this:

```text
1_lag_detection/    STEP 1 -- detect (diagnostics + results)
    plots/          per-chunk PWB diagnostic figures      (--save-plots)
    plots_summary/  batch-level overview figures          (--save-plots)
    detect_and_remove_tlag_decisions.txt         start here: the lag applied to
                                                 each output file, and why
    detect_and_remove_tlag_summary.csv           one row per chunk
    detect_and_remove_tlag_summary_columns.md    data dictionary for that CSV
    detect_and_remove_tlag_checkpoint.csv        phase-1 snapshot
    detect_and_remove_tlag_remove_checkpoint.csv phase-2 snapshot

2_lag_removed/      STEP 2 -- remove (the deliverable)
    Lag-corrected chunk files, and only those.

run_settings.txt    every setting used, each with a one-line explanation
detect_remove_tui_settings.yaml   reloadable by `dyco tui`
log.txt             plain-text console log
README.txt          this layout, written into the folder itself
```

**`2_lag_removed/` is the one to hand to your flux software.** The subdirectory names come from
`--detect-subdir` and `--data-subdir` if you want different ones.

`pwb-batch` and `apply-batch` write their own `log.txt` beside their results. `pwb-batch`'s results
file is called `tlag_results.csv`, which is what `apply-batch --results-csv` expects.

:::{note}
**Lag is expressed in seconds.** The covariance-maximization method removed in v3 counted records
instead, so older notes, plots and result files may mean something different by "lag".
:::

## Which lag was applied, and why

`detect_and_remove_tlag_decisions.txt` answers both, one block per written file:

```text
CH-CHA_202110080730.csv
  CH4     +2.20 s (+44 rec)  detected here and reliable (S1): HDI 0.40 s < 0.50 s
  N2O     +1.85 s (+37 rec)  nothing had been detected yet at this point in the run,
                             so the lag was filled 2 period(s) back from
                             CH-CHA_202110080900.csv, the first period that did

CH-CHA_202110080800.csv
  CH4     +2.20 s (+44 rec)  carried 1 period(s) forward from CH-CHA_202110080730.csv:
                             HDI 2.70 s, wider than the 1.00 s prefilter
  N2O     +1.85 s (+37 rec)  borrowed from CH4: HDI 9.15 s, wider than the 1.00 s
                             prefilter, and no detection of its own within
                             2 period(s) either side
```

The thresholds behind the decisions head the file, periods that produced no output file are listed
at the foot, and a tally closes it. The same sentences are in the summary CSV as
`{gas}_lag_reason`, if you would rather join them to the numbers.

## The summary CSV

`detect_and_remove_tlag_summary.csv` has one row per chunk: the detected lag, its HDI, the
reliability flag, the PWBOPT columns and the records actually applied. Its schema mirrors
`tlag_results.csv` from `pwb-batch`, plus the columns this pipeline adds at the removal step.

**Every column is described in `detect_and_remove_tlag_summary_columns.md`, written next to the CSV
by the same run.** That file is generated from the run's own settings — it names your gases, your
sampling rate and the specific lag column that was applied — so it is always more accurate than a
list in these docs could be. Read it first.

The parts worth knowing before you open either file:

`{gas}_lag_applied_s`
: **The lag that was actually removed**, in seconds. Derived from the record shift the pipeline
  made, so it describes the files on disk rather than what was asked for — and it is always a whole
  number of records, which is why it can sit up to half a record from the requested lag. Six other
  lag columns exist per gas; they are the working steps that led to this one.

`{gas}_lag_reason`
: Why that value was chosen, in words. The same text as in the decisions report above.

`{gas}_tlag_s`
: The **raw** per-chunk detection, before PWBOPT. This is not necessarily what was removed. A
  wide-HDI chunk's raw lag can be spurious, which is the entire reason PWBOPT exists.

`{gas}_hdi_range_s`
: The width of the 95% interval, and the number every PWBOPT decision turns on. Below
  `--hdi-thresh` the detection is accepted outright (S1). `{gas}_hdi_lo_s` and `{gas}_hdi_hi_s` are
  the bounds it was computed from, and `{gas}_is_reliable` is the same test as a boolean.

`{gas}_n_valid`
: How many records of this gas the chunk actually holds. `0` means the analyser was offline for the
  whole period: nothing was detected, nothing was carried or borrowed in, and the column was written
  through untouched.

`{gas}_best_combination`
: Which of the four pre-whitening combinations won: `cw` and `wc` are the gas against vertical wind,
  `ct` and `tc` the gas against sonic temperature. Strong fluxes usually win on `cw`/`wc`. A trace
  gas that keeps falling back to the temperature pair is telling you its own signal against `W` is
  too weak to work with.

`{gas}_tlag_final_pf_s`
: The PWBOPT-optimised, pre-filtered, gap-filled lag — the default value of
  `--lag-column-template`, and so by default the column that was *requested* for removal. The data
  dictionary flags whichever column your run used.

`{gas}_lag_source`
: Where the period's lag came from: `own` (the gas detected it here, or carried it from one of its
  own nearby periods), `from:CO2` (borrowed from a donor gas), `median` (the last-resort median of
  rejected detections), `no_data` (the gas was missing for the whole period, so no lag was needed and
  none was applied), or `none`. See [PWBOPT](method.md#periods-still-without-a-lag).

`{gas}_carry_periods`
: How far that lag travelled: `0` if it was detected in this very period, `n` if it came from `n`
  periods away, empty if it came from somewhere other than the gas's own carry. `own` alone cannot
  distinguish a fresh detection from an inherited one; this can.

`{gas}_applied_records`
: The shift in records, which is `{gas}_lag_applied_s * hz`.

`status` and `{gas}_status`
: Two different things. The row-level `status` is the chunk's fate — only `ok` rows produced an
  output file, while `skipped:short`, `skipped:duplicate` and `error` rows are reported for
  traceability and write nothing. The per-gas `{gas}_status` is that gas's phase-2 outcome within an
  otherwise fine chunk: `ok`, `skipped:lag_nan` (no finite PWBOPT lag to apply), or `pending` (the
  chunk never reached phase 2).

:::{note}
**Rows that wrote no file carry no lag.** For any row whose `status` is not `ok`, the final, applied
and carry columns are left empty and `{gas}_lag_source` reads `none`. There is no data there to
align, so a number in those columns would suggest a correction that never happened. The detection
columns are untouched.
:::

:::{tip}
A run where many periods show `median` in `{gas}_lag_source` is telling you that gas could not locate
its own lag and had no donor. Give it one with `@lagfrom=`, and set `--max-carry` alongside it —
without a carry limit the gas keeps reaching every period with its own lag and the donor is never
consulted.
:::

## Diagnostic plots

`--save-plots` writes two kinds of figure. Both are off by default, because they cost time and disk
on a long run.

### One figure per chunk and gas, in `plots/`

Three panels, left to right (`PreWhiteningBootstrap.plot`):

1. **The pre-whitened cross-correlation**, grey stems with the smoothed line over them, a Bartlett
   significance band, and a red marker at the detected lag. If the peak does not clear the band,
   there was nothing to find in that period.
2. **The raw cross-covariance**, same layout, with the same lag marked. This is the curve the old
   covariance-maximization method worked on, so comparing the two panels shows what pre-whitening
   bought.
3. **The bootstrap lag distribution** for the winning combination: a histogram of the peak found in
   each resample, the 95% interval shaded, and the mode marked. A tight cluster is a reliable lag; a
   spread-out or multi-peaked one is what a wide interval looks like. The panel title names the
   combination that won.

### Batch overviews, in `plots_summary/`

Five panels per gas, plus one cross-gas comparison figure (`PwbBatchDetection.plot_summary`):

1. Detected lags over the run, coloured by S1/S2/S3 flag.
2. The final gap-filled lags, with the S1/S2 detections that anchor them drawn as filled markers and
   the applied lag as open circles. This is the panel that shows carry and borrowing at a glance.
3. Interval width per period, against the S1 threshold and the pre-filter threshold as lines.
4. Flag counts per period, standard rule beside pre-filtered.
5. A histogram of every detected lag, with the mode marked.

The comparison figure puts all gases on one scatter with a density curve per gas, which is where a
systematic offset between two gases in the same tube shows up.

## log.txt and run_settings.txt

Every CLI writes a `log.txt` to its output folder: the run header, the per-file and per-chunk lines,
and the finish time. The animated progress display is deliberately kept out of it, so the log stays
readable as plain text.

`run_settings.txt` is written at the *start* of the run and records every setting with a one-line
explanation of what it does. Between it and `detect_remove_tui_settings.yaml`, a finished run carries
enough to reproduce itself.

## Feeding the output to flux software

:::{important}
Downstream flux processing must run with time-lag maximization **disabled**. The lag has already
been removed from the files in `2_lag_removed/`.
:::

<!-- TODO: an EddyPro-specific note -- which setting, where. -->
