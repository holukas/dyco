# What a run writes

## Folder layout

A `dyco detect-remove` run fills `--output-dir` like this:

```text
1_lag_detection/    STEP 1 -- detect (diagnostics + results)
    plots/          per-chunk PWB diagnostic figures      (--save-plots)
    plots_summary/  batch-level overview figures          (--save-plots)
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

## The summary CSV

`detect_and_remove_tlag_summary.csv` has one row per chunk: the detected lag, its HDI, the
reliability flag, the PWBOPT columns and the records actually applied. Its schema mirrors
`tlag_results.csv` from `pwb-batch`, plus the columns this pipeline adds at the removal step.

**Every column is described in `detect_and_remove_tlag_summary_columns.md`, written next to the CSV
by the same run.** That file is generated from the run's own settings — it names your gases, your
sampling rate and the specific lag column that was applied — so it is always more accurate than a
list in these docs could be. Read it first.

The parts worth knowing before you open either file:

`tlag_s`
: The **raw** per-chunk detection, before PWBOPT. This is not necessarily what was removed. A
  wide-HDI chunk's raw lag can be spurious, which is the entire reason PWBOPT exists.

`{gas}_tlag_final_pf_s`
: The PWBOPT-optimised, pre-filtered, gap-filled lag — the default value of
  `--lag-column-template`, and so by default **the column that was actually applied**. The data
  dictionary flags whichever column your run used.

`{gas}_lag_source`
: Where the period's lag came from: `own` (the gas detected it), `from:CO2` (borrowed from a donor
  gas), or `median` (the last-resort median of rejected detections). See
  [PWBOPT](method.md#periods-with-nothing-to-carry-forward).

`{gas}_applied_records`
: The shift in records. The applied lag in seconds is `{gas}_applied_records / hz`.

`status` and `{gas}_status`
: Two different things. The row-level `status` is the chunk's fate — only `ok` rows produced an
  output file, while `skipped:short`, `skipped:duplicate` and `error` rows are reported for
  traceability and write nothing. The per-gas `{gas}_status` is that gas's phase-2 outcome within an
  otherwise fine chunk: `ok`, `skipped:lag_nan` (no finite PWBOPT lag to apply), or `pending` (the
  chunk never reached phase 2).

:::{tip}
A run where many periods show `median` in `{gas}_lag_source` is telling you that gas could not locate
its own lag and had no donor. Give it one with `@lagfrom=`.
:::

## Diagnostic plots

`--save-plots` writes per-chunk PWB figures to `plots/` and batch-level overviews to
`plots_summary/`.

<!-- TODO: what each panel shows and how to read it. The batch overview has
     three: detected lags coloured by S1/S2/S3 flag, final gap-filled lags with
     S1/S2 anchor points, and HDI range bars against the S1 and pre-filter
     threshold lines. See PwbBatchDetection.plot_batch_summary in pwb.py. -->

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
