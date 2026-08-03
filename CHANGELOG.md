# CHANGELOG

## v3.0.0 | unreleased

`dyco` replaces its time-lag detection method and becomes a standalone package.

**Pre-whitening with block-bootstrap (PWB) is now the only method.** Following Vitale et al. (2024),
the lag is estimated after removing serial autocorrelation with an AR(p) filter, and its reliability is
quantified by block-bootstrap resampling: each detection carries a 95% highest-density interval, and the
PWBOPT decision rule (S1/S2/S3) substitutes a trustworthy neighbouring lag where a period's own detection
cannot be trusted. This addresses the case the covariance-maximization method cannot: low-SNR gases such
as N<sub>2</sub>O and CH<sub>4</sub>, where the cross-correlation function is too noisy to locate the peak
reliably. Lags are expressed in **seconds** here, not in number of records.

**[BREAKING] The v1/v2 covariance-maximization method is removed.** It pooled detections into a daily
median look-up table and normalized them toward a target lag, without saying how much any single
detection could be trusted. That is the difficulty with low-SNR gases, and it is what PWB addresses.
Existing v2 workflows do not carry over: the two methods take different parameters, so there is no
flag-for-flag migration. See **Removed** below.

**`diive` is no longer a dependency.** v2.0.3 introduced it to avoid duplicating code; that coupling broke
`dyco` four separate ways when `diive` restructured internally — two import paths moved, and the Python and
pandas floors diverged to the point where the two could not be installed together. The shared code now lives
here, and the small generic helpers are bundled in `dyco/_vendor/` with their provenance recorded.

### Added

- `dyco.pwb` — `PreWhiteningBootstrap`, `PwbBatchDetection`, `PwboptLagPlot`. Four CCF combinations per
  period (scalar/reference × which series is AR-filtered); sonic temperature is required, since the
  `T_SONIC` combinations often expose a cleaner peak for gases with a weak scalar × `W` signal
- `dyco.pipeline` — `PerFilePipeline`, `process_one_file`. Splits long raw files into wall-clock-aligned
  chunks, rotates each in memory, detects per chunk, applies PWBOPT across the whole sequence, then writes
  one lag-corrected file per chunk
- `dyco.apply_tlag` — `TlagApplier`, removes lags listed in an existing `tlag_results.csv`
- `dyco.tui` — Textual terminal UI over the pipeline, with live validation, a preflight check and a
  `--demo` mode that needs no data
- Per-gas time-lag search windows, so a long-inlet gas such as H<sub>2</sub>O can use a wider window than
  the dry gases in the same run
- `dyco.maxcov` — `MaxCovariance`, the covariance-maximization estimator, previously imported from
  `diive`. `FluxDetectionLimit` builds on it, and it is usable on its own
- `dyco.rotation` — `WindDoubleRotation`, `reynolds_decomposition`
- `dyco.split` — `FileSplitter`, `FileSplitterMulti`, splitting long raw files into shorter parts with
  optional rotation
- `dyco.detectionlimit` — `FluxDetectionLimit`, the smallest flux distinguishable from noise, read off the
  far tail of the same cross-covariance function used for lag detection
- `dyco.rawio` — one place that knows how to open a raw data file. Raw EC data is delimited text
  (`.csv`, `.dat`, `.txt`) that is routinely shipped compressed, so `.gz`, `.bz2`, `.xz` and `.zip`
  now read and write transparently everywhere: the pipeline, `apply-batch`, the TUI's column scan
  and preflight check, and the file splitter. A zipped raw file must hold exactly one member;
  an archive of many files is a different thing and says so
- `dyco._vendor` — self-contained copies of the small helpers formerly imported from `diive`
- A real raw file for the examples: `examples/data/CH-LAE_202507251300.csv.gz`, a 1-hour 20 Hz
  excerpt from CH-LAE, plus `examples/detect_remove_tlag_realdata.py`, which runs the full
  detect-and-remove pipeline over it. Running this is what surfaced the three gzip faults below
- `--wdt` on `dyco detect-remove`: width, in records, of the centred rolling mean applied to each
  bootstrap CCF before its peak is taken. The default of 5 follows RFlux; the paper's equation 6
  specifies `hz/2 + 1` (11 at 20 Hz). Previously the value was fixed at 5 with no way to reach the
  paper's. It matters: on the bundled CH-LAE hour, `--wdt 11` widens the 95% HDI from 0.00/0.05 s to
  0.30/0.20 s, and the S1 reliability threshold is 0.5 s
- `--output-suffix` on `dyco detect-remove` (**Output as** in the TUI): the extension the corrected
  chunks carry. Reading and writing follow from it, so compression never has to be stated separately.
  Three ways to give it, all with the leading dot:
  - the whole extension — `.csv`, `.csv.gz`, `.dat.zip`, `.txt` — used as written
  - the text format alone — `.csv` — which drops compression: `file1.csv.zip` gives `file1.csv`
  - the compression alone — `.zip` — which keeps the input's text format: `file1.csv` gives
    `file1.csv.zip`, not `file1.zip`
  `auto` (the default) reuses the input's own extension. The dot is required, so a one-part
  extension is written the same way as a two-part one. A compression dyco cannot write (`.zst`,
  `.7z`) is refused rather than quietly producing plain text under a name that promises otherwise
- **A gas can take its lag from another gas**, for the periods where its own detection could not be
  trusted: `--scalar "N2O:n2o@lagfrom=CO2"`, or the **Lag from** field in the TUI, which starts with
  every gas pointing at itself. A gas keeps every lag PWBOPT accepts for it (S1/S2) and every lag it
  may still carry forward from an earlier period; past that it takes the donor's lag *for that same
  period*. Own-lag-first at both tiers is deliberate — two gases down one tube have systematically
  different delays, so borrowing swaps a stale number for a biased one, and `--max-carry` is what
  says when staleness has become the bigger error. Without a donor, a trace gas that never detects
  reliably falls back to the median of detections PWBOPT has just rejected, which on real noise is
  frequently a negative lag no tube can produce. Chains resolve donor-first; circular ones are
  refused. **Pair it with `--max-carry`**: with no carry limit a gas carries its own lag forever and
  the donor reaches only the periods before its first detection — dyco warns when that happens
- **`--max-carry N`** (**Max carry** in the TUI): the longest distance, in averaging periods, that
  PWBOPT's S3 rule may carry a lag. The published rule is unbounded — one good half hour can supply
  every later period in the run, however far away. Beyond the limit the lag expires (`S3_expired`)
  and the period falls through to the donor gas or the median instead. `{gas}_carry_periods` reports
  the distance for every period: `0` = detected in that period, `n` = carried `n` periods, empty =
  came from somewhere else. Default unlimited, i.e. the published behaviour
- **`{gas}_lag_applied_s` in the summary** — the lag that was actually removed, in seconds. The
  summary carries six lag columns per gas and none of them said plainly which one reached the data;
  this one is read back off the record shift, so it is true of the files on disk rather than of the
  request. Alongside it, `{gas}_lag_reason` gives the decision in words
- **`detect_and_remove_tlag_decisions.txt`** — one block per output file naming the lag applied to
  each gas and why: detected here and reliable, accepted for continuity, carried *n* periods forward
  from a named period, borrowed from a named gas, back-filled, or the median last resort. The
  thresholds behind the decisions head the file, and a tally closes it
- **`pwb-batch` and `apply-batch` now write `log.txt`** too, next to their results, the way
  `detect-remove` already did: run header, per-file lines, and the finish time
- The `detect-remove` log header names a borrowed lag (`lag from CO2 where N2O has none`), so the
  per-chunk lines cannot show a gas matching its donor without saying why
- `{gas}_lag_source` in the summary: `own`, `from:CO2` or `median` per period, so a borrowed lag is
  never mistaken for a detected one
- The TUI title bar shows the version, read from package metadata rather than hardcoded
- A test suite: `tests/`, 130 tests. `dyco` previously had none. `tests/test_pwb_reference.py` pins
  the pre-whitening chain to the numbers RFlux v3.2.0 produces on the same input — unit-root
  decision, AR order, AR coefficients, pre-whitened CCF peak and raw cross-covariance all agree to
  12 significant digits. Three cases: two synthetic, one per branch of the unit-root test, and the
  bundled **real** CH-LAE half hour, where AR orders reach 133 / 87 / 312 against the synthetic
  cases' 1 to 5. Fixtures and the R script that produced the frozen values are in `tests/data/`

### Changed

- **A period that produced no output file no longer gets a lag.** Short, duplicate and errored
  chunks write nothing, so `{gas}_tlag_final_s`, `{gas}_tlag_final_pf_s`, `{gas}_carry_periods` and
  `{gas}_lag_applied_s` are left empty for them and `{gas}_lag_source` reads `none`. A number in
  those rows suggested something had been corrected there. The detection columns are untouched
- **The TUI is now the recommended way to run dyco.** A detect-and-remove run takes around thirty
  settings; the TUI validates as you type, picks column names off a real file, and previews the run
  with a preflight check. The CLI is unchanged and remains the right choice for scripting
- **Chunk filename placeholders changed.** `{stem}` is the input filename with every suffix removed,
  and `{suffix}` is the extension the *output* should carry — `--output-suffix`, resolved. One
  template therefore names files the same way whatever the input was called. Previously a `.csv.gz`
  input put the `.csv` inside `{stem}` and left `{suffix}` as `.gz`, so
  `{stem}_chunk{index:02d}{suffix}` produced `site_202401010000.csv_chunk00.gz`; it now produces
  `site_202401010000_chunk00.csv.gz`. Setting `--output-suffix` with a template that has no
  `{suffix}` is an error rather than a silently ignored setting
- **New unified `dyco` command.** One front door dispatching to every workflow:
  `dyco detect-remove`, `dyco tui`, `dyco pwb-batch`, `dyco apply-batch`. The four
  standalone `dyco-*` scripts keep working unchanged
- **Console scripts renamed**: `dyco-detect-remove`, `dyco-detect-remove-tui`, `dyco-pwb-batch`,
  `dyco-apply-batch`
- **[BREAKING] The v2 top-level CLI is gone**, with the method it drove. It took short flags directly
  (`dyco REF LAG TGT -lsw 1000 -lsi 3 ...`). An old-style command line is detected and answered with a
  pointer to `dyco detect-remove` rather than a parse error
- **TUI settings file moved** from `~/.diive/detect_remove_tui.yaml` to `~/.dyco/detect_remove_tui.yaml`.
  An existing settings file is not found until it is moved
- **PWB detection is tuned around its one hot spot.** Profiling a 30-minute 20 Hz chunk puts ~96% of the
  run inside the block bootstrap's batched cross-correlation, and ~3% in reading the raw file. The FFT is
  now padded to `next_fast_len(N + lag_max)` instead of the next power of two — 36288 rather than 65536 for
  that chunk, and the bulk of the gain. The CCF is normalised after being sliced to the kept lag window
  rather than across the full transform, and centring, zero-padding and the sum of squares fold into a
  single pass over the buffer. That takes one chunk-gas detection from 0.71 s to 0.41 s against the first
  working implementation, and the bundled real-data example from about a minute to about ten seconds.
  Results are bit-identical, checked across both unit-root branches, the real CH-LAE chunk, windowed and
  unwindowed searches, 10 Hz, and chunks shorter than one bootstrap block. `scipy.fft` replaces
  `numpy.fft`, and was already a dependency via `scipy.signal`
- Python requirement raised to `>=3.12,<3.14` (was `>=3.11,<3.12`)
- pandas requirement raised to `>=3.0.0` (was `>=2.2.3,<3.0.0`)
- Build backend switched from `poetry-core` to `hatchling`; `uv` is now used for dependency management
- New dependencies: `numpy`, `polars`, `pyarrow`, `textual`, `pyyaml`

### Fixed

- **A quoted column-name row made every column look missing.** Loggers commonly write the header as
  `"TIMESTAMP","u","CH4"` and the data rows bare. `pandas` strips those quotes when it parses the
  data, but dyco split the header on the separator alone — in six places across `pipeline.py`,
  `apply_tlag.py` and `tui.py`, none of which stripped quotes — so the frame was labelled `"u"`,
  quote characters included, and no `--col-u` or `--scalar` name a user would think to pass could
  match. Every such file failed with `columns missing` while every column was in fact present.
  Header splitting now lives in one place, `rawio.split_header_line`, alongside the compression
  dispatch and for the same reason; a separator inside a quoted field no longer splits it either.
  Found on real CZ-Lnz QCL files

- **A literal `Inf` in a data column killed the averaging period.** Loggers write them — the CZ-Lnz
  QCL record carries four `-Inf` in one H₂O column. An infinite concentration is not a measurement,
  but nothing treated it as missing: it is not a string `--na-values` can match (and `argparse`
  refuses `-Inf` as a value outright, reading the leading dash as an option name), it survives the
  `-9999` filtering, and `na.approx` interpolating a gap adjacent to one *spreads* it into the gap —
  four values became fourteen. It then surfaced far downstream as `array must not contain infs or
  NaNs` out of scipy's `detrend`, naming neither the column nor the period. Non-finite values are now
  folded into NaN on read, so they are interpolated across like any other gap, and `{gas}_n_valid`
  counts only usable records. A column of nothing but `Inf` takes the same `no_data` path as an empty
  one

- **A gas missing for a whole averaging period killed the entire file.** An analyser offline for a
  period writes its fill value down the whole column — about one period in eight of the CZ-Lnz QCL
  record this was found on. `_na_approx` handed the empty column to `np.interp`, which raised *array
  of sample points is empty*; the error surfaced at file level, so one dead gas took every other gas
  and every chunk of that file with it, and nothing was written. An empty column now means what it
  says: there is no lag to find and none to apply, since shifting it would move nothing. The gas is
  skipped for that period, `{gas}_n_valid` records how many records were present, and
  `{gas}_lag_source` reads `no_data`. That marking survives PWBOPT deliberately — S3 carry and
  `@lagfrom=` exist to fill periods whose detection was *rejected* and cannot otherwise tell those
  apart from periods that had no data to detect in. Other gases in the same period, and periods
  where the gas is only partly missing, are unaffected

- **The PWB raw cross-covariance was read off the differenced series.** When the Breitung
  variance-ratio test rejects stationarity, all three series are first-differenced before AR
  fitting — but the differenced arrays were then also used for the raw cross-covariance, which R
  computes from the *original* series. `cov_pwb` became a covariance of increments: on a drifting
  record, two orders of magnitude too small and free to carry the opposite sign (measured: -0.017
  where R gives 4.040 at the same lag). The detected lag was never affected, only the reported
  covariance and the second diagnostic panel

- **An even CCF smoothing width raised `ValueError`.** The centred rolling mean assumed an odd
  window, so the paper's `hz/2 + 1` was unusable at 10 Hz (= 6). Even widths now follow zoo's
  `align="center"` convention, putting the extra sample after the centre, and a window wider than
  the series returns all-NaN instead of a shape error

- **`dyco apply-batch` crashed on a legacy Windows console** before doing any work: the Rich progress
  spinner defaults to braille characters, which cp1252 cannot encode, so the run died with
  `UnicodeEncodeError` instead of starting. All three CLIs now use the ASCII spinner

- **The TUI's column scan and preflight check produced garbage on compressed input.** Both used a
  plain `open()`, so a `.csv.gz` was decoded as text: five replacement-character "column names"
  from the compressed bytes, no exception, and a preflight that then reported every configured
  column as missing — telling you your setup was broken when the run itself would have worked

- **`dyco apply-batch` could not read or write compressed files.** It used a plain `open()` on both
  ends, so a `.csv.gz` input — which dyco's own file splitter produces — failed with a bare
  `StopIteration` and no message. It now handles gzip on both ends, and a file shorter than the
  header block reports which flag to check

- **`dyco apply-batch` wrote mixed line terminators.** Preserved header lines went out with the LF
  they picked up from the text-mode read while the data used `--lineterm`, so `--lineterm "\r\n"`
  produced an LF header above CRLF data. The same fault was already fixed on the `detect-remove`
  side; both now re-terminate the header

- **`_count_data_rows` lost the last row of a file with no trailing newline**, and so disagreed with
  `_estimate_data_rows` on the same file (2 vs 3). It is the counter used for every compressed
  input, so gzipped files were systematically one row short in chunk planning and the preflight
  check

- **`files.read_raw_data` refused compressed files.** It dispatched on
  `Path(filepath).suffix`, which for `raw.csv.gz` is `.gz`, so every compressed file
  raised *"File extension must be '.csv' or '.parquet'"*. This is the reader
  `FileSplitter` uses — and `FileSplitterMulti` writes `.csv.gz` when
  `compress_splits=True`, so the splitter's own output could not be read back in.
  Dispatch now ignores compression suffixes (new `files.data_suffix`), and the error
  for a genuinely unsupported format names what it saw

- **Gzip-compressed raw files could not be handled by the PWB pipeline** — three
  further faults in a *separate* reader, all from `pandas` inferring compression from
  the suffix while the surrounding plain `open()` calls did not:

  1. *Reading.* Header rows were parsed from compressed bytes, raising a
     column-count mismatch before the pipeline could start.
  2. *Chunk planning.* `_estimate_data_rows` scaled a bytes-per-line measured on
     decompressed content by `stat().st_size`, which is the **compressed** size.
     On a 6-hour 20 Hz file that gave 86 500 rows instead of 432 008 — so the run
     processed 3 chunks of 12, **exited successfully, and reported nothing
     unusual**. Compressed files now use the exact row count.
  3. *Writing.* The chunk filename template carries the input's suffix through to
     the output, so a `.gz` input produced a `.gz` *name* while the writer wrote
     plain text — a file whose extension lied. The writer now compresses when the
     name says `.gz`.

  This mattered because dyco's own file splitter writes `.csv.gz` when
  `compress_splits=True`, so the toolchain produced files its own pipeline could
  not read. All read and write paths now dispatch on the suffix
  (`_open_text` / `_open_binary` / `_open_text_write`). Regression tests in
  `tests/test_rawio_gzip.py` write the same content plain and gzipped and require
  identical results end to end

### Removed

- **[BREAKING] The covariance-maximization method**, i.e. everything reached through the `Dyco` class:
  `dyco.dyco` (`Dyco`), `dyco.loop` (`Loop`), `dyco.lag` (`AdjustLagsearchWindow`), `dyco.analyze`
  (`AnalyzeLags`), `dyco.correction` (`RemoveLags`), `dyco.plot` and `dyco.setup`. With them go the
  iterative window narrowing, the daily median look-up table, the target-lag normalization, the
  `outdirs` numbered output tree, and the rolling z-score outlier filter (`dyco._vendor.outliers`).
  `dyco cm` exits with a pointer to `dyco detect-remove`. To run the old method, install `dyco==2.0.3`,
  noting that it depends on `diive` and no longer installs cleanly against current `diive` versions.
  `MaxCovariance` (`dyco.maxcov`) stays: `FluxDetectionLimit` uses it, and it is useful on its own
- `files.read_segment_lagtimes_file` and `files.add_data_stats`, which only served that path.
  `dyco._vendor.filedetector.add_data_stats` is now the only function of that name
- The `example/` directory (two JOSS-era scripts driving the old CLI, and their input archive) and the
  `images/dyco_v2_*.png` figures that illustrated the removed workflow. `examples/` is unaffected
- `diive` dependency

### Notes

Downstream flux processing must run with EC time-lag maximization **disabled** — the lag has already been
removed.

The published JOSS paper (`paper/`) describes the covariance-maximization method as it stood in `v1.1.2`,
the version released for that publication, and is left as the historical record. The PWB method has its own
publication (Vitale et al. 2024).

Vitale, D., Fratini, G., Helfter, C., Hörtnagl, L., et al., 2024. A pre-whitening with block-bootstrap
cross-correlation procedure for temporal alignment of data sampled by eddy covariance systems. Environ.
Ecol. Stat. 31, 219–244. https://doi.org/10.1007/s10651-024-00615-9

## v2.0.3 | 6 May 2025

`dyco` uses eddy covariance raw data files as input and produces lag-compensated raw data files as output.

Version `2` changes the previous workflow.

`dyco` identifies and corrects time lags between variables. It iteratively searches for lags between two variables,
e.g., `W` (turbulent vertical wind) and `S` (scalar used for time lag detection, e.g. CO<sub>2</sub> or CH<sub>4</sub>),
starting with a broad time window and progressively narrowing it based on the distribution of found lags. This iterative
refinement helps pinpoint consistent lags, suggesting strong covariance. Lag searches can be performed on short segments
of a long file. After collecting all identified lags, `dyco` filters outliers and creates a look-up table of daily time
lags. This table is then used to shift variables in the input files, correcting for the identified lags. While `S` is
typically used for lag detection, the correction can be applied to other variables as needed. Lags are expressed in "
number of records"; the corresponding time depends on the data's recording frequency.

This update also implements [diive](https://github.com/holukas/diive) as a required dependency. The advantage of this
implementation is that existing (and better tested) code in `diive` does not have to be duplicated for `dyco`
(although the copy-paste approach has its merits), the drawback is that yes there is another dependency. I will try
not to break things.

For a more detailed explanation of the `dyco` processing chain please see the README file.

### Changes

- The minimum time window for lag search is now min. 20 records, which corresponds to +/- 0.5s for 20 Hz
  data. If the automatically detected window is smaller than 20 records it is automatically expanded.
  For example:
    - if the lag is searched between -8 and -3 records: `[-8, -3]` is expanded incrementally until the range between
      the two values is >= 20, `[-16, 5]`. Since the increase is always done on both sides of the search window,
      the resulting range in this example is 21 records.
    - Another example: `[-5, 5]` is expanded to `[-10, 10]`
      (`lag.AdjustLagsearchWindow.adjust_lgs_winsize`)
- When creating the look-up table for daily median lags, missing median values are now filled with the
  rolling median in a 5-day window, centered around the missing value. (`analyze.AnalyzeLags.make_lut_agg`)
- Added `diive` library to dependencies
- Several functions are now handled
  by `diive`: `calc_true_resolution`, `create_timestamp`, `search_files`, `FileDetector`, `MaxCovariance`
- Plots are now explicitely closed after export to avoid memory issues
- Parquet files can now be used for data input. If found files have the extension `.parquet`, the correct function to
  read the file is used.

## v1.2.0 | 6 Mar 2024

- Refactored code to work with newest package versions
- Several small bugfixes
- Now using `poetry` for dependency management
- Now using Python `3.9.18`
- All dependencies were updated to their newest possible versions
- Added example for using the class `DynamicLagCompensation` to run `dyco` directly from
  code (`example.example_kwargs.example`)

## v1.1.2 | 16 Jun 2021

### Release version for publication in JOSS

- JOSS: https://joss.theoj.org/
- DYCO open review: openjournals/joss-reviews#2575
