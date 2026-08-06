# CHANGELOG

## v3.0.0 | 6 Aug 2026

A new detection method, no `diive` dependency, and a command line that does not carry over from v2.

**[BREAKING] Pre-whitening with block-bootstrap (PWB) replaces covariance maximization.** Following
Vitale et al. (2024), an AR(p) filter strips the serial autocorrelation before the cross-correlation
is computed, and block-bootstrap resampling gives every detection a 95% highest-density interval.
The PWBOPT rule (S1/S2/S3) substitutes a trustworthy neighbouring lag where a period's own detection
cannot be trusted. This is what low-SNR gases such as N<sub>2</sub>O and CH<sub>4</sub> need: their
cross-correlation function is too noisy to locate the peak reliably, and the old method pooled
detections into a daily median look-up table without saying how far any single one could be trusted.
Lags are now in **seconds**, not records. The two methods take different parameters, so there is no
flag-for-flag migration.

**[BREAKING] `diive` is no longer a dependency.** The coupling introduced in v2.0.3 broke `dyco`
four ways when `diive` restructured: two import paths moved, and the Python and pandas floors
diverged until the two could not be installed together. That code lives here now, with the small
generic helpers in `dyco/_vendor/`.

### Added

- `dyco.pwb`: `PreWhiteningBootstrap`, `PwbBatchDetection`, `PwboptLagPlot`. Four CCF combinations
  per period. Sonic temperature is required, since the `T_SONIC` combinations often expose a cleaner
  peak than a weak scalar against `W`
- `dyco.pipeline`: `PerFilePipeline`, `process_one_file`. Splits raw files into wall-clock-aligned
  chunks, rotates each in memory, detects per chunk, applies PWBOPT across the sequence, writes one
  corrected file per chunk
- `dyco.tui`: a Textual terminal UI over the pipeline, with live validation, a preflight check and a
  `--demo` mode that needs no data
- `dyco.apply_tlag`: `TlagApplier`, removes lags listed in an existing `tlag_results.csv`
- `dyco.rawio`: one place that opens raw files. `.gz`, `.bz2`, `.xz` and `.zip` read and write
  transparently everywhere. A zipped raw file must hold exactly one member
- `dyco.maxcov`, `dyco.rotation`, `dyco.split`, `dyco.detectionlimit`: the covariance-maximization
  estimator, double rotation, the file splitter and the flux detection limit. Formerly imported from
  `diive`, each usable on its own
- Per-gas search windows, so a long-inlet gas such as H<sub>2</sub>O can use a wider window than the
  dry gases in the same run
- **A gas can take its lag from another gas** for periods its own detection cannot cover:
  `--scalar "N2O:n2o@lagfrom=CO2"`, or **Lag from** in the TUI. Own lag wins at every tier, since two
  gases down one tube have systematically different delays. Chains resolve donor-first and circular
  ones are refused. Pair it with `--max-carry`, or the gas carries its own lag forever and the donor
  never gets a turn
- **`--max-carry N`**: how far, in averaging periods, PWBOPT's S3 rule may carry a lag. The published
  rule is unbounded, so one good half hour can supply every later period in a run. Past the limit the
  lag expires and the period falls through to the donor gas or the median. `{gas}_carry_periods`
  reports the distance. Default unlimited, i.e. the published behaviour
- **`--wdt`**: width of the centred rolling mean applied to each bootstrap CCF before its peak is
  taken. Default 5, following RFlux. The paper's `hz/2 + 1` was previously unreachable, and it
  matters: on the bundled CH-LAE hour `--wdt 11` widens the 95% HDI from 0.00/0.05 s to 0.30/0.20 s,
  against an S1 threshold of 0.5 s
- **`--output-suffix`** (**Output as** in the TUI): the extension the corrected chunks carry, with
  reading and writing following from it. Give the whole extension (`.csv.gz`), the text format alone
  (`.csv`, dropping compression), or the compression alone (`.zip`, keeping the input's format). The
  leading dot is required, `auto` reuses the input's. A compression dyco cannot write is refused
  rather than quietly writing plain text under a name that promises otherwise
- **`{gas}_lag_applied_s`** in the summary: the lag actually removed, read back off the record shift,
  so it describes the files on disk rather than the request. Six lag columns per gas existed and none
  said plainly which one reached the data. `{gas}_lag_reason` gives the decision in words, and
  `{gas}_lag_source` reads `own`, `from:CO2`, `median` or `no_data`
- **`detect_and_remove_tlag_decisions.txt`**: one block per output file naming the lag applied to
  each gas and why. Thresholds head the file, a tally closes it
- `pwb-batch` and `apply-batch` write `log.txt` beside their results, as `detect-remove` already did
- A test suite. `dyco` had none. `tests/test_pwb_reference.py` pins the pre-whitening chain to RFlux
  v3.2.0 at 12 significant digits on the unit-root decision, AR order and coefficients, pre-whitened
  CCF peak and raw cross-covariance, across both branches of the unit-root test and a real CH-LAE
  half hour where the AR orders reach 133 / 87 / 312
- Documentation on Read the Docs, with the command reference generated from the argparse parsers so
  it cannot drift from `--help`, and two worked examples following a bundled raw file each from input
  to output
- `.github/workflows/tests.yml`: the suite, a `-W -j auto` documentation build and `uv build`, on
  3.12 and 3.13

### Changed

- **[BREAKING] The v2 top-level CLI is gone**, with the method it drove. An old-style command line
  (`dyco REF LAG TGT -lsw 1000 ...`) is answered with a pointer to `dyco detect-remove` rather than a
  parse error
- **[BREAKING] Chunk filename placeholders changed.** `{stem}` is the input name with every suffix
  removed and `{suffix}` is the resolved output extension, so one template names files the same way
  whatever the input was called. `{stem}_chunk{index:02d}{suffix}` on a `.csv.gz` input previously
  gave `site_202401010000.csv_chunk00.gz` and now gives `site_202401010000_chunk00.csv.gz`. Setting
  `--output-suffix` with a template that has no `{suffix}` is an error
- **New unified `dyco` command**, dispatching to `detect-remove`, `tui`, `pwb-batch` and
  `apply-batch`. The standalone scripts keep working, under new names (`dyco-detect-remove`,
  `dyco-detect-remove-tui`, `dyco-pwb-batch`, `dyco-apply-batch`)
- **The TUI is the recommended way to run dyco.** A detect-and-remove run takes around thirty
  settings. The CLI is unchanged and is still the right choice for scripting
- **A period that produced no output file no longer gets a lag.** Short, duplicate and errored chunks
  write nothing, so their lag columns are left empty and `{gas}_lag_source` reads `none`. A number
  there suggested something had been corrected. Detection columns are untouched
- **Stopping a run writes the chunks it had already detected.** Stop used to skip phase 2 outright,
  so a run stopped after hours of detection left a summary and plots but not one corrected file.
  Pressing Stop again during alignment skips that too and keeps what is written. PWBOPT sees a
  truncated sequence in a stopped run, so its output is provisional
- **PWB detection is about 1.7x faster.** Profiling puts ~96% of a chunk inside the block bootstrap's
  batched cross-correlation. The FFT is padded to `next_fast_len(N + lag_max)` rather than the next
  power of two, which is most of the gain; the CCF is normalised after slicing to the kept lag
  window; centring, zero-padding and the sum of squares fold into one pass. One chunk-gas detection
  goes from 0.71 s to 0.41 s, and the bundled real-data example from about a minute to about ten
  seconds. Results are bit-identical
- **TUI settings moved** from `~/.diive/detect_remove_tui.yaml` to `~/.dyco/detect_remove_tui.yaml`.
  An existing file is not found until it is moved
- Python `>=3.12,<3.14` (was `>=3.11,<3.12`), pandas `>=3.0.0` (was `>=2.2.3,<3.0.0`). Build backend
  is hatchling rather than poetry-core, dependencies are managed with uv, and `numpy`, `polars`,
  `pyarrow`, `textual` and `pyyaml` are new

### Fixed

- **A gzipped input was silently processed in part.** `_estimate_data_rows` scaled a bytes-per-line
  measured on decompressed content by the *compressed* file size, so a 6-hour 20 Hz file was planned
  as 86,500 rows instead of 432,008. The run processed 3 chunks of 12, exited successfully and
  reported nothing unusual. Two more faults sat beside it in the same reader: header rows were parsed
  from compressed bytes, and a `.gz` input produced a `.gz` filename holding plain text. dyco's own
  splitter writes `.csv.gz`, so the toolchain produced files its own pipeline could not read.
  `files.read_raw_data` refused them outright for the same reason, dispatching on `.gz` as though it
  were the data format
- **A quoted column-name row made every column look missing.** Loggers write the header as
  `"TIMESTAMP","u","CH4"` and the data rows bare. `pandas` strips those quotes, but dyco split the
  header on the separator alone in six places, so columns were labelled `"u"` and no `--col-u` or
  `--scalar` name could match. Every such file failed with `columns missing` while every column was
  present. Splitting now lives in `rawio.split_header_line`, and a separator inside a quoted field no
  longer splits it
- **A literal `Inf` in a data column killed the averaging period.** Loggers write them. It is not a
  string `--na-values` can match, `argparse` refuses `-Inf` as a value outright, and interpolating a
  gap beside one spreads it further. It surfaced far downstream as `array must not contain infs or
  NaNs` out of scipy's `detrend`, naming neither column nor period. Non-finite values now fold into
  NaN on read, and `{gas}_n_valid` counts only usable records
- **A gas missing for a whole period killed the entire file.** An analyser offline for a period
  writes its fill value down the column, about one period in eight of the record this was found on.
  The empty column reached `np.interp`, and the error surfaced at file level, so one dead gas took
  every other gas and every chunk with it. An empty column now means there is no lag to find and none
  to apply: the gas is skipped for that period and `{gas}_lag_source` reads `no_data`. That marking
  survives PWBOPT deliberately, since S3 carry and `@lagfrom=` exist to fill periods whose detection
  was *rejected* and cannot otherwise tell the two apart
- **The PWB raw cross-covariance was read off the differenced series.** When the Breitung test
  rejects stationarity all three series are first-differenced before AR fitting, and those arrays
  were then also used for the raw cross-covariance, which R computes from the original series.
  `cov_pwb` became a covariance of increments: two orders of magnitude too small on a drifting record
  and free to carry the opposite sign (-0.017 where R gives 4.040). The detected lag was unaffected
- **The hosted documentation build could not have worked.** `sphinx-argparse` declares itself
  parallel-safe and registers a domain with no `merge_domaindata`, so Sphinx splits the read and dies
  merging it. Read the Docs builds with `-j auto`; a Windows checkout cannot, so every local build
  was serial and the failure invisible. `docs/conf.py` supplies the method, guarded so an upstream
  fix wins
- **The progress bar filled to 100% long before the run ended**, with the ETA at zero. Phase 1
  dispatches at least one chunk past EOF per file so a sampling error can never drop a trailing
  chunk. Those phantoms finish instantly and were counted while the total was not. The display alone
  was affected
- **An even CCF smoothing width raised `ValueError`**, making the paper's `hz/2 + 1` unusable at
  10 Hz. Even widths now follow zoo's `align="center"` convention, and a window wider than the series
  returns all-NaN
- **`dyco apply-batch` could not read or write compressed files**, failing with a bare
  `StopIteration`, and wrote an LF header above CRLF data when a CRLF `--lineterm` was set. It also
  died on a legacy Windows console before doing any work, because Rich's default braille spinner is
  not cp1252-encodable. All three CLIs now use the ASCII spinner
- **The TUI's column scan and preflight check produced garbage on compressed input**, reading a
  `.csv.gz` as text and then reporting every configured column as missing, when the run itself would
  have worked
- **`_count_data_rows` lost the last row of a file with no trailing newline**, so gzipped files were
  systematically one row short in chunk planning and the preflight check

### Removed

- **[BREAKING] The covariance-maximization method**, i.e. everything reached through the `Dyco`
  class: `dyco.dyco`, `dyco.loop`, `dyco.lag`, `dyco.analyze`, `dyco.correction`, `dyco.plot` and
  `dyco.setup`. With them go the iterative window narrowing, the daily median look-up table, the
  target-lag normalization, the `outdirs` numbered output tree and the rolling z-score outlier
  filter. `dyco cm` exits with a pointer to `dyco detect-remove`. To run the old method install
  `dyco==2.0.3`, which depends on `diive` and no longer installs cleanly against current `diive`.
  `MaxCovariance` stays: `FluxDetectionLimit` uses it, and it is useful on its own
- `files.read_segment_lagtimes_file` and `files.add_data_stats`, which only served that path
- The `example/` directory and the `images/dyco_v2_*.png` figures. `examples/` is unaffected
- The `diive` dependency

### Notes

Downstream flux processing must run with EC time-lag maximization **disabled**. The lag is already
gone from the data.

The JOSS paper (`paper/`) describes the covariance-maximization method as it stood in `v1.1.2` and is
left as the historical record. PWB has its own publication:

Vitale, D., Fratini, G., Helfter, C., Hörtnagl, L., et al., 2024. A pre-whitening with block-bootstrap
cross-correlation procedure for temporal alignment of data sampled by eddy covariance systems. Environ.
Ecol. Stat. 31, 219-244. https://doi.org/10.1007/s10651-024-00615-9

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
