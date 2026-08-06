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

- **The new detection method.** Each averaging period is estimated four times over, from different
  pairings of the gas, the vertical wind and sonic temperature. Sonic temperature is now required:
  where a gas signal is weak, its pairing often shows a cleaner peak than the wind does
- **One command that does the whole job.** `dyco detect-remove` cuts a long raw file into averaging
  periods, rotates each one, finds the lag in each, decides across the whole run which of those lags
  can be trusted, then writes one corrected file per period
- **A terminal interface**, and the recommended way to run dyco. It checks settings as you type,
  reads column names off a real file, and previews a run before it starts. `dyco tui --demo` runs
  without any data at all
- **Lags can be removed from a previous run's results** without detecting again (`dyco apply-batch`)
- **Compressed raw files work everywhere.** `.gz`, `.bz2`, `.xz` and `.zip` are read and written
  throughout: the pipeline, both batch commands, the TUI's column scan and preflight check, and the
  file splitter. A zipped file must hold exactly one member
- **Four tools that used to come from `diive`**, each usable on its own: covariance maximization,
  double rotation, the file splitter, and the flux detection limit
- **A separate search window per gas**, so a gas on a long inlet such as water vapour can search
  wider than the dry gases in the same run
- **A gas can borrow another gas's lag** for the periods its own detection cannot cover:
  `--scalar "N2O:n2o@lagfrom=CO2"`, or **Lag from** in the TUI. A gas always prefers its own lag,
  because two gases down the same tube have systematically different delays. Chains are resolved
  donor first, and circular ones are refused. Pair it with `--max-carry`, or a gas carries its own
  lag forever and the donor never gets a turn
- **A limit on how far a trusted lag travels** (`--max-carry N`), counted in averaging periods. The
  published rule has no limit, so a single good half hour can supply every later period in a run.
  Past the limit the lag expires and the period falls back to the donor gas or the median. The
  summary reports the distance for each period. Unlimited by default, matching the published rule
- **Control over the CCF smoothing width** (`--wdt`). The default of 5 follows RFlux; the width the
  paper specifies was previously out of reach. It matters: on the bundled CH-LAE hour, `--wdt 11`
  widens the 95% uncertainty interval from 0.00/0.05 s to 0.30/0.20 s, against a 0.5 s threshold for
  calling a detection reliable
- **Control over the output file type** (`--output-suffix`, **Output as** in the TUI). Give the whole
  extension (`.csv.gz`), the text format alone (`.csv`, which drops compression), or the compression
  alone (`.zip`, which keeps the input's format). The leading dot is required, and `auto` reuses the
  input's. A compression dyco cannot write is refused rather than quietly writing plain text under a
  name that promises otherwise
- **The summary says which lag actually reached the data** (`{gas}_lag_applied_s`), measured from the
  shift itself rather than from what was requested. There were six lag columns per gas and none of
  them answered that. Beside it, `{gas}_lag_reason` gives the decision in words and
  `{gas}_lag_source` says where the lag came from: the gas itself, another gas, the median, or
  nothing at all
- **A report explaining every decision** (`detect_and_remove_tlag_decisions.txt`): one block per
  output file, naming the lag applied to each gas and why. The thresholds behind the decisions head
  the file and a tally closes it
- **Every command writes a `log.txt`** beside its results, which only `detect-remove` did before
- **A test suite.** dyco had none. The pre-whitening chain is pinned to the numbers RFlux v3.2.0
  produces on the same input, agreeing to 12 significant digits, over both branches of the
  stationarity test and a real CH-LAE half hour
- **Documentation on Read the Docs.** The command reference is generated from the parsers, so it
  cannot drift from `--help`, and two worked examples follow a bundled raw file each from input to
  output
- **Continuous integration**: the test suite, a strict documentation build and a package build, on
  Python 3.12 and 3.13

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
- **Detection is about 1.7x faster.** Almost all of a run sits in one place, the block bootstrap's
  cross-correlation, and three changes there account for the gain: a better-chosen transform length,
  normalising only the part of the result that is kept, and folding three passes over the data into
  one. Detecting one gas in one period goes from 0.71 s to 0.41 s, and the bundled real-data example
  from about a minute to about ten seconds. Results are bit-identical
- **TUI settings moved** from `~/.diive/detect_remove_tui.yaml` to `~/.dyco/detect_remove_tui.yaml`.
  An existing file is not found until it is moved
- Python `>=3.12,<3.14` (was `>=3.11,<3.12`), pandas `>=3.0.0` (was `>=2.2.3,<3.0.0`). Build backend
  is hatchling rather than poetry-core, dependencies are managed with uv, and `numpy`, `polars`,
  `pyarrow`, `textual` and `pyyaml` are new

### Fixed

- **A gzipped input was silently processed in part.** To plan its periods, dyco estimated how many
  rows a file held by measuring a line length and dividing the file size by it. For a compressed file
  it measured the line after decompression and the size before, so a 6-hour 20 Hz file was planned as
  86,500 rows instead of 432,008. The run processed 3 periods of 12, exited successfully and reported
  nothing unusual. Two more faults sat beside it: header rows were read from the compressed bytes,
  and a `.gz` input produced a `.gz` filename holding plain text. dyco's own file splitter writes
  `.csv.gz`, so it produced files its own pipeline could not read, and the splitter's reader rejected
  them too, treating `.gz` as though it were the data format
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
- **Counting rows missed the last one when a file had no trailing newline**, so compressed files were
  systematically one row short in period planning and in the preflight check

### Removed

- **[BREAKING] The covariance-maximization method**, i.e. everything reached through the `Dyco`
  class: `dyco.dyco`, `dyco.loop`, `dyco.lag`, `dyco.analyze`, `dyco.correction`, `dyco.plot` and
  `dyco.setup`. With them go the iterative window narrowing, the daily median look-up table, the
  target-lag normalization, the `outdirs` numbered output tree and the rolling z-score outlier
  filter. `dyco cm` exits with a pointer to `dyco detect-remove`. To run the old method install
  `dyco==2.0.3`, which depends on `diive` and no longer installs cleanly against current `diive`.
  `MaxCovariance` stays: `FluxDetectionLimit` uses it, and it is useful on its own
- Two helper functions that served only that path (`files.read_segment_lagtimes_file`,
  `files.add_data_stats`)
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
