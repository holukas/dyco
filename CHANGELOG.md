# CHANGELOG

## v3.0.0 | unreleased

`dyco` gains a second time-lag detection method and becomes a standalone package.

**Pre-whitening with block-bootstrap (PWB) is now the primary method.** Following Vitale et al. (2024),
the lag is estimated after removing serial autocorrelation with an AR(p) filter, and its reliability is
quantified by block-bootstrap resampling: each detection carries a 95% highest-density interval, and the
PWBOPT decision rule (S1/S2/S3) substitutes a trustworthy neighbouring lag where a period's own detection
cannot be trusted. This addresses the case the covariance-maximization method cannot: low-SNR gases such
as N<sub>2</sub>O and CH<sub>4</sub>, where the cross-correlation function is too noisy to locate the peak
reliably. Lags are expressed in **seconds** here, not in number of records.

**The v2 covariance-maximization method is retained, unchanged.** Existing workflows keep working. The two
methods answer the same question differently and neither replaces the other: PWB judges each averaging
period on its own evidence, while the v2 path pools detections into a daily median and normalizes toward a
target lag.

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
- `dyco.maxcov` — `MaxCovariance`, the v2 lag estimator, previously imported from `diive`
- `dyco.rotation` — `WindDoubleRotation`, `reynolds_decomposition`
- `dyco.split` — `FileSplitter`, `FileSplitterMulti`, splitting long raw files into shorter parts with
  optional rotation
- `dyco.detectionlimit` — `FluxDetectionLimit`, the smallest flux distinguishable from noise, read off the
  far tail of the same cross-covariance function used for lag detection
- `dyco._vendor` — self-contained copies of the small helpers formerly imported from `diive`
- A real raw file for the examples: `examples/data/CH-LAE_202507251300.csv.gz`, a 1-hour 20 Hz
  excerpt from CH-LAE, plus `examples/detect_remove_tlag_realdata.py`, which runs the full
  detect-and-remove pipeline over it. Running this is what surfaced the three gzip faults below
- A test suite: `tests/`, 104 tests. `dyco` previously had none

### Changed

- **New unified `dyco` command.** One front door dispatching to every workflow:
  `dyco detect-remove`, `dyco tui`, `dyco pwb-batch`, `dyco apply-batch`, `dyco cm`. The four
  standalone `dyco-*` scripts keep working unchanged
- **Console scripts renamed**: `dyco-detect-remove`, `dyco-detect-remove-tui`, `dyco-pwb-batch`,
  `dyco-apply-batch`
- **[BREAKING] The v2 top-level CLI is gone.** It took short flags directly (`dyco REF LAG TGT
  -lsw 1000 -lsi 3 ...`) and drove the covariance-maximization method. That work is now
  `dyco cm` with long flag names, and the options that were `0`/`1` integers are real switches.
  An old-style command line is detected and answered with a pointer rather than a parse error.
  Mapping:

  | v2 | v3 (`dyco cm`) |
  |---|---|
  | `-fnd` | `--filename-date-format` |
  | `-fnp` | `--file-pattern` |
  | `-flim` | `--limit-files` |
  | `-fgr` | `--file-generation-res` |
  | `-fdur` | `--file-duration` |
  | `-dtf` | `--timestamp-format` |
  | `-dres` | `--nominal-timeres` |
  | `-lss` | `--segment-duration` |
  | `-lsw` | `--lag-winsize` |
  | `-lsi` | `--n-iterations` |
  | `-lsf 1` / `-lsf 0` | `--remove-fringe-bins` / `--no-remove-fringe-bins` |
  | `-lsp` | `--perc-threshold` |
  | `-lt` | `--target-lag` |
  | `-del 1` | `--delete-previous` |
- **TUI settings file moved** from `~/.diive/detect_remove_tui.yaml` to `~/.dyco/detect_remove_tui.yaml`.
  An existing settings file is not found until it is moved
- Python requirement raised to `>=3.12,<3.14` (was `>=3.11,<3.12`)
- pandas requirement raised to `>=3.0.0` (was `>=2.2.3,<3.0.0`)
- Build backend switched from `poetry-core` to `hatchling`; `uv` is now used for dependency management
- New dependencies: `numpy`, `polars`, `pyarrow`, `textual`, `pyyaml`
- Rolling z-score outlier removal in `analyze.AnalyzeLags` no longer regularizes an irregular index before
  filtering. The lags it screens are indexed by segment start time and are inherently irregular, since
  segments are missing wherever a raw file was missing or its peak was low quality; regularizing inserted
  rows the caller never had and produced a flag that did not align with the input. Results may differ from
  v2.0.3 where the previous frequency detection succeeded

### Fixed

- **The v2 CLI could not run on pandas 3 at all.** Its `--file-generation-res`,
  `--file-duration` and `--segment-duration` equivalents defaulted to `'30T'`, and
  pandas 3 removed the `T` alias — every invocation raised
  `ValueError: invalid unit abbreviation: T` before doing any work. Defaults are now
  `'30min'`, and an explicit `'30T'` is rejected with a message naming the replacement

- **The v2 CLI compared durations as strings.** `--segment-duration` was checked against
  `--file-duration` with `>`, so `'10min' > '30min'` compared lexically: valid combinations
  were rejected and invalid ones let through. Now compared as `Timedelta`

- **`analyze.AnalyzeLags.make_lut_instantaneous` never filled missing lags.**
  The `fillna` that substitutes the default lag for dates with no detection
  discarded its result instead of assigning it, so the branch logged
  *"Filling missing lags with default lag"* while leaving those dates missing.
  Any run that hit missing lags behaved differently from what its log claimed.
  Now assigned

- **`files.read_raw_data` refused compressed files.** It dispatched on
  `Path(filepath).suffix`, which for `raw.csv.gz` is `.gz`, so every compressed file
  raised *"File extension must be '.csv' or '.parquet'"*. This is the reader the v2
  path and `FileSplitter` use — and `FileSplitterMulti` writes `.csv.gz` when
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

- `diive` dependency

### Notes

Downstream flux processing must run with EC time-lag maximization **disabled** — the lag has already been
removed.

The published JOSS paper (`paper/`) describes the v1/v2 covariance-maximization method and is left as the
historical record. The PWB method has its own publication (Vitale et al. 2024).

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
