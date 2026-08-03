# CLAUDE.md — dyco Development Guide

Time-lag detection and compensation for eddy covariance raw data.
See `CHANGELOG.md` for version history.

> This file is the single standing reference: conventions, architecture, open
> items and the rules that outlive any one task. For what changed and when, read
> `CHANGELOG.md` and `git log` — not a status file that goes stale between
> commits.

---

## [CRITICAL] PWB is the only method

dyco carries **one** lag-detection method: pre-whitening block-bootstrap.

| | |
|---|---|
| Entry point | `dyco tui` (recommended) or `dyco detect-remove` (`dyco/pipeline.py`) |
| Detection | `dyco/pwb.py` |
| Lag selection | PWBOPT S1/S2/S3, per chunk; a gas borrows another's lag (`@lagfrom=`) for any period it has no accepted detection in, and `--max-carry` bounds how far S3 may carry one |
| Which lag was applied | `{gas}_lag_applied_s` in the summary, and `detect_and_remove_tlag_decisions.txt` for the reasoning. Periods that wrote no file carry no lag at all |
| Removal | `dyco/apply_tlag.py` `TlagApplier` |
| Tests | `tests/test_pwb.py` + 10 more, 198 total |

**The v2 covariance-maximization method was removed on 2026-08-01**, at the
user's instruction, along with `dyco.py`, `loop.py`, `lag.py`, `analyze.py`,
`correction.py`, `plot.py`, `setup.py` and `_vendor/outliers.py`. It pooled
detections into a daily median LUT and normalized toward a target lag, with no
per-detection confidence, which is what low-SNR gases need. `dyco cm` now exits
with a pointer. The code is in git history and in the `2.0.3` release.

Do not reintroduce it, and do not restore a module from history to "fix" a
missing import. If something the PWB path needs turns out to have lived only
there, port the specific piece and say so.

`dyco/maxcov.py` (`MaxCovariance`) is **not** part of that removal. It is the
covariance-maximization estimator itself, it is what `FluxDetectionLimit` reads
its noise from, and it is tested. It stays.

**PWB is measured against the original R code.** `tests/test_pwb_reference.py`
pins the deterministic half of `pwb.py` to RFlux v3.2.0's `tlag_detection.R` at
12 significant digits: both branches of the unit-root test, and the bundled real
CH-LAE half hour, where the AR orders reach 133 / 87 / 312 and the synthetic
cases only reach 5. The remaining differences are catalogued with severities in
the `pwb.py` module docstring — read that before changing the algorithm, and
rerun `tests/data/pwb_reference_rflux.R` if you do.

That comparison also settled two things worth remembering. The differencing
branch fires on ordinary data (T_SONIC drifts over half an hour), so it is not
the edge case it looks like. And on that half hour the *unwindowed* PWB
detection is unreliable in **both** implementations — the cross-covariance peaks
at -7.45 s, which no tube delay can be. The pipeline gets a usable 8.45 s
because `--lws 0 --uws 10` confines the search to physical lags; that window is
doing real work, and an S1 flag from a windowed search is a weaker claim than
from an unwindowed one.

## [READ FIRST] v3 state

The migration is **done**: all of diive's `flux/hires/` plus `filesplitter.py`
now live here, and the diive dependency is gone.

`cli.py` is now the unified `dyco` dispatcher (`detect-remove`, `tui`, `pwb-batch`,
`apply-batch`); the standalone `dyco-*` scripts still work.

## [DONE] dyco no longer depends on diive

As of 2026-08-01 the diive dependency is gone. All seven former imports are
satisfied locally, the four breakages that made dyco uninstallable are fixed
(both dead import paths, plus the Python and pandas floors), and every module
imports in a clean environment with no diive present.

What that involved:

- **`dyco/_vendor/`** — leaf utilities copied from diive: `times`
  (`create_timestamp`, `calc_true_resolution`), `fileio` (`search_files`,
  `read_parquet`), `filedetector` (`FileDetector`, `add_data_stats`), `frames`
  (`trim_frame`), `plotstyle` (`default_format` + theme constants) and
  `console`. `outliers` (a port of `zScoreRolling`) was here too and went with
  the v2 path — it had no other caller.
- **`dyco/maxcov.py`, `dyco/rotation.py`, `dyco/detectionlimit.py`,
  `dyco/split.py`** — domain code moved over from diive. Not vendored helpers;
  these are dyco's own now.

**Verified end-to-end on real data.** `examples/detect_remove_tlag_realdata.py`
runs the full PWB pipeline over a bundled 1-hour CH-LAE file: correct chunking,
genuinely gzipped output, headers byte-identical to the source, and scalar
columns shifted by exactly the applied lag. `FluxDetectionLimit` also reproduces
diive's numbers to 10 decimal places.

---

## Behavioral Guidelines

**Bias toward caution over speed. For trivial tasks, use judgment.**

**Think before coding:** State assumptions explicitly. If multiple
interpretations exist, present them. If something is unclear, ask. Push back
when a simpler approach exists.

**Simplicity first:** Minimum code to solve the problem. No speculative
features, abstractions for single-use code, or error handling for impossible
scenarios.

**Surgical changes:** Touch only what you must. Don't "improve" adjacent code.
Match existing style. **Mention dead code — don't delete it** (nothing is
catalogued as dead right now; the rule stands for whatever turns up). Remove
only imports and variables that YOUR changes made unused.

**Goal-driven execution:** For multi-step tasks, state a plan with verifiable
success criteria before coding.

---

## Development Workflow

**[CRITICAL] NEVER COMMIT.** Never run `git commit`, `git add` or `git push`.
The user stages and commits exclusively.

**[CRITICAL] NEVER BUMP THE VERSION NUMBER.** The user does this manually.

**Do NOT:** run package-manager commands without explicit approval, skip
pre-commit hooks (`--no-verify`), force-push, or add Claude as a commit
co-author.

**Commit message style** (when asked to draft one): one-line title under 50
chars, then bullet points.

---

## Environment

**Python** `>=3.12,<3.14`. **Build** hatchling, **deps** uv. `pyproject.toml`
reads `version = "3.0.0"`. The CHANGELOG entry is still `unreleased` — v3 is
not published yet. **Do not touch the version again; the user owns it.**

```bash
uv sync
uv run pytest tests/ -q                                  # 198 passed
uv run python examples/detect_remove_tlag_realdata.py    # real-data end-to-end
uv run dyco                                              # list all workflows
```

**Companion repo:** `F:\dev\diive`. Nothing here imports it any more, and
nothing should. It is only useful as historical reference for where this code
came from.

---

## Current Architecture

~10,600 lines across 11 modules plus ~400 in `_vendor/`. Pipeline:
**split into chunks → rotate → detect per chunk → PWBOPT across the sequence →
shift and write**.

| Module | LOC | Role |
|---|---|---|
| `dyco/pipeline.py` | 3,436 | `PerFilePipeline`, `process_one_file`. The primary workflow, and its own raw-file reader/writer |
| `dyco/pwb.py` | 3,089 | `PreWhiteningBootstrap`, `PwbBatchDetection`, `PwboptLagPlot`. The detection method itself |
| `dyco/tui.py` | 2,110 | Textual UI over the pipeline. `--demo` needs no data |
| `dyco/apply_tlag.py` | 833 | `TlagApplier` — remove lags listed in an existing `tlag_results.csv` |
| `dyco/split.py` | 524 | `FileSplitter`, `FileSplitterMulti` — divide a long raw file into averaging-period parts |
| `dyco/detectionlimit.py` | 476 | `FluxDetectionLimit` — minimum detectable flux, read off the far tail of the cross-covariance function |
| `dyco/maxcov.py` | 417 | `MaxCovariance` — covariance-maximization lag estimator. `FluxDetectionLimit` builds on it |
| `dyco/files.py` | 243 | Raw CSV/parquet reading for `split.py`, incl. header-vs-data column reconciliation |
| `dyco/rotation.py` | 138 | `WindDoubleRotation`, `reynolds_decomposition`. Chunks are rotated before the search |
| `dyco/rawio.py` | 240 | Opening raw files, compressed or not (`.gz`, `.bz2`, `.xz`, `.zip`). Every reader and writer goes through it |
| `dyco/cli.py` | 125 | Unified `dyco` dispatcher |
| `dyco/_vendor/` | ~400 | Leaf utilities copied from diive; see its `__init__.py` for the rationale |
| `dyco/__init__.py` | 2 | A comment. **No public API is defined** |

**Lag is expressed in seconds.** The removed v2 path counted records instead, so
older notes, plots and result files may mean something different by "lag".

**Never import from diive again.** If something is needed from there, copy it
into `_vendor/` with a provenance note, or reimplement it. The cross-repo
coupling is what broke dyco four ways, and it is deliberately gone.

---

## Performance

**Everything that matters is in one function.** Profiling a 30-minute 20 Hz
chunk puts ~96% of the run inside `pwb.py`'s `_batch_ccf_fft` — the block
bootstrap's batched cross-correlation. Reading the raw file is ~3%. AR fitting,
smoothing, the KDE mode and the HDI are rounding error. Profile before touching
anything; the intuitive targets are all in that 4%.

The optimisation pass on 2026-08-02 gave **~1.7x** on a quiet machine (0.708 s
-> 0.409 s per chunk-gas, best of 7) and ~2.8x on the median. The gap widens
under load because the old padding allocated nearly twice the memory. Three
changes, all in `_batch_ccf_fft` and its caller:

- **FFT length is `next_fast_len(N + lag_max)`, not the next power of two.**
  For a 30-min 20 Hz chunk that is 36288 instead of 65536 — 1.8x less
  transform, and the single biggest win. The `n_fft >= N + lag_max` bound is
  *tight*: below it, circular wraparound contaminates the kept lag window. Do
  not "simplify" this back to a bit-length shift.
- **Slice, then normalise.** Only `2*lag_max+1` of the `n_fft` columns survive.
- **Centring, zero-padding and the sum of squares fold into one pass** over the
  ~30 MB buffer, which also drops the copy the FFT would make to pad.

`scipy.fft` replaced `numpy.fft` here. It was already a dependency via
`scipy.signal`, so this cost nothing.

### Measured dead ends — do not re-attempt

- **Gathering the bootstrap blocks straight into the padded FFT buffer.** Looks
  like the obvious next step, and is a *regression*: 0.42 s -> 0.51 s.
  `np.take` into a strided view drops off numpy's fast path and costs more
  (21 ms) than a plain advanced-index gather plus the centring pass (15 ms). A
  reused contiguous scratch is also slower (17.5 ms). The benchmark compared
  four strategies whose output was identical; there is a note in the code.
- **Threading the FFT** (`scipy.fft(workers=…)`). Flat from 1 to 8 threads —
  the kernel is memory-bandwidth bound, not compute bound.
- **Sharing bootstrap draws across the four combinations.** `cw`/`ct` share the
  scalar series, so one forward FFT could be saved, but only by reusing the
  same block indices. That correlates combinations that R draws independently
  and breaks reproducibility. Deliberately not done.
- **`np.minimum(idx, n-1)` in `_block_bootstrap`** was a no-op and is gone.
  Block starts top out at `n-L` and offsets at `L-1`, so the largest index is
  exactly `n-1` in every regime, including `n < L`. The trailing partial block
  is handled by truncating to `n` columns. Do not re-add it "for safety".

### Verifying an optimisation

The reference tests pin R parity, but they do not pin *unchanged behaviour* —
a perf change can pass them and still move results. Dump the full results
(every reported field, all `n_bootstrap` peak lags, AR orders, HDI bounds)
across a spread of configs before and after, and diff. The 2026-08-02 pass used
19: both unit-root branches, the real CH-LAE chunk, windowed and unwindowed
searches, odd block and smoothing widths, 10 Hz, and four short chunks spanning
`n < L`. All three changes came out **bit-identical** — zero difference in 1338
float values, not merely within tolerance. Aim for that; a change that only
*nearly* matches deserves an explanation.

### Still on the table

Neither is urgent, and both are smaller than they look:

- Per-chunk `pd.read_csv(skiprows=…)` re-decompresses everything ahead of it in
  a gzipped file, and parses all columns when detection needs six. Real, but
  ~50 ms against ~1.7 s of detection.
- Importing `dyco.pwb` takes ~1.3 s, of which matplotlib is ~0.2 s, and every
  worker pays it once per run under Windows spawn. Deferring the matplotlib
  import would need the plotting methods reworked.

---

## Open

Two items, neither urgent. Anything else belongs in the GitHub issue tracker.

**`rawio` unification, part two.** The *opening* layer is done — `rawio.py`
owns compression for every module. What is still split is *parsing*:
`dyco/files.py` (used by `split.py`) reads parquet and reconciles a column-count
mismatch, while `pipeline.py`'s `_read_raw_file` / `_write_raw_file` handle
arbitrary metadata rows, preserve line endings and can write. Folding the first
into the second is low priority — `files.py` has exactly one consumer, and the
defect class that motivated it lived in the opening layer, which is now shared.

**Release chores.** `CITATION.cff` now carries `version: 3.0.0` and the Zenodo
concept DOI `10.5281/zenodo.4964067`, which resolves to the latest version and
is the one to cite for all versions; its `date-released:` is still commented
out. The `CHANGELOG.md` heading is `## v3.0.0 | unreleased` — both dates go in
at release, deliberately not before. Check `status.svg` still points somewhere
valid. **The version in `pyproject.toml` is the user's; do not touch it** — but
`CITATION.cff` has to be kept in step with it.

---

## Known Dead Code

None catalogued. The two known-dead blocks (`plot.py`'s `SummaryPlots`,
`setup.py`'s `FilesDetector`) went with the v2 path.

**The standing rule still holds: mention dead code, don't delete it.** Flag it
and let the user decide.

---

## Known Defects

None catalogued. The `analyze.py` defects were fixed, then the module was
removed; the matplotlib 3.9 breakage (`plt.cm.get_cmap`, `Axes.plot_date`) lived
entirely in `loop.py`, `analyze.py` and `plot.py`, all of which are gone.

Worth carrying forward: nothing real has ever been caught by reading this code.
The matplotlib breakage, the three gzip faults and all five defects from the
RFlux comparison were found by *running* it — against real data, or against the
reference implementation. Prefer that over inspection.

The same held for speed. Every plausible-sounding target in the 2026-08-02 perf
pass — the file reading, FFT threading, fusing the bootstrap gather — was worth
nothing or worse, and the one that paid was a single constant. See
**Performance** above; the rule is the same: measure rather than reason.

---

## Gotchas

- **Compression is a storage detail, not a format.** `.gz`, `.bz2`, `.xz` and
  `.zip` are read as the text inside them. `--output-suffix` sets the whole
  output extension (`.csv`, `.csv.gz`), independent of the input; the name
  template's `{suffix}` expands to it, and `{stem}` is the input name with
  every suffix stripped.
- **Compression is `rawio.py`'s job, nobody else's.** `pipeline.py`,
  `apply_tlag.py` and `tui.py` each grew their own `open()` calls, and each
  broke on compressed input independently — the TUI's silently, returning
  mojibake column names. If you add a reader, go through `rawio`.
- **Header splitting is `rawio.split_header_line`'s job**, for the same
  reason. The header row was split on the raw separator in six places across
  four modules, none of which stripped quotes, so a logger that writes
  `"TIMESTAMP","u","CH4"` produced columns named `"u"` and every run died with
  every column "missing" from a file whose columns were all there. pandas
  strips those quotes on the data rows, so the header has to agree with it.
  Fixed 2026-08-03 against real CZ-Lnz QCL files.
- **Non-finite is missing.** Loggers write literal `Inf`/`-Inf` into data
  columns (CZ-Lnz H2O does). `--na-values` cannot catch it — argparse won't
  even accept `-Inf` as a value — and `na.approx` *spreads* it into any gap
  beside it before scipy's `detrend` finally rejects the array, naming neither
  column nor period. `pwb._finite_or_nan` folds inf into NaN at the point the
  columns are read; `{gas}_n_valid` counts finite records only, so an all-`Inf`
  column takes the same `no_data` path as an empty one.
- **A gas missing for a whole period gets no lag, and that has to survive
  PWBOPT.** An offline analyser writes its fill value down the entire column
  (~1 period in 8 on CZ-Lnz). There is nothing to detect and nothing to shift,
  so detection is skipped, `{gas}_n_valid` is 0 and `{gas}_lag_source` is
  `no_data`. PWBOPT's S3 carry and `@lagfrom=` fill periods whose detection was
  *rejected* and cannot distinguish those from periods with no data, so they
  would happily carry a lag into an empty column — the mask that stops them is
  applied to the summary columns only, **not** to `donors[label]`, because a
  gas that does have data may still legitimately borrow this gas's
  interpolated lag for that period.
- **Never donate a sticky gas's lag to an inert one** via `@lagfrom=`. H2O
  adsorbs and desorbs on the tube wall, so its lag runs longer than the
  flow-through delay and moves with humidity and tube age; CO2, CH4 and N2O
  ride the flow. H2O is frequently the *most reliable* detector in a dataset,
  which makes it the tempting donor and the wrong one — the borrowed
  wall-interaction delay biases the recipient's flux. Match donor to recipient
  by behaviour, not by detection rate.
- **Two raw-file *parsers* still exist**: `files.py` (parquet, column
  reconciliation) and `pipeline.py` (metadata rows, line endings, writing). See
  **Open** above. Only the opening layer was unified.
- Two near-identical `add_data_stats` functions used to exist. Only
  `_vendor/filedetector.py:24` remains; `files.py`'s six-argument variant went
  with the v2 path.
- **Every CLI writes `log.txt`** to its output folder: the run header, the
  per-file/per-chunk lines, and the finish time. The animated progress display
  is deliberately kept out of it.
- **Textual tests must wait for the state they assert.** `Input.Changed` is
  delivered through the message queue, so setting a value and calling
  `pilot.pause()` once is a race -- `test_tui_win_field_autosync` failed about
  one run in six that way. Use the `_settle` helper in `tests/test_pwb.py`.
- **The TUI settings path is `~/.dyco/detect_remove_tui.yaml`.** An existing
  `~/.diive/` config from before the migration will not be found.
- **Do not truncate pytest output.** Piping the suite through `tail -5` hides
  which test failed; use `-rf`, or read `.pytest_cache/v/cache/lastfailed`.

---

## Testing

`tests/` holds **198 tests plus 120 subtests**, seeded by diive's
`test_echires.py` (1,297 lines) and extended with gzip, CLI and R-reference
suites.

```bash
uv run pytest tests/ -q     # 198 passed, 120 subtests
```

When adding tests: use flexible assertion ranges for anything involving
stochastic components (the block bootstrap in the PWB code). Do not mock file
I/O in integration tests — this library's whole job is reading and writing real
files. And note that the test suite has never been what caught the real
breakages: the gzip faults and the matplotlib 3.9 removals were all found by
running the code.

---

## Coding Standards

- **Input validation** — only at system boundaries (CLI args, user files).
  Trust internal code.
- **Error handling** — let exceptions propagate unless you can recover. Never
  `sys.exit()` from library code — the removed `analyze.py` did, with no status,
  so a failed run exited `0` and looked successful to the calling shell.
- **Comments** — only WHY, not WHAT. Hidden constraints, workarounds,
  non-obvious logic.
- **Console strings must be cp1252-safe** (Windows stdout): use ASCII `->`,
  not `→`. This bites through libraries too: Rich's default spinner is braille,
  and it crashed `dyco apply-batch` with `UnicodeEncodeError` before the run
  started. All three CLIs pass `SpinnerColumn('line')`.

### Module docstring format

```python
"""
MODULE_NAME: DESCRIPTIVE_TITLE
================================

Brief one-line description of scope and purpose.

Part of the dyco package: https://github.com/holukas/dyco
"""
```

Existing modules carry the GPL-3.0 header block instead. Keep it when editing
them; new modules should have both.

---

## Text Writing Standards

Use the `/llm-detox` skill for all written content — documentation, comments,
commit messages, README prose, CHANGELOG entries.

---

## Publication Status

dyco is a **published, citable package**. The citation is:

> Hörtnagl, L., (2021). DYCO: A Python package to dynamically detect and
> compensate for time lags in ecosystem time series. *Journal of Open Source
> Software*, 6(62), 2575, https://doi.org/10.21105/joss.02575

Single author, 2021 — not "Hörtnagl et al. (2020)". The `date: 31 Jul 2020` in
`paper/paper.md` is the submission date, not the publication year. That source
lives in `paper/` with `paper.bib`; `CITATION.cff` carries the same reference as
`preferred-citation`, plus the author's ORCID.

**The paper describes `v1.1.2`** — released 16 Jun 2021 for that publication —
i.e. the covariance-maximization method, which v3 removed. Anyone reading the
paper and then this code is reading about two different algorithms; say so
wherever the paper is cited.

**Do not edit `paper/`** without explicit instruction — it is the published
record. The newer pre-whitening block-bootstrap method has its own manuscript
repo (`holukas/ms_fluxnet_ch4_n2o_timelag`).

---

**Last Updated:** 2026-08-02 | **Version:** v3.0.0 (unreleased)
