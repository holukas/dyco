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
| Lag selection | PWBOPT S1/S2/S3, per chunk; a gas may borrow another's lag where it has none (`@lagfrom=`) |
| Removal | `dyco/apply_tlag.py` `TlagApplier` |
| Tests | `tests/test_pwb.py` + 7 more, 143 total |

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
uv run pytest tests/ -q                                  # 143 passed
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
| `dyco/pwb.py` | 3,062 | `PreWhiteningBootstrap`, `PwbBatchDetection`, `PwboptLagPlot`. The detection method itself |
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

## Open

Two items, neither urgent. Anything else belongs in the GitHub issue tracker.

**`rawio` unification, part two.** The *opening* layer is done — `rawio.py`
owns compression for every module. What is still split is *parsing*:
`dyco/files.py` (used by `split.py`) reads parquet and reconciles a column-count
mismatch, while `pipeline.py`'s `_read_raw_file` / `_write_raw_file` handle
arbitrary metadata rows, preserve line endings and can write. Folding the first
into the second is low priority — `files.py` has exactly one consumer, and the
defect class that motivated it lived in the opening layer, which is now shared.

**Release chores.** `CITATION.cff` needs its `version:` and a Zenodo DOI (the
`doi:` field is commented out). The `CHANGELOG.md` heading is
`## v3.0.0 | unreleased` — the date goes in at release, deliberately not before.
Check `status.svg` still points somewhere valid. **The version in
`pyproject.toml` is the user's; do not touch it.**

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

`tests/` holds **143 tests plus 117 subtests**, seeded by diive's
`test_echires.py` (1,297 lines) and extended with gzip, CLI and R-reference
suites.

```bash
uv run pytest tests/ -q     # 143 passed, 117 subtests
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

dyco is a **published, citable package**. `paper/paper.md` is a JOSS-format
paper (*"DYCO: A Python package to dynamically detect and compensate for time
lags in ecosystem time series"*, 2020) with `paper.bib`; `CITATION.cff` carries
the author's ORCID.

**Do not edit `paper/`** without explicit instruction — it is the published
record. The newer pre-whitening block-bootstrap method has its own manuscript
repo (`holukas/ms_fluxnet_ch4_n2o_timelag`).

---

**Last Updated:** 2026-08-02 | **Version:** v3.0.0 (unreleased)
