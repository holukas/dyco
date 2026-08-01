# CLAUDE.md — dyco Development Guide

Time-lag detection and compensation for eddy covariance raw data.
See `CHANGELOG.md` for version history.

> **New to this repo? Read `HANDOVER.md` first.** It carries the current state,
> what is uncommitted, what is open and in what order, and the traps worth
> knowing before you spend time on them. This file is the standing reference:
> conventions, architecture, and the rules that outlive any one task.

---

## [CRITICAL] PWB is the only method

dyco carries **one** lag-detection method: pre-whitening block-bootstrap.

| | |
|---|---|
| Entry point | `dyco detect-remove` (`dyco/pipeline.py`) |
| Detection | `dyco/pwb.py` |
| Lag selection | PWBOPT S1/S2/S3, per chunk |
| Removal | `dyco/apply_tlag.py` `TlagApplier` |
| Tests | `tests/test_pwb.py` + 5 more, 98 total |

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

## [READ FIRST] v3 state

The migration is **done**: all of diive's `flux/hires/` plus `filesplitter.py`
now live here, and the diive dependency is gone.

`HANDOVER.md` carries the current state and what is still open. `MIGRATION_V3.md`
is historical; its §4 proposed deleting the v2 modules, which is what eventually
happened, though for a different reason and on the user's call.

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
Match existing style. **Mention dead code — don't delete it** (this repo has
several known dead blocks, see below). Remove only imports and variables that
YOUR changes made unused.

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
uv run pytest tests/ -q                                  # 98 passed
uv run python examples/detect_remove_tlag_realdata.py    # real-data end-to-end
uv run dyco                                              # list all workflows
```

**Companion repo:** `F:\dev\diive`. Nothing here imports it any more, and
nothing should. It is only useful as historical reference for where this code
came from.

---

## Current Architecture

~9,900 lines across 10 modules plus ~400 in `_vendor/`. Pipeline:
**split into chunks → rotate → detect per chunk → PWBOPT across the sequence →
shift and write**.

| Module | LOC | Role |
|---|---|---|
| `dyco/pipeline.py` | 3,006 | `PerFilePipeline`, `process_one_file`. The primary workflow, and its own raw-file reader/writer |
| `dyco/pwb.py` | 2,541 | `PreWhiteningBootstrap`, `PwbBatchDetection`, `PwboptLagPlot`. The detection method itself |
| `dyco/tui.py` | 1,808 | Textual UI over the pipeline. `--demo` needs no data |
| `dyco/apply_tlag.py` | 683 | `TlagApplier` — remove lags listed in an existing `tlag_results.csv` |
| `dyco/split.py` | 524 | `FileSplitter`, `FileSplitterMulti` — divide a long raw file into averaging-period parts |
| `dyco/detectionlimit.py` | 476 | `FluxDetectionLimit` — minimum detectable flux, read off the far tail of the cross-covariance function |
| `dyco/maxcov.py` | 417 | `MaxCovariance` — covariance-maximization lag estimator. `FluxDetectionLimit` builds on it |
| `dyco/files.py` | 195 | Raw CSV/parquet reading for `split.py`, incl. header-vs-data column reconciliation |
| `dyco/rotation.py` | 138 | `WindDoubleRotation`, `reynolds_decomposition`. Chunks are rotated before the search |
| `dyco/cli.py` | 101 | Unified `dyco` dispatcher |
| `dyco/_vendor/` | ~400 | Leaf utilities copied from diive; see its `__init__.py` for the rationale |
| `dyco/__init__.py` | 2 | A comment. **No public API is defined** |

**Lag is expressed in seconds.** The removed v2 path counted records instead, so
older notes, plots and result files may mean something different by "lag".

**Never import from diive again.** If something is needed from there, copy it
into `_vendor/` with a provenance note, or reimplement it. The cross-repo
coupling is what broke dyco four ways, and it is deliberately gone.

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

Worth carrying forward: that breakage was found by *running* the code, not by
the test suite, and so were the three gzip faults. Neither would have surfaced
from reading.

---

## Gotchas

- **Two raw-file readers exist.** `pipeline.py`'s `_read_raw_file` /
  `_write_raw_file` serve the PWB path; `files.py` serves `split.py`. They share
  no code, which is why the same class of gzip gap had to be found and fixed in
  each. `files.py` reads parquet and reconciles a column-count mismatch;
  `pipeline.py` handles arbitrary metadata rows, preserves line endings, and can
  write. See `HANDOVER.md` §5.3.
- Two near-identical `add_data_stats` functions used to exist. Only
  `_vendor/filedetector.py:24` remains; `files.py`'s six-argument variant went
  with the v2 path.

---

## Testing

`tests/` holds **98 tests plus 31 subtests**, seeded by diive's
`test_echires.py` (1,297 lines) and extended with gzip and CLI suites.

```bash
uv run pytest tests/ -q     # 98 passed, 31 subtests
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
  not `→`.

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

**Last Updated:** 2026-08-01 | **Version:** v3.0.0 (unreleased)
