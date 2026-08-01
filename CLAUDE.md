# CLAUDE.md — dyco Development Guide

Time-lag detection and compensation for eddy covariance raw data.
See `CHANGELOG.md` for version history.

> **New to this repo? Read `HANDOVER.md` first.** It carries the current state,
> what is uncommitted, what is open and in what order, and the traps worth
> knowing before you spend time on them. This file is the standing reference:
> conventions, architecture, known dead code and known defects.

---

## [CRITICAL] PWB is the primary method; the v2 method is retained

dyco now carries **two** lag-detection methods. Do not treat them as equals when
choosing defaults, docs order, or where new work goes:

| | Primary | Retained |
|---|---|---|
| Method | **Pre-whitening block-bootstrap (PWB)** | Iterative max-covariance window narrowing |
| Entry point | `dyco detect-remove` (`dyco/pipeline.py`) | `dyco cm` -> `dyco/dyco.py` `Dyco` |
| Detection | `dyco/pwb.py` | `dyco/maxcov.py` + `dyco/lag.py` |
| Lag selection | PWBOPT S1/S2/S3, per chunk | histogram narrowing -> daily median LUT (`dyco/analyze.py`) |
| Removal | `dyco/apply_tlag.py` `TlagApplier` | `dyco/correction.py` `RemoveLags` |
| Tests | `tests/test_pwb.py` + 5 more, 104 total | none, by decision |

New features, docs and defaults go to the PWB path. **The v2 path stays
available and must keep working** - do not delete `dyco.py`, `loop.py`,
`analyze.py`, `correction.py`, `setup.py`, `lag.py` or `plot.py`. An earlier
draft of `MIGRATION_V3.md` proposed deleting five of them; that is superseded.

**But do not invest in it either.** Decided: the v2 path is kept working, not
extended - no new tests, no reference run. Its z-score deviation
(`_vendor/outliers.py`) therefore stays unvalidated *by decision*, not by
oversight.

The two methods answer the same question differently. PWBOPT decides per chunk
whether a detection is trustworthy; the v2 path pools detections into a daily
median and normalizes toward a target lag. They are not interchangeable, and
neither is a drop-in replacement for the other.

## [READ FIRST] v3 state

The migration is **done**: all of diive's `flux/hires/` plus `filesplitter.py`
now live here, and the diive dependency is gone.

`HANDOVER.md` carries the current state and what is still open. `MIGRATION_V3.md`
is historical, and parts of it are explicitly superseded — in particular its
proposal to delete five of dyco's modules, which is void (see the rule above).

`cli.py` is now the unified `dyco` dispatcher (`detect-remove`, `tui`, `pwb-batch`,
`apply-batch`, `cm`); the standalone `dyco-*` scripts still work.

## [DONE] dyco no longer depends on diive

As of 2026-08-01 the diive dependency is gone. All seven former imports are
satisfied locally, the four breakages that made dyco uninstallable are fixed
(both dead import paths, plus the Python and pandas floors), and every module
imports in a clean environment with no diive present.

What that involved:

- **`dyco/_vendor/`** — leaf utilities copied from diive: `times`
  (`create_timestamp`, `calc_true_resolution`), `fileio` (`search_files`,
  `read_parquet`), `filedetector` (`FileDetector`, `add_data_stats`), `frames`
  (`trim_frame`), `plotstyle` (`default_format` + theme constants), `console`,
  and `outliers` (a port of `zScoreRolling`).
- **`dyco/maxcov.py`, `dyco/rotation.py`, `dyco/detectionlimit.py`,
  `dyco/split.py`** — domain code moved over from diive. Not vendored helpers;
  these are dyco's own now.

**Verified end-to-end on real data.** `examples/detect_remove_tlag_realdata.py`
runs the full PWB pipeline over a bundled 1-hour CH-LAE file: correct chunking,
genuinely gzipped output, headers byte-identical to the source, and scalar
columns shifted by exactly the applied lag. `FluxDetectionLimit` also reproduces
diive's numbers to 10 decimal places.

The **v2 path** has had no such run, by decision.

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
uv run pytest tests/ -q                                  # 104 passed
uv run python examples/detect_remove_tlag_realdata.py    # real-data end-to-end
uv run dyco                                              # list all workflows
```

**Companion repo:** `F:\dev\diive`. Nothing here imports it any more, and
nothing should. It is only useful as historical reference for where this code
came from.

---

## Current Architecture (v2)

~3,400 lines across 10 modules. Pipeline: **detect lags → analyze → build
lookup table → shift files**.

| Module | LOC | Role |
|---|---|---|
| `dyco/dyco.py` | ~490 | `Dyco` orchestrator. `detect_lags()` → `analyze_lags()` → `remove_lags()` |
| `dyco/loop.py` | 750 | `Loop` — iterate files, then segments within each file; calls `MaxCovariance` per segment |
| `dyco/lag.py` | 350 | `AdjustLagsearchWindow` — histogram-based iterative narrowing of the search window. **dyco's core IP** |
| `dyco/analyze.py` | 415 | `AnalyzeLags` — outlier filtering, daily median lookup table, normalization correction |
| `dyco/correction.py` | 155 | `RemoveLags` — shift target variables by the LUT lag, write output files |
| `dyco/files.py` | 292 | Raw CSV/parquet reading, header-vs-data column reconciliation |
| `dyco/plot.py` | 300 | `default_format`/`format_spines`/`setup_fig_ax` (live) + `SummaryPlots` (dead) |
| `dyco/setup.py` | ~260 | **Not packaging** — see gotchas. Output dirs, logger, run ID |
| `dyco/cli.py` | 250 | **Rewritten in v3.** Unified `dyco` dispatcher; the v2 workflow is `dyco cm` |
| `dyco/__init__.py` | 1 | A single commented-out line. **No public API is defined** |

**Lag is expressed in "number of records", not seconds.** At 20 Hz, 1000
records = 50 s. Negative lag means the scalar lags behind the wind.

### Moved in from diive (2026-08-01)

Not part of the v2 design. These arrived when the diive dependency was severed.

| Module | LOC | Role |
|---|---|---|
| `dyco/maxcov.py` | 501 | `MaxCovariance` — the lag estimator `loop.py` drives. v2 imported this from diive; dyco had no estimator of its own |
| `dyco/rotation.py` | 179 | `WindDoubleRotation`, `reynolds_decomposition`. Prerequisite for lag detection — chunks are rotated before the search |
| `dyco/detectionlimit.py` | 570 | `FluxDetectionLimit` — minimum detectable flux, read off the far tail of the same cross-covariance function |
| `dyco/split.py` | 578 | `FileSplitter`, `FileSplitterMulti` — divide a long raw file into averaging-period parts. Was unreferenced dead code in diive |
| `dyco/_vendor/` | ~480 | Leaf utilities copied from diive; see its `__init__.py` for the rationale |

**Never import from diive again.** If something is needed from there, copy it
into `_vendor/` with a provenance note, or reimplement it. The cross-repo
coupling is what broke dyco four ways, and it is deliberately gone.

`_vendor/outliers.py` carries a **documented behavioural deviation** from
diive's `zScoreRolling` — it does not regularize an irregular index. Read the
module docstring before changing it.

---

## Known Dead Code

Present in the repo, unreferenced. **Do not delete without asking** — flag it
and let the user decide. All of it is slated for removal by the v3 migration.

| Where | What |
|---|---|
| `plot.py:32–264` | `SummaryPlots` — never referenced. Also broken 3 ways: `:46` `NameError` (`setup_dyco` not imported), `:168` `TypeError` (passes `iteration=`/`phase=` that `loop.py:169` does not accept), `:248` `AttributeError` (`AnalyzeLags.filter_dataframe` does not exist). Leftover **v1** code using a "Phase 1–3" model that v2 replaced with "Steps 1–7" |
| `setup.py:147` | `FilesDetector` — never referenced. Superseded local copy; `dyco.py:371` uses diive's `FileDetector` instead |

---

## Known Defects

Pre-existing, in live code. Fix during the v3 port, not opportunistically.

`analyze.py:246` (the `fillna` whose result was never assigned) was **fixed in
v3** after confirming no published result relied on that branch. The rest stand.

| Location | Defect |
|---|---|
| `analyze.py:80` | `sys.exit()` inside a library. Should raise |
| `analyze.py:221` | `ABS_LIMIT = 50` hardcoded. Should be a parameter |
| `analyze.py:62` | `__init__` calls `self.run()` — work in the constructor |

---

## Gotchas

- **`dyco/setup.py` is not a packaging file.** It holds `set_dirs`,
  `create_logger`, `CreateOutputDirs`, `FilesDetector`, `generate_run_id`,
  `set_logfile_path`. Do not treat it as setuptools config. It was once named
  `setup_dyco` — `plot.py:46` still refers to that old name, which is one
  reason `SummaryPlots` cannot run.
- **Output directories are addressed by string key** through an `outdirs` dict
  built in `setup.CreateOutputDirs`: `'0_log'`, `'1_overview'`,
  `'2_covariances'`, `'3_covariances_plots'`, `'4_time_lags_overview'`,
  `'5_time_lags_overview_histograms'`, `'6_time_lags_overview_timeseries'`,
  `'7_time_lags_lookup_table'`, `'8_time_lags_corrected_files'`. Renaming a key
  breaks callers silently — grep before touching.
- **A `logger` is threaded explicitly** through most constructors rather than
  obtained per-module. Keep that pattern until v3 changes it deliberately.
- **`analyze.py` calls into `loop.py`** (`:162` → `Loop.plot_segment_lagtimes_ts`).
  The v3 plan deletes `loop.py`, so that plot must be ported first.
- Two near-identical `add_data_stats` functions exist — `files.py:146` here and
  `filedetector.add_data_stats` in diive. Common ancestry; only one survives v3.

---

## Testing

**There are currently no tests.** No `tests/` directory, no test files.

The v3 migration brings `F:\dev\diive\tests\test_echires.py` (1,297 lines)
across, which becomes the initial suite. Getting it green is a hard gate in the
migration sequence — nothing proceeds past it.

When adding tests: use flexible assertion ranges for anything involving
stochastic components (the block bootstrap in the incoming PWB code). Do not
mock file I/O in integration tests — this library's whole job is reading and
writing real files.

---

## Coding Standards

- **Input validation** — only at system boundaries (CLI args, user files).
  Trust internal code.
- **Error handling** — let exceptions propagate unless you can recover. Never
  `sys.exit()` from library code (see Known Defects).
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
