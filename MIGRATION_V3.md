# dyco v3 — migration plan

Working document for the v3 rebuild. Written 2026-08-01 against dyco `indev`
at commit `a69a9b2` (identical to `origin/main`, clean tree).

**This document is self-contained.** It assumes no prior context. Read §0 and §1
before touching anything.

---

## SUPERSEDED — read `CLAUDE.md` first

Two decisions in this document have been overtaken:

1. **PWB is the only method**, not one of two equal backends. The v2 iterative
   max-covariance path was removed on 2026-08-01 at the user's instruction.
   (In between it was briefly "retained but not extended" — that is also
   superseded.)
2. **§4's "delete five modules" turned out right**, but not for the reason given
   here, and it undercounted: `dyco.py`, `loop.py`, `lag.py`, `analyze.py`,
   `correction.py`, `plot.py` and `setup.py` are all gone. Read §4 as history,
   not as instructions.

Everything about the file moves, the vendoring and the diive-side removals still
holds.

## STATUS — the move is complete

**The diive dependency is severed** (2026-08-01, uncommitted). All seven former
imports resolve locally; every dyco module imports in a clean environment with
no diive installed; `pyproject.toml` is on hatchling + uv with Python
`>=3.12,<3.14` and pandas `>=3.0.0`, and no longer lists diive.

Landed in dyco:

| New | Source | Notes |
|---|---|---|
| `dyco/_vendor/` (~480 lines) | diive leaf utilities | `times`, `fileio`, `filedetector`, `frames`, `plotstyle`, `console`, `outliers` |
| `dyco/maxcov.py` | `flux/hires/lag.py` | `MaxCovariance` |
| `dyco/rotation.py` | `flux/hires/windrotation.py` | copy — diive still needs its own |
| `dyco/detectionlimit.py` | `flux/hires/fluxdetectionlimit.py` | moved |
| `dyco/split.py` | `core/io/filesplitter.py` | moved + rewired to dyco's reader |
| `tests/` | diive `test_echires.py` | 3 tests, green with no diive present |

Removed from diive (all verified to have no remaining consumers): the whole
`flux/hires/` package, `core/io/filesplitter.py` and `core/io/filedetector.py`.

**Kept in diive on purpose:** `create_timestamp`, `calc_true_resolution` and
`trim_frame`. They were orphaned by the move and briefly removed, then restored
at the user's request - the judgement being that they are generic enough to
belong in diive regardless of who currently calls them. dyco's `_vendor/times.py`
and `_vendor/frames.py` are therefore **copies, not relocations**. Two live
copies of each now exist; a bug fixed in one should be checked against the other.

**All of `diive/flux/hires/` is now in dyco** (2026-08-01, uncommitted), plus
`core/io/filesplitter.py`. The package no longer exists in diive; `dv.flux` has
lost `hires`, `MaxCovariance`, `FluxDetectionLimit`, `WindDoubleRotation`,
`reynolds_decomposition`, `PreWhiteningBootstrap`, `PwbBatchDetection`,
`PwboptLagPlot`, `TlagApplier`, `PerFilePipeline` and `process_one_file`.
diive also dropped `textual` and `polars` and 4 of its 5 console scripts.

| dyco module | From |
|---|---|
| `pwb.py` | `lag_pwb.py` |
| `pipeline.py` | `detect_and_remove_tlag.py` |
| `apply_tlag.py` | `apply_tlag.py` |
| `tui.py` | `detect_and_remove_tlag_tui.py` |
| `maxcov.py` | `lag.py` |
| `rotation.py` | `windrotation.py` |
| `detectionlimit.py` | `fluxdetectionlimit.py` |
| `split.py` | `core/io/filesplitter.py` |
| `_vendor/` | leaf utilities |
| `tests/test_pwb.py` | `tests/test_echires.py` |
| `examples/` (9 files) | `examples/flux/hires/` |

Console scripts renamed: `dyco-detect-remove`, `dyco-detect-remove-tui`,
`dyco-pwb-batch`, `dyco-apply-batch`. The TUI settings file moved from
`~/.diive/` to `~/.dyco/`.

**Verified:** 77 tests + 22 subtests pass in a clean Python 3.12 environment
with no diive installed; all four console-script entry points resolve; the v2
path still imports.

**Not verified:** step 1 was skipped — no v2 reference run exists, so nothing is
checked against pre-migration *pipeline* output. This now matters mainly for the
retained v2 path (decision 8's z-score deviation feeds its lookup table). The
PWB path arrived with its own test suite, which is why the risk is lopsided.

---

## 0. Start here

You are in `F:\dev\dyco` on branch `indev`. The companion repo is
`F:\dev\diive` — you will read from it constantly and write to it only at
step 10.

**dyco does not currently run.** Four independent breakages (§2). The first
task is to make v2 work again so it can produce reference outputs to verify the
rebuild against. Do not skip that — every later verification depends on it.

First session checklist:

1. Read §1 (context), §2 (breakages), §6 (decisions).
2. Confirm `F:\dev\diive` exists and is on a branch you can read.
3. Do step 1 of §10: fix the pins and the two dead imports, run v2, capture the
   reference daily lookup table.
4. Stop and confirm the reference output looks right before proceeding.

**Working agreement:** never run `git commit`, `git add` or `git push` — staging
and committing is the user's job exclusively. Never bump version numbers
manually. State assumptions before coding; ask when a choice is genuinely open.

---

## 1. Context

### What dyco is

`dyco` (dynamic lag compensation) takes eddy covariance raw data files as input
and produces lag-compensated raw data files as output. It detects the time lag
between vertical wind `W` and a scalar `S` (CO2, CH4, N2O) by iteratively
narrowing a covariance search window, filters outlier lags into a daily lookup
table, then shifts variables in the input files to remove the lag.

It is a **published, citable package**: `paper/paper.md` is a JOSS-format paper
(*"DYCO: A Python package to dynamically detect and compensate for time lags in
ecosystem time series"*, 2020), with `paper.bib` and a `CITATION.cff` carrying
the author's ORCID. The name is not up for discussion.

### What diive is

`diive` (`F:\dev\diive`) is a large time-series processing library for
post-processed 30-minute flux data — gap-filling, QA/QC, flux processing chain,
a desktop GUI. Over the years it accumulated a `flux/hires/` subpackage for
**raw high-frequency** EC data that does not belong there.

### The goal

Draw a clean boundary:

- **dyco** owns everything operating on raw high-frequency EC files on disk.
- **diive** keeps everything operating on post-processed 30-minute time series.

In practice: all of `F:\dev\diive\diive\flux\hires\` plus
`F:\dev\diive\diive\core\io\filesplitter.py` moves into dyco, dyco's own
orchestration is replaced by the (newer, better) equivalents coming in, and
dyco stops depending on diive entirely.

### Why this is the right shape

- dyco already imports `MaxCovariance` **from** diive — it has no lag estimator
  of its own. The two repos are one toolchain split across a repo boundary.
- That cross-repo coupling is what broke dyco (§2). Co-locating the primitive
  with its caller removes the failure mode.
- 89% of the merged package is lag work (~10,400 of ~11,700 domain lines), so
  the dyco name still describes it. Only `FluxDetectionLimit` (541 lines, 4.6%)
  is off-theme.
- diive sheds 10,804 lines (10.7%), the `textual` and `polars` dependencies, and
  4 of its 5 console scripts.

---

## 2. dyco's four breakages

`pip install dyco` alongside current diive is unsatisfiable.

| # | Breakage | Detail |
|---|---|---|
| 1 | Dead import | `dyco/loop.py:30` → `diive.pkgs.echires.lag.MaxCovariance`. `diive/pkgs/` now holds only `analysis` and `features`; `echires` became `flux/hires` |
| 2 | Dead import | `dyco/analyze.py:27` → `diive.pkgs.outlierdetection.zscore.zScoreRolling`. Now at `F:\dev\diive\diive\preprocessing\outlier_detection\zscore.py:249` |
| 3 | Python pin | dyco `requires-python = ">=3.11,<3.12"` vs diive `">=3.12,<3.14"`. Mutually exclusive |
| 4 | pandas pin | dyco `pandas (>=2.2.3,<3.0.0)` vs diive `pandas>=3.0.0,<4.0.0`. Mutually exclusive |

Pins 3 and 4 must be bumped regardless of anything else — the incoming hires
code is written against Python 3.12 and pandas 3.

The `zScoreRolling` **API** still matches what dyco calls: `calc(repeat=True)`,
`.fig`, `.get_flag()` all still exist. Only the path is stale.

---

## 3. What moves in from diive

All paths relative to `F:\dev\diive\`.

### A. `diive\flux\hires\` — 10,232 lines, all of it

| File | LOC | Contains |
|---|---|---|
| `windrotation.py` | 173 | `WindDoubleRotation`, `reynolds_decomposition` |
| `lag.py` | 496 | `MaxCovariance` |
| `fluxdetectionlimit.py` | 541 | `FluxDetectionLimit` |
| `lag_pwb.py` | 2,938 | `PreWhiteningBootstrap`, `PwbBatchDetection`, `PwboptLagPlot` |
| `apply_tlag.py` | 787 | `TlagApplier` |
| `detect_and_remove_tlag.py` | 3,263 | `PerFilePipeline`, `process_one_file` |
| `detect_and_remove_tlag_tui.py` | 2,012 | `DetectRemoveTUI` (Textual) |

Also `tests\test_echires.py` (1,297 lines) and `examples\flux\hires\`
(9 files + `tlag_results_synthetic.csv`).

### B. `diive\core\io\filesplitter.py` — 572 lines

`FileSplitter` / `FileSplitterMulti`. **Dead code in diive** — nothing imports
it; its only callers are two `__main__` demos inside the file. It also violates
diive's own layering rule by importing from `flux/`, which is inert only
because nothing reaches it.

---

## 4. What dyco keeps and loses

| Module | LOC | Fate |
|---|---|---|
| `dyco/lag.py` (`AdjustLagsearchWindow`) | 350 | **Keep** → `lag/window.py`. dyco's real IP |
| `dyco/analyze.py` | 415 | **Keep, reworked** → `lag/lookup.py`. Not a clean move — see §5 |
| `dyco/correction.py` (`RemoveLags`) | 155 | **Fold** into `lag/apply.py` with `TlagApplier` |
| `dyco/loop.py` | 750 | **Delete** — superseded by `PerFilePipeline`. First port `plot_segment_lagtimes_ts` (`:169`), which `analyze.py:162` depends on |
| `dyco/files.py` | 292 | **Delete** — header reconciliation folds into `rawio.py` |
| `dyco/plot.py` | 300 | **Delete.** `SummaryPlots` (`:32–264`) is already dead (§5). Live part is `default_format`/`format_spines`/`setup_fig_ax` (`:267–299`, 9 call sites), replaced by vendored `_internal/plotstyle.py` |
| `dyco/dyco.py` | ~490 | **Delete** — orchestrator superseded by `PerFilePipeline` |
| `dyco/setup.py` | ~260 | **Delete** — folder scaffolding. Note `FilesDetector` (`:147`) is already dead: never referenced, superseded by diive's `FileDetector` which `dyco.py:371` uses instead |
| `dyco/cli.py` | ~170 | **Rewrite** — built around the deleted orchestrator |
| `dyco/__init__.py` | 1 | One commented-out line. **No public API to preserve** — define one freely |

Five of ten modules are deleted. **v3 is a rebuild around `AdjustLagsearchWindow`
and `analyze.py`, not an extension of v2.** Budget accordingly.

---

## 5. Known defects in the code being migrated

Found by reading the files in full. None are caused by this migration; all
predate it. Fix during the port rather than carrying them over.

### `dyco/analyze.py`

| Line | Defect |
|---|---|
| `:246` | `lut_df['INSTANTANEOUS_LAG'].fillna(lut_df['DEFAULT_LAG'])` — **result never assigned, the call is a no-op.** The "fill missing lags with default lag" path has never done anything, while the logger reports that it did. Check whether any published result relied on this branch before changing it |
| `:80` | `sys.exit()` inside a library when the LUT comes back empty. Should raise |
| `:162` | Calls `loop.Loop.plot_segment_lagtimes_ts` — depends on the module being deleted. Port that plot first, or drop the call |
| `:221` | `ABS_LIMIT = 50` hardcoded. Should be a parameter |
| `:62` | `__init__` calls `self.run()` — work in the constructor |

### `dyco/plot.py`

`SummaryPlots` is **never referenced** anywhere in the repo — the only
occurrence is its own class statement. It also cannot run:

| Line | Defect |
|---|---|
| `:46` | `setup_dyco.create_logger(...)` but only `setup` is imported → `NameError` |
| `:168` | `plot_segment_lagtimes_ts(iteration=, phase=)` — `loop.py:169` accepts neither → `TypeError` |
| `:248` | `analyze.AnalyzeLags.filter_dataframe(...)` does not exist (it is `Loop.filter_series`, `loop.py:482`) → `AttributeError` |

It is leftover **v1** code built on a "Phase 1–3" model that v2 replaced with
"Steps 1–7". Deleting it removes dead broken code, not working code.

### `dyco/setup.py`

`FilesDetector` (`:147`, ~155 lines) is also **never referenced**. It is the
superseded local copy of file detection; `dyco.py:371` uses diive's
`FileDetector` instead. Same story as `SummaryPlots` — left behind when v2
switched to the diive implementation.

**So two of dyco's ten modules are substantially dead already**, which is worth
knowing before mourning the five that §4 deletes.

---

## 6. Decisions

### Locked

| # | Decision | Rationale |
|---|---|---|
| 1 | Name stays `dyco`, version v3 | Published package with a paper and CITATION.cff |
| 2 | Poetry → hatchling + uv | Matches diive's toolchain. dyco currently uses `poetry-core` + `poetry.lock` + `[tool.poetry.group.dev.dependencies]` |
| 3 | Bump to Python `>=3.12,<3.14`, pandas `>=3.0.0,<4.0.0` | Forced — §2 |
| 4 | No diive dependency | See §6b |
| 5 | Break the v2 CLI | **DECIDED: break it.** Document the migration in CHANGELOG. Note the v2 orchestrator itself is retained, so `cli.py` is rewritten rather than deleted |
| 6 | Merge direction: diive's code lands **here** | It is 3× larger than dyco. The repo's centre of gravity shifts; that is fine |

### Open — decide before or during the work

| # | Question | Notes |
|---|---|---|
| ~~7~~ | `paper/` | **DECIDED: leave it** as the published v1/v2 record. Do not edit. The PWB method has its own manuscript (`holukas/ms_fluxnet_ch4_n2o_timelag`) |
| 8 | `zScoreRolling`: vendor or reimplement? | **Measure, don't guess.** See §6b |
| 9 | `CITATION.cff` | Update `version:` at release; mint a Zenodo DOI if not already (`doi:` is currently commented out) |

### 6b. Does diive remain a dependency?

**No — but it is a decision, not automatic.** All six dyco → diive imports
resolve after v3:

| Import | After v3 |
|---|---|
| `MaxCovariance` (`loop.py:30`) | **Moves in.** No longer external |
| `FileDetector` (`dyco.py:22`) | Module deleted; folds into `rawio` (§9) |
| `search_files` (`dyco.py:23`) | `pathlib.Path.rglob` |
| `load_parquet` (`files.py:25`) | `pd.read_parquet` — `files.py` deleted |
| `calc_true_resolution`, `create_timestamp` (`loop.py:29`) | Vendor; ~200 self-contained lines |
| `zScoreRolling` (`analyze.py:27`) | **The only real fork** — decision 8 |

**Decision 8 in detail.** Vendoring `zScoreRolling` costs ~936 lines
(`zscore.py` 350 + `flagbase.py` 348 + `funcs.py` 127 + `common.py` 60 +
`prints.py` 51) for one call site — nearly double the entire rest of §8.
`analyze._remove_outliers` (`:251–275`) only needs: rolling z-score on the
high-quality-peak series with `repeat=True`, keep `flag == 0`. That is ~15 lines
of pandas.

Write the 15 lines, then **diff the resulting LUT against the v2 reference
captured in step 1**. Identical → done. Not identical → investigate, and vendor
only if the difference cannot be explained.

**Why drop the dependency at all.** Not code volume — vendoring 400–600 lines to
avoid a dependency is a bad trade on volume alone. It is that diive's internal
restructuring has broken dyco four ways and dyco currently cannot be installed.
Once `MaxCovariance` lives here, the shared code is gone and what remains is
generic utilities any package would write itself. The coupling stops earning
its keep.

---

## 7. Target layout

Restructure in place on `indev`.

```
F:\dev\dyco\
├── pyproject.toml           hatchling + uv, Python >=3.12,<3.14
├── README.md                rewrite — see §12
├── CHANGELOG.md   LICENSE   CITATION.cff   MIGRATION_V3.md
├── paper/                   unchanged, historical record
├── images/                  logo keeps; workflow figures regenerate
├── dyco/
│   ├── __init__.py          define the v3 public API (v2's is empty)
│   ├── rawio.py             unified reader/writer — §9
│   ├── split.py             <- diive filesplitter.py + dyco loop_segments
│   ├── rotation.py          <- diive windrotation.py
│   ├── detectionlimit.py    <- diive fluxdetectionlimit.py
│   ├── lag/
│   │   ├── __init__.py
│   │   ├── maxcov.py        <- diive lag.py (MaxCovariance)
│   │   ├── pwb.py           <- diive lag_pwb.py
│   │   ├── window.py        <- dyco lag.py (AdjustLagsearchWindow)
│   │   ├── lookup.py        <- dyco analyze.py, reworked per §5
│   │   ├── apply.py         <- diive apply_tlag.py + dyco correction.py
│   │   ├── pipeline.py      <- diive detect_and_remove_tlag.py
│   │   └── tui.py           <- diive detect_and_remove_tlag_tui.py
│   └── _internal/
│       ├── console.py       vendored Rich helpers
│       └── plotstyle.py     vendored default_format + theme constants
├── tests/                   new — dyco v2 has none
└── examples/                <- diive examples\flux\hires\ + dyco example\
```

Two detection methods behind one pipeline for the first time: PWB (`pwb.py`) and
iterative max-covariance narrowing (`maxcov.py` + `window.py`). Directly
relevant to `holukas/ms-timelag-comparison`.

Console scripts: `dyco-pwb-batch`, `dyco-apply-batch`, `dyco-detect-remove`,
`dyco-detect-remove-tui`.

Dependencies: numpy, pandas, scipy, matplotlib, polars, textual, pyyaml, rich,
pyarrow. **No diive.**

---

## 8. Vendoring list

Source paths relative to `F:\dev\diive\`.

| Needed | Source | Action |
|---|---|---|
| `warn`/`info`/`detail`/`console` | `diive\core\utils\console.py` (209 lines) | Vendor whole file → `_internal/console.py` |
| `default_format` | `diive\core\plotting\plotfuncs.py:102` **or** `dyco/plot.py:267` | Vendor once. Compare both first — dyco's copy may be lighter |
| `LightTheme` constants | `diive\core\plotting\styles\` | Vendor only the constants actually read |
| `air_temp_from_sonic_temp` | `diive\variables\thermodynamic.py:167` | Inline: `ta = sonic_temp / (1 + 0.32 * h2o)`. Keep the Striednig 2020 citation comment |
| `create_timestamp` | `diive\core\times\times.py:2030` | Vendor; self-contained |
| `calc_true_resolution` | `diive\core\times\times.py:2000` | Vendor; self-contained |
| `trim_frame` | `diive\core\dfun\frames.py:206` | Vendor; small |
| `search_files` | `diive\core\io\filereader.py` | Replace with `pathlib.Path.rglob` |
| `load_parquet` | `diive\core\io\files.py:116` | `pd.read_parquet` unless diive's timestamp handling is needed — check |
| `save_parquet` | `diive\core\io\files.py` | `df.to_parquet` |
| `FileDetector`, `add_data_stats` | `diive\core\io\filedetector.py` (294 lines) | See §9 |
| `zScoreRolling` | `diive\preprocessing\outlier_detection\zscore.py:249` | **Not a leaf** — subclasses `FlagBase`, pulls 5 further modules (~936 lines). Reimplement instead — decision 8 |

Roughly 400–600 lines vendored, all leaf utilities — **provided decision 8 goes
the reimplement way.**

---

## 9. The `rawio` unification — the risk item

Everything else in this plan is copy plus import fixes. This is design work, and
it is where the estimate's variance lives. Budget it separately.

Three raw-file readers must become one:

1. **dyco `files.read_raw_data_csv`** plus `length_data_cols`,
   `length_header_cols`, `data_vs_header`, `generate_missing_cols`. Handles data
   rows outnumbering header columns by synthesizing `unknown_N` names. Also
   reads parquet.
2. **diive `ReadFileType` / `filedetector`** — used by `FileSplitter` at
   `:135`, `:147`, `:444`.
3. **diive hires `_read_raw_file` / `_write_raw_file`**
   (`diive\flux\hires\detect_and_remove_tlag.py:246` and `:318`) — the newest,
   and the only one preserving CRLF/LF line terminators.

**Start from (3), fold in (1)'s header reconciliation, drop (2).**

`filedetector.py` is 294 lines and only `FileDetector` + `add_data_stats` are
used across both callers. Scope those two before choosing vendor vs rewrite; if
`FileDetector` drags in the rest of the module, rewrite file discovery against
`Path.rglob` instead.

Note that dyco's `files.add_data_stats` (`:146`) and diive's
`filedetector.add_data_stats` are near-identical — same name, same signature
shape. They have common ancestry. Only one survives.

---

## 10. Sequencing

`F:\dev\diive` is untouched until step 10.

1. **Make v2 run again, and capture the baseline.** Bump `pyproject.toml` to
   hatchling + uv, Python `>=3.12,<3.14`, pandas `>=3.0.0`. Fix **both** dead
   imports (`loop.py:30`, `analyze.py:27`) against current diive. Run v2 on real
   data and capture the reference outputs — the **daily lookup table above all**,
   since decision 8 and step 8 both diff against it. Archive that output outside
   the working tree. *Everything downstream depends on this baseline. Do not
   skip it.*
2. **Vendor the §8 shims** and prove they import with diive absent. This is what
   severs the dependency — do it before any hires code lands.
3. **Move the 7 hires files**, import fixes only. Order: `windrotation`, `lag`,
   `fluxdetectionlimit`, `lag_pwb`, `apply_tlag`, `detect_and_remove_tlag`,
   `detect_and_remove_tlag_tui`.
4. **Lift `rawio.py`** out of `pipeline.py` (the `_read_raw_file` /
   `_write_raw_file` pair).
5. **Move `F:\dev\diive\tests\test_echires.py`** → `tests/`; get it green.
   **Gate — nothing proceeds until it passes.**
6. **Build the unified `rawio`** (§9). Re-run step 5's tests.
7. **Port `FileSplitter`** onto `rawio`. Write its first tests — it has none
   today (242 uncovered statements in diive's coverage audit).
8. **Rewire dyco's own modules:**
   - `AdjustLagsearchWindow` → `lag/window.py` (clean move)
   - Port `loop.py:169` `plot_segment_lagtimes_ts` — the one live thing in the
     module about to be deleted
   - `analyze.py` → `lag/lookup.py` **with the §5 fixes**: replace
     `_remove_outliers` per decision 8 and diff the LUT against step 1's
     reference; assign the `:246` `fillna`; raise instead of `sys.exit()` at
     `:80`; parameterize `ABS_LIMIT`; move `run()` out of `__init__`
   - `RemoveLags` → `lag/apply.py`, folded in with `TlagApplier`
   - Delete `loop.py`, `files.py`, `plot.py`, `setup.py`, `dyco.py`
   - Rewrite `cli.py`
9. **Define `dyco/__init__.py`**; merge examples; docs (§12).
10. **Only now: strip diive** (§11). Separate repo, separate commits.
11. **Release v3**; update `CITATION.cff`.

---

## 11. diive-side removals — SEPARATE REPO, STEP 10 ONLY

Do none of this until dyco v3 is green. All paths relative to `F:\dev\diive\`.

**Code**
- Delete `diive\flux\hires\` and `diive\core\io\filesplitter.py`
- `diive\flux\__init__.py`: drop line 11 (`from diive.flux import hires`),
  lines 30–39 (the hires re-exports), and `__all__` entries `'hires'` (`:54`)
  and `:71–80` (`FluxDetectionLimit`, `MaxCovariance`, `PreWhiteningBootstrap`,
  `PwbBatchDetection`, `PwboptLagPlot`, `TlagApplier`, `PerFilePipeline`,
  `process_one_file`, `WindDoubleRotation`, `reynolds_decomposition`)

**pyproject.toml**
- Drop `textual>=8.2.7` and `polars>=1.26.0,<2.0.0` from `[project] dependencies`
- Drop 4 of 5 `[project.scripts]`; keep only `diive-gui`

**Tests / examples**
- Delete `tests\test_echires.py` and `examples\flux\hires\`
- `examples\run_all_examples.py` lines 134–142
- `examples\CATALOG.md` lines 213–221
- `examples\README.md` file count; `examples\flux\` category README

**Docs**
- `docs\conf.py:116` remove `'echires'`; `docs\contributing.rst:374`
- `docs\auto_examples\echires\` and `docs\sg_execution_times.rst` — regenerate

**Project docs**
- `CLAUDE.md`: drop the "High-Resolution EC Analysis (hires)" section
  (~`:311–340`), the `dv.flux` API-table entries for the hires classes, and the
  `flux/` line in the project-structure block (`:55`); add a pointer to dyco
- `CONTRIBUTING.md:303`
- `COVERAGE_GAPS.md`: delete Tier 6, the `core/io/filesplitter.py` row (`:170`),
  the `flux/hires/lag_pwb.py` reference (`:69`)
- `CHANGELOG.md`: breaking-change entry naming every removed public symbol

**Verified as needing no change**
- `packaging\diive_gui.spec` — no hires references
- `diive\__init__.py` `_LAZY_SUBMODULES` — `hires` was never listed

---

## 12. Docs work in dyco

`README.md` is 24.7 KB across 20 sections, and "Workflow in `v2`" through
"Step 7" describe the deleted orchestrator step by step with figures from
`images/`. That is a rewrite, not an edit.

| Item | Work |
|---|---|
| README "Workflow in v2" → "Steps 1–7" | Rewrite for the v3 pipeline; regenerate figures |
| README Installation, Usage/Code, Usage/CLI | Rewrite — new install, new API, new console scripts |
| README Motivation, Scientific background, References | Largely reusable; extend for PWB |
| `CHANGELOG.md` | v3 entry: breaking changes, merged scope, v2 CLI migration |
| `CITATION.cff` | `version:`, Zenodo DOI |
| `paper/` | Leave as-is (decision 7) |
| `status.svg` badge | Check whether it still points anywhere valid |

---

## 13. Verification

**dyco v3**
- [ ] `test_echires.py` passes (modulo import paths)
- [ ] All 4 console scripts run `--help`; TUI opens with `--demo`
- [ ] `FileSplitter` round-trip matches its pre-migration output byte-for-byte
- [ ] A v2 run reproduces the step-1 reference daily lookup table
- [ ] Clean venv install with **no diive present**
- [ ] `uv sync` works after the Poetry → hatchling migration

**diive** (after step 10)
- [ ] `uv run pytest tests/ -v` green
- [ ] `uv sync` no longer resolves `textual` or `polars`
- [ ] `import diive` timing unchanged or better (baseline 0.96 s)
- [ ] GUI launches, all tabs build
- [ ] Sphinx builds without the echires gallery
- [ ] `grep -rn "hires\|echires" --include="*.py"` returns only
      `aggregated_as_hires` and `series_hires_*` in
      `diive\preprocessing\qaqc\meteoscreening.py` (both unrelated)

---

## 14. Effort

| Phase | Estimate |
|---|---|
| pyproject migration, repair v2, vendor shims (1–2) | 1 day |
| Move hires + tests green (3–5) | 1–2 days |
| Unified `rawio` (6) | 1–2 days |
| `FileSplitter` port + tests (7) | 1 day |
| dyco rewire + CLI rewrite + §5 fixes (8) | 2–3 days |
| `__init__`, examples, docs (9, §12) | 1.5–2 days |
| diive strip + docs (10) | 1 day |
| Release (11) | 0.5 day |
| **Total** | **2–2.5 weeks focused** |

---

## 15. Appendix: provenance of these facts

Established 2026-08-01 by reading both repos. **Line numbers drift** — re-grep
rather than trusting a reference that does not match.

Verified directly:
- dyco's six diive imports and which are broken
- `SummaryPlots` is unreferenced; the three plot helpers have 9 live call sites
- `zScoreRolling`'s current API (`calc`, `.fig`, `.get_flag`) still matches
  dyco's usage; its vendoring cost is ~936 lines across 5 modules
- Nothing in diive imports `flux/hires` except `diive\flux\__init__.py`
- Nothing anywhere imports `filesplitter.py`
- `packaging\diive_gui.spec` has no hires references
- The Python and pandas pin conflicts

Read in full: `dyco/loop.py`, `dyco/files.py`, `dyco/analyze.py`,
`dyco/plot.py`, `diive/flux/__init__.py`, `diive/flux/hires/__init__.py`,
`diive/core/io/filesplitter.py` (imports and call sites).

**Not read in full — verify before relying on:** `dyco/dyco.py`, `dyco/cli.py`,
`dyco/setup.py`, `dyco/lag.py`, `dyco/correction.py`, and the bodies of the
seven hires modules. Their LOC counts and public names are confirmed; their
internals are not.
