# Handover — dyco v3

Written 2026-08-01. **Read this first, then `CLAUDE.md`.** `MIGRATION_V3.md` is
historical reference only — parts of it are explicitly superseded.

You are in `F:\dev\dyco`, branch `indev`, last commit `a69a9b2`.

---

## 1. The one thing to know

**Both repos are committed and clean.**

The v3 work landed as eight commits on `indev` on top of `a69a9b2`, and the
matching removals landed as six commits in `F:\dev\diive` on top of `6e09f693`.
dyco no longer depends on diive, so the two histories stand alone.

`pyproject.toml` reads `3.0.0`; the CHANGELOG heading is still
`## v3.0.0 | unreleased`. Nothing is published.

**Do not commit.** The user stages and commits exclusively. Do not run
`git commit`, `git add` or `git push`. Do not bump version numbers. (The v3
commits were made on an explicit one-off instruction; the standing rule is
unchanged.)

---

## 2. What just happened

All raw high-frequency eddy covariance code moved from `diive` into `dyco`, and
`dyco` stopped depending on `diive`.

Before this, `dyco` could not run at all: it imported `MaxCovariance` and
`zScoreRolling` from `diive` at paths that no longer existed, and its Python
(`>=3.11,<3.12`) and pandas (`<3.0.0`) floors had diverged from diive's to the
point where the two could not be installed together. Four independent breakages,
all from one cross-repo dependency. Removing that dependency is the point of the
whole exercise, not a side effect.

### Arrived in dyco

| Module | LOC | From diive |
|---|---|---|
| `pipeline.py` | 3,263 | `flux/hires/detect_and_remove_tlag.py` |
| `pwb.py` | 2,937 | `flux/hires/lag_pwb.py` |
| `tui.py` | 2,012 | `flux/hires/detect_and_remove_tlag_tui.py` |
| `apply_tlag.py` | 787 | `flux/hires/apply_tlag.py` |
| `split.py` | 593 | `core/io/filesplitter.py` |
| `detectionlimit.py` | 570 | `flux/hires/fluxdetectionlimit.py` |
| `maxcov.py` | 502 | `flux/hires/lag.py` |
| `rotation.py` | 179 | `flux/hires/windrotation.py` |
| `_vendor/` (7 modules) | 498 | leaf utilities |
| `tests/` (104 tests) | — | `tests/test_echires.py` + extracted blocks, gzip + CLI suites |
| `examples/` (10 scripts + data) | — | `examples/flux/hires/` + a real-data example |

### Removed from diive

`flux/hires/` (whole package), `core/io/filesplitter.py`,
`core/io/filedetector.py`. Dropped `textual` and `polars`; 4 of 5 console
scripts. `dv.flux` lost 11 public symbols.

**Kept in diive deliberately:** `create_timestamp`, `calc_true_resolution`,
`trim_frame`. These were orphaned by the move and briefly deleted, then restored
at the user's request. dyco's `_vendor/times.py` and `_vendor/frames.py` are
therefore **copies, not relocations** — two live copies now exist and can drift.

---

## 3. Rules that must not be broken

1. **Never import from `diive`.** If something is needed, copy it into
   `_vendor/` with a provenance note, or reimplement it. The cross-repo coupling
   is what broke dyco four ways.
2. **PWB is the primary method.** New features, docs order and defaults go to
   the PWB path (`pipeline.py`, `pwb.py`).
3. **The v2 covariance-maximization method stays available and working.** Do not
   delete `dyco.py`, `loop.py`, `analyze.py`, `correction.py`, `setup.py`,
   `lag.py` or `plot.py`. `MIGRATION_V3.md` §4 proposes deleting five of them —
   that is void.
4. **Mention dead code, don't delete it.** Two known-dead blocks are catalogued
   in `CLAUDE.md`; leave them unless the user says otherwise.

---

## 4. Verification

```bash
uv sync
uv run pytest tests/ -q          # expect: 104 passed, 34 subtests
```

To prove the diive dependency is really gone, run in an isolated environment
with diive absent:

```bash
uv run --no-project --isolated --python 3.12 \
  --with pandas --with numpy --with matplotlib --with scipy \
  --with polars --with pyarrow --with textual --with pyyaml --with rich --with pytest \
  python -m pytest tests/ -q
```

**Current status: 104 passed, 34 subtests, in that isolated environment.**

diive's suite is at **653 passed, 695 subtests**. Its `1 failed, 9 errors` are
one gitignored parquet fixture absent from the working copy — pre-existing, not
caused by the migration.

---

## 5. What is open, in priority order

### 5.1 The v2 (CM) path has no tests and no reference run

**PWB has now been run end to end on real data** (`examples/data/`, a 1-hour
CH-LAE excerpt) and verified: correct chunking, genuinely gzipped output, headers
byte-identical to source, and scalar columns shifted by exactly the applied lag
(169 records = 8.45 s). That run found three gzip bugs no unit test reached.

**Decision taken: the v2 path stays working but is not extended.** No new tests,
no reference run. This is a deliberate accepted risk, not an oversight — see 5.2.

### 5.2 The z-score deviation is unvalidated

`_vendor/outliers.py` is a port of diive's `zScoreRolling` with one deliberate
difference: it does **not** regularize an irregular index. diive's `FlagBase`
called `series.asfreq(detected_freq)` when the index had no frequency; the lags
this filter screens are indexed by segment start time and are inherently
irregular, so regularizing inserted rows the caller never had and returned a
flag that did not align with the input.

Verified on synthetic data: 330 irregular rows in, 330 aligned rows out. Under
diive's version that call would have raised `IndexingError`, so this looks like a
genuine bug fix. But it can shift the daily lookup table the v2 path produces,
and without a reference run nobody knows by how much.

**This stays unvalidated by decision** — the v2 path is kept working but not
extended. Read the module docstring before touching it.

### 5.3 Two raw-file readers still exist

`dyco/files.py` (v2 path + `split.py`) and `pipeline.py`'s `_read_raw_file` /
`_write_raw_file` (PWB path). They share no code, which is why the same class of
gzip gap had to be found and fixed in each independently. Capabilities differ:

| | `files.py` | `pipeline.py` |
|---|---|---|
| formats | CSV **and** parquet | CSV only |
| header rows | **one** only | arbitrary metadata rows, preserved |
| column-count mismatch | reconciles (`unknown_N`) | raises |
| CRLF/LF | not preserved | preserved |
| writer | none | yes |

Unification would take `pipeline.py`'s as the base and fold in parquet plus the
column reconciliation. **Priority has dropped**: the v2 path is the main consumer
of `files.py` and is frozen by decision, so this is a rewrite under frozen code
for no gain to the primary path. The concrete broken behaviour (compressed input)
is already fixed in both. Revisit only if the v2 path is unfrozen or `split.py`
grows more use.

### 5.4 Known defects in `analyze.py` — all fixed

- ~~`:246` — `fillna` result never assigned~~ **FIXED.** Confirmed with the user
  that no published result relied on that branch.
- ~~`:80` — `sys.exit()` inside a library~~ **FIXED.** Raises `ValueError`. Note
  the old call passed no status, so it exited `0`: a failed run looked successful
  to the calling shell.
- ~~`:221` — `ABS_LIMIT = 50` hardcoded~~ **FIXED.** Now an `abs_limit=50`
  parameter. It lives in `make_lut_instantaneous`, which nothing calls — its only
  reference is the dead `SummaryPlots` — so this parameterizes dead code.
- ~~`:62` — `__init__` calls `self.run()`~~ **FIXED.** Breaking for anyone
  constructing `AnalyzeLags` directly: call `run()` before `get_lut()`.
  `Dyco.analyze_lags` was updated, so `dyco cm` is unaffected.

Verified with a synthetic-lag script rather than a test, since the v2 path stays
untested by decision. The LUT came out at 9 rows with a median of -201 records
from lags drawn at -200 ± 5.

### 5.4b The v2 plotting is broken on matplotlib 3.9+

Found while verifying the above. `plt.cm.get_cmap` (`loop.py:211`) and
`Axes.plot_date` (`loop.py:221`) were both removed in matplotlib 3.9; the pinned
version is 3.11.1. Both sit on the live path, so **`Dyco.analyze_lags()` cannot
currently complete** — it raises `AttributeError` at the plotting step after the
look-up table has been built.

Five more `plot_date` calls sit in dead code (`analyze.py:114, 119, 125` and
`plot.py:99, 105, 111`).

Not fixed: it is a change to frozen code and was outside the task. The fix is
`matplotlib.colormaps['rainbow'].resampled(n)` and `ax.plot(x, y, fmt, ...)`.
This is the clearest argument yet that "kept working" and "never run" are not the
same thing.

### 5.5 Release chores

- `pyproject.toml` reads `version = "3.0.0"`. **Do not change it again; the user
  owns the version.**
- `CHANGELOG.md` heading is `## v3.0.0 | unreleased`. The date goes in at
  release, deliberately not before.
- `CITATION.cff` needs its `version:` and a Zenodo DOI (the `doi:` field is
  commented out).
- `status.svg` badge — check it still points somewhere valid.

### 5.6 Decisions taken

- **`paper/`: leave it.** It is the published v1/v2 record. Do not edit it. The
  PWB method has its own publication (Vitale et al. 2024).
- **v2 CLI: broken and rewritten.** `cli.py` is now the unified `dyco` dispatcher;
  the v2 workflow lives at `dyco cm` with long flag names. Migration table is in
  the CHANGELOG. **Done.**
- **v2 method: keep working, do not extend.** No new tests, no reference run.
- **Example data: trimmed and committed.** `examples/data/` holds a 1-hour
  CH-LAE excerpt (1.4 MB, two chunks) rather than the 6-hour original (8 MB).
- **`analyze.py:246`: fixed.** Confirmed no published result relied on it.

---

## 6. Traps that will waste your time

- **GUI tests skip silently.** In diive, `uv sync` without `--extra gui` leaves
  PySide6 out and `test_gui.py` vanishes from the run rather than failing. A
  before/after comparison on that basis reads as "passes on clean, fails on
  modified" and sends you hunting a bug that does not exist.
- **`diive/configs/exampledata/*.parquet` is gitignored and untracked.** A fresh
  checkout of diive lacks it, so 10 tests error on a missing file. Copy the
  parquet files across before comparing anything.
- **Do not truncate pytest output.** Piping a 25-minute suite through `tail -5`
  hides which test failed. Use `-rf`, or read `.pytest_cache/v/cache/lastfailed`,
  which has the answer in a second.
- **`dyco/setup.py` is not a packaging file.** It holds `set_dirs`,
  `create_logger`, `CreateOutputDirs`, `generate_run_id`. It was once named
  `setup_dyco`, and `plot.py:46` still refers to that old name — one of three
  reasons `SummaryPlots` cannot run.
- **`outdirs` is keyed by string** (`'0_log'` … `'8_time_lags_corrected_files'`,
  built in `setup.CreateOutputDirs`). Renaming a key breaks callers silently;
  there is no symbol to grep for.
- **The TUI settings path changed** to `~/.dyco/detect_remove_tui.yaml`. An
  existing `~/.diive/` config will not be found.

---

## 7. Layout

```
F:\dev\dyco\
├── HANDOVER.md          this file
├── CLAUDE.md            conventions, architecture, dead code, defects
├── MIGRATION_V3.md      historical — parts superseded, see its header
├── README.md            rewritten: PWB primary, v2 retained
├── CHANGELOG.md         v3.0.0 | unreleased
├── paper/               published JOSS paper — do not edit
├── dyco/
│   ├── pipeline.py pwb.py tui.py apply_tlag.py     PWB path (primary)
│   ├── dyco.py loop.py lag.py analyze.py           v2 path (retained)
│   │   correction.py setup.py cli.py plot.py files.py
│   ├── maxcov.py rotation.py split.py detectionlimit.py   shared / tools
│   └── _vendor/         copies of former diive helpers
├── tests/               104 tests + data/
└── examples/            10 scripts + data/ (1-hour CH-LAE raw file)
```

Console scripts: `dyco-detect-remove` (main), `dyco-detect-remove-tui`,
`dyco-pwb-batch`, `dyco-apply-batch`.

---

## 8. If you only do one thing

`rawio` unification (§5.3) is the last substantial item. Everything else in §5 is
decided, deferred by decision, or a release chore.

If you are instead looking for what to be careful about: §6, and the fact that
the three gzip bugs were all found by running the thing, not by the test suite.
