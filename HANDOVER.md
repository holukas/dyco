# Handover — dyco v3

Written 2026-08-01. **Read this first, then `CLAUDE.md`.** `MIGRATION_V3.md` is
historical reference only — parts of it are explicitly superseded.

You are in `F:\dev\dyco`, branch `indev`, last commit `53fa552`.

---

## 1. The one thing to know

**The v2 covariance-maximization method is gone, and that removal is
uncommitted.**

On 2026-08-01 the user decided PWB is the only method. Deleted from the working
tree: `dyco.py`, `loop.py`, `lag.py`, `analyze.py`, `correction.py`, `plot.py`,
`setup.py`, `_vendor/outliers.py`, the `dyco cm` subcommand and its tests, plus
`files.read_segment_lagtimes_file` and `files.add_data_stats`, which served
nothing else. Docs updated to say the method existed up to v2 and is no longer
available. 98 tests pass. Nothing else in the working tree is modified.

Earlier v3 work is already committed: eight commits on `indev`, and the matching
removals in `F:\dev\diive` on top of `6e09f693`. dyco no longer depends on diive,
so the two histories stand alone.

`pyproject.toml` reads `3.0.0`; the CHANGELOG heading is still
`## v3.0.0 | unreleased`. Nothing is published, so the removed method never
shipped under a v3 number — v2.0.3 on PyPI is the last release that has it.

**Do not commit.** The user stages and commits exclusively. Do not run
`git commit`, `git add` or `git push`. Do not bump version numbers. (The v3
commits were made on an explicit one-off instruction; the standing rule is
unchanged.)

---

## 2. What happened just before that

All raw high-frequency eddy covariance code moved from `diive` into `dyco`, and
`dyco` stopped depending on `diive`. This is the committed part of v3; the v2
removal in §1 sits on top of it.

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
2. **PWB is the only method.** All work goes to the PWB path (`pipeline.py`,
   `pwb.py`, `apply_tlag.py`, `tui.py`).
3. **Do not reintroduce the covariance-maximization method**, and do not restore
   a module from git history to satisfy a missing import. If the PWB path turns
   out to need something that lived only there, port that specific piece and say
   so. `maxcov.py` was never part of the removal — `FluxDetectionLimit` uses it.
4. **Mention dead code, don't delete it.** Nothing is catalogued as dead right
   now; the rule stands for whatever turns up next.

---

## 4. Verification

```bash
uv sync
uv run pytest tests/ -q          # expect: 98 passed, 31 subtests
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

### 5.1 Two raw-file readers still exist

`dyco/files.py` (used by `split.py`) and `pipeline.py`'s `_read_raw_file` /
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
column reconciliation. **Low priority.** `files.py` now has exactly one consumer,
`split.py`, and the concrete broken behaviour (compressed input) is already fixed
in both. Worth doing if `split.py` grows more use, or as tidying — not for
correctness.

### 5.2 Nothing else from the v2 removal is outstanding

The removal took its open items with it: the untested v2 path, the unvalidated
z-score deviation in `_vendor/outliers.py`, and the matplotlib 3.9 breakage in
`loop.py` that meant `Dyco.analyze_lags()` could not complete. None of that code
exists any more. It is in git history if a question about past results comes up.

### 5.3 Release chores

- `pyproject.toml` reads `version = "3.0.0"`. **Do not change it again; the user
  owns the version.**
- `CHANGELOG.md` heading is `## v3.0.0 | unreleased`. The date goes in at
  release, deliberately not before.
- `CITATION.cff` needs its `version:` and a Zenodo DOI (the `doi:` field is
  commented out).
- `status.svg` badge — check it still points somewhere valid.

### 5.4 Decisions taken

- **v2 method: removed.** The user's call, 2026-08-01: "pwb is the boss".
  Superseded the earlier "keep it working, do not extend" decision, which had
  left an entire path in the package that nothing ran or tested.
- **`paper/`: leave it.** It is the published v1/v2 record, and it describes the
  method that was just removed — which is a reason to leave it alone, not to
  amend it. The PWB method has its own publication (Vitale et al. 2024).
- **v2 CLI: gone with the method.** `cli.py` is the unified `dyco` dispatcher.
  `dyco cm` and the old short flags are still recognized, only to answer with a
  pointer to `dyco detect-remove`.
- **Example data: trimmed and committed.** `examples/data/` holds a 1-hour
  CH-LAE excerpt (1.4 MB, two chunks) rather than the 6-hour original (8 MB).

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
- **`dyco/setup.py` was never a packaging file** — it held `set_dirs`,
  `create_logger`, `CreateOutputDirs`, `generate_run_id`, and it went with the v2
  path. Worth knowing only if you meet it in git history and assume it was build
  config.
- **The TUI settings path changed** to `~/.dyco/detect_remove_tui.yaml`. An
  existing `~/.diive/` config will not be found.

---

## 7. Layout

```
F:\dev\dyco\
├── HANDOVER.md          this file
├── CLAUDE.md            conventions, architecture, standing rules
├── MIGRATION_V3.md      historical — parts superseded, see its header
├── README.md            PWB only; the v2 method is documented as removed
├── CHANGELOG.md         v3.0.0 | unreleased
├── paper/               published JOSS paper — do not edit
├── dyco/
│   ├── pipeline.py pwb.py tui.py apply_tlag.py cli.py     PWB path
│   ├── maxcov.py rotation.py split.py detectionlimit.py files.py   tools
│   └── _vendor/         copies of former diive helpers
├── tests/               98 tests + data/
└── examples/            10 scripts + data/ (1-hour CH-LAE raw file)
```

Console scripts: `dyco-detect-remove` (main), `dyco-detect-remove-tui`,
`dyco-pwb-batch`, `dyco-apply-batch`.

---

## 8. If you only do one thing

Get the v2 removal reviewed and staged — it is the whole uncommitted diff, and
it touches every doc in the repo. After that, §5.3's release chores are what is
left; `rawio` unification (§5.1) is optional tidying.

If you are instead looking for what to be careful about: §6, and the fact that
the three gzip bugs were all found by running the thing, not by the test suite.
