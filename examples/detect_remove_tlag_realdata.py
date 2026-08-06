"""
==========================================================================
Per-Chunk Time Lag Detection and Removal on a real raw file (CH-LAE, 20 Hz)
==========================================================================

The companion example ``detect_remove_tlag.py`` builds synthetic files with a
known tube delay, so its answer can be checked against the truth. This one runs
the same pipeline on a **real** raw file bundled with the repository, so you can
see what the output actually looks like on measured data - where there is no
known-correct answer to compare against.

Input: ``examples/data/CH-LAE_202507251300.csv.gz``

    Site        CH-LAE (Laegeren), Switzerland
    Start       2025-07-25 13:00
    Duration    1 hour (72 000 records at 20 Hz)
    Sonic       HS50-B      -> U, V, W, T_SONIC
    Analyzer    IRGA72-A    -> H2O_DRY, CO2_DRY
    Header      3 rows (names, units, instrument)

One hour at 30-minute chunks gives **two output files**. The bundled file is a
1-hour excerpt of a 6-hour raw file, trimmed to keep the repository small; the
full file would yield twelve chunks. Chunking matters because rotation angles
and tube delay both drift over hours, so a single estimate per multi-hour file
is the wrong granularity.

Note the column names carry instrument tags in brackets
(``W_[HS50-B]``, ``CO2_DRY_[IRGA72-A]``). That is normal for raw logger output
and needs no special handling - the pipeline takes column names as given.

Run time
--------
This is real work, not a toy: PWB runs a block-bootstrap per gas per chunk, so
expect something like ten seconds for the two bundled chunks. ``n_bootstrap`` is
lowered from the paper's default here to keep the example tolerable; raise it
for production use.

Output goes to a temporary directory that is removed at the end. Set the
``DYCO_OUT`` environment variable to a real path to keep the results instead.

[WINDOWS] Run this as a **file**, not by piping it into the interpreter. It uses
``n_workers=4``, and ``ProcessPoolExecutor`` on Windows spawns children that
re-import ``__main__`` — with the script on stdin there is nothing to re-import
and the run hangs with the workers idle. That is why the ``if __name__ ==
'__main__'`` guard below is load-bearing rather than decorative.

# See Also
dyco.pipeline.PerFilePipeline : The class this example drives.
examples/detect_remove_tlag.py : Same pipeline, synthetic data, known answer.
"""

# %%
# Imports
# -------
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from dyco.pipeline import PerFilePipeline

# %%
# Input file and settings
# -----------------------
# The bundled raw file. ``PerFilePipeline`` scans a *directory*, so the input
# directory is the folder holding it.
DATA_FILE = Path(__file__).parent / 'data' / 'CH-LAE_202507251300.csv.gz'

HZ = 20  # Sampling rate of the raw data
CHUNK_SECONDS = 1800  # 30-minute averaging periods
N_BOOTSTRAP = 100  # Lowered for runtime; the paper uses more

# Column names as they appear in the file's first header row.
COL_U = 'U_[HS50-B]'
COL_V = 'V_[HS50-B]'
COL_W = 'W_[HS50-B]'
COL_TSONIC = 'T_SONIC_[HS50-B]'

# Gas label -> column name. The label is used as a prefix in the results.
SCALARS = {
    'CO2': 'CO2_DRY_[IRGA72-A]',
    'H2O': 'H2O_DRY_[IRGA72-A]',
}


# %%
# Run the pipeline
# ----------------
def main():
    """Run detect + remove over the bundled raw file."""
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Raw file not found: {DATA_FILE}\n"
            f"It ships with the repository under examples/data/.")

    output_root = Path(os.environ.get('DYCO_OUT') or tempfile.mkdtemp(prefix='dyco_realdata_'))
    keep = bool(os.environ.get('DYCO_OUT'))
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        pipeline = PerFilePipeline(
            input_dir=DATA_FILE.parent,
            output_dir=output_root,
            col_u=COL_U, col_v=COL_V, col_w=COL_W, col_tsonic=COL_TSONIC,
            scalars=SCALARS,
            hz=HZ,
            chunk_seconds=CHUNK_SECONDS,
            min_chunk_seconds=300,

            # Closed-path tube delay is physically positive, so the search
            # window keeps only positive lags. H2O has a longer effective delay
            # than the dry gases because of sorption on the tube walls, so it
            # gets a wider window of its own.
            lag_max_s=10.0,
            lws=0.0,
            uws=10.0,
            per_gas_lag={'H2O': {'lag_max_s': 30.0, 'lws': 0.0, 'uws': 30.0}},

            n_bootstrap=N_BOOTSTRAP,

            # File format: one file, gzip-compressed CSV, 3 header rows. Row 0
            # carries the column names; rows 1-2 (units, instrument) are the
            # 'extra' header rows that are preserved in the output.
            file_pattern='*.csv.gz',
            skiprows=0,
            extra_rows=2,
            sep=',',

            # Name each output chunk by its own wall-clock start. The regex's
            # capture group is parsed with the format below.
            start_time_regex=r'(\d{12})',
            start_time_format='%Y%m%d%H%M',
            chunk_name_template='CH-LAE_{starttime}{suffix}',

            n_workers=4,
            random_state=42,
        )
        summary = pipeline.run()

        # %%
        # Inspect the results
        # -------------------
        print(f"\nChunks written: {len(list((output_root / '2_lag_removed').glob('*')))}")
        print(f"Output root:    {output_root}")

        results_csv = output_root / '1_lag_detection' / 'detect_and_remove_tlag_summary.csv'
        if not results_csv.is_file():
            raise FileNotFoundError(
                f"Expected the detection summary at {results_csv}, but it is not "
                f"there. Present instead: "
                f"{[p.name for p in (output_root / '1_lag_detection').glob('*')]}")
        res = pd.read_csv(results_csv)
        print(f"\nDetection table: {len(res)} rows")

        # For each gas: the raw per-chunk detection, and the PWBOPT lag that was
        # actually applied. Where they differ, PWBOPT judged the chunk's own
        # detection untrustworthy and substituted a neighbouring one.
        for label in SCALARS:
            raw = f'{label.lower()}_tlag_s'
            applied = f'{label.lower()}_tlag_final_pf_s'
            missing = [c for c in (raw, applied) if c not in res.columns]
            if missing:
                raise KeyError(
                    f"{missing} not in the summary. Columns present: "
                    f"{sorted(res.columns)[:12]} ...")
            print(f"\n{label}:")
            print(f"  raw detections    median {res[raw].median():.3f} s, "
                  f"{res[raw].notna().sum()}/{len(res)} succeeded")
            print(f"  lag applied       median {res[applied].median():.3f} s")

        print(f"\nSummary:\n{summary}")

    finally:
        if not keep:
            shutil.rmtree(output_root, ignore_errors=True)
        else:
            print(f"\nResults kept in: {output_root}")


if __name__ == '__main__':
    main()
