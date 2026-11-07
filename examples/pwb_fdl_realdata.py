"""
===========================================================================
Time lag and flux detection limit for CH4 and N2O on a real file (CH-CHA)
===========================================================================

Runs ``PwbFdlBatch`` on a bundled 6-hour raw file: each 30-minute period gets
a PWB time lag per gas, PWBOPT picks the lag across the sequence, and the flux
is read at that lag and compared with the period's flux detection limit
(Langford et al. 2015).

Input: ``examples/data/CH-CHA_202110081300.csv.gz``

    Site        CH-CHA (Chamau), Switzerland
    Start       2021-10-08 13:00
    Duration    6 hours (431 987 records at 20 Hz) -> twelve 30-min periods
    Sonic       R350-B     -> U, V, W, T_SONIC (K)
    Analyzer    LGR-A      -> CH4_DRY, N2O_DRY, H2O (all umol mol-1)
    Pressure    IRGA75-A   -> PRESS_BOX (hPa), the ambient pressure
    Header      3 rows (names, units, instrument)

Units for the detection limit. ``scale`` turns the gases into nmol mol-1, so
the detection limit and the flux come out in nmol m-2 s-1; H2O into mol mol-1
and pressure into Pa. Do not use ``PRESS_CELL_[LGR-A]``: about 60 hPa, that is
the analyser cell, not the air.

CH4 borrows N2O's lag. Both are measured in the same LGR cell through the same
tube, and neither sticks to the tube wall, so their delays are the same. On this
file N2O is detected reliably (S1) in every period and CH4 in none.

Output goes to a temporary directory that is removed at the end. Set the
``DYCO_OUT`` environment variable to a real path to keep the results instead.

[WINDOWS] Run this as a file. The workers are spawned processes, so the
``if __name__ == '__main__'`` guard below is required.
"""

# %%
# Imports
# -------
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from dyco.pwbfdl import PwbFdlBatch

RAW = Path(__file__).parent / 'data' / 'CH-CHA_202110081300.csv.gz'


def main():
    keep = os.environ.get('DYCO_OUT')
    out = Path(keep) if keep else Path(tempfile.mkdtemp(prefix='dyco_pwbfdl_'))

    # %%
    # Run
    # ---
    batch = PwbFdlBatch(
        files=[RAW],
        scalars={'CH4': 'CH4_DRY_[LGR-A]', 'N2O': 'N2O_DRY_[LGR-A]'},
        col_u='U_[R350-B]', col_v='V_[R350-B]', col_w='W_[R350-B]',
        col_tsonic='T_SONIC_[R350-B]',
        col_h2o='H2O_[LGR-A]',
        col_press='PRESS_BOX_[IRGA75-A]',
        hz=20,
        scale={
            'CH4_DRY_[LGR-A]': 1e3,       # umol mol-1 -> nmol mol-1
            'N2O_DRY_[LGR-A]': 1e3,       # umol mol-1 -> nmol mol-1
            'H2O_[LGR-A]': 1e-6,          # umol mol-1 -> mol mol-1
            'PRESS_BOX_[IRGA75-A]': 100,  # hPa -> Pa
        },
        skiprows=0, extra_rows=2,         # names row, then units + instrument rows
        start_time_regex=r'(\d{12})', start_time_format='%Y%m%d%H%M',
        lws=0, uws=10,                    # physical tube delays only
        n_bootstrap=99,
        lag_fallback={'CH4': 'N2O'},
        max_carry=4,                      # own lag reaches 2 h at most, then the donor's
        random_state=42,
        n_workers=4,
        output_dir=out,
    )
    res = batch.run(on_progress=lambda done, total, row: print(
        f'{done:2d}/{total} {row["period"]} {row["status"]}'))

    # %%
    # Results
    # -------
    # ``{gas}_lag_s`` is the lag the flux was read at, ``{gas}_fdl`` the
    # detection limit (3 x RMSE of the cross-covariance at +/-160 to 180 s),
    # and ``{gas}_above_fdl`` whether the flux is distinguishable from noise.
    pd.set_option('display.width', 200)
    for gas in ('n2o', 'ch4'):
        cols = ['period', f'{gas}_flag_pf', f'{gas}_lag_source', f'{gas}_lag_s',
                f'{gas}_flux', f'{gas}_fdl', f'{gas}_flux_to_fdl', f'{gas}_above_fdl']
        print(f'\n{gas.upper()} [nmol m-2 s-1]')
        print(res[cols].round(3).to_string(index=False))

    print(f'\nWritten: {out / "pwb_fdl_results.csv"}')
    if not keep:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == '__main__':
    main()
