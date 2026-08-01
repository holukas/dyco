"""
EXAMPLEDATA: TEST FIXTURES
===========================

Loaders for the raw data files under `tests/data/`.

The 10 Hz file came across from diive together with `FluxDetectionLimit`, whose
test needs it. It is the same file diive exposes as
`load_exampledata_GENERIC_TXT_EDDY_COVARIANCE_10Hz`.

Part of the dyco package: https://github.com/holukas/dyco
"""

from pathlib import Path

import pandas as pd

DIR_PATH = Path(__file__).parent / 'data'


def load_exampledata_10hz_ec() -> pd.DataFrame:
    """Load the 10 Hz eddy covariance example file (generic TXT, LGR data)."""
    filepath = DIR_PATH / 'exampledata_GENERIC-TXT-EDDY-COVARIANCE-10Hz-2023-06-24-03-30_LGRData.txt'
    return pd.read_csv(filepath, sep=',')
