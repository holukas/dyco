"""
PWB_REFERENCE_GENERATE: BUILD THE R-COMPARISON FIXTURES
=======================================================

Writes the two synthetic raw-data fixtures that `tests/test_pwb_reference.py`
checks dyco's PWB implementation against. Both are 5 min of 20 Hz data with a
known 169-record (8.45 s) tube delay built into the scalar:

    pwb_reference_stationary.csv.gz    all three series pass the Breitung
                                       variance-ratio test -> no differencing
    pwb_reference_differencing.csv.gz  the scalar carries a random walk, so the
                                       test fails and all series are
                                       first-differenced before AR fitting

The second case exists because the differencing branch is otherwise never
executed: real turbulent EC data virtually always passes the test, and it is
where a defect in the raw cross-covariance went unnoticed (dyco read `cov_pwb`
off the differenced series; R reads it off the original).

Regenerating the fixtures invalidates the frozen R values in the test. Rerun
`pwb_reference_rflux.R` afterwards and update them.

Part of the dyco package: https://github.com/holukas/dyco
"""

from pathlib import Path

import numpy as np
import pandas as pd

HZ = 20
N = HZ * 60 * 5           # 5 min
LAG_RECORDS = 169         # 8.45 s tube delay
BURN = 600                # discarded AR spin-up
SEED = 20240401

# AR(1) persistence of the three synthetic series. Kept below ~0.95: at n=6000
# a series with phi=0.97 is close enough to a unit root that the Breitung test
# rejects stationarity, which would leave both fixtures on the differencing
# branch and lose the contrast between them.
PHI_W = 0.90
PHI_NOISE = 0.80
PHI_TSONIC = 0.70


def _ar1(rng, phi: float, n: int, sd: float = 1.0) -> np.ndarray:
    """One AR(1) realisation, x_t = phi*x_{t-1} + e_t."""
    e = rng.normal(0, sd, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def build(nonstationary: bool) -> pd.DataFrame:
    """Synthetic (scalar, tsonic, w) with the scalar lagged behind w.

    The scalar is a scaled, delayed copy of w plus its own coloured noise and a
    concentration offset; T_SONIC shares the same turbulence as w, which is why
    the scalar x T_SONIC combinations can also see the lag.
    """
    rng = np.random.default_rng(SEED)
    n = N + LAG_RECORDS + BURN

    w = _ar1(rng, PHI_W, n)
    noise = _ar1(rng, PHI_NOISE, n)

    scalar = np.full(n, np.nan)
    scalar[LAG_RECORDS:] = 0.30 * w[:-LAG_RECORDS]
    scalar = scalar + noise + 330.0
    if nonstationary:
        # A random walk has a unit root, so the Breitung test rejects
        # stationarity and every series is first-differenced.
        scalar = scalar + np.cumsum(rng.normal(0, 0.05, n))

    tsonic = 0.55 * w + _ar1(rng, PHI_TSONIC, n) + 18.0

    sl = slice(BURN, BURN + N)
    return pd.DataFrame({'scalar': scalar[sl], 'tsonic': tsonic[sl], 'w': w[sl]})


def main() -> None:
    here = Path(__file__).parent
    for nonstat, name in ((False, 'pwb_reference_stationary.csv.gz'),
                          (True, 'pwb_reference_differencing.csv.gz')):
        df = build(nonstationary=nonstat)
        out = here / name
        # 6 decimals keeps the file small while leaving ~9 significant digits,
        # and both R and Python then read byte-identical inputs.
        df.to_csv(out, index=False, float_format='%.6f')
        print(f'wrote {out.name}  rows={len(df)}  lag={LAG_RECORDS} rec '
              f'= {LAG_RECORDS / HZ:.2f} s')


if __name__ == '__main__':
    main()
