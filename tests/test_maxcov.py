"""
TEST_MAXCOV: MAXIMUM COVARIANCE LAG DETECTION
==============================================

Moved from diive `tests/test_echires.py` together with `MaxCovariance` itself.

Part of the dyco package: https://github.com/holukas/dyco
"""

import unittest

import numpy as np
import pandas as pd

from dyco.maxcov import MaxCovariance


class TestMaxCovarianceLagDetection(unittest.TestCase):
    """MaxCovariance must recover a lag that was put in deliberately."""

    @staticmethod
    def _shifted_pair(n: int, lag: int) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 1, n)
        # A positive lag means the scalar arrives *later* than the reference.
        lagged = pd.Series(reference).shift(lag).fillna(0.0).to_numpy()
        return pd.DataFrame({'w': reference, 'c': lagged})

    def _detected_shift(self, df, winsize_from=-50, winsize_to=50) -> int:
        mc = MaxCovariance(df=df, var_reference='w', var_lagged='c',
                           lgs_winsize_from=winsize_from, lgs_winsize_to=winsize_to)
        mc.run()
        cov_df, _ = mc.get()
        peak = cov_df.loc[cov_df['flag_peak_max_cov_abs'], 'shift']
        return int(peak.iloc[0])

    def test_recovers_injected_lag_including_sign(self):
        # The sign convention is documented on MaxCovariance.__init__: a positive
        # lag means var_lagged arrives later. Pin it, so a sign flip is caught.
        for injected in (0, 5, -7, 20):
            with self.subTest(injected=injected):
                df = self._shifted_pair(n=3000, lag=injected)
                self.assertEqual(self._detected_shift(df), injected)

    def test_peak_stays_inside_the_search_window(self):
        # A lag beyond the window cannot be reported; whatever is picked must
        # still lie within the requested bounds rather than run off the end.
        df = self._shifted_pair(n=3000, lag=40)
        found = self._detected_shift(df, winsize_from=-10, winsize_to=10)
        self.assertGreaterEqual(found, -10)
        self.assertLessEqual(found, 10)


if __name__ == '__main__':
    unittest.main()
