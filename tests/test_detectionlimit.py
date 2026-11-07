"""
TEST_DETECTIONLIMIT: FLUX DETECTION LIMIT
==========================================

Moved from diive `tests/test_echires.py` together with `FluxDetectionLimit`.
The 10 Hz fixture in `tests/data/` came across with it.

Part of the dyco package: https://github.com/holukas/dyco
"""

import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')  # Headless: the class builds a covariance plot.

from dyco.detectionlimit import FluxDetectionLimit
from tests.exampledata import load_exampledata_10hz_ec


class TestFluxDetectionLimit(unittest.TestCase):

    def test_flux_detection_limit(self):
        df = load_exampledata_10hz_ec()
        df = df[['x', 'y', 'z', 'N2Od', 'Ts', 'H2O']].copy()
        df['pressure'] = 100000
        df['N2Od'] = df['N2Od'].multiply(10 ** 3)  # umol mol-1 -> nmol mol-1
        df['H2O'] = df['H2O'].div(10 ** 6)  # umol mol-1 -> mol mol-1
        df['Ts'] = df['Ts'].add(273.15)  # degC -> K
        fdl = FluxDetectionLimit(
            df=df,
            u_col='x',  # m s-1
            v_col='y',  # m s-1
            w_col='z',  # m s-1
            c_col='N2Od',  # nmol mol-1 (ppb)
            ts_col='Ts',  # K
            h2o_col='H2O',  # mol mol-1
            press_col='pressure',  # Pa
            noise_range=20,  # seconds
            default_lag=2.8,  # seconds
            lag_range=[-180, 180],  # seconds
            lag_stepsize=1,  # records
            sampling_rate=10,  # Hz
            title_covariance_plot="Covariance vs time lag for example file")
        fdl.run()
        results = fdl.get_detection_limit()
        self.assertEqual(len(fdl.hires_df.columns), 11)
        self.assertIn("e", fdl.hires_df.columns)  # e_col
        self.assertIn("pd", fdl.hires_df.columns)  # pd_col
        self.assertIsInstance(fdl.hires_df, pd.DataFrame)
        self.assertEqual(fdl.lag_from, -1800)
        self.assertEqual(fdl.lag_to, 1800)
        self.assertIn('flux_detection_limit', results)
        self.assertIn('flux_noise_rmse', results)
        # Langford et al. (2015) eq. 4 and 9, recomputed independently in plain numpy.
        # diive's estimator gave 0.6433393208791166: it re-centred each lag on its own overlap.
        self.assertAlmostEqual(results['flux_detection_limit'], 1.9852402168688271, places=10)
        self.assertAlmostEqual(results['flux_noise_rmse'], 0.6617467389562757, places=10)

        # Verify that the flux conversion factor is applied correctly
        flux_conversion_factor = fdl.cov_df['cov_flux'] / fdl.cov_df['cov']
        calculated_flux_conversion_factor = (
                1 / ((8.31446261815324 * fdl.hires_df['Ts'].mean()) / fdl.hires_df['pd'].mean()))
        self.assertEqual(flux_conversion_factor.iloc[0], calculated_flux_conversion_factor)

        # The detection limit follows the 3 * RMSE rule
        detection_limit = fdl.results['flux_detection_limit']
        noise_rmse = fdl.results['flux_noise_rmse']
        self.assertEqual(detection_limit, 3 * noise_rmse)

    @staticmethod
    def _lagged_scalar(lag_records: int, n: int = 6000) -> pd.DataFrame:
        """10 Hz data where the scalar arrives `lag_records` after the vertical wind."""
        rng = np.random.default_rng(1)
        w = np.convolve(rng.normal(0, 0.3, n + lag_records), np.ones(10) / 10, mode='same')
        c = 400 + 5.0 * np.roll(w, lag_records) + rng.normal(0, 0.5, n + lag_records)
        return pd.DataFrame({
            'u': 2 + rng.normal(0, 0.5, n), 'v': rng.normal(0, 0.3, n),
            'w': w[lag_records:], 'c': c[lag_records:],
            'Ts': 293.0 + rng.normal(0, 0.1, n), 'H2O': 0.01, 'p': 1e5})

    @staticmethod
    def _fdl(df, default_lag, lag_stepsize=1):
        return FluxDetectionLimit(
            df=df, u_col='u', v_col='v', w_col='w', c_col='c', ts_col='Ts', h2o_col='H2O',
            press_col='p', default_lag=default_lag, noise_range=5, lag_range=[-20, 20],
            lag_stepsize=lag_stepsize, sampling_rate=10)

    def test_signal_read_at_positive_default_lag(self):
        """A scalar arriving 3 s late is read at +3 s, where the covariance peaks."""
        fdl = self._fdl(self._lagged_scalar(lag_records=30), default_lag=3.0)
        fdl.run()
        r = fdl.results
        self.assertEqual(r['cov_max_shift'], 30)
        self.assertEqual(r['flux_signal_at_default_lag'], r['flux_signal_at_cov_max_lag'])
        self.assertGreater(r['signal_to_detection_limit'], 1)

    def test_default_lag_off_grid_raises(self):
        fdl = self._fdl(self._lagged_scalar(lag_records=30), default_lag=2.8, lag_stepsize=10)
        with self.assertRaisesRegex(ValueError, 'not on the lag grid'):
            fdl.run()


if __name__ == '__main__':
    unittest.main()
