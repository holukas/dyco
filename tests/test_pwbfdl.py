"""
TEST_PWBFDL: TIME LAG AND FLUX DETECTION LIMIT TOGETHER
=======================================================

`PwbFdl` on one synthetic period, and `PwbFdlBatch` over a small raw file on
disk, read and written for real.

Part of the dyco package: https://github.com/holukas/dyco
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')

from dyco.pwbfdl import PwbFdl, PwbFdlBatch

HZ = 10
FAST = dict(hz=HZ, lag_max_s=5.0, n_bootstrap=9, fdl_lag_range_s=60, noise_range_s=10)


def _synthetic(n: int, lag_records: dict, seed: int = 0) -> pd.DataFrame:
    """Unrotated 10 Hz data; each gas arrives `lag_records[gas]` after w."""
    rng = np.random.default_rng(seed)
    pad = max(lag_records.values())
    w = np.convolve(rng.normal(0, 0.3, n + pad), np.ones(10) / 10, mode='same')
    df = pd.DataFrame({
        'u': 2 + rng.normal(0, 0.5, n),
        'v': 0.5 + rng.normal(0, 0.3, n),
        'w': 0.05 + w[pad:],
        'ts': 293.0 + 2.0 * w[pad:] + rng.normal(0, 0.05, n),
        'h2o': 10_000 + rng.normal(0, 20, n),       # umol mol-1
        'p': np.full(n, 970.0),                      # hPa
    })
    for gas, lag in lag_records.items():
        df[gas] = 2.0 + 0.05 * np.roll(w, lag)[pad:] + rng.normal(0, 0.002, n)  # umol mol-1
    return df


COLS = dict(col_u='u', col_v='v', col_w='w', col_tsonic='ts', col_h2o='h2o', col_press='p')
SCALE = {'ch4': 1e3, 'n2o': 1e3, 'h2o': 1e-6, 'p': 100}


class TestPwbFdl(unittest.TestCase):

    def test_lag_limit_and_flux_from_one_period(self):
        df = _synthetic(6000, {'ch4': 15, 'n2o': 20})
        p = PwbFdl(df, scalars={'CH4': 'ch4', 'N2O': 'n2o'}, scale=SCALE,
                   lws=0, uws=5, random_state=1, **COLS, **FAST)
        row = p.run()
        self.assertAlmostEqual(row['ch4_tlag_s'], 1.5, delta=0.1)
        self.assertAlmostEqual(row['n2o_tlag_s'], 2.0, delta=0.1)
        for g in ('ch4', 'n2o'):
            self.assertGreater(row[f'{g}_fdl'], 0)
            self.assertEqual(row[f'{g}_fdl'], 3 * row[f'{g}_fdl_noise_rmse'])
            self.assertEqual(row[f'{g}_flux_at_tlag'],
                             p.flux_at_lag(g.upper(), row[f'{g}_tlag_s']))
            # A strong, clean signal sits far above the noise.
            self.assertGreater(row[f'{g}_flux_to_fdl_at_tlag'], 5)
        curve = p.flux_curve('N2O')
        self.assertEqual(len(curve), 2 * 60 * HZ + 1)
        self.assertEqual(curve[60 * HZ + 20], row['n2o_flux_at_tlag'])

    def test_gas_without_data_gets_nothing(self):
        df = _synthetic(3000, {'ch4': 15})
        df['ch4'] = np.nan
        row = PwbFdl(df, scalars={'CH4': 'ch4'}, scale=SCALE, **COLS, **FAST).run()
        self.assertEqual(row['ch4_n_valid'], 0)
        self.assertTrue(np.isnan(row['ch4_tlag_s']))
        self.assertTrue(np.isnan(row['ch4_fdl']))

    def test_unscaled_pressure_is_refused(self):
        df = _synthetic(3000, {'ch4': 15})
        scale = {k: v for k, v in SCALE.items() if k != 'p'}
        with self.assertRaisesRegex(ValueError, 'pressure column'):
            PwbFdl(df, scalars={'CH4': 'ch4'}, scale=scale, **COLS, **FAST).run()


class TestPwbFdlBatch(unittest.TestCase):

    def test_batch_over_a_raw_file(self):
        """Three 5-minute periods from one file, one with CH4 offline, and a short tail."""
        df = _synthetic(9200, {'ch4': 15, 'n2o': 15}, seed=2)
        df.loc[3000:5999, 'ch4'] = -9999
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'SITE_202107221100.csv'
            with open(path, 'w', newline='') as fh:
                fh.write(','.join(df.columns) + '\n')
                fh.write(','.join(['[unit]'] * df.shape[1]) + '\n')
                df.to_csv(fh, header=False, index=False)
            out = Path(tmp) / 'out'
            b = PwbFdlBatch(
                files=[path], scalars={'CH4': 'ch4', 'N2O': 'n2o'}, scale=SCALE,
                extra_rows=1, chunk_seconds=300, min_chunk_seconds=60,
                start_time_regex=r'(\d{12})', lws=0, uws=5, random_state=3,
                n_workers=1, output_dir=out, **COLS, **FAST)
            res = b.run()
            self.assertTrue((out / 'pwb_fdl_results.csv').is_file())

        self.assertEqual(list(res['period']),
                         ['2021-07-22 11:00', '2021-07-22 11:05', '2021-07-22 11:10',
                          '2021-07-22 11:15'])
        self.assertEqual(list(res['status']), ['ok', 'ok', 'ok', 'skipped:short'])
        # A period that was never looked at is `none`, not `no_data`, as in detect-remove.
        self.assertEqual(res.iloc[3]['n2o_lag_source'], 'none')
        self.assertTrue(np.isnan(res.iloc[3]['n2o_flux']))
        res = res.iloc[:3]
        np.testing.assert_allclose(res['n2o_lag_s'], 1.5, atol=0.1)
        self.assertTrue(res['n2o_above_fdl'].astype(bool).all())
        # The flux is read at the PWBOPT lag, off the same curve as the limit.
        np.testing.assert_array_equal(res['n2o_flux_to_fdl'],
                                      res['n2o_flux'].abs() / res['n2o_fdl'])

        offline = res.iloc[1]
        self.assertEqual(offline['ch4_n_valid'], 0)
        self.assertEqual(offline['ch4_lag_source'], 'no_data')
        self.assertTrue(np.isnan(offline['ch4_lag_s']))
        self.assertTrue(np.isnan(offline['ch4_flux']))
        self.assertIn('no lag needed', offline['ch4_lag_reason'])


if __name__ == '__main__':
    unittest.main()
