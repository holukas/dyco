"""
TEST_MISSING_GAS: A GAS THAT IS ABSENT FOR A WHOLE PERIOD NEEDS NO LAG
=======================================================================

An analyser offline for an averaging period writes its fill value down the
whole column. That is an ordinary state of a raw record -- roughly one period
in eight of the CZ-Lnz QCL series this was found on -- but `_na_approx` fed the
empty column to `np.interp`, which raised "array of sample points is empty".
The error surfaced at file level, so a single dead gas took every other gas and
every chunk of that file down with it, and nothing was written.

There is no lag to find in an empty column and none to apply: shifting it would
move nothing. So the gas is skipped for that period and the period is marked
`no_data`, which also has to survive PWBOPT -- S3 carry and `@lagfrom=` exist
to fill periods whose detection was *rejected*, and cannot tell that apart from
a period that had no data to detect in.

Part of the dyco package: https://github.com/holukas/dyco
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from dyco.pipeline import PerFilePipeline
from dyco.pwb import PreWhiteningBootstrap, _finite_or_nan, _na_approx

_HZ = 20
_CHUNK_S = 60
_N = 3000          # 150 s at 20 Hz -> two full 60 s chunks + a rejected tail
_CHUNK_ROWS = _CHUNK_S * _HZ


def _raw_frame(n: int = _N, records: int = 20, seed: int = 3) -> pd.DataFrame:
    """EC-shaped data whose scalars lag the wind by `records` rows."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    lagged = np.r_[np.zeros(records), w[:-records]]
    return pd.DataFrame({
        'u': rng.standard_normal(n), 'v': rng.standard_normal(n), 'w': w,
        'ts': 0.8 * w + 0.2 * rng.standard_normal(n),
        'ch4': lagged + 0.2 * rng.standard_normal(n),
        'n2o': lagged + 0.2 * rng.standard_normal(n),
    })


def _write(path: Path, df: pd.DataFrame) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(','.join(df.columns) + '\n')
        df.to_csv(fh, index=False, header=False, lineterminator='\n')


class TestNaApproxAllMissing(unittest.TestCase):
    """The interpolator must not raise on a column with nothing to interpolate."""

    def test_all_nan_passes_through(self):
        x = np.full(100, np.nan)
        out = _na_approx(x)
        self.assertTrue(np.isnan(out).all())

    def test_partial_nan_still_interpolates(self):
        x = np.array([1.0, np.nan, 3.0])
        np.testing.assert_allclose(_na_approx(x), [1.0, 2.0, 3.0])


class TestNonFiniteIsMissing(unittest.TestCase):
    """`-Inf` in a data column is a missing value, not a measurement.

    The CZ-Lnz QCL record writes literal `-Inf` into H2O. It is not a string
    `--na-values` can match (and argparse will not even accept `-Inf` as a
    value), it survives the `-9999` filtering, and `na.approx` spreads it into
    any gap next to it -- surfacing much later as `array must not contain infs
    or NaNs` from scipy's detrend.
    """

    def test_inf_becomes_nan(self):
        x = np.array([1.0, np.inf, 3.0, -np.inf, np.nan])
        out = _finite_or_nan(x)
        self.assertTrue(np.isnan(out[[1, 3, 4]]).all())
        np.testing.assert_allclose(out[[0, 2]], [1.0, 3.0])

    def test_inf_does_not_spread_through_interpolation(self):
        # -inf next to a gap: na.approx would otherwise fill the gap with -inf.
        x = np.array([1.0, -np.inf, np.nan, 4.0])
        out = _na_approx(_finite_or_nan(x))
        self.assertTrue(np.isfinite(out).all(), out)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0])

    def _frame(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(n)
        return pd.DataFrame({
            'W_rot': w,
            'CH4': np.r_[np.zeros(20), w[:-20]] + 0.2 * rng.standard_normal(n),
            'T_SONIC': 0.8 * w + 0.2 * rng.standard_normal(n),
        })

    def test_a_few_infs_no_longer_fail_the_run(self):
        df = self._frame()
        df.loc[[100, 500, 501, 900], 'CH4'] = -np.inf
        pwb = PreWhiteningBootstrap(
            df=df, var_w='W_rot', var_scalar='CH4', var_tsonic='T_SONIC',
            hz=_HZ, n_bootstrap=19, random_state=42)
        pwb.run()   # used to raise ValueError from scipy detrend
        self.assertTrue(np.isfinite(pwb.results['tlag_s']))

    def test_a_column_of_only_infs_reads_as_empty(self):
        df = self._frame()
        df['CH4'] = -np.inf
        pwb = PreWhiteningBootstrap(
            df=df, var_w='W_rot', var_scalar='CH4', var_tsonic='T_SONIC',
            hz=_HZ, n_bootstrap=19, segment_name='period_B')
        with self.assertRaises(ValueError) as ctx:
            pwb.run()
        self.assertIn('CH4', str(ctx.exception))
        self.assertIn('empty', str(ctx.exception))


class TestPwbNamesTheEmptyColumn(unittest.TestCase):
    """Direct PWB use gets a message that says which column was empty."""

    def _frame(self, dead: str) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        n = 2000
        df = pd.DataFrame({
            'W_rot': rng.standard_normal(n),
            'CH4': rng.standard_normal(n),
            'T_SONIC': rng.standard_normal(n),
        })
        df[dead] = np.nan
        return df

    def test_empty_scalar_raises_by_name(self):
        pwb = PreWhiteningBootstrap(
            df=self._frame('CH4'), var_w='W_rot', var_scalar='CH4',
            var_tsonic='T_SONIC', hz=_HZ, n_bootstrap=5,
            segment_name='period_A')
        with self.assertRaises(ValueError) as ctx:
            pwb.run()
        self.assertIn('CH4', str(ctx.exception))
        self.assertIn('period_A', str(ctx.exception))

    def test_empty_wind_raises_by_name(self):
        pwb = PreWhiteningBootstrap(
            df=self._frame('W_rot'), var_w='W_rot', var_scalar='CH4',
            var_tsonic='T_SONIC', hz=_HZ, n_bootstrap=5)
        with self.assertRaises(ValueError) as ctx:
            pwb.run()
        self.assertIn('W_rot', str(ctx.exception))


class TestDeadGasDoesNotSinkTheRun(unittest.TestCase):
    """N2O is absent for the second chunk only; CH4 is fine throughout.

    A donor is configured on purpose (`lag_fallback`), so the no-data period
    is one PWBOPT would otherwise have been eager to fill.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()
        ind = Path(cls._tmp.name) / 'in'
        cls.out = Path(cls._tmp.name) / 'out'
        ind.mkdir()
        df = _raw_frame()
        # Analyser offline for the whole of chunk 2.
        df.loc[_CHUNK_ROWS:2 * _CHUNK_ROWS - 1, 'n2o'] = -9999.0
        cls.raw = df
        _write(ind / 'site_202401010000.csv', df)
        cls.summary = PerFilePipeline(
            ind, cls.out, 'u', 'v', 'w', 'ts',
            {'CH4': 'ch4', 'N2O': 'n2o'}, hz=_HZ,
            n_bootstrap=19, chunk_seconds=_CHUNK_S, min_chunk_seconds=40,
            sep=',', extra_rows=0, n_workers=1, file_pattern='*.csv',
            lag_fallback={'N2O': 'CH4'}, max_carry=1,
            random_state=42).run()
        cls.detect_dir = cls.out / '1_lag_detection'
        # Row order follows chunk_index; chunk 1 is the dead one.
        cls.ok = cls.summary[cls.summary['status'] == 'ok'].reset_index(drop=True)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_both_chunks_were_written(self):
        # The whole point: one dead gas used to take the entire file with it.
        self.assertEqual(len(self.ok), 2, self.summary['status'].tolist())
        written = sorted(p.name for p in (self.out / '2_lag_removed').glob('*.csv'))
        self.assertEqual(len(written), 2, written)

    def test_n_valid_counts_the_live_and_dead_periods(self):
        self.assertEqual(int(self.ok.loc[0, 'n2o_n_valid']), _CHUNK_ROWS)
        self.assertEqual(int(self.ok.loc[1, 'n2o_n_valid']), 0)
        self.assertEqual(int(self.ok.loc[1, 'ch4_n_valid']), _CHUNK_ROWS)

    def test_dead_period_gets_no_lag_despite_a_donor(self):
        self.assertEqual(self.ok.loc[1, 'n2o_lag_source'], 'no_data')
        self.assertTrue(np.isnan(self.ok.loc[1, 'n2o_tlag_final_pf_s']))
        self.assertTrue(np.isnan(self.ok.loc[1, 'n2o_tlag_final_s']))

    def test_dead_period_reports_it_in_words(self):
        reason = str(self.ok.loc[1, 'n2o_lag_reason'])
        self.assertIn('missing', reason)
        self.assertIn('no lag', reason)
        report = (self.detect_dir
                  / 'detect_and_remove_tlag_decisions.txt').read_text(
            encoding='utf-8')
        self.assertIn('missing', report)

    def test_dead_column_is_written_through_untouched(self):
        self.assertEqual(self.ok.loc[1, 'n2o_status'], 'skipped:lag_nan')
        out_file = sorted((self.out / '2_lag_removed').glob('*.csv'))[1]
        got = pd.read_csv(out_file, na_values=['-9999', '-9999.0'])
        self.assertTrue(got['n2o'].isna().all())

    def test_the_live_gas_is_unaffected(self):
        self.assertEqual(self.ok.loc[1, 'ch4_status'], 'ok')
        self.assertTrue(np.isfinite(self.ok.loc[1, 'ch4_tlag_final_pf_s']))

    def test_the_live_period_of_the_dead_gas_still_detects(self):
        self.assertEqual(self.ok.loc[0, 'n2o_status'], 'ok')
        self.assertTrue(np.isfinite(self.ok.loc[0, 'n2o_tlag_final_pf_s']))


class TestInfEndToEnd(unittest.TestCase):
    """A raw file carrying literal `-Inf` runs, and `n_valid` excludes them."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()
        ind = Path(cls._tmp.name) / 'in'
        cls.out = Path(cls._tmp.name) / 'out'
        ind.mkdir()
        df = _raw_frame()
        # A handful of -Inf in chunk 1, and a whole chunk of them in chunk 2.
        df.loc[[10, 11, 700, 1199], 'ch4'] = -np.inf
        df.loc[_CHUNK_ROWS:2 * _CHUNK_ROWS - 1, 'n2o'] = -np.inf
        _write(ind / 'site_202401010000.csv', df)
        cls.summary = PerFilePipeline(
            ind, cls.out, 'u', 'v', 'w', 'ts',
            {'CH4': 'ch4', 'N2O': 'n2o'}, hz=_HZ,
            n_bootstrap=19, chunk_seconds=_CHUNK_S, min_chunk_seconds=40,
            sep=',', extra_rows=0, n_workers=1, file_pattern='*.csv',
            random_state=42).run()
        cls.ok = cls.summary[cls.summary['status'] == 'ok'].reset_index(drop=True)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_run_completes(self):
        # Before the fix the -Inf chunk died with "array must not contain infs
        # or NaNs" out of scipy's detrend.
        self.assertEqual(len(self.ok), 2, self.summary['status'].tolist())
        self.assertNotIn('error', self.summary['status'].tolist())

    def test_scattered_infs_are_not_counted_as_valid(self):
        self.assertEqual(int(self.ok.loc[0, 'ch4_n_valid']), _CHUNK_ROWS - 4)
        self.assertTrue(np.isfinite(self.ok.loc[0, 'ch4_tlag_final_pf_s']))

    def test_an_all_inf_chunk_is_no_data(self):
        self.assertEqual(int(self.ok.loc[1, 'n2o_n_valid']), 0)
        self.assertEqual(self.ok.loc[1, 'n2o_lag_source'], 'no_data')


if __name__ == '__main__':
    unittest.main()
