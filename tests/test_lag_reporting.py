"""
TEST_LAG_REPORTING: WHICH LAG WAS APPLIED, AND WHY
==================================================

The summary CSV carries six lag columns per gas, and reconstructing which one
reached the data -- and on what grounds -- meant reading the source. These
tests pin the three outputs that answer it directly: the applied-lag column,
the decisions report, and the rule that a period with no output file gets no
lag at all.

Part of the dyco package: https://github.com/holukas/dyco
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from dyco.pipeline import PerFilePipeline

_HZ = 20
_CHUNK_S = 60


def _raw_frame(n: int, records: int = 20, seed: int = 3) -> pd.DataFrame:
    """EC-shaped data whose scalar lags the wind by `records` rows."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    return pd.DataFrame({
        'u': rng.standard_normal(n), 'v': rng.standard_normal(n), 'w': w,
        'ts': 0.8 * w + 0.2 * rng.standard_normal(n),
        'ch4': np.r_[np.zeros(records), w[:-records]]
        + 0.2 * rng.standard_normal(n),
    })


def _write(path: Path, df: pd.DataFrame) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(','.join(df.columns) + '\n')
        df.to_csv(fh, index=False, header=False, lineterminator='\n')


class TestAppliedLagAndReasons(unittest.TestCase):
    """One run, three questions: what was applied, why, and where nothing was."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()
        ind = Path(cls._tmp.name) / 'in'
        cls.out = Path(cls._tmp.name) / 'out'
        ind.mkdir()
        # 3000 rows at 20 Hz = 150 s: two full 60 s chunks plus a 30 s tail
        # that min_chunk_seconds=40 rejects -- so the run has both a written
        # period and a gap period, which is what case C is about.
        _write(ind / 'site_202401010000.csv', _raw_frame(3000))
        cls.summary = PerFilePipeline(
            ind, cls.out, 'u', 'v', 'w', 'ts', {'CH4': 'ch4'}, hz=_HZ,
            n_bootstrap=19, chunk_seconds=_CHUNK_S, min_chunk_seconds=40,
            sep=',', extra_rows=0, n_workers=1, file_pattern='*.csv',
            random_state=42).run()
        cls.detect_dir = cls.out / '1_lag_detection'

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_run_has_both_a_written_period_and_a_gap_period(self):
        status = self.summary['status'].astype(str).tolist()
        self.assertEqual(status.count('ok'), 2, status)
        self.assertEqual(status.count('skipped:short'), 1, status)

    # ---- A: the applied lag, stated plainly ----
    def test_the_applied_column_matches_the_shift_that_was_made(self):
        # The lag actually removed is a whole number of records, so it can
        # differ from the requested lag; this column is the one that is true
        # of the data on disk.
        ok = self.summary[self.summary['status'] == 'ok']
        applied = ok['ch4_lag_applied_s'].to_numpy(dtype=float)
        records = ok['ch4_applied_records'].to_numpy(dtype=float)
        self.assertFalse(np.isnan(applied).any())
        np.testing.assert_allclose(applied, records / _HZ)
        # and it really is the lag the source data carries (20 records = 1.0 s)
        np.testing.assert_allclose(applied, 1.0, atol=0.3)

    def test_the_applied_column_is_documented_in_the_data_dictionary(self):
        doc = (self.detect_dir
               / 'detect_and_remove_tlag_summary_columns.md').read_text(
            encoding='utf-8')
        # The per-gas table lists the template names, not the expanded ones.
        self.assertIn('{gas}_lag_applied_s', doc)
        self.assertIn('{gas}_carry_periods', doc)
        self.assertIn('{gas}_lag_reason', doc)
        self.assertIn('ch4_', doc)     # the run's own prefix is named too

    # ---- B: the reasons ----
    def test_the_decisions_report_covers_every_written_file(self):
        report = (self.detect_dir
                  / 'detect_and_remove_tlag_decisions.txt').read_text(
            encoding='utf-8')
        written = self.summary.loc[self.summary['status'] == 'ok', 'period']
        for period in written:
            self.assertIn(str(period), report)
        # the applied lag appears next to a reason, not on its own
        self.assertIn('rec)', report)
        self.assertRegex(report, r'detected here|carried|borrowed|back-filled')
        # the thresholds behind the decision are stated, so the reader does
        # not have to go and find the settings file
        self.assertIn('S1', report)
        self.assertIn('prefilter' if 'prefilter' in report else 'dropped',
                      report)

    def test_the_report_names_the_periods_that_got_no_file(self):
        report = (self.detect_dir
                  / 'detect_and_remove_tlag_decisions.txt').read_text(
            encoding='utf-8')
        self.assertIn('NO OUTPUT FILE', report)
        self.assertIn('skipped:short', report)

    def test_every_row_carries_its_reason_in_the_summary_too(self):
        reasons = self.summary['ch4_lag_reason'].astype(str)
        self.assertTrue((reasons.str.len() > 0).all())
        gap = self.summary[self.summary['status'] != 'ok']
        self.assertIn('no file written', gap['ch4_lag_reason'].iloc[0])

    # ---- C: no lag where no file was written ----
    def test_a_period_with_no_output_file_gets_no_lag(self):
        gap = self.summary[self.summary['status'] != 'ok']
        self.assertEqual(len(gap), 1)
        row = gap.iloc[0]
        for col in ('ch4_tlag_final_s', 'ch4_tlag_final_pf_s',
                    'ch4_lag_applied_s', 'ch4_carry_periods'):
            self.assertTrue(pd.isna(row[col]), f'{col} = {row[col]!r}')
        self.assertEqual(row['ch4_lag_source'], 'none')

    def test_the_written_periods_keep_their_lags(self):
        ok = self.summary[self.summary['status'] == 'ok']
        self.assertFalse(ok['ch4_tlag_final_pf_s'].isna().any())
        self.assertFalse(ok['ch4_lag_applied_s'].isna().any())


class TestBorrowedLagsAreExplained(unittest.TestCase):
    """A borrowed lag has to say so, in the summary and in the report."""

    def test_a_noisy_gas_borrows_per_period_and_the_report_says_from_whom(self):
        rng = np.random.default_rng(5)
        n = 4800
        w = rng.standard_normal(n)
        df = pd.DataFrame({
            'u': rng.standard_normal(n), 'v': rng.standard_normal(n), 'w': w,
            'ts': 0.8 * w + 0.2 * rng.standard_normal(n),
            'co2': np.r_[np.zeros(30), w[:-30]] + 0.05 * rng.standard_normal(n),
            'n2o': rng.standard_normal(n),        # pure noise, never detects
        })
        with TemporaryDirectory() as tmp:
            ind, out = Path(tmp) / 'in', Path(tmp) / 'out'
            ind.mkdir()
            _write(ind / 'site_202401010000.csv', df)
            summary = PerFilePipeline(
                ind, out, 'u', 'v', 'w', 'ts', {'CO2': 'co2', 'N2O': 'n2o'},
                hz=_HZ, lag_max_s=5.0, n_bootstrap=19, chunk_seconds=_CHUNK_S,
                min_chunk_seconds=30, sep=',', extra_rows=0, n_workers=1,
                file_pattern='*.csv', lag_fallback={'CO2': 'CO2', 'N2O': 'CO2'},
                random_state=42).run()
            report = (out / '1_lag_detection'
                      / 'detect_and_remove_tlag_decisions.txt').read_text(
                encoding='utf-8')

        sources = summary['n2o_lag_source'].astype(str)
        self.assertTrue((sources == 'from:CO2').any(), sources.tolist())
        # a borrowed period takes the donor's lag for that same period
        borrowed = sources == 'from:CO2'
        np.testing.assert_allclose(
            summary.loc[borrowed, 'n2o_tlag_final_pf_s'].to_numpy(dtype=float),
            summary.loc[borrowed, 'co2_tlag_final_pf_s'].to_numpy(dtype=float))
        # and nothing is left to guess in the report
        self.assertIn('borrowed from CO2', report)
        self.assertIn('from:CO2', report)      # the tally at the foot


class TestCarryLimitEndToEnd(unittest.TestCase):
    """--max-carry has to reach the summary, not just apply_pwbopt."""

    def test_the_carry_distance_is_reported_per_period(self):
        with TemporaryDirectory() as tmp:
            ind, out = Path(tmp) / 'in', Path(tmp) / 'out'
            ind.mkdir()
            _write(ind / 'site_202401010000.csv', _raw_frame(4800))
            summary = PerFilePipeline(
                ind, out, 'u', 'v', 'w', 'ts', {'CH4': 'ch4'}, hz=_HZ,
                n_bootstrap=19, chunk_seconds=_CHUNK_S, min_chunk_seconds=30,
                sep=',', extra_rows=0, n_workers=1, file_pattern='*.csv',
                max_carry=1, random_state=42).run()
            settings = (out / 'run_settings.txt').read_text(encoding='utf-8')

        carry = summary['ch4_carry_periods'].to_numpy(dtype=float)
        finite = carry[~np.isnan(carry)]
        self.assertTrue(len(finite), 'no period reported a carry distance')
        # the limit is what it says: nothing travels further than 1 period
        self.assertLessEqual(finite.max(), 1)
        self.assertIn('max_carry', settings)


if __name__ == '__main__':
    unittest.main()
