"""
TEST_VALID_RANGE: VALUES OUTSIDE A GAS'S VALID RANGE ARE IGNORED FOR DETECTION
==============================================================================

A logger fault at CH-LAE (sonicreadHS before v7.08, two IRGAs) wrote LI-7500
values into the LI-7200 record about once per second, so `CO2_DRY` read about
15 instead of about 400 umol mol-1 in 3.9 % of rows, with a clean status byte.
EddyPro drops such rows with its absolute-limits test; dyco detected lags on
them. A per-gas valid range makes detection treat those values as missing,
the same rule it already applies to inf.

It is for detection only. The written files keep every value, shifted like the
rest: cleaning the data for the flux calculation is the flux software's job.

Part of the dyco package: https://github.com/holukas/dyco
"""

import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import pandas as pd

from dyco.pipeline import PerFilePipeline, parse_scalar_spec
from dyco.pwb import (PreWhiteningBootstrap, _out_of_range, _pwb_file_worker,
                      format_valid_range, out_of_range_lines)
from tests.test_pwb import _settle

_HZ = 20
_CHUNK_S = 60
_CHUNK_ROWS = _CHUNK_S * _HZ
_N = 3000          # two full 60 s chunks + a rejected tail
_OUTLIER = 15.0    # what the LI-7500 density looks like in the LI-7200 column


class TestSpecParsing(unittest.TestCase):

    def test_both_bounds(self):
        self.assertEqual(
            parse_scalar_spec('CO2:CO2_DRY_[IRGA72-B]@min=200;max=900'),
            ('CO2', 'CO2_DRY_[IRGA72-B]',
             {'valid_min': 200.0, 'valid_max': 900.0}))

    def test_either_bound_may_be_left_out(self):
        self.assertEqual(parse_scalar_spec('H2O:h2o@min=0'),
                         ('H2O', 'h2o', {'valid_min': 0.0}))
        self.assertEqual(parse_scalar_spec('H2O:h2o@max=40'),
                         ('H2O', 'h2o', {'valid_max': 40.0}))

    def test_mixes_with_the_other_options(self):
        self.assertEqual(
            parse_scalar_spec('N2O:n2o@lagfrom=CO2;min=0.032;max=1000;uws=5'),
            ('N2O', 'n2o', {'lagfrom': 'CO2', 'valid_min': 0.032,
                            'valid_max': 1000.0, 'uws': 5.0}))

    def test_bad_input(self):
        for bad in ('CO2:co2@min=abc', 'CO2:co2@min=', 'CO2:co2@min=900;max=200',
                    'CO2:co2@min=5;max=5', 'CO2:co2@max=nan', 'CO2:co2@min=inf',
                    'CO2:co2@minimum=1'):
            with self.subTest(token=bad), self.assertRaises(ValueError):
                parse_scalar_spec(bad)

    def test_unknown_key_names_min_and_max(self):
        with self.assertRaises(ValueError) as ctx:
            parse_scalar_spec('CO2:co2@low=1')
        self.assertIn('min, max', str(ctx.exception))

    def test_format_valid_range(self):
        self.assertEqual(format_valid_range(200.0, 900.0), '200 to 900')
        self.assertEqual(format_valid_range(0.032, None), 'at least 0.032')
        self.assertEqual(format_valid_range(None, 40.0), 'at most 40')
        self.assertEqual(format_valid_range(None, None), 'no limit')


def _frame(seed=5, n=12000, k=40, amp=0.07, every=25):
    """W, sonic T and a CO2-like scalar near 400 that lags W by `k` records.

    The coupling is weak on purpose: near the detection limit is where a
    handful of wild values decides whether the lag is found. Returns the clean
    frame and a copy with every `every`-th scalar value replaced by
    `_OUTLIER` (25 -> 4 %, close to the 3.9 % seen at CH-LAE).
    """
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    w = np.empty(n)
    w[0] = e[0]
    for i in range(1, n):
        w[i] = 0.6 * w[i - 1] + e[i]
    s = 400 + amp * np.r_[np.zeros(k), w[:-k]] + rng.standard_normal(n)
    t = 0.7 * w + 0.5 * rng.standard_normal(n)
    clean = pd.DataFrame({'w': w, 's': s, 't': t})
    bad = clean.copy()
    bad.loc[::every, 's'] = _OUTLIER
    return clean, bad


def _run(df, **kw):
    pwb = PreWhiteningBootstrap(df, 'w', 's', 't', hz=_HZ, lag_max_s=10,
                                n_bootstrap=29, random_state=1, **kw)
    pwb.run()
    return pwb


class TestDetection(unittest.TestCase):

    def test_mask_helper(self):
        x = np.array([1.0, 5.0, np.nan, 10.0, -3.0])
        np.testing.assert_array_equal(_out_of_range(x, 0, 6),
                                      [False, False, False, True, True])
        np.testing.assert_array_equal(_out_of_range(x, None, 6),
                                      [False, False, False, True, False])
        np.testing.assert_array_equal(_out_of_range(x, 0, None),
                                      [False, False, False, False, True])
        self.assertFalse(_out_of_range(x, None, None).any())

    def test_bounds_are_inclusive(self):
        x = np.array([200.0, 900.0])
        self.assertFalse(_out_of_range(x, 200, 900).any())

    def test_in_range_data_give_bit_identical_results(self):
        clean, _ = _frame()
        plain = _run(clean)
        ranged = _run(clean, valid_min=200, valid_max=900)
        self.assertEqual(ranged.results['n_out_of_range'], 0)
        skip = {'valid_min', 'valid_max', 'n_out_of_range'}
        for key, val in plain.results.items():
            if key in skip:
                continue
            with self.subTest(key=key):
                self.assertEqual(repr(val), repr(ranged.results[key]))
        np.testing.assert_array_equal(plain._bootstrap_lags,
                                      ranged._bootstrap_lags)

    def test_out_of_range_values_are_counted(self):
        _, bad = _frame()
        pwb = _run(bad, valid_min=200, valid_max=900)
        self.assertEqual(pwb.results['n_out_of_range'], len(bad) // 25)

    def test_only_the_scalar_is_range_checked(self):
        # W and T_SONIC sit far outside 200..900 and must not be touched.
        clean, _ = _frame()
        pwb = _run(clean, valid_min=200, valid_max=900)
        self.assertTrue(np.isfinite(pwb.results['tlag_s']))

    def test_every_value_out_of_range_reads_as_empty(self):
        clean, _ = _frame(n=2000)
        clean['s'] = _OUTLIER
        pwb = PreWhiteningBootstrap(clean, 'w', 's', 't', hz=_HZ,
                                    n_bootstrap=5, valid_min=200,
                                    segment_name='period_X')
        with self.assertRaises(ValueError) as ctx:
            pwb.run()
        self.assertIn('valid range', str(ctx.exception))
        self.assertIn('period_X', str(ctx.exception))

    def test_an_empty_range_is_refused(self):
        clean, _ = _frame(n=2000)
        with self.assertRaises(ValueError):
            PreWhiteningBootstrap(clean, 'w', 's', 't', valid_min=900,
                                  valid_max=200)

    def test_the_range_recovers_a_lag_the_outliers_spoil(self):
        clean, bad = _frame()
        true_lag = 40 / _HZ
        reference = _run(clean)
        spoilt = _run(bad)
        recovered = _run(bad, valid_min=200, valid_max=900)
        # The clean series detects the lag reliably, so this is a fair test.
        self.assertTrue(reference.is_reliable)
        self.assertAlmostEqual(reference.tlag_s, true_lag, delta=0.1)
        # 4 % outliers: the bootstrap no longer agrees on the lag.
        self.assertFalse(spoilt.is_reliable)
        self.assertGreater(spoilt.hdi_range_s, 4 * recovered.hdi_range_s)
        # With the range they are missing values, and the lag comes back.
        self.assertTrue(recovered.is_reliable)
        self.assertAlmostEqual(recovered.tlag_s, true_lag, delta=0.1)


class TestBatchWorker(unittest.TestCase):
    """`dyco pwb-batch` applies the same range and reports the count."""

    def test_worker_counts_and_masks(self):
        _, bad = _frame(n=6000)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / '20240101-0000_rotated.txt'
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write('w t co2\n')
                bad[['w', 't', 's']].to_csv(fh, sep=' ', index=False,
                                            header=False)
            args = (str(path), {'CO2': 'co2'}, 'w', 't',
                    _HZ, 10.0, 9, 20.0, [0, 1, 2], ['w', 't', 'co2'], 0,
                    ['-9999'], 0.3, None, False, 1, True, None, None,
                    {'CO2': (200.0, 900.0)})
            row = _pwb_file_worker(args)
        self.assertEqual(row['co2_n_out_of_range'], 6000 // 25)
        self.assertNotIn('co2_error', row)


def _raw_frame(n=_N, records=20, seed=3):
    """EC-shaped raw data; co2 lags the wind by `records` rows."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    lagged = np.r_[np.zeros(records), w[:-records]]
    return pd.DataFrame({
        'u': rng.standard_normal(n), 'v': rng.standard_normal(n), 'w': w,
        'ts': 0.8 * w + 0.2 * rng.standard_normal(n),
        'co2': np.round(400 + 2 * lagged + 0.2 * rng.standard_normal(n), 4),
        'ch4': np.round(2 + lagged + 0.2 * rng.standard_normal(n), 4),
    })


def _write(path, df):
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(','.join(df.columns) + '\n')
        df.to_csv(fh, index=False, header=False, lineterminator='\n')


class TestPipeline(unittest.TestCase):
    """Chunk 1 carries 4 % outliers, chunk 2 is nothing but outliers."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()
        ind = Path(cls._tmp.name) / 'in'
        cls.out = Path(cls._tmp.name) / 'out'
        ind.mkdir()
        df = _raw_frame()
        df.loc[0:_CHUNK_ROWS - 1:25, 'co2'] = _OUTLIER
        df.loc[_CHUNK_ROWS:2 * _CHUNK_ROWS - 1, 'co2'] = _OUTLIER
        cls.raw = df
        _write(ind / 'site_202401010000.csv', df)
        cls.pipeline = PerFilePipeline(
            ind, cls.out, 'u', 'v', 'w', 'ts',
            {'CO2': 'co2', 'CH4': 'ch4'}, hz=_HZ,
            n_bootstrap=19, chunk_seconds=_CHUNK_S, min_chunk_seconds=40,
            sep=',', extra_rows=0, n_workers=1, file_pattern='*.csv',
            valid_ranges={'CO2': (200.0, 900.0)}, random_state=42)
        cls.summary = cls.pipeline.run()
        cls.ok = cls.summary[cls.summary['status'] == 'ok'].reset_index(drop=True)
        cls.files = sorted((cls.out / '2_lag_removed').glob('*.csv'))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_counts(self):
        self.assertEqual(int(self.ok.loc[0, 'co2_n_out_of_range']),
                         _CHUNK_ROWS // 25)
        self.assertEqual(int(self.ok.loc[0, 'co2_n_valid']),
                         _CHUNK_ROWS - _CHUNK_ROWS // 25)
        self.assertEqual(int(self.ok.loc[1, 'co2_n_out_of_range']), _CHUNK_ROWS)
        self.assertEqual(int(self.ok.loc[1, 'co2_n_valid']), 0)
        # A gas without a range is never range-checked.
        self.assertEqual(int(self.ok.loc[0, 'ch4_n_out_of_range']), 0)
        self.assertEqual(int(self.ok.loc[0, 'ch4_n_valid']), _CHUNK_ROWS)

    def test_the_spoilt_chunk_still_detects(self):
        self.assertAlmostEqual(self.ok.loc[0, 'co2_tlag_s'], 1.0, delta=0.1)

    def test_an_all_out_of_range_chunk_is_no_data(self):
        self.assertEqual(self.ok.loc[1, 'co2_lag_source'], 'no_data')
        self.assertTrue(np.isnan(self.ok.loc[1, 'co2_tlag_final_pf_s']))
        self.assertIn('valid range', str(self.ok.loc[1, 'co2_lag_reason']))

    def test_both_chunks_were_written(self):
        self.assertEqual(len(self.files), 2, self.summary['status'].tolist())

    def test_the_outliers_are_still_in_the_output(self):
        # Detection only: the written chunk is the input shifted by the
        # applied lag, out-of-range values and all.
        shift = int(self.ok.loc[0, 'co2_applied_records'])
        got = pd.read_csv(self.files[0], na_values=['-9999'])['co2'].to_numpy()
        want = self.raw['co2'].to_numpy()[shift:_CHUNK_ROWS]
        np.testing.assert_allclose(got[:len(want)], want)
        self.assertEqual(int((got == _OUTLIER).sum()),
                         int((want == _OUTLIER).sum()))
        self.assertGreater(int((got == _OUTLIER).sum()), 0)

    def test_the_no_data_chunk_is_written_unchanged(self):
        got = pd.read_csv(self.files[1])['co2'].to_numpy()
        np.testing.assert_array_equal(got, np.full(_CHUNK_ROWS, _OUTLIER))

    def test_log_line(self):
        lines = out_of_range_lines(self.summary, ['CO2', 'CH4'],
                                   self.pipeline.valid_ranges)
        self.assertEqual(len(lines), 1)
        self.assertIn('CO2', lines[0])
        self.assertIn(f'{_CHUNK_ROWS + _CHUNK_ROWS // 25} records', lines[0])
        self.assertIn('200 to 900', lines[0])
        self.assertIn('2 of 2 periods', lines[0])
        lines[0].encode('cp1252')

    def test_run_settings_name_the_range(self):
        text = (self.out / 'run_settings.txt').read_text(encoding='utf-8')
        self.assertIn('CO2: 200 to 900', text)


class TestPipelineValidation(unittest.TestCase):

    def test_bad_ranges_are_refused(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            with self.assertRaises(ValueError):
                PerFilePipeline(d, d, 'u', 'v', 'w', 'ts', {'CO2': 'co2'},
                                valid_ranges={'XX': (0, 1)})
            with self.assertRaises(ValueError):
                PerFilePipeline(d, d, 'u', 'v', 'w', 'ts', {'CO2': 'co2'},
                                valid_ranges={'CO2': (900, 200)})
            p = PerFilePipeline(d, d, 'u', 'v', 'w', 'ts', {'CO2': 'co2'},
                                valid_ranges={'CO2': (None, None)})
            self.assertEqual(p.valid_ranges, {})


# --------------------------------------------------------------------------
# TUI
# --------------------------------------------------------------------------

def _write_units_file(folder: Path) -> None:
    """A raw CSV with a units row and an instrument row, as loggers write."""
    cols = ['u', 'v', 'w', 'ts', 'co2', 'h2o', 'n2o', 'ch4d']
    units = ['[m+1_s-1]'] * 3 + ['[K]', '[umol+1_mol-1]', '[mmol+1_mol-1]',
                                 '[nmol+1_mol-1]', '[mmol+1_m-3]']
    with open(folder / 'site_202401010000.csv', 'w', encoding='utf-8') as fh:
        fh.write(','.join(cols) + '\n')
        fh.write(','.join(units) + '\n')
        fh.write(','.join(['[X]'] * len(cols)) + '\n')
        fh.write(','.join(['1'] * len(cols)) + '\n')


class TestTuiSuggestions(unittest.TestCase):

    def setUp(self):
        try:
            from dyco import tui
        except Exception:
            self.skipTest('textual TUI not importable')
        self.tui = tui

    def test_units_spelled_several_ways(self):
        kind = self.tui._unit_kind
        for unit in ('[umol+1_mol-1]', 'umol mol-1', 'umol mol^-1', 'umol/mol',
                     'µmol/mol', 'ppm'):
            self.assertEqual(kind(unit), 'umol/mol', unit)
        self.assertEqual(kind('[mmol+1_mol-1]'), 'mmol/mol')
        for unit in ('[mmol+1_m-3]', '[mmol+1 m-3]', 'mmol/m^3'):
            self.assertEqual(kind(unit), 'density', unit)
        for unit in ('[nmol+1_mol-1]', 'ppb', '[%]', ''):
            self.assertIsNone(kind(unit), unit)

    def test_suggestions(self):
        s = self.tui.suggest_valid_range
        units = {'CO2_DRY': '[umol+1_mol-1]', 'h2o': '[mmol+1_mol-1]',
                 'ch4': 'ppm', 'n2o': 'umol/mol', 'CO2_CONC': '[mmol+1_m-3]',
                 'x': '[umol+1_mol-1]'}
        self.assertEqual(s('CO2', 'CO2_DRY', units)[0], (200.0, 900.0))
        self.assertEqual(s('H2O', 'h2o', units)[0], (0.0, 40.0))
        self.assertEqual(s('CH4', 'ch4', units)[0], (0.17, 1000.0))
        self.assertEqual(s('N2O', 'n2o', units)[0], (0.032, 1000.0))
        # The gas is found in the column name when the label does not say.
        self.assertEqual(s('A', 'CO2_DRY', units)[0], (200.0, 900.0))
        rng, why = s('CO2', 'CO2_CONC', units)
        self.assertIsNone(rng)
        self.assertIn('density', why)
        self.assertEqual(s('CO2', 'missing', units), (None, 'unit unknown'))
        self.assertIsNone(s('X', 'x', units)[0])
        # A recognised gas in the other gas's unit is not guessed either.
        self.assertIsNone(s('H2O', 'n2o', units)[0])

    def test_text_round_trip_and_errors(self):
        parse, fmt = self.tui.parse_valid_ranges, self.tui.format_valid_ranges
        text = 'CO2:[200,900],H2O:[0,],N2O:[,1000],CH4:[,]'
        ranges = parse(text)
        self.assertEqual(ranges, {'CO2': (200.0, 900.0), 'H2O': (0.0, None),
                                  'N2O': (None, 1000.0), 'CH4': (None, None)})
        self.assertEqual(fmt(ranges), text)
        self.assertEqual(parse(''), {})
        for bad in ('CO2:200,900', 'CO2:[900,200]', 'CO2:[a,b]', 'junk'):
            with self.subTest(text=bad), self.assertRaises(ValueError):
                parse(bad)


class TestTuiField(unittest.TestCase):

    def setUp(self):
        try:
            from dyco import tui  # noqa: F401
        except Exception:
            self.skipTest('textual TUI not importable')

    def test_seed_edit_reseed_collect_and_persist(self):
        from dyco import tui
        from textual.widgets import Button, Input

        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ind = Path(tmp.name) / 'in'
        ind.mkdir()
        _write_units_file(ind)
        settings = Path(tmp.name) / 'tui.yaml'

        async def first_session():
            app = tui.DetectRemoveTUI(demo=False)
            async with app.run_test(size=(140, 60)) as pilot:
                await pilot.pause()
                rng = lambda: app.query_one('#validrange', Input).value
                app.query_one('#input_dir', Input).value = str(ind)
                app.query_one('#output_dir', Input).value = str(
                    Path(tmp.name) / 'out')
                scal = app.query_one('#scalars', Input)

                # Seeded from the units row; unknown and density units get [,].
                scal.value = 'CO2:co2,H2O:h2o,N2O:n2o,CH4:ch4d'
                want = 'CO2:[200,900],H2O:[0,40],N2O:[,],CH4:[,]'
                seeded = await _settle(pilot, rng, want)

                # An edit survives a change of gases; a dropped gas goes.
                app.query_one('#validrange', Input).value = \
                    'CO2:[300,800],H2O:[0,40],N2O:[,],CH4:[,]'
                scal.value = 'CO2:co2,H2O:h2o,N2O:n2o'
                want = 'CO2:[300,800],H2O:[0,40],N2O:[,]'
                kept = await _settle(pilot, rng, want)

                # A gas typed before its column is complete is seeded once it is.
                scal.value = 'CO2:co2,H2O:h2o,N2O:n2o,C2:c'
                await pilot.pause()
                scal.value = 'CO2:co2,H2O:h2o,N2O:n2o,C2:co2'
                want = 'CO2:[300,800],H2O:[0,40],N2O:[,],C2:[200,900]'
                late = await _settle(pilot, rng, want)

                # The button re-seeds every gas from its unit.
                scal.value = 'CO2:co2,H2O:h2o,N2O:n2o'
                await _settle(pilot, rng, 'CO2:[300,800],H2O:[0,40],N2O:[,]')
                app.query_one('#reseed_validrange', Button).press()
                want = 'CO2:[200,900],H2O:[0,40],N2O:[,]'
                reseeded = await _settle(pilot, rng, want)
                # ...and leaves the search windows alone.
                win = app.query_one('#winranges', Input).value

                # _collect passes real ranges only, and refuses a bad one.
                cfg = app._collect()
                app.query_one('#validrange', Input).value = 'CO2:[900,200]'
                await pilot.pause()
                with self.assertRaises(ValueError):
                    app._collect()

                app.query_one('#validrange', Input).value = \
                    'CO2:[250,850],H2O:[0,40],N2O:[,]'
                await pilot.pause()
                app._save_settings(announce=False)
                return seeded, kept, late, reseeded, win, cfg

        async def second_session():
            app = tui.DetectRemoveTUI(demo=False)
            async with app.run_test(size=(140, 60)) as pilot:
                rng = lambda: app.query_one('#validrange', Input).value
                want = 'CO2:[250,850],H2O:[0,40],N2O:[,]'
                return await _settle(pilot, rng, want)

        with mock.patch.object(tui, '_SETTINGS_PATH', settings):
            seeded, kept, late, reseeded, win, cfg = asyncio.run(first_session())
            restored = asyncio.run(second_session())

        self.assertEqual(seeded, 'CO2:[200,900],H2O:[0,40],N2O:[,],CH4:[,]')
        self.assertEqual(kept, 'CO2:[300,800],H2O:[0,40],N2O:[,]')
        self.assertEqual(late, 'CO2:[300,800],H2O:[0,40],N2O:[,],C2:[200,900]')
        self.assertEqual(reseeded, 'CO2:[200,900],H2O:[0,40],N2O:[,]')
        self.assertEqual(win, 'CO2:[-10,10],H2O:[-10,10],N2O:[-10,10]')
        self.assertEqual(cfg['valid_ranges'],
                         {'CO2': (200.0, 900.0), 'H2O': (0.0, 40.0)})
        # Saved and loaded back as typed, not re-seeded from the units.
        self.assertEqual(restored, 'CO2:[250,850],H2O:[0,40],N2O:[,]')

    def test_cli_settings_yaml_carries_the_ranges(self):
        import yaml
        from dyco import tui
        from dyco.pipeline import _build_parser
        args = _build_parser().parse_args([
            '--input-dir', '.', '--output-dir', '.',
            '--col-u', 'u', '--col-v', 'v', '--col-w', 'w', '--col-tsonic', 'ts',
            '--scalar', 'CO2:co2@min=200;max=900', '--scalar', 'N2O:n2o'])
        with TemporaryDirectory() as tmp:
            path = tui.write_run_settings_yaml(
                tmp, args, {'CO2': 'co2', 'N2O': 'n2o'}, {}, {},
                {'CO2': (200.0, 900.0)})
            data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        # N2O had no range in the run, so it must come back as none, not
        # seeded from its unit when the file is loaded into the TUI.
        self.assertEqual(data['validrange'], 'CO2:[200,900],N2O:[,]')


if __name__ == '__main__':
    unittest.main()
