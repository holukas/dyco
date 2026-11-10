"""
TEST_PWBFDL_CLI: THE `dyco fdl` COMMAND
=======================================

`dyco fdl` and `dyco-fdl` run end to end on a small synthetic raw file written
to a temporary directory, read and written for real.

Part of the dyco package: https://github.com/holukas/dyco
"""

import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')

from dyco import cli
from dyco.pwbfdl import _cli_main, _scale_token

HZ = 10


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


def _write_raw(folder: Path) -> Path:
    """Three 5-minute periods in one file: a header row and one units row."""
    df = _synthetic(3 * 300 * HZ, {'ch4': 15, 'n2o': 20}, seed=4)
    path = folder / 'SITE_202107221100.csv'
    with open(path, 'w', newline='') as fh:
        fh.write(','.join(df.columns) + '\n')
        fh.write(','.join(['[unit]'] * df.shape[1]) + '\n')
        df.to_csv(fh, header=False, index=False)
    return path


def _args(inp: Path, out: Path, *extra) -> list:
    return [
        '--input-dir', str(inp), '--output-dir', str(out),
        '--col-u', 'u', '--col-v', 'v', '--col-w', 'w', '--col-tsonic', 'ts',
        '--col-h2o', 'h2o', '--col-press', 'p',
        '--extra-rows', '1', '--start-time-regex', r'(\d{12})',
        '--start-time-format', '%Y%m%d%H%M',
        '--hz', str(HZ), '--chunk-seconds', '300', '--min-chunk-seconds', '60',
        '--n-bootstrap', '9', '--lag-max', '5', '--fdl-lag-range', '60',
        '--noise-range', '10', '--n-workers', '1', '--random-state', '7',
        *extra]


SCALE = ['--scale', 'ch4=1e3', '--scale', 'n2o=1e3',
         '--scale', 'h2o=1e-6', '--scale', 'p=100']


def _run(argv: list, via_dispatcher: bool) -> tuple:
    """Run the command, return (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with redirect_stdout(out), redirect_stderr(err):
        try:
            if via_dispatcher:
                cli.main(['fdl', *argv])
            else:
                with mock.patch.object(sys, 'argv', ['dyco-fdl', *argv]):
                    _cli_main()
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
    return code, out.getvalue(), err.getvalue()


class TestFdlCli(unittest.TestCase):

    def test_dispatcher_run_with_a_per_gas_window(self):
        """The global window excludes the true lags; N2O's own window does not."""
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write_raw(tmp)
            out = tmp / 'out'
            code, stdout, _ = _run(_args(
                tmp, out, '--scalar', 'CH4:ch4@lagfrom=N2O',
                '--scalar', 'N2O:n2o@lws=0;uws=5', '--lws', '-5', '--uws', '-1',
                '--max-carry', '2', *SCALE), via_dispatcher=True)
            self.assertEqual(code, 0, stdout)
            res = pd.read_csv(out / 'pwb_fdl_results.csv')
            log = (out / 'log.txt').read_text(encoding='utf-8')
            for gas in ('ch4', 'n2o'):
                for suffix in ('.png', '.pdf'):
                    self.assertTrue((out / f'fdl_{gas}{suffix}').is_file(), gas + suffix)

        self.assertEqual(list(res['period']),
                         ['2021-07-22 11:00', '2021-07-22 11:05', '2021-07-22 11:10'])
        self.assertTrue(res['status'].eq('ok').all())
        # N2O searched 0..5 s and found its true 2.0 s; CH4 kept the global -5..-1 s.
        np.testing.assert_allclose(res['n2o_tlag_s'], 2.0, atol=0.1)
        self.assertTrue(res['ch4_tlag_s'].between(-5, -1).all())
        self.assertTrue(res['n2o_above_fdl'].astype(bool).all())

        self.assertIn('started : ', log)
        self.assertIn('lag from N2O where CH4 has none', log)
        self.assertIn('finished: ', log)
        for period in res['period']:
            self.assertIn(period, log)
        self.assertIn('N2O lag 2.00 s [own]', stdout)
        self.assertIn('(HDI ', stdout)
        # The end summary goes to the log as well as the console.
        for text in ('Summary', 'above : 3 of 3 period(s) (100.0 %) with abs(flux) above',
                     'per period: median', 'FDL of the mean', 'own detections: 3, HDI',
                     'day 06-18 h', 'check : flux / RE'):
            self.assertIn(text, stdout)
            self.assertIn(text, log)
        self.assertTrue(stdout.isascii())

    def test_standalone_entry_point(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write_raw(tmp)
            out = tmp / 'out'
            code, stdout, _ = _run(_args(tmp, out, '--scalar', 'N2O:n2o', *SCALE),
                                   via_dispatcher=False)
            self.assertEqual(code, 0, stdout)
            res = pd.read_csv(out / 'pwb_fdl_results.csv')
            self.assertTrue((out / 'log.txt').is_file())
        self.assertEqual(len(res), 3)
        self.assertNotIn('ch4_fdl', res.columns)
        self.assertTrue((res['n2o_fdl'] > 0).all())

    def test_a_unit_mistake_is_one_clear_error(self):
        """Pressure left in hPa fails every period the same way: say it once."""
        scale = SCALE[:-2]
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write_raw(tmp)
            out = tmp / 'out'
            code, _, stderr = _run(_args(tmp, out, '--scalar', 'N2O:n2o', *scale),
                                   via_dispatcher=True)
            self.assertTrue((out / 'log.txt').is_file())
        self.assertEqual(code, 1)
        self.assertIn('every period failed', stderr)
        self.assertIn('pressure column', stderr)
        self.assertIn('--scale COLUMN=FACTOR', stderr)
        self.assertNotIn('Traceback', stderr)

    def test_unknown_column_is_refused_before_the_run(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write_raw(tmp)
            out = tmp / 'out'
            code, _, stderr = _run(
                _args(tmp, out, '--scalar', 'N2O:n2o', *SCALE, '--scale', 'nope=2'),
                via_dispatcher=True)
            self.assertFalse(out.exists())
        self.assertEqual(code, 1)
        self.assertIn("--scale 'nope'", stderr)

    def test_scale_parsing(self):
        self.assertEqual(_scale_token('PRESS_BOX_[IRGA75-A]=100'),
                         ('PRESS_BOX_[IRGA75-A]', 100.0))
        # A column name may itself contain "=": the factor follows the last one.
        self.assertEqual(_scale_token('a=b=1e-6'), ('a=b', 1e-6))
        with TemporaryDirectory() as tmp:
            code, _, stderr = _run(_args(Path(tmp), Path(tmp) / 'out', '--scalar',
                                         'N2O:n2o', '--scale', 'n2o=lots'),
                                   via_dispatcher=True)
        self.assertEqual(code, 2)
        self.assertIn('must be a number', stderr)

    def test_help(self):
        code, stdout, _ = _run(['--help'], via_dispatcher=True)
        self.assertEqual(code, 0)
        self.assertIn('usage: dyco-fdl', stdout)
        self.assertIn('pwb_fdl_results.csv', stdout)
        self.assertIn('--fdl-lag-range', stdout)


if __name__ == '__main__':
    unittest.main()
