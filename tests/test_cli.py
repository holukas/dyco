"""
TEST_CLI: THE UNIFIED `dyco` DISPATCHER
========================================

The dispatcher delegates by rewriting `sys.argv`, so these tests mostly check
that the right entry point is reached with the right arguments and that
`sys.argv` is restored afterwards - a leak there would corrupt whatever ran next.

The `cm` subcommand replaces the v2 top-level CLI, so its validation is checked
directly rather than through `Dyco`, which needs real data.

Part of the dyco package: https://github.com/holukas/dyco
"""

import contextlib
import io
import sys
import unittest
from unittest import mock

from dyco import cli


class TestDispatch(unittest.TestCase):

    def test_no_args_prints_usage_and_exits_zero(self):
        with mock.patch('sys.stdout') as out:
            cli.main([])  # must not raise SystemExit
        printed = ''.join(c.args[0] for c in out.write.call_args_list if c.args)
        for expected in ('detect-remove', 'tui', 'pwb-batch', 'apply-batch', 'cm'):
            self.assertIn(expected, printed)

    def test_each_subcommand_reaches_its_entry_point(self):
        for command, (module, func, prog, _) in cli._DELEGATED.items():
            with self.subTest(command=command):
                with mock.patch(f'{module}.{func}') as entry:
                    cli.main([command, '--flag', 'value'])
                entry.assert_called_once_with()

    def test_delegated_entry_sees_its_own_argv(self):
        seen = {}

        def capture():
            seen['argv'] = list(sys.argv)

        with mock.patch('dyco.pipeline._cli_main', side_effect=capture):
            cli.main(['detect-remove', '--input-dir', 'in', '--hz', '20'])
        # argv[0] is the standalone command name, so a delegated --help names it.
        self.assertEqual(seen['argv'],
                         ['dyco-detect-remove', '--input-dir', 'in', '--hz', '20'])

    def test_argv_is_restored_after_delegation(self):
        before = list(sys.argv)
        with mock.patch('dyco.pipeline._cli_main'):
            cli.main(['detect-remove', '--anything'])
        self.assertEqual(sys.argv, before)

    def test_argv_is_restored_even_when_the_entry_point_raises(self):
        before = list(sys.argv)
        with mock.patch('dyco.pipeline._cli_main', side_effect=SystemExit(2)):
            with self.assertRaises(SystemExit):
                cli.main(['detect-remove', '--bad'])
        self.assertEqual(sys.argv, before)

    def test_unknown_command_exits_nonzero(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(['nonsense'])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_a_v2_command_line_gets_a_migration_pointer(self):
        # The v2 CLI took these at the top level. Users will try them.
        with self.assertRaises(SystemExit) as ctx:
            cli.main(['-lsw', '1000', '-lsi', '3'])
        msg = str(ctx.exception)
        self.assertIn('v2 command line', msg)
        self.assertIn('cm', msg)


class TestCmSubcommand(unittest.TestCase):

    BASE = ['cm', 'W', 'CH4', 'CH4', '-i', 'in', '-o', 'out']

    def _run(self, extra):
        with mock.patch('dyco.dyco.Dyco') as dyco_cls:
            cli.main(self.BASE + extra)
        return dyco_cls.call_args.kwargs

    def test_defaults_reach_dyco(self):
        kw = self._run([])
        self.assertEqual(kw['var_reference'], 'W')
        self.assertEqual(kw['var_target'], ['CH4'])
        self.assertEqual(kw['lag_n_iter'], 3)
        self.assertEqual(kw['lag_hist_remove_fringe_bins'], True)
        self.assertEqual(kw['del_previous_results'], False)

    def test_segment_duration_defaults_to_file_duration(self):
        kw = self._run(['--file-duration', '60min'])
        self.assertEqual(kw['lag_segment_dur'], '60min')

    def test_boolean_flags_are_real_switches(self):
        # v2 took 0/1 integers for these.
        self.assertFalse(self._run(['--no-remove-fringe-bins'])['lag_hist_remove_fringe_bins'])
        self.assertTrue(self._run(['--delete-previous'])['del_previous_results'])

    def test_segment_longer_than_file_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._run(['--file-duration', '30min', '--segment-duration', '60min'])

    def test_durations_are_compared_as_durations_not_strings(self):
        # v2 compared these as strings, so '10min' > '30min' lexically and this
        # valid combination was rejected.
        kw = self._run(['--file-duration', '30min', '--segment-duration', '10min'])
        self.assertEqual(kw['lag_segment_dur'], '10min')

    def test_pandas_2_frequency_alias_is_rejected_with_a_hint(self):
        # pandas 3 removed the 'T' alias; the v2 defaults still used '30T'.
        # argparse.error() writes to stderr and raises SystemExit(2), so the
        # message is not on the exception - capture the stream.
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            self._run(['--file-duration', '30T'])
        self.assertIn('30min', err.getvalue())

    def test_out_of_range_values_are_rejected(self):
        for bad in (['--n-iterations', '0'], ['--perc-threshold', '2'],
                    ['--limit-files', '-1']):
            with self.subTest(bad=bad):
                with self.assertRaises(SystemExit):
                    self._run(bad)


if __name__ == '__main__':
    unittest.main()
