"""
TEST_CLI: THE UNIFIED `dyco` DISPATCHER
========================================

The dispatcher delegates by rewriting `sys.argv`, so these tests mostly check
that the right entry point is reached with the right arguments and that
`sys.argv` is restored afterwards - a leak there would corrupt whatever ran next.

The covariance-maximization method was removed in v3.0.0, so the dispatcher also
has to answer an old command line with a pointer rather than a parse error.

Part of the dyco package: https://github.com/holukas/dyco
"""

import sys
import unittest
from unittest import mock

from dyco import cli


class TestDispatch(unittest.TestCase):

    def test_no_args_prints_usage_and_exits_zero(self):
        with mock.patch('sys.stdout') as out:
            cli.main([])  # must not raise SystemExit
        printed = ''.join(c.args[0] for c in out.write.call_args_list if c.args)
        for expected in ('detect-remove', 'tui', 'pwb-batch', 'apply-batch'):
            self.assertIn(expected, printed)
        self.assertNotIn('cm ', printed)

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
        self.assertIn('detect-remove', msg)

    def test_cm_is_gone_and_says_so(self):
        # `cm` shipped in no release, but the removal is worth naming: a bare
        # "unknown command" would read as a typo, not as a method that went away.
        with self.assertRaises(SystemExit) as ctx:
            cli.main(['cm', 'W', 'CH4', 'CH4'])
        msg = str(ctx.exception)
        self.assertIn('covariance-maximization', msg)
        self.assertIn('detect-remove', msg)

