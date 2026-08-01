"""
CLI: UNIFIED `dyco` COMMAND
============================

One entry point that dispatches to every dyco workflow::

    dyco detect-remove ...   split, detect and remove in one pass  (primary)
    dyco tui ...             the same pipeline behind a terminal UI
    dyco pwb-batch ...       detect only, across pre-split files
    dyco apply-batch ...     remove lags from an existing results CSV
    dyco cm ...              the covariance-maximization method (retained)

The first four delegate to the entry points that back the standalone
``dyco-*`` console scripts, which keep working unchanged - this module only
adds a single discoverable front door. Delegation rewrites ``sys.argv`` rather
than re-declaring each sub-parser, so the delegated parsers stay the one
definition of their own options and ``dyco detect-remove --help`` prints exactly
what ``dyco-detect-remove --help`` does.

[BREAKING, v3] The v2 CLI is gone. It took short flags (``-lsw``, ``-lsi``,
``-lsf`` ...) directly at the top level and drove the covariance-maximization
method. Its work now lives under the ``cm`` subcommand with long flag names, and
the boolean options that were ``0``/``1`` integers are real switches. It also
could not run on pandas 3 at all: its ``'30T'`` frequency defaults raise
``ValueError: invalid unit abbreviation: T``, since pandas removed the ``T``
alias in favour of ``min``.

Old (v2)                        New (v3)
-----------------------------   ---------------------------------------
dyco REF LAG TGT -i in -o out   dyco cm REF LAG TGT -i in -o out
  -fnd FMT                        --filename-date-format FMT
  -fnp PATTERN                    --file-pattern PATTERN
  -flim N                         --limit-files N
  -fgr 30T                        --file-generation-res 30min
  -fdur 30T                       --file-duration 30min
  -dtf FMT                        --timestamp-format FMT
  -dres 0.05                      --nominal-timeres 0.05
  -lss 30T                        --segment-duration 30min
  -lsw 1000                       --lag-winsize 1000
  -lsi 3                          --n-iterations 3
  -lsf 1 / -lsf 0                 --remove-fringe-bins / --no-remove-fringe-bins
  -lsp 0.9                        --perc-threshold 0.9
  -lt 0                           --target-lag 0
  -del 1                          --delete-previous

Part of the dyco package: https://github.com/holukas/dyco
"""

import argparse
import sys
from pathlib import Path

# Subcommand -> (module, entry function, prog name for --help, one-line summary).
# The module is imported lazily so that, for example, `dyco detect-remove` does
# not pull in Textual.
_DELEGATED = {
    'detect-remove': ('dyco.pipeline', '_cli_main', 'dyco-detect-remove',
                      'Split raw files into chunks, detect the lag per chunk, remove it. Primary.'),
    'tui': ('dyco.tui', '_tui_main', 'dyco-detect-remove-tui',
            'Terminal UI over detect-remove. --demo needs no input data.'),
    'pwb-batch': ('dyco.pwb', '_cli_main', 'dyco-pwb-batch',
                  'Detect lags only, across many already-split files.'),
    'apply-batch': ('dyco.apply_tlag', '_cli_main', 'dyco-apply-batch',
                    'Remove lags listed in an existing tlag_results.csv.'),
}

_CM_SUMMARY = 'Covariance-maximization method (the v2 workflow). Retained, not extended.'


def _usage() -> str:
    lines = ['dyco - dynamic lag compensation', '',
             'usage: dyco <command> [options]', '', 'commands:']
    width = max(len(k) for k in list(_DELEGATED) + ['cm'])
    for name, (_, _, _, summary) in _DELEGATED.items():
        lines.append(f'  {name:<{width}}  {summary}')
    lines.append(f'  {"cm":<{width}}  {_CM_SUMMARY}')
    lines += ['', 'Run `dyco <command> --help` for a command\'s options.',
              '', 'Each command is also available standalone:',
              '  dyco-detect-remove, dyco-detect-remove-tui, dyco-pwb-batch, dyco-apply-batch']
    return '\n'.join(lines)


def _delegate(module_name: str, func_name: str, prog: str, argv: list):
    """Run another module's CLI entry point with *argv* as its arguments.

    The delegated entry points call ``parse_args()`` with no arguments, so they
    read ``sys.argv``. Swapping it here means their parsers - which are the
    tested, documented definition of each command's options - need no changes,
    and their ``--help`` still names the standalone command.
    """
    from importlib import import_module
    entry = getattr(import_module(module_name), func_name)
    saved = sys.argv
    sys.argv = [prog] + list(argv)
    try:
        return entry()
    finally:
        sys.argv = saved


# ---------------------------------------------------------------------------
# cm - the retained covariance-maximization workflow
# ---------------------------------------------------------------------------

def _build_cm_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='dyco cm',
        description=_CM_SUMMARY + ' Lags are in number of records, not seconds.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('var_reference', type=str,
                   help='Column name of the unlagged reference variable. Lags are determined '
                        'relative to this signal.')
    p.add_argument('var_lagged', type=str,
                   help='Column name of the lagged variable, whose lag relative to '
                        'var_reference is determined.')
    p.add_argument('var_target', nargs='+',
                   help='Column name(s) the detected lag is applied to. May include '
                        'var_lagged itself. Example: var1 var2 var3')

    p.add_argument('-i', '--input-dir', type=Path, required=True,
                   help='Folder holding the raw data files.')
    p.add_argument('-o', '--output-dir', type=Path, required=True,
                   help='Folder results are written to.')

    p.add_argument('--filename-date-format', type=str, default='%Y%m%d%H%M%S',
                   help='Datetime format of the timestamp in each filename. Files in the input '
                        'folder must carry datetime information in their name. Example for '
                        '20161015123000.csv: %%Y%%m%%d%%H%%M%%S')
    p.add_argument('--file-pattern', type=str, default='*.csv',
                   help='Glob selecting which files to process.')
    p.add_argument('--limit-files', type=int, default=0,
                   help='Use only the first N found files. 0 uses all of them.')
    p.add_argument('--file-generation-res', type=str, default='30min',
                   help='Interval at which files were generated, as a pandas frequency string.')
    p.add_argument('--file-duration', type=str, default='30min',
                   help='Duration of the data in one file, as a pandas frequency string.')

    p.add_argument('--timestamp-format', type=str, default='%Y-%m-%d %H:%M:%S.%f',
                   help='Datetime format of the timestamp on each data row. Example for '
                        '2016-10-24 10:00:00.024999: %%Y-%%m-%%d %%H:%%M:%%S.%%f')
    p.add_argument('--nominal-timeres', type=float, default=0.05,
                   help='Nominal time resolution in seconds per record. 0.05 for 20 Hz.')

    p.add_argument('--segment-duration', type=str, default=None,
                   help='Segment length for lag detection within a file. Must be shorter than '
                        'or equal to --file-duration. Defaults to --file-duration, i.e. one '
                        'lag per file.')
    p.add_argument('--lag-winsize', type=int, default=1000,
                   help='Initial half-width of the lag search window, in records.')
    p.add_argument('--n-iterations', type=int, default=3,
                   help='Number of window-narrowing iterations. Must be at least 1.')
    p.add_argument('--remove-fringe-bins', action=argparse.BooleanOptionalAction, default=True,
                   help='Drop the outermost bins of the found-lag histogram before narrowing.')
    p.add_argument('--perc-threshold', type=float, default=0.9,
                   help='Cumulative share of found lags the narrowed window must enclose. '
                        'Between 0.1 and 1.')
    p.add_argument('--target-lag', type=int, default=0,
                   help='Lag, in records, that all target variables are normalized to.')
    p.add_argument('--delete-previous', action='store_true',
                   help='Delete previous results in the output folder instead of continuing '
                        'from them.')
    return p


def _validate_cm(args, parser: argparse.ArgumentParser):
    """Check the value constraints argparse cannot express."""
    import pandas as pd

    if args.limit_files < 0:
        parser.error('--limit-files must be 0 or a positive integer.')
    if args.n_iterations < 1:
        parser.error('--n-iterations must be at least 1.')
    if not (0.1 <= args.perc_threshold <= 1):
        parser.error('--perc-threshold must be between 0.1 and 1.')

    # Unspecified segment duration means one lag per file.
    if args.segment_duration is None:
        args.segment_duration = args.file_duration

    # v2 compared these as strings, so '10min' > '30min' lexically and the check
    # fired on valid input while missing real mistakes. Compare durations.
    try:
        seg = pd.Timedelta(args.segment_duration)
        dur = pd.Timedelta(args.file_duration)
    except ValueError as e:
        parser.error(f'could not parse a duration: {e}. Note pandas 3 removed the "T" '
                     f'alias - use "30min", not "30T".')
        raise  # unreachable; parser.error exits
    if seg > dur:
        parser.error(f'--segment-duration ({args.segment_duration}) must be shorter than or '
                     f'equal to --file-duration ({args.file_duration}).')
    return args


def _run_cm(argv: list) -> None:
    parser = _build_cm_parser()
    args = _validate_cm(parser.parse_args(argv), parser)

    from dyco.dyco import Dyco
    Dyco(var_reference=args.var_reference,
         var_lagged=args.var_lagged,
         var_target=args.var_target,
         indir=args.input_dir,
         outdir=args.output_dir,
         filename_date_format=args.filename_date_format,
         filename_pattern=args.file_pattern,
         files_how_many=args.limit_files,
         file_generation_res=args.file_generation_res,
         file_duration=args.file_duration,
         data_timestamp_format=args.timestamp_format,
         data_nominal_timeres=args.nominal_timeres,
         lag_segment_dur=args.segment_duration,
         lag_winsize=args.lag_winsize,
         lag_n_iter=args.n_iterations,
         lag_hist_remove_fringe_bins=args.remove_fringe_bins,
         lag_hist_perc_thres=args.perc_threshold,
         target_lag=args.target_lag,
         del_previous_results=args.delete_previous)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Flags the v2 CLI took at the top level. Seeing one here means an old command
# line, which deserves a pointer rather than "unknown command".
_V2_FLAGS = {'-fnd', '-fnp', '-flim', '-fgr', '-fdur', '-dtf', '-dres',
             '-lss', '-lsw', '-lsi', '-lsf', '-lsp', '-lt', '-del'}


def main(argv: list = None) -> None:
    """Dispatch to a subcommand. See the module docstring."""
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ('-h', '--help', 'help'):
        print(_usage())
        return
    if argv[0] in ('-V', '--version'):
        from importlib.metadata import version, PackageNotFoundError
        try:
            print(version('dyco'))
        except PackageNotFoundError:
            print('unknown (dyco is not installed)')
        return

    command, rest = argv[0], argv[1:]

    if command in _DELEGATED:
        module, func, prog, _ = _DELEGATED[command]
        _delegate(module, func, prog, rest)
        return
    if command == 'cm':
        _run_cm(rest)
        return

    if command.startswith('-'):
        hint = ''
        if command in _V2_FLAGS or any(a in _V2_FLAGS for a in argv):
            hint = ("\n\nThat looks like a v2 command line. The v2 flags moved under the 'cm' "
                    "subcommand and were renamed in v3 - see `dyco cm --help`, or the "
                    "mapping table in the CHANGELOG.")
        sys.exit(f"dyco: expected a command, got the option {command!r}.{hint}\n\n{_usage()}")

    sys.exit(f"dyco: unknown command {command!r}.\n\n{_usage()}")


if __name__ == '__main__':
    main()
