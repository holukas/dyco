"""
CLI: UNIFIED `dyco` COMMAND
============================

One entry point that dispatches to every dyco workflow::

    dyco detect-remove ...   split, detect and remove in one pass  (primary)
    dyco tui ...             the same pipeline behind a terminal UI
    dyco pwb-batch ...       detect only, across pre-split files
    dyco apply-batch ...     remove lags from an existing results CSV

Each delegates to the entry point that backs the standalone ``dyco-*`` console
script, which keeps working unchanged - this module only adds a single
discoverable front door. Delegation rewrites ``sys.argv`` rather than
re-declaring each sub-parser, so the delegated parsers stay the one definition
of their own options and ``dyco detect-remove --help`` prints exactly what
``dyco-detect-remove --help`` does.

[BREAKING, v3] The covariance-maximization method is gone, and with it the v2
CLI. Up to v2 the top-level command took short flags (``-lsw``, ``-lsi``,
``-lsf`` ...) directly and drove that method. Pre-whitening block-bootstrap
replaces it: use ``dyco detect-remove``. There is no flag-for-flag mapping - the
two methods take different parameters. The old spellings are still recognized
here, only so an old command line gets a pointer instead of a parse error.

Part of the dyco package: https://github.com/holukas/dyco
"""

import sys

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

def _usage() -> str:
    lines = ['dyco - dynamic lag compensation', '',
             'usage: dyco <command> [options]', '', 'commands:']
    width = max(len(k) for k in _DELEGATED)
    for name, (_, _, _, summary) in _DELEGATED.items():
        lines.append(f'  {name:<{width}}  {summary}')
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
# Dispatcher
# ---------------------------------------------------------------------------

# Flags the v2 CLI took at the top level. Seeing one here means an old command
# line, which deserves a pointer rather than "unknown command".
_V2_FLAGS = {'-fnd', '-fnp', '-flim', '-fgr', '-fdur', '-dtf', '-dres',
             '-lss', '-lsw', '-lsi', '-lsf', '-lsp', '-lt', '-del'}

_CM_GONE = (
    "The covariance-maximization method was removed in v3.0.0. Pre-whitening "
    "block-bootstrap replaces it: `dyco detect-remove`. Its parameters are not a "
    "renaming of the old ones - the two methods differ in what they take. See the "
    "CHANGELOG and `dyco detect-remove --help`.")


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
        sys.exit(f"dyco: {_CM_GONE}")

    if command.startswith('-'):
        hint = ''
        if command in _V2_FLAGS or any(a in _V2_FLAGS for a in argv):
            hint = f"\n\nThat looks like a v2 command line. {_CM_GONE}"
        sys.exit(f"dyco: expected a command, got the option {command!r}.{hint}\n\n{_usage()}")

    sys.exit(f"dyco: unknown command {command!r}.\n\n{_usage()}")


if __name__ == '__main__':
    main()
