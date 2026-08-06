"""
CONSOLE: MINIMAL PROGRESS OUTPUT
=================================

Stand-ins for the console helpers dyco used to import from diive
(`core/utils/console.py`). diive's versions render through Rich and honour a
package-wide verbosity level; these keep the same call signatures and the same
verbosity thresholds but write plain text, so dyco pulls in no console library.

Verbosity levels match diive: 0 silent, 1 errors only, 2 progress (default),
3 debug.

Keep printed strings cp1252-safe (Windows stdout): use ASCII `->`, not an arrow
glyph.

Part of the dyco package: https://github.com/holukas/dyco
"""

import sys

VERBOSE_SILENT = 0
VERBOSE_ERROR = 1
VERBOSE_PROGRESS = 2
VERBOSE_DEBUG = 3

_VERBOSE = VERBOSE_PROGRESS


def set_verbosity(level: int) -> None:
    """Set the package-wide verbosity level."""
    global _VERBOSE
    _VERBOSE = level


def _emit(msg: str, min_level: int, verbose: int = None, stream=None) -> None:
    level = _VERBOSE if verbose is None else verbose
    if level >= min_level:
        print(msg, file=stream if stream else sys.stdout)


def info(msg: str, verbose: int = None) -> None:
    """Key progress or result. Shown at PROGRESS and above."""
    _emit(msg, VERBOSE_PROGRESS, verbose)


def success(msg: str, verbose: int = None) -> None:
    """Operation completed. Shown at PROGRESS and above."""
    _emit(msg, VERBOSE_PROGRESS, verbose)


def detail(msg: str, verbose: int = None) -> None:
    """Inner-loop detail. Shown at DEBUG only."""
    _emit(msg, VERBOSE_DEBUG, verbose)


def warn(msg: str, verbose: int = None) -> None:
    """Warning. Shown at ERROR and above, i.e. almost always."""
    _emit(f"(!)WARNING: {msg}", VERBOSE_ERROR, verbose, stream=sys.stderr)


def error(msg: str, verbose: int = None) -> None:
    """Error. Shown at ERROR and above, i.e. almost always."""
    _emit(f"(!)ERROR: {msg}", VERBOSE_ERROR, verbose, stream=sys.stderr)


def rule(title: str, verbose: int = None) -> None:
    """Section header. Shown at PROGRESS and above."""
    _emit(f"\n{'=' * 70}\n{title}\n{'=' * 70}", VERBOSE_PROGRESS, verbose)
