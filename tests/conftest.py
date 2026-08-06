"""
CONFTEST: SHARED TEST FIXTURES
==============================

Keeps the test suite out of the developer's own dyco configuration.

Part of the dyco package: https://github.com/holukas/dyco
"""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope='session')
def isolate_tui_settings(tmp_path_factory):
    """Point the TUI's settings file at a throwaway location for the session.

    ``DetectRemoveTUI`` loads ``~/.dyco/detect_remove_tui.yaml`` on mount and
    writes it on Save and at the start of every run, so without this the suite
    both reads and overwrites whatever the developer last had in the TUI. Both
    directions caused real trouble: ``test_tui_win_field_autosync`` failed
    because a saved ``CH4:[0,10]`` window was loaded instead of the default,
    and a test that pressed Save replaced a real configuration.

    Session-scoped and autouse: no test has any business touching the real
    file, and a test that wants a settings file of its own can write into the
    same temporary directory.
    """
    try:
        from dyco import tui
    except Exception:
        yield           # textual not installed; nothing to isolate
        return
    original = tui._SETTINGS_PATH
    tui._SETTINGS_PATH = Path(tmp_path_factory.mktemp('dyco_home')) / 'tui.yaml'
    try:
        yield
    finally:
        tui._SETTINGS_PATH = original
