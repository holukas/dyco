"""
TEST_PACKAGING: RELEASE METADATA
=================================

The version is stated in three files that nothing else keeps in step: `pyproject.toml`,
`CITATION.cff` and the top entry of `CHANGELOG.md`. Zenodo builds its record from the second and
PyPI from the first, so a release that bumps one and forgets another is archived under a number it
does not carry. These tests are the thing that notices.

Skipped entirely when the repository root is not present, so an installed wheel can still run the
suite without failing on files a wheel does not ship.

Part of the dyco package: https://github.com/holukas/dyco
"""

import re
import tomllib
from datetime import datetime
from importlib.metadata import version as installed_version
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (_REPO_ROOT / 'pyproject.toml').is_file(),
    reason='not running from a source tree',
)


def _pyproject() -> dict:
    with open(_REPO_ROOT / 'pyproject.toml', 'rb') as f:
        return tomllib.load(f)


def _citation() -> dict:
    return yaml.safe_load((_REPO_ROOT / 'CITATION.cff').read_text(encoding='utf-8'))


def _changelog_heading() -> re.Match:
    """The first `## vX.Y.Z | <when>` line in CHANGELOG.md."""
    text = (_REPO_ROOT / 'CHANGELOG.md').read_text(encoding='utf-8')
    match = re.search(r'^##\s+v(?P<version>\S+)\s*\|\s*(?P<when>.+?)\s*$', text, re.MULTILINE)
    assert match, 'CHANGELOG.md has no `## vX.Y.Z | <when>` heading'
    return match


def test_installed_distribution_matches_pyproject():
    assert installed_version('dyco') == _pyproject()['project']['version']


def test_citation_matches_pyproject():
    # CITATION.cff writes the version unquoted, so YAML may hand back a float for a number like
    # 3.0 -- compare as text.
    assert str(_citation()['version']) == _pyproject()['project']['version']


def test_changelog_heads_the_current_version():
    assert _changelog_heading()['version'] == _pyproject()['project']['version']


def test_release_dates_agree():
    """Once the changelog entry is dated, CITATION.cff has to carry the same date.

    An entry still marked `unreleased` is the ordinary state during development and says nothing
    about `date-released`, so there is nothing to compare yet.
    """
    when = _changelog_heading()['when']
    if 'unreleased' in when.lower():
        pytest.skip('the current entry is not released yet')

    released = _citation().get('date-released')
    assert released is not None, (
        'CHANGELOG.md dates this version but CITATION.cff has no `date-released`'
    )
    # `date-released: 2026-08-06` parses as a date; the changelog writes `6 Aug 2026`.
    if isinstance(released, str):
        released = datetime.strptime(released, '%Y-%m-%d').date()
    assert datetime.strptime(when, '%d %b %Y').date() == released
