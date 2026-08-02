"""
CONF: SPHINX CONFIGURATION
===========================

Build configuration for the dyco documentation site.

The command reference is generated from the argparse parsers themselves, via
``sphinx-argparse`` pointed at the ``_build_parser`` function in each CLI
module. That is deliberate: there are ~77 flags across the four entry points,
and a hand-written copy of them would drift from ``--help`` within a release.

Part of the dyco package: https://github.com/holukas/dyco
"""

import sys
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# sphinx-argparse imports the CLI modules to reach their parsers. Putting the
# repository root on the path means the docs build against the working tree,
# installed or not.
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

project = 'dyco'
author = 'Lukas Hörtnagl'
copyright = '2020-2026, Lukas Hörtnagl'

# The version lives in pyproject.toml and nowhere else. Read it rather than
# repeat it -- another copy is another thing to forget at release time.
with open(_REPO_ROOT / 'pyproject.toml', 'rb') as _f:
    release = tomllib.load(_f)['project']['version']
version = release

# ---------------------------------------------------------------------------
# Sphinx
# ---------------------------------------------------------------------------

extensions = [
    'myst_parser',          # pages are Markdown, matching README and CHANGELOG
    'sphinxarg.ext',        # the generated command reference
    'sphinxcontrib.mermaid',  # the PWB workflow flowchart
]

myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

# Lets one page link to a heading on another, e.g. (cli/detect-remove.md#io).
myst_heading_anchors = 3

templates_path = []
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

html_theme = 'furo'
html_title = f'dyco {release}'
html_static_path = []
