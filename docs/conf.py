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

# The PWB flowchart has more nodes than fit legibly in the content column, so
# it needs to be readable at something other than the width Sphinx gives it.
# Zoom pulls in d3 from a CDN; mermaid itself is already loaded that way, so
# this adds no new class of dependency. The fullscreen button is on by default
# and is the better route for readers who dislike scroll-to-zoom.
mermaid_d3_zoom = True
mermaid_height = '700px'

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

# ---------------------------------------------------------------------------
# The parallel read
# ---------------------------------------------------------------------------

# `sphinx-argparse` declares itself safe to read in parallel and registers a domain that does not
# implement `merge_domaindata`, so Sphinx splits the read across workers and then dies merging what
# they produced. Read the Docs builds with `-j auto`, so this is not hypothetical.
#
# Declaring the extension unsafe instead only moves the failure: Sphinx then warns that it is
# unsafe and that it is reading serially, and `fail_on_warning` turns both warnings into the error
# that fails the build. Neither carries a type that `suppress_warnings` could reach.
#
# So the method is supplied. The domain keeps two collections, and every entry in both is a tuple
# whose fourth element is the document it came from -- exactly what a merge needs: take the entries
# belonging to the documents this worker read, and leave the rest.
#
# None of this is visible from a Windows checkout. Parallel reading needs `os.fork`, so
# `sphinx.util.parallel.parallel_available` is False here and a local build is serial whatever it
# is asked for. `.github/workflows/tests.yml` builds with `-j auto` on Linux for that reason.
#
# Remove all of it once the extension implements the method itself. Learned in `fluxatlas`, which
# hit this on the hosted build and nowhere else.
def _merge_argparse_domaindata(self, docnames, otherdata):
    docnames = set(docnames)
    for entry in otherdata.get('commands', ()):
        if entry[3] in docnames:
            self.data['commands'].append(entry)
    for group, entries in otherdata.get('commands-by-group', {}).items():
        kept = [entry for entry in entries if entry[3] in docnames]
        if kept:
            self.data['commands-by-group'].setdefault(group, []).extend(kept)


def setup(app):
    from sphinx.domains import Domain
    from sphinxarg.ext import ArgParseDomain

    # Only where the extension still inherits the base class's `raise NotImplementedError`, so a
    # released fix upstream wins over this one rather than being shadowed by it.
    if ArgParseDomain.merge_domaindata is Domain.merge_domaindata:
        ArgParseDomain.merge_domaindata = _merge_argparse_domaindata
