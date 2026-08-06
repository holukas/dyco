# dyco tui

Terminal UI over [`dyco detect-remove`](detect-remove.md). See [The terminal UI](../tui.md) for what
it does and why it is the recommended entry point.

Also available as the standalone `dyco-detect-remove-tui`.

## Options

```
usage: dyco tui [-h] [--demo]

Textual UI for dyco-detect-remove. Pass --demo to preview the interface
without input data.

options:
  -h, --help  show this help message and exit
  --demo      Run a synthetic pipeline (no input data required).
```

<!-- Written by hand, unlike the other command pages. dyco/tui.py builds its
     parser inside _tui_main() rather than in a module-level factory, so
     sphinx-argparse cannot reach it. Two flags is not worth refactoring the
     module for; if that parser is ever extracted, replace this block with the
     .. argparse:: directive used on the other pages. -->
