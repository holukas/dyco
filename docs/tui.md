# The terminal UI

```bash
dyco tui
```

![The dyco terminal UI](https://raw.githubusercontent.com/holukas/dyco/refs/heads/main/images/dyco_tui_v3.0.0.png)

*Settings on the left, the run on the right. This is `dyco tui --demo`, which needs no data.*

**This is the recommended way to run dyco.** A detect-and-remove run takes on the order of thirty
settings: column names, file format, chunk length, search windows per gas, PWBOPT thresholds. Getting
one of them wrong is easy on a command line and obvious in a form. The TUI validates as you type,
scans your first file so you can pick column names from a list instead of typing bracketed names by
hand, and runs a preflight **Check** that reads a real file and reports what it found before anything
is processed. It saves and reloads settings, and every run writes its configuration back out, so a
run started in the TUI can be repeated exactly.

`dyco tui --demo` explores the interface with no data at all.

## Its relationship to the command line

The TUI drives one command, `dyco detect-remove`, which is the whole job in one pass. That covers the
normal case, including [taking a weak gas's lag from a strong one](cli/index.md#taking-a-gass-lag-from-another-gas).

**The CLI and the TUI take the same format settings.** The TUI is a front end over the same parser,
building its configuration from the identical arguments, so anything you can describe in the TUI you
can pass on the command line. The difference is that the TUI validates as you type and can scan a
file to show you its columns first.

Every TUI run writes a `detect_remove_tui_settings.yaml` next to its results, and the TUI can reload
it. Build the configuration interactively, then automate the command.

## Settings file location

The TUI's saved settings live at `~/.dyco/detect_remove_tui.yaml`.

:::{note}
An existing `~/.diive/` configuration from before the v3 migration will not be found.
:::

<!-- TODO: a walkthrough of the panes, the Check preflight and what it reports,
     and the per-gas window editor. Screenshots of each rather than one overview
     shot. -->
