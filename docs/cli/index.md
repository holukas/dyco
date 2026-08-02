# Command reference

Everything is reachable through one command:

```bash
dyco                    # list the workflows
dyco <command> --help   # options for one of them
```

| Command | Does | When |
|---|---|---|
| [`dyco tui`](tui.md) | `detect-remove` behind a form, with live validation and a preflight check. | **Start here.** |
| [`dyco detect-remove`](detect-remove.md) | **The main command.** Split long raw files into averaging-period chunks, rotate, detect the lag per chunk, then remove it. One pass, one output folder. | Almost always. Scripting, or when you prefer a command line. |
| [`dyco pwb-batch`](pwb-batch.md) | Detect only, on files that are *already* split into averaging periods. Writes `tlag_results.csv` and stops. | Step 1 of the two-step route. |
| [`dyco apply-batch`](apply-batch.md) | Remove lags listed in an existing `tlag_results.csv`. Detects nothing. | Step 2 of the two-step route. |

Each also exists standalone: `dyco-detect-remove`, `dyco-detect-remove-tui`, `dyco-pwb-batch`,
`dyco-apply-batch`.

Everything else in `dyco` is a library API with no command of its own: the file splitter, the flux
detection limit, the covariance-maximization estimator.

:::{note}
The pages below are generated from the argparse parsers themselves, so they always match what
`--help` prints.
:::

```{toctree}
:maxdepth: 1

detect-remove
pwb-batch
apply-batch
tui
```

## Taking a gas's lag from another gas

A trace gas can be too noisy to locate its own lag. When that happens PWBOPT rejects the detections,
and the last resort is the *median of those same rejected numbers*, which on real data is often a
negative lag no tube can produce. A reference gas travelling the same tube is a better answer.

Say so per gas, with `@lagfrom=` (**Lag from** in the TUI):

```bash
dyco detect-remove ... --scalar "CO2:CO2_DRY_[IRGA72-A]" --scalar "N2O:N2O_DRY_[QCL-C2]@lagfrom=CO2"
```

N₂O keeps every lag it *detects and PWBOPT accepts* (S1 or S2), and it keeps its own lag carried
forward from an earlier period for as long as that lag is allowed to travel — see the carry limit
below. Only past that does it take the CO₂ lag, **for that same period**, so a donor lag that drifts
is followed rather than flattened into a constant. The summary records the choice for every period
in `{gas}_lag_source` (`own`, `from:CO2`, `median`), so a borrowed lag is never mistaken for a
detected one.

The gas's own lag comes first at both tiers on purpose. Two gases down one tube still have different
delays — a systematic 0.35 s between CH₄ and N₂O is ordinary — so borrowing swaps a stale number for
a biased one. It is worth doing once the gas's own lag is old enough that staleness is the bigger
error, and that is a judgement the carry limit expresses.

Chains work (`CH4` from `N2O` from `CO2`); circular ones are rejected.

## Limiting how far a lag may be carried

PWBOPT's S3 rule gives a period with no usable detection the nearest earlier optimal lag, with no
limit on the distance — one good half hour can supply the rest of a week. `--max-carry N`
(**Max carry** in the TUI) caps it at N averaging periods. Beyond that the lag expires
(`{gas}_flag_* = S3_expired`) and the period falls through to the donor gas, or to the median.

`{gas}_carry_periods` reports the distance for every period: `0` where the lag was detected in that
very period, `n` where it travelled `n` periods, empty where it came from somewhere else entirely.

The default is unlimited, which is the published behaviour. It also means a donor is nearly idle:
with no limit the gas carries its own lag forever and only the periods before its first detection
are left to borrow. **`@lagfrom=` and `--max-carry` are meant to be set together**, and dyco warns
when a donor is named without one.

## The two-step route

`pwb-batch` to detect, then `apply-batch`, where `--scalar LABEL:column` reads the lag of one gas and
shifts the column of another. It applies when your files are already split into averaging periods by
other software, leaving nothing for `detect-remove` to chunk.

:::{important}
PWB detection needs **wind-rotation-corrected** high-frequency data. `dyco detect-remove` handles
this itself. Files fed to `dyco pwb-batch` must already be rotated (double rotation or planar fit,
e.g. EddyPro "Advanced" rotated output). A non-zero mean `W` corrupts the cross-correlation.
:::

## Input file formats

| Flag | Handles |
|---|---|
| `--sep` | Field separator. `,` by default; `\t` for TSV, `\s+` for whitespace-aligned. |
| `--skiprows` | Metadata lines **before** the column-name row. `0` for a plain CSV with names on line 1; `9` for EddyPro rotated output. |
| `--extra-rows` | Rows **after** the header but before the data, such as units and instrument tags. Default `2`. They are preserved byte-for-byte in the output. |
| `--na-values` / `--na-rep` | What counts as missing on the way in, what is written for it on the way out. |
| `--lineterm` | `auto` reproduces the input's CRLF or LF. Force it with `\r\n` or `\n`. |
| `--file-pattern` | Which files to read. Compression is transparent: `.gz`, `.bz2`, `.xz` and `.zip` are read as the text they contain. |
| `--output-suffix` | The extension the written chunks carry, dot included. Give the whole thing (`.csv`, `.csv.gz`, `.dat.zip`), just the text format (`.csv`, which writes plain text and drops any compression), or just the compression (`.zip`, which keeps the input's text format, so `file1.csv` gives `file1.csv.zip`). `auto` (default) reuses the input's own extension. A suffix without the leading dot is refused, so `.csv` is written the same way as `.csv.gz`. |

Two limits worth knowing. This path reads **delimited text only**; Parquet is read by
`dyco.files.read_raw_data`, which serves the file splitter, not this pipeline. And it needs **no
data-timestamp column**: chunking is driven by `--hz` and record count, and the wall-clock time comes
from the filename via `--start-time-regex`.
