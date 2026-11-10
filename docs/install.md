# Installation

Requires Python 3.12 or 3.13.

```bash
pip install dyco
```

## From a checkout

Working on `dyco` itself, or running the bundled examples, needs the repository:

```bash
git clone https://github.com/holukas/dyco.git
cd dyco
uv sync
```

Prefix the commands below with `uv run` when working this way.

## Checking the install

```bash
dyco
```

That prints the list of workflows. `dyco --version` prints the installed version. `dyco <command>
--help` lists the options of one workflow, for example `dyco fdl --help`.

## Using only the flux detection limit

The [flux detection limit](fdl.md) needs the same install, `pip install dyco`; there is no lighter
variant or optional extra. Every dependency (pandas, numpy, scipy, matplotlib, pyarrow, textual,
pyyaml) is installed with it, including those of the terminal UI, which the detection limit does not
use. It runs on the command line as `dyco fdl` (or the standalone `dyco-fdl`, see
[the command reference](cli/fdl.md)), or from Python with `from dyco.pwbfdl import PwbFdlBatch` for
raw files, or `PwbFdl` for one period already in memory.

To keep it apart from other Python projects, install it into a virtual environment. With the
standard library:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux and macOS: source .venv/bin/activate
pip install dyco
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
uv pip install dyco
uv run dyco fdl --help
```

`dyco fdl --help` prints every option of the detection-limit run.

:::{important}
**On Windows, a script that runs `PwbFdlBatch` with more than one worker needs the
`if __name__ == '__main__':` guard** around the code that starts the run. The workers are separate
processes, and Windows starts each one by importing your script again; without the guard each of
them starts the batch over. `n_workers=1` needs no guard. The command line handles this itself.
:::

## Running without data

```bash
dyco tui --demo
```

The terminal UI runs a synthetic pipeline and needs no input files at all. A real 1-hour raw file
from CH-LAE (Lägeren) also ships with the repository (a checkout, not the PyPI package), so the
full pipeline can be exercised without supplying your own data:

```bash
uv run python examples/detect_remove_tlag_realdata.py
```

That detects and removes the tube delay for CO₂ and H₂O across two 30-minute chunks, and prints what
was detected against what was actually applied. Set `DYCO_OUT` to a path to keep the results. Expect
about 20 s. `examples/detect_remove_tlag.py` is the synthetic counterpart, where the lag is known
in advance and can be checked.
