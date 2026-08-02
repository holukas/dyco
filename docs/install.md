# Installation

Requires Python 3.12 or 3.13.

```bash
git clone https://github.com/holukas/dyco.git
cd dyco
uv sync
```

`dyco` has no dependency on `diive`. Everything it needs is bundled.

## Checking the install

```bash
uv run dyco
```

That prints the list of workflows. `dyco --version` prints the installed version.

## Running without data

```bash
uv run dyco tui --demo
```

The terminal UI runs a synthetic pipeline and needs no input files at all. A real 1-hour raw file
from CH-LAE (Lägeren) also ships with the repository, so the full pipeline can be exercised without
supplying your own data:

```bash
uv run python examples/detect_remove_tlag_realdata.py
```

That detects and removes the tube delay for CO₂ and H₂O across two 30-minute chunks, and prints what
was detected against what was actually applied. Set `DYCO_OUT` to a path to keep the results. Expect
about a minute. `examples/detect_remove_tlag.py` is the synthetic counterpart, where the lag is known
in advance and can be checked.

<!-- TODO: PyPI install instructions once v3 is released. Until then the only
     route is a clone, because the published 2.0.3 is a different package. -->
