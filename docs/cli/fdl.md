# dyco fdl

Compute the flux detection limit (Langford et al. 2015) for each averaging period and gas, and the
flux at the lag PWBOPT chose. Cuts raw files into periods, rotates the wind and detects the lag
the same way as [`dyco detect-remove`](detect-remove.md), then writes `pwb_fdl_results.csv`,
`log.txt` (ending with the numbers to report) and one figure per gas, `fdl_{gas}.png` and `.pdf`,
to `--output-dir`. It writes no lag-corrected data files.

Also available as the standalone `dyco-fdl`.

[Flux detection limit](../fdl.md) is the guide: what the numbers mean, the units each column needs,
what to report, two worked examples (the CH-OE2 one with figures), and the pitfalls.

:::{important}
The detection limit needs sonic temperature in K (or `--tsonic-unit degC`), H₂O as a mole fraction
in mol mol⁻¹, and the **ambient** pressure in Pa. Use `--scale COLUMN=FACTOR` to convert columns into
these units. A period whose means are implausible is recorded as an error, and when every period
fails that way the command reports the error once and exits with status 1.
:::

## All options

```{eval-rst}
.. argparse::
   :module: dyco.pwbfdl
   :func: _build_parser
   :prog: dyco fdl
```
