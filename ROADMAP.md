# Roadmap

Planned work, grouped by the release it is meant for. What changed in past releases is in
`CHANGELOG.md`.

## Next patch release (3.1.1)

Small fixes that change results someone may already have, so each gets its own changelog entry.

- **Give the short period at a file's end its own label.** A file that runs a few records past its
  length ends in a sub-second period that is skipped but carries the next file's start time, so
  `period` is not unique in the results.
- **Derive the bootstrap seeds the same way in `dyco fdl` and `dyco detect-remove`.** Today the same
  files and `--random-state` can give slightly different lags in the two commands.
- **Align the `block_length_s` default of `PwbFdlBatch` with the command line.** The class uses
  `2 * lag_max_s`, the command line a fixed 20 s; they agree only at the default `--lag-max`.

## Next minor release (3.2.0)

- **A fixed lag per gas for `dyco fdl`**, for example `--scalar "CH4:...@lagfix=1.6"`, used where
  the gas has no reliable lag of its own. Langford et al. (2015) recommend a prescribed lag for
  low-SNR gases, and the CH-OE2 example shows the gap: without it, CH4 fluxes are read at the median
  of rejected detections.
- **A reference-gas mode**, for example `--lag-gas CH4 --lag-gas-for N2O`: PWB runs only on the
  reference gas, PWBOPT gives it a complete lag series, and the named gases are read at that series.
- **Report the instrument-noise part of the detection limit** (Langford et al. 2015, Sects. 2.1 and
  2.3), so a user can see how much of the limit is due to the analyser.
- **The detection limit in `dyco detect-remove` and the TUI**, as an option, so one run removes the
  lag and reports each period's limit. Phase 1 already holds the rotated chunk, and the FDL needs
  only that plus H2O and pressure.

## Validation

- **Compare the detection limit with innFLUX** (Striednig et al. 2020) on one period, as PWB is
  compared with RFlux.
- **Add an RFlux parity case with fill values read as missing.** The current real-data case keeps
  12 `-9999` values in CO2 as numbers, which is what a normal run never sees.

## Housekeeping

- **Report the v3.0.0 Zenodo failures to Zenodo support** with the `error_id`s, since the cause is
  still unknown.
- **Remove `example()` and `example2()` from `dyco/split.py`**, which use hard-coded local paths.
- **Replace the TUI screenshot**, which predates the Valid range field.
- **Fold `dyco/files.py` into the pipeline's raw-file reader** (`rawio` unification, part two).
  Opening compressed files is already shared; parsing is not.
- **Rotate the wind once per period in `PwbFdl`.** `FluxDetectionLimit` rotates it again with the
  same result.
- **Defer the matplotlib import in `dyco.pwb`**, about 0.2 s that every worker pays under Windows.
