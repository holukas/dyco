# Flux detection limit

`dyco fdl` determines, for each averaging period and each gas, whether the flux can be distinguished
from random noise. It computes two quantities per period:

- the **flux detection limit** (FDL): the smallest flux that stands out from the random error of
  that period, following Langford et al. (2015);
- the **flux** itself, read at the time lag that PWB and PWBOPT chose for that gas and period.

It writes one results table, `pwb_fdl_results.csv`, and no data files. Removing the lag from the raw
data is the job of [`dyco detect-remove`](cli/detect-remove.md); see
[Relation to dyco detect-remove](#relation-to-dyco-detect-remove).

The method is most useful for gases with a low signal-to-noise ratio, such as N₂O and CH₄ over many
ecosystems, where a half-hourly flux is often small compared with its random error.

## How it is computed

Each averaging period is processed on its own: the wind is double-rotated once, PWB detects the lag
of every gas, and the cross-covariance between the rotated vertical wind and each gas is computed at
every lag out to ±180 s. Then PWBOPT chooses the final lag of each gas across the whole sequence of
periods, as `dyco detect-remove` does, and the flux is read off that period's cross-covariance at
that lag.

### The cross-covariance

For a lag of `t` records, with `w` the rotated vertical wind and `c` the gas, both over the whole
averaging period of `n` records (Langford et al. 2015, eq. 4):

```text
cov(t) = 1/(n - |t|) * sum over i of (w[i] - mean(w)) * (c[i + t] - mean(c))
```

The means are those of the whole period (block averaging), and the sum runs over the `n - |t|` pairs
that overlap. Missing records are skipped, in the means and in each sum. A positive `t` means the gas
arrives after the wind, which is what a tube delay looks like, and is the same sign convention as
the PWB lag. The covariance is computed for all lags at once by FFT.

The covariance is turned into a flux with the molar density of dry air:

```text
flux(t) = cov(t) * p_d / (R * T_a)

e   = x_H2O * p              partial pressure of water vapour, Pa
p_d = mean(p - e)            dry-air partial pressure, Pa
T_a = mean(T_s / (1 + 0.32 * x_H2O))   air temperature from sonic temperature, K
R   = 8.314 J mol-1 K-1
```

The flux is in the gas's unit times m⁻² s⁻¹: a gas given as a dry mole fraction in nmol mol⁻¹ gives
a flux in nmol m⁻² s⁻¹. Sonic temperature, H₂O and pressure enter the conversion only through their
period means, so they need to be correct on average, not record by record.

### The random error and the detection limit

At lags far beyond any tube delay, the wind and the gas cannot still be physically related, so the
covariance found there is random error. dyco takes two windows at the ends of the lag range, by
default -180 to -160 s and +160 to +180 s, and combines the scatter and the offset from zero of the
flux in each (Langford et al. 2015, eq. 9):

```text
RE_RMSE = sqrt( 0.5 * ( sd_left**2 + mean_left**2 + sd_right**2 + mean_right**2 ) )

FDL = 3 * RE_RMSE
```

`sd_left` and `mean_left` are the standard deviation and the mean of `flux(t)` over every lag in the
left window, and likewise on the right. The factor 3 is the α of Langford et al. for the 99th
percentile and is fixed in dyco. The windows are those of Striednig et al. (2020); Langford et al.
use ±150 to 180 s in their example. `--fdl-lag-range` (default 180 s) sets the outer edge and
`--noise-range` (default 20 s) the width of each window.

The detection limit does not depend on the time lag. The flux does.

### Choice of lag for the flux

The usual way to find a lag is to take the position of the largest absolute covariance inside a
search window. For a strong flux that works. For a flux near zero it does not: the largest of many
noisy values tends to lie further from zero than a typical one, so the flux is pushed away from
zero, up or down. Over many periods, a gas with a true flux near zero then shows two groups of fluxes
on either side of zero and few close to it. Langford et al. (2015) call this mirroring (their Figs.
6, 8 and 9) and recommend reading low signal-to-noise fluxes at a prescribed lag instead.

dyco reads every flux at the lag PWBOPT chose. Where PWB detected the lag in that very period and
PWBOPT accepted it (S1, or S2 when it agrees with the preceding lag), the flux is read at that
detection; a narrow interval means the peak was clear in that period. Otherwise the lag comes from a
neighbouring period of the same gas, or from a donor gas, so the noise of the period itself played
no part in choosing it. In those periods it acts as the prescribed lag that Langford et al.
recommend, decided period by period rather than fixed for a whole campaign. See
[PWBOPT](method.md#pwbopt-s1-s2-s3) for how the lag is chosen.

The results also keep `{gas}_flux_at_tlag`, the flux at the period's own raw PWB detection before
PWBOPT. Comparing it with `{gas}_flux` shows how much the choice of lag changed the flux.

## Inputs and units

PWB works on correlations and does not depend on units. The detection limit does, because it
converts covariance into flux with the dry-air density. It needs:

| Input | Flag | Unit the FDL needs | Notes |
|---|---|---|---|
| Wind `u`, `v`, `w` | `--col-u` `--col-v` `--col-w` | m s⁻¹ | Unrotated. dyco rotates each period itself. |
| Sonic temperature | `--col-tsonic` | K | Or degC with `--tsonic-unit degC`. Also used by PWB. |
| H₂O | `--col-h2o` | mol mol⁻¹ (a mole fraction) | Converts sonic to air temperature, and gives the dry-air pressure. |
| Air pressure | `--col-press` | Pa | The **ambient** pressure, not an analyser cell pressure. |
| Each gas | `--scalar LABEL:column` | any mole fraction unit | Use the **dry** mole fraction. The unit you scale it to is the unit of the flux and the FDL. |

`--scale COLUMN=FACTOR` multiplies a column by a factor before the detection limit is computed.
Repeat it once per column. It applies to the FDL step only: PWB sees the columns as they are in the
file. Common factors:

| From | To | Factor |
|---|---|---|
| µmol mol⁻¹ (ppm) of a gas | nmol mol⁻¹ (ppb) | `1e3` |
| µmol mol⁻¹ of H₂O | mol mol⁻¹ | `1e-6` |
| mmol mol⁻¹ of H₂O | mol mol⁻¹ | `1e-3` |
| hPa | Pa | `100` |
| kPa | Pa | `1000` |

For example `--scale "N2O_DRY_[LGR-A]=1e3" --scale "H2O_[LGR-A]=1e-6" --scale "PRESS_BOX_[IRGA75-A]=100"`.

A gas's valid range (`@min=` and `@max=` in `--scalar`) stays in the column's own, unscaled units,
the same units as with `dyco detect-remove`. Values outside it are treated as missing for the lag
and for the detection limit alike.

**Implausible means are refused.** In each period, after scaling, the means of sonic temperature,
H₂O and pressure must lie in these ranges. A period where one does not gets `status` `error`, with a
message naming the column and the unit it needs. When every period fails this way, which is what a
missing `--scale` does, `dyco fdl` reports the error once and exits with status 1:

| Input | Accepted mean |
|---|---|
| Sonic temperature | 200 to 340 K |
| H₂O | 0 to 0.08 mol mol⁻¹ |
| Pressure | 40 000 to 110 000 Pa |

The checks exist because a unit error here does not otherwise cause a failure. H₂O in mmol mol⁻¹
instead of mol mol⁻¹ makes the dry-air pressure negative, and pressure in hPa instead of Pa makes
every detection limit a hundred times too small. Both would look like ordinary numbers in the
results.

The gas column is not checked. Without its `--scale`, the flux and the FDL come out in
µmol m⁻² s⁻¹ instead of nmol m⁻² s⁻¹, which is a different unit, not an error.

## Running the calculation

### On the command line

The flags are the same as for `dyco detect-remove` where the two overlap (file layout, chunking, the
PWB search, PWBOPT, the per-gas `--scalar` syntax), plus `--col-h2o`, `--col-press`, `--scale`,
`--tsonic-unit`, `--fdl-lag-range` and `--noise-range`. The `detect-remove` flags for written data
files do not exist here. Every option is listed in the [command reference](cli/fdl.md).

Before any period is processed, the command reads the header of the first file and stops if a column
named in `--col-*`, `--scalar` or `--scale` is not there. A misspelt `--scale` column would otherwise
be ignored silently.

The command for the bundled CH-CHA file, from the root of a repository checkout:

```bash
dyco fdl --input-dir examples/data --file-pattern "CH-CHA_*.csv.gz" --output-dir out_fdl --col-u "U_[R350-B]" --col-v "V_[R350-B]" --col-w "W_[R350-B]" --col-tsonic "T_SONIC_[R350-B]" --col-h2o "H2O_[LGR-A]" --col-press "PRESS_BOX_[IRGA75-A]" --scalar "N2O:N2O_DRY_[LGR-A]" --scalar "CH4:CH4_DRY_[LGR-A]@lagfrom=N2O" --scale "N2O_DRY_[LGR-A]=1e3" --scale "CH4_DRY_[LGR-A]=1e3" --scale "H2O_[LGR-A]=1e-6" --scale "PRESS_BOX_[IRGA75-A]=100" --extra-rows 2 --start-time-regex "(\d{12})" --start-time-format "%Y%m%d%H%M" --lws 0 --uws 10 --max-carry 4 --random-state 42
```

Reading it in groups:

| Flags | What they say |
|---|---|
| `--input-dir` `--file-pattern` `--output-dir` | Where the raw files are, which of them to take, where the results go. |
| `--col-u/v/w` `--col-tsonic` | The sonic anemometer. |
| `--col-h2o` `--col-press` | H₂O and the ambient pressure, needed for the dry-air density. Here H₂O from the LGR and the box pressure of the IRGA75. |
| `--scalar` | One per gas. `@lagfrom=N2O` lets CH₄ take N₂O's lag where it has none of its own. |
| `--scale` | Gases from µmol mol⁻¹ to nmol mol⁻¹, H₂O from µmol mol⁻¹ to mol mol⁻¹, pressure from hPa to Pa. |
| `--extra-rows` `--start-time-regex` `--start-time-format` | Two header rows after the column names (units, instrument), and the start time from the file name. |
| `--lws 0 --uws 10` | Search only lags of 0 to 10 s, the delays a closed-path tube can produce. |
| `--max-carry 4` | A gas's own lag may be carried across at most 4 periods (2 hours) before the donor's is used. Without a carry limit the donor is rarely used. |
| `--random-state` | Makes the bootstrap, and so the results, reproducible. |

PWBOPT carries lags from one period to the next, so the order of the periods affects the result.
Name the files so that they sort in time order; a timestamp at the start of the name does this.

### From Python

`PwbFdlBatch` does what the command does, on a list of files:

```python
from pathlib import Path
from dyco.pwbfdl import PwbFdlBatch

if __name__ == '__main__':
    batch = PwbFdlBatch(
        files=sorted(Path('examples/data').glob('CH-CHA_*.csv.gz')),   # in time order
        scalars={'N2O': 'N2O_DRY_[LGR-A]', 'CH4': 'CH4_DRY_[LGR-A]'},
        col_u='U_[R350-B]', col_v='V_[R350-B]', col_w='W_[R350-B]',
        col_tsonic='T_SONIC_[R350-B]',
        col_h2o='H2O_[LGR-A]',
        col_press='PRESS_BOX_[IRGA75-A]',
        hz=20,
        scale={'N2O_DRY_[LGR-A]': 1e3, 'CH4_DRY_[LGR-A]': 1e3,   # -> nmol mol-1
               'H2O_[LGR-A]': 1e-6,                             # -> mol mol-1
               'PRESS_BOX_[IRGA75-A]': 100},                    # -> Pa
        extra_rows=2,
        start_time_regex=r'(\d{12})', start_time_format='%Y%m%d%H%M',
        lws=0, uws=10,
        lag_fallback={'CH4': 'N2O'},   # the command line's @lagfrom=
        max_carry=4,
        random_state=42,
        output_dir='out_fdl',          # writes out_fdl/pwb_fdl_results.csv
    )
    results = batch.run()              # a DataFrame, one row per period
```

The keyword arguments follow the command-line flags, except for the per-gas settings of `--scalar`:
a donor is `lag_fallback={'CH4': 'N2O'}`, a valid range is `valid_ranges={'CH4': (0.17, 1000)}`, and
a search window of its own is `gas_lag_overrides={'H2O': {'lag_max_s': 30, 'uws': 25}}` (keys
`lag_max_s`, `block_length_s`, `lws`, `uws`). Two defaults differ from the command line:
`extra_rows` is `0` and `start_time_format` is `'%Y%m%d%H%M'`, so give both explicitly. A period
whose units are implausible gets `status` `error`; `strict=True` raises instead. `n_workers=1` runs
everything in the calling process, which is easier to debug. `run(on_progress=f)` calls
`f(completed, total, row)` as each period finishes.

:::{important}
**On Windows, keep the `if __name__ == '__main__':` guard** unless `n_workers=1`. The workers are
separate processes, and Windows starts each one by importing the script again. Without the guard,
every worker starts the whole batch again.
:::

`PwbFdl` takes the same arguments for one averaging period already in memory, as a DataFrame of raw,
unrotated data. It reads the flux at that period's own PWB lag, since without a sequence there is no
PWBOPT. The [Library API](library.md#time-lag-and-detection-limit-together) describes both classes
and their parameters. `examples/pwb_fdl_realdata.py` in a repository checkout runs the batch on the
CH-CHA file and prints the results per gas.

## Output

In `--output-dir`:

`pwb_fdl_results.csv`
: One row per averaging period, in time order. The columns are listed below.

`log.txt`
: The run header, one line per period, and the finish time, as for the other commands. Written by
  the command line; `PwbFdlBatch` writes the CSV only.

### Columns per period

| Column | What it is |
|---|---|
| `period` | The start of the period, `YYYY-MM-DD HH:MM`. Without a start time from the file name it is `{file stem}#{chunk index}`, for example `site_data#03`. |
| `file` | The raw file the period was cut from. |
| `chunk_index` | Position of the period within that file, from 0. |
| `timestamp` | The start of the period in ISO format. Empty without a start time from the file name. |
| `status` | `ok`; `skipped:short` (fewer records than `--min-chunk-seconds`); or `error` (see `error`). |
| `error` | What went wrong when `status` is `error`, or why a period was skipped. `worker lost: ...` means the worker process died, often from lack of memory. |
| `n_records` | Number of records in the period. |
| `theta_deg`, `phi_deg` | The two double-rotation angles, in degrees. |

### Columns per gas

`{gas}` is the gas label in lower case: `--scalar "N2O:..."` gives `n2o_fdl`.

| Column | What it is |
|---|---|
| `{gas}_lag_s` | **The lag the flux was read at**, in s: the PWBOPT lag (`{gas}_tlag_final_pf_s`) rounded to a whole record. Empty where the gas has no data in the period, or the period was skipped or failed. |
| `{gas}_lag_reason` | Why the lag is what it is, in words: detected here, carried, borrowed, no data. |
| `{gas}_fdl` | **The flux detection limit**, `3 * {gas}_fdl_noise_rmse`, in the gas's scaled unit times m⁻² s⁻¹. |
| `{gas}_fdl_noise_rmse` | The random error `RE_RMSE` from the far-lag windows, same unit. |
| `{gas}_flux` | **The flux** at `{gas}_lag_s`, same unit. Positive is upward. |
| `{gas}_flux_to_fdl` | `abs({gas}_flux) / {gas}_fdl`. |
| `{gas}_above_fdl` | `True` when `{gas}_flux_to_fdl` is above 1, `False` when not, empty when there is no ratio. |
| `{gas}_flux_at_tlag` | The flux at the period's own raw PWB lag (`{gas}_tlag_s`), before PWBOPT. |
| `{gas}_flux_to_fdl_at_tlag` | `abs({gas}_flux_at_tlag) / {gas}_fdl`. |
| `{gas}_n_valid` | Records of the gas that are present and inside its valid range. `0` also for periods that were skipped or failed. |
| `{gas}_n_out_of_range` | Records outside the gas's valid range, treated as missing. |
| `{gas}_error` | Text when PWB (`PWB: ...`) or the detection limit (`FDL: ...`) failed for this gas in an otherwise good period. A gas whose PWB failed still gets its detection limit, but no flux at its own lag. |

The PWB and PWBOPT columns are the same as in the `dyco detect-remove` summary and mean the same
thing; [What a run writes](output.md#the-summary-csv) describes them. They are `{gas}_tlag_s`,
`{gas}_hdi_lo_s`, `{gas}_hdi_hi_s`, `{gas}_hdi_range_s`, `{gas}_is_reliable`, `{gas}_tlag_pw_s`,
`{gas}_cov_pwb`, `{gas}_ar_order`, `{gas}_best_combination`, `{gas}_pwbopt_s_std`, `{gas}_flag_std`,
`{gas}_pwbopt_s_pf`, `{gas}_flag_pf`, `{gas}_tlag_final_s`, `{gas}_tlag_final_pf_s`,
`{gas}_lag_source` and `{gas}_carry_periods`. The columns about written data (`{gas}_lag_applied_s`,
`{gas}_applied_records`, `{gas}_status`) do not exist here, since nothing is written. For a period
that was skipped or failed, `{gas}_lag_source` reads `none`, as in `dyco detect-remove`; `no_data`
means the gas had no usable record in a period that was read.

## Interpreting the results

### Above and below the detection limit

`{gas}_above_fdl` is `True` when the flux of that period is larger, in absolute value, than its
detection limit: at the 99 % level it is distinguishable from random noise. That is a statement
about one period.

**A flux below the detection limit is still a measurement.** Langford et al. (2015) recommend
reporting it, with its detection limit, and keeping it in averages. Dropping the periods below the
limit, or setting them to zero, biases any average: the periods that remain are the ones where noise
happened to push the flux further from zero.

### Detection limit of an average

The random errors of different periods are independent, so they partly cancel in an average. The
detection limit of an average of `N` periods is (Langford et al. 2015, eq. 10):

```text
FDL_average = sqrt( sum of FDL_i**2 ) / N
```

Twelve periods that each have a detection limit of 1 nmol m⁻² s⁻¹ give a 6-hour average with a
detection limit of `sqrt(12) / 12`, about 0.29 nmol m⁻² s⁻¹. A gas that is below the limit in every
half hour can still have a daily or seasonal mean that is clearly above it. With the results file:

```python
import numpy as np
import pandas as pd

r = pd.read_csv('out_fdl/pwb_fdl_results.csv')
ok = r['ch4_flux'].notna() & r['ch4_fdl'].notna()
mean_flux = r.loc[ok, 'ch4_flux'].mean()
fdl_of_mean = np.sqrt((r.loc[ok, 'ch4_fdl'] ** 2).sum()) / ok.sum()
```

### Components of the random error

`RE_RMSE` is the **total random error** of the period's flux: the instrument noise and the random
sampling error of turbulence together, not the noise of the analyser alone. Langford et al. (2015,
Sects. 2.1 and 2.3) also estimate the instrument noise separately; dyco does not, so it does not
report how much of the limit is due to the analyser.

### Effect of a trend in the gas

dyco removes the period mean (block averaging) and does not detrend. If a gas drifts over the
period, the covariance at far lags keeps an offset from zero, and the `mean_left` and `mean_right`
terms make the detection limit larger (Langford et al. 2015, Fig. 2b). A period with a strong trend
therefore gets a conservative, larger limit. If your flux software detrends linearly, its flux for
the same period can differ from `{gas}_flux` for the same reason.

### Relation to final fluxes

`{gas}_flux` is the covariance at the chosen lag times the dry-air density. Nothing else is applied:
no spectral corrections, no quality flags, no storage term. Use it to compare with the detection
limit, which is computed the same way, and take final fluxes from your flux software.

## Worked example: CH-CHA

The repository checkout includes `examples/data/CH-CHA_202110081300.csv.gz`: six hours of 20 Hz data
from CH-CHA (Chamau, Switzerland), starting 8 October 2021 at 13:00, which gives twelve 30-minute
periods.

| Columns | Instrument | Unit in the file |
|---|---|---|
| `U_`, `V_`, `W_`, `T_SONIC_[R350-B]` | Sonic anemometer | m s⁻¹, K |
| `N2O_DRY_`, `CH4_DRY_`, `H2O_[LGR-A]` | LGR analyser | µmol mol⁻¹ |
| `PRESS_BOX_[IRGA75-A]` | Box pressure of an open-path IRGA, the ambient pressure | hPa |

The file has three header rows: column names, units and instrument, so `--extra-rows 2`.

The command is the one under [On the command line](#on-the-command-line). CH₄ takes N₂O's lag
because both are measured in the same LGR cell, through the same tube, and neither sticks to the
tube wall, so their delays are the same. N₂O detects well on this file and CH₄ does not.

`examples/pwb_fdl_realdata.py`, with 99 bootstrap resamples (the default) and `random_state=42`,
gives:

| | N₂O | CH₄ |
|---|---|---|
| Lag detection | Its own lag accepted in every period: 11 S1, 1 S2 | Never reliable; every lag borrowed from N₂O |
| Lag | 1.75 to 1.95 s | N₂O's |
| Flux, nmol m⁻² s⁻¹ | 1.0 to 2.8 | -4.7 to +5.2 |
| FDL, nmol m⁻² s⁻¹ | 0.3 to 1.5 | 1.7 to 18 |
| Periods above the FDL | 12 of 12 | 0 of 12 |

N₂O was emitted throughout, clearly above its detection limit in every half hour. CH₄ cannot be
distinguished from noise in any single half hour: its fluxes scatter on both sides of zero, and its
detection limit is larger than any of them. For CH₄, the six-hour average and its detection limit
(eq. 10 above) are the next quantities to examine.

The script lists CH₄ before N₂O; the command above lists N₂O first. Each gas's bootstrap seed
depends on its position in the list, so the two runs draw different resamples and their numbers can
differ slightly.

## Pitfalls

**An analyser's cell pressure is not the air pressure.** The LGR writes `PRESS_CELL_[LGR-A]`, about
60 hPa: the pressure inside the measurement cell. The detection limit needs the ambient pressure,
here the IRGA75 box pressure. A cell pressure fails the plausibility check in every period, so the
run ends with an error instead of writing wrong numbers.

**Densities are not mole fractions.** Open-path IRGAs often write CO₂ and H₂O as densities, in
mmol m⁻³ (in this file `CO2_CONC_` and `H2O_CONC_[IRGA75-A]`). `--scale` cannot turn a density into a
mole fraction, because that needs the temperature and pressure of each record. Convert such a column
before running dyco, or use a mole fraction from another instrument, as the example does with the
LGR's H₂O.

**An analyser logged faster than it measures repeats its values.** The LGR here measures at 10 Hz and
is logged at 20 Hz, so each value appears twice. That is fine. Give `--hz` as the rate of the file
(20), since lags are counted in records of the file.

**The search window matters for weak gases.** Without `--lws` and `--uws`, the search runs from
minus to plus `--lag-max`, and a weak gas can settle on a negative lag that no tube can produce. See
[Effect of the search window](method.md#effect-of-the-search-window).

**Do not give H₂O's lag to an inert gas.** H₂O interacts with the tube wall, so its lag is longer
than the travel time through the tube and changes with humidity. See
[Taking a gas's lag from another gas](cli/index.md#taking-a-gass-lag-from-another-gas).

**Runtime.** Most of the time is spent in PWB. The cross-covariance for the detection limit is
computed for all lags at once and adds little. Expect about the same time per period and gas as
`dyco detect-remove` takes, of the order of a second on one core, spread over `--n-workers`
processes.

**Memory.** Each worker holds one period at a time. The main process keeps, for every period and
gas, the flux at each lag inside the widest search window until PWBOPT has run: with `--lag-max 10`
at 20 Hz that is 401 values, about 3 kB, so a year of half hours for two gases needs about 100 MB.
If workers die, rows show `status` `error` and `worker lost` in `error`; lower `--n-workers`.

## Relation to dyco detect-remove

The two commands are independent; neither needs the other to have run first. Both cut files into
periods the same way and choose the lag with the same PWB detection and PWBOPT rules. On the same
files with the same settings, they choose the same lags up to the randomness of the bootstrap: they
derive their seeds from `--random-state` differently, so they draw different resamples. Reliable
detections agree closely; uncertain ones can differ.

:::{note}
**Planned.** The TUI does not compute the detection limit yet. A future option could compute it
within the PWB run of `dyco detect-remove` and the TUI, so that one run gives both the corrected
files and the detection limits.
:::

## References

Langford, B., Acton, W., Ammann, C., Valach, A., & Nemitz, E. (2015). Eddy-covariance data with low
signal-to-noise ratio: time-lag determination, uncertainties and limit of detection. *Atmospheric
Measurement Techniques*, 8(10), 4197-4213. <https://doi.org/10.5194/amt-8-4197-2015>

Striednig, M., Graus, M., Märk, T. D., & Karl, T. G. (2020). InnFLUX: an open-source code for
conventional and disjunct eddy covariance analysis of trace gas measurements: an urban test case.
*Atmospheric Measurement Techniques*, 13(3), 1447-1465. <https://doi.org/10.5194/amt-13-1447-2020>

Vitale, D., Fratini, G., Helfter, C., Hörtnagl, L., Kohonen, K.-M., Mammarella, I., Nemitz, E.,
Nicolini, G., Rebmann, C., Sabbatini, S., & Papale, D. (2024). A pre-whitening with block-bootstrap
cross-correlation procedure for temporal alignment of data sampled by eddy covariance systems.
*Environmental and Ecological Statistics*, 31, 219-244. <https://doi.org/10.1007/s10651-024-00615-9>
