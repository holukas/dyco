# dyco detect-remove

**The main command.** Read raw files, cut them into averaging-period chunks, rotate each chunk in
memory, detect its time lag, remove that lag from the *unrotated* data, and write the result.

Also available as the standalone `dyco-detect-remove`.

## A complete command

This is the bundled CH-LAE example. It processes `examples/data/CH-LAE_202507251300.csv.gz`, one hour
of 20 Hz data, gzipped, with a 3-row header, and writes two 30-minute lag-corrected chunks:

```bash
dyco detect-remove --input-dir examples/data --output-dir ./dyco_out --file-pattern "*.csv.gz" --col-u "U_[HS50-B]" --col-v "V_[HS50-B]" --col-w "W_[HS50-B]" --col-tsonic "T_SONIC_[HS50-B]" --scalar "CO2:CO2_DRY_[IRGA72-A]" --scalar "H2O:H2O_DRY_[IRGA72-A]@lag=30;uws=30" --hz 20 --chunk-seconds 1800 --lag-max 10 --lws 0 --uws 10 --n-bootstrap 100 --skiprows 0 --extra-rows 2 --sep "," --start-time-regex "(\d{12})" --start-time-format "%Y%m%d%H%M" --chunk-name-template "CH-LAE_{starttime}{suffix}" --n-workers 4 --random-state 42
```

Reading it in groups:

| Flags | What they say |
|---|---|
| `--input-dir` `--output-dir` `--file-pattern` | Where the raw files are, where results go, which of them to take. |
| `--col-u/v/w` `--col-tsonic` | The four wind columns. `T_SONIC` is **required**, because PWB tries it as an alternative reference. |
| `--scalar LABEL:column` | One per gas, repeated. `LABEL` becomes the prefix in the results (`co2_tlag_s`). `@lag=30;uws=30` gives H₂O its own wider window, since sorption on the tube walls delays it beyond the dry gases. |
| `--hz` `--chunk-seconds` | Sampling rate, and the averaging period each file is cut into. |
| `--lag-max` `--lws` `--uws` | Search window in **seconds**. `0` to `10` keeps only positive lags, because a closed-path tube delay cannot be negative. |
| `--skiprows` `--extra-rows` `--sep` | The file format. See [Input file formats](index.md#input-file-formats). |
| `--start-time-regex` `--start-time-format` `--chunk-name-template` | Read the start time out of the filename so each output chunk can be named for its own wall-clock time. |
| `--n-workers` `--random-state` | Parallelism, and a seed that makes the bootstrap reproducible. |

## Per-gas search windows

Gases with different inlet geometry need different search windows. Each gas can have its own:

```bash
dyco detect-remove --scalar "CH4:ch4" --scalar "H2O:h2o@lag=30;uws=25" --lws 0 --uws 5
```

A positive-only window keeps only physical tube delays (a closed-path delay is always > 0). A
long-inlet gas such as H₂O can use a wider window than the dry gases in the same run, which matters
because EddyPro applies a single lag setting to all gases downstream. Keep the expected lag near the
middle of the window; detections pinned to a boundary are unreliable and are discarded.

## A valid range per gas

A gas can be given the range its values can physically take, in the column's own units:

```bash
dyco detect-remove --scalar "CO2:CO2_DRY_[IRGA72-B]@min=200;max=900" --scalar "H2O:H2O_DRY_[IRGA72-B]@min=0;max=40"
```

Either bound may be left out (`@min=0`). A value outside the range is not a measurement, so lag
detection treats it as missing, the same way it treats `Inf`: the gap is interpolated over before the
search. `{gas}_n_out_of_range` in the summary counts these values per period, and `log.txt` names
each gas that had any. A period where every value is outside the range is treated like a period with
no data (`no_data`).

The range exists because loggers can write values that pass every status check and are still wrong.
At CH-LAE in 2016 and 2017, a logger fault copied LI-7500 values into the LI-7200 record about once a
second, so the closed-path CO₂ column read about 15 instead of about 400 µmol mol⁻¹ in 3.9 % of the
rows. EddyPro removes such values with its absolute-limits test. Without a range, dyco searched for
the lag on the corrupted series.

**The range is for detection only.** The written files keep every value, out-of-range ones
included, shifted by the lag like the rest. A range never repairs the output: removing bad values
before the flux calculation is the job of the flux software, so keep its own limits switched on.

Suggested values, from EddyPro's absolute-limit defaults:

| Gas | Unit | Range |
|---|---|---|
| CO₂ | µmol mol⁻¹ | 200 to 900 |
| H₂O | mmol mol⁻¹ | 0 to 40 |
| CH₄ | µmol mol⁻¹ | 0.17 to 1000 |
| N₂O | µmol mol⁻¹ | 0.032 to 1000 |

They only make sense in these units. A density (mmol m⁻³) needs different limits. The TUI's
**Valid range** field fills these in for you from the units row of the first input file. A gas whose
unit it cannot identify, or whose unit is a density, gets `[,]`, which means no range.

:::{important}
Downstream flux processing must run with time-lag maximization **disabled**. The lag has already
been removed.
:::

## All options

```{eval-rst}
.. argparse::
   :module: dyco.pipeline
   :func: _build_parser
   :prog: dyco detect-remove
```
