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
