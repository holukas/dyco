# Worked example: a 20 Hz IRGA file

One hour of 20 Hz eddy covariance from CH-LAE (Laegeren, Switzerland), recorded on 25 July 2025 from
13:00. The instruments are an HS50 sonic and a closed-path IRGA72 measuring CO₂ and H₂O. The file is
`examples/data/CH-LAE_202507251300.csv.gz`, it ships with dyco, and it holds 72000 records in 1.4 MB
of gzip. It is a 1-hour excerpt of a 6-hour raw file, trimmed to keep the repository small.

It differs from the [10 Hz QCL example](example-qcl-10hz.md) in four ways: the input is compressed,
the header is three rows deep, there is no timestamp column, and one file becomes two averaging
periods.

## The raw file

```
U_[HS50-B],V_[HS50-B],W_[HS50-B],T_SONIC_[HS50-B],INC_X_[HS50-B],INC_Y_[HS50-B],...
[m+1_s-1],[m+1_s-1],[m+1_s-1],[K],[deg],[deg],...
[HS50-B],[HS50-B],[HS50-B],[HS50-B],[HS50-B],[HS50-B],...
0.23,-0.52,0.11,291.23,20.47,-1.29,...
0.23,-0.55,0.15,291.26,21.75,-2.75,...
```

The file has 19 columns and six are shown. The two gases sit further right, named
`H2O_DRY_[IRGA72-A]` and `CO2_DRY_[IRGA72-A]`.

| What the file shows | Setting |
|---|---|
| Three header rows: names, units, instrument | `--skiprows 0 --extra-rows 2`, both defaults |
| Comma-separated | `--sep ,` |
| Sonic columns are `U_[HS50-B]`, `V_[HS50-B]`, `W_[HS50-B]`, `T_SONIC_[HS50-B]` | `--col-u`, `--col-v`, `--col-w`, `--col-tsonic` |
| Gases are `CO2_DRY_[IRGA72-A]`, `H2O_DRY_[IRGA72-A]` | one `--scalar` each |
| `.csv.gz` | nothing. Use `--file-pattern "*.csv.gz"` to select it |
| No timestamp column | `--hz 20`, plus a start time read from the filename |

The instrument tags in brackets need no escaping. dyco takes column names as given, so
`--col-w "W_[HS50-B]"` works. The quotes are there for the shell, not for dyco.

:::{note}
`--extra-rows` counts the header rows after the names row. Three header rows means
`--extra-rows 2`, which is the default. The QCL example sets it to `0` because that file has only
the names row. Get this wrong and the units row is read as data, which shifts every column by one
record.
:::

## There is no timestamp column

Nothing in the data says when a record was measured. The time base is the filename plus `--hz`:

- `--start-time-regex "(\d{12})"` pulls `202507251300` out of the name, and
  `--start-time-format "%Y%m%d%H%M"` reads it as 2025-07-25 13:00. That is when record 1 was taken.
- Chunk boundaries are counted in records. `--chunk-seconds 1800` at 20 Hz is 36000 records, so the
  hour splits at record 36001 and the second chunk is labelled 13:30.

Both settings matter more here than in a file with timestamps. A wrong `--hz` puts the chunk boundary
in the wrong place and mislabels every output file, and no timestamp column exists to contradict it.

## One file, two averaging periods

`--chunk-seconds 1800` gives two chunks of exactly 36000 records. Detection runs on each separately,
and the double rotation is fitted separately too. On this hour the angles are not close:

| Chunk | θ | φ |
|---|---|---|
| 13:00 | -94.0° | -1.1° |
| 13:30 | -123.4° | +21.4° |

Half an hour apart, the wind direction swings 29° and the tilt angle moves with it. Treating a
multi-hour file as one period would average both away. This is why `dyco detect-remove` splits before
it detects.

## The search window

A closed-path tube delay is physically positive, so the search keeps only positive lags:
`--lws 0 --uws 10` inside a `--lag-max 10`.

H₂O gets a wider window of its own, `@lag=30;uws=30;block=60`. Water sticks to the tube wall and
comes off again, so its effective delay runs longer than the flow-through time and moves with
humidity and tube age. On this hour the wider window turns out to be unnecessary, since H₂O lands at
8.25 to 8.70 s against CO₂'s 8.45 s. It costs nothing, and it is what covers a humid afternoon.

Note the per-gas `block=60`. R's rule is a bootstrap block of `2 × lag_max`, so a window widened to
30 s needs a block widened to 60 s. The global `--block-length 20` matches the global `--lag-max 10`
in the same way.

## The command

```bash
dyco detect-remove --input-dir ./data --output-dir ./dyco_out --file-pattern "*.csv.gz" --col-u "U_[HS50-B]" --col-v "V_[HS50-B]" --col-w "W_[HS50-B]" --col-tsonic "T_SONIC_[HS50-B]" --scalar "CO2:CO2_DRY_[IRGA72-A]" --scalar "H2O:H2O_DRY_[IRGA72-A]@lag=30;uws=30;block=60" --hz 20 --lag-max 10 --lws 0 --uws 10 --block-length 20 --n-bootstrap 99 --skiprows 0 --extra-rows 2 --sep "," --chunk-seconds 1800 --start-time-regex "(\d{12})" --start-time-format "%Y%m%d%H%M" --chunk-name-template "CH-LAE_{starttime}{suffix}" --random-state 42
```

That takes about 8 seconds for the two chunks. `examples/detect_remove_tlag_realdata.py` runs the
same settings through `PerFilePipeline` if you would rather call the library.

## What the run found

| Chunk | CO₂ | HDI | H₂O | HDI |
|---|---|---|---|---|
| 13:00 | 8.45 s | 0.00 s | 8.70 s | 0.25 s |
| 13:30 | 8.45 s | 0.05 s | 8.25 s | 0.35 s |

Four detections, four S1 flags, `lag_source` of `own` throughout. Nothing is carried or borrowed.

CO₂'s HDI of 0.00 s in the first chunk is the strongest result the method can return. The 95%
interval has collapsed onto a single value, so the bootstrap replicates agree to the record. The AR
order there is 299, which means the filter removed a lot of autocorrelation and the peak was still
clean.

:::{note}
`--wdt` sets the smoothing width applied to each bootstrap CCF before its peak is read. The default
of 5 follows RFlux. The paper's rule is `hz/2 + 1`, which is 11 at 20 Hz. On this file the two
disagree:

| | CO₂ 13:00 | CO₂ 13:30 | H₂O 13:00 | H₂O 13:30 |
|---|---|---|---|---|
| `--wdt 5` | 8.45 s (0.00) | 8.45 s (0.05) | 8.70 s (0.25) | 8.25 s (0.35) |
| `--wdt 11` | 8.40 s (0.30) | 8.55 s (0.20) | 8.85 s (0.20) | 8.85 s (0.15) |

Wider smoothing tightens H₂O and makes it agree across both chunks. It loosens CO₂ and makes it
disagree. Every difference is at or under 0.30 s, well inside the 0.5 s reliability threshold, and
one hour is no basis for choosing. The command above uses the default, which is what the bundled
example script does.
:::

## What the run wrote

Two files, named for their own start times, in the same format and compression as the input:

```
2_lag_removed/CH-LAE_202507251300.csv.gz    699 KB
2_lag_removed/CH-LAE_202507251330.csv.gz    671 KB
```

Compression is a storage detail, not a format. dyco reads `.gz`, `.bz2`, `.xz` and `.zip` as the text
inside them. `--output-suffix` sets the whole output extension independently of the input, and it
defaults to reusing the input's, which is why gzip in gives gzip out. Use `--output-suffix .csv` for
plain text.

All three header rows come back byte for byte, units and instrument tags included, and the columns
dyco did not shift keep their values. Three details to know before you diff an output file against
its input:

- The last 169 records of `CO2_DRY_[IRGA72-A]` are `-9999`, and the last 174 of
  `H2O_DRY_[IRGA72-A]`. Shifting a column forward by 8.45 s at 20 Hz leaves 169 records at the end
  with nothing to draw from. The data is in the next chunk. This is the cost of removing a lag, not
  a defect.
- Missing values are normalized to `--na-rep`. This file writes `-9999.0` and dyco writes back
  `-9999`, in every column, including ones it never touched.
- Numbers are written in their shortest exact form. `18.439999999999998` in the input is `18.44` in
  the output. The float is the same and reads back identically, only the text is shorter.

So parse the output file rather than expecting it byte for byte.

## Where to go next

[What a run writes](output.md) covers the summary CSV, the decisions report and the diagnostic plots.
The [10 Hz QCL example](example-qcl-10hz.md) works through a harder case: three gases, one of which
almost never detects on its own, and a tube delay that changes mid-record.
