# Library API

These arrived in v3 and support the lag work, but are useful on their own. None has a command of its
own — import them.

:::{note}
`dyco/__init__.py` defines no public API. Import from the modules directly, as shown below.
:::

## Splitting and rotating raw files

`FileSplitter` / `FileSplitterMulti` divide long raw files into shorter time-based parts, optionally
applying double rotation and writing the turbulent departures alongside. Output as CSV (optionally
gzipped) or Parquet.

```python
from dyco.split import FileSplitterMulti

fsm = FileSplitterMulti(
    outdir='./splits',
    searchdirs='./raw',
    filename_pattern='*.csv',
    filename_date_format='CH-DAV_%Y%m%d%H%M.csv',   # how the time is written in the name
    file_generation_freq='1h',                      # one input file per hour
    data_split_duration='30min',                    # cut each into two
    data_nominal_res=0.05,                          # 20 Hz, in seconds per record
    splits_output_format='csv',
    compress_splits=True,                           # write .csv.gz
)
fsm.run()
```

The parameters worth knowing:

`filename_date_format` / `file_generation_freq`
: How to read the start time out of the filename, and how much data one file holds. Together they
  say where each split sits in wall-clock time.

`data_nominal_res`
: Seconds per record, not hertz. `0.05` is 20 Hz, `0.1` is 10 Hz.

`data_timestamp_format`
: The format of a timestamp column in the file, or `None`, which is the usual case. With `None` the
  index is rebuilt from the record count.

`splits_output_format` / `compress_splits`
: `'csv'` or `'parquet'`, and whether to gzip the result afterwards.

`rotation`, with `u_var` / `v_var` / `w_var` / `c_var`
: Rotate the wind and write the turbulent departures alongside the original columns.

`split_trim` / `split_trim_var` / `outfile_limit_n_rows` / `files_split_how_many`
: Drop incomplete splits, cap the rows per output file, and stop after the first *n* input files.

:::{note}
**This path reads Parquet; `detect-remove` does not.** The splitter goes through
`dyco.files.read_raw_data`, which handles both delimited text and `.parquet`. The detect-remove
pipeline has its own reader and takes delimited text only, compressed or not. So a Parquet raw file
can be split here but has to be written back out as CSV before the pipeline will read it.
:::

## Wind rotation

`WindDoubleRotation` and `reynolds_decomposition`: double rotation for sonic anemometer tilt
correction, and turbulent departures `x' = x - mean(x)`.

```python
from dyco.rotation import WindDoubleRotation, reynolds_decomposition

wr = WindDoubleRotation(u=df['u'], v=df['v'], w=df['w'])
w_prime = reynolds_decomposition(wr.w2)
c_prime = reynolds_decomposition(df['N2O'])
```

The rotation runs in the constructor, so the results are there as soon as you have the object:
`wr.u2`, `wr.v2` and `wr.w2` are the rotated components, and `wr.theta` and `wr.phi` are the two
angles in radians. After rotation `mean(v2)` and `mean(w2)` are both about zero, which is the point:
it separates mean transport from the turbulent part.

Rotation and Reynolds decomposition are deliberately separate steps. Rotating does not give you
fluctuations; you take those afterwards, from the rotated components.

The `detect-remove` pipeline uses this internally, in memory. Rotated data never reaches disk: it
feeds the lag search, and the lag is then applied to the original unrotated columns.

## Flux detection limit

`FluxDetectionLimit` estimates the smallest flux distinguishable from noise, following Langford et
al. (2015). It reads the noise from the far tail of the same cross-covariance function used for lag
detection, which is why it lives here.

```python
from dyco.detectionlimit import FluxDetectionLimit

fdl = FluxDetectionLimit(
    df=df, u_col='u', v_col='v', w_col='w', c_col='N2O',
    ts_col='Ts', h2o_col='H2O', press_col='Pressure',
    default_lag=1.0,          # seconds, the lag the flux is read at
    noise_range=20,           # seconds, width of the far-tail noise windows
    lag_range=[-180, 180],    # seconds, how far out the covariance is computed
    lag_stepsize=10,          # records
    sampling_rate=10,         # Hz
)
fdl.run()
results = fdl.get_detection_limit()
```

The idea is that at very large lags, wind and scalar cannot still be physically related, so whatever
covariance remains out there is noise. The scatter of those far-tail covariances gives an RMSE, and
the detection limit is three times it (Langford et al. 2015).

`get_detection_limit()` returns a dict:

| Key | What it is |
|---|---|
| `flux_detection_limit` | The smallest flux distinguishable from noise: `3 * flux_noise_rmse`. |
| `flux_noise_rmse` | Scatter of the covariances in the far-lag windows. |
| `flux_signal_at_default_lag` | The flux read at `default_lag`. This is the signal. |
| `flux_signal_at_cov_max_lag` | The flux at the covariance peak instead. |
| `cov_max_shift` / `cov_max_ix` | Where that peak sits, in records and as a row index. |
| `signal_to_noise` | `abs(signal) / flux_noise_rmse`. |
| `signal_to_detection_limit` | `abs(signal) / flux_detection_limit`. Below 1 means the flux for this period is not distinguishable from noise. |

**Units follow your scalar.** Feed it N₂O in nmol mol⁻¹ and the detection limit comes back in
nmol m⁻² s⁻¹; feed it CO₂ in µmol mol⁻¹ and you get µmol m⁻² s⁻¹. The conversion uses the ideal gas
law with the mean air temperature and dry air pressure, so the temperature, water vapour and pressure
columns all have to be real.

`get_fig_cov()` returns the covariance figure when `create_covariance_plot=True`, which is the
quickest way to see whether the far tail really is flat. `examples/detectionlimit.py` runs the whole
thing on synthetic data.

## Covariance maximization

`MaxCovariance` is the covariance-maximization lag estimator, which `FluxDetectionLimit` builds on.

```python
from dyco.maxcov import MaxCovariance

mc = MaxCovariance(
    df=df,
    var_reference='w',        # the wind
    var_lagged='N2O',         # the gas, whose lag is measured against it
    lgs_winsize_from=-1000,   # search window, in records, not seconds
    lgs_winsize_to=1000,
    shift_stepsize=1,
)
mc.run()
cov_df, peak = mc.get()
```

`get()` returns the covariance against lag as a DataFrame, plus the properties of the peak it
detected. The search window is in **records**, unlike the PWB path, which takes seconds throughout.

A positive lag means `var_lagged` arrives *later* than the reference, which is what a closed-path
tube delay looks like. `examples/maxcov.py` runs it end to end.

:::{warning}
This is not the removed v2 *workflow*. The estimator survives because the detection limit needs it;
the daily-median lookup table and target-lag normalization built on top of it are gone. See
[Migrating from v2](migrating-from-v2.md).
:::
