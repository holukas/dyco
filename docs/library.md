# Library API

The classes below support the lag detection and can also be used on their own. The flux detection
limit also has a command, `dyco fdl`; [Flux detection limit](fdl.md) is its guide.

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

Main parameters:

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
al. (2015). It reads the noise from the far tails of the cross-covariance function between the
vertical wind and the scalar. That function is computed for all lags at once by FFT
(`dyco.maxcov.cross_covariance`), with whole-period means and divided by the number of overlapping
pairs (Langford et al. 2015, eq. 4).

```python
from dyco.detectionlimit import FluxDetectionLimit

fdl = FluxDetectionLimit(
    df=df, u_col='u', v_col='v', w_col='w', c_col='N2O',
    ts_col='Ts', h2o_col='H2O', press_col='Pressure',
    default_lag=1.0,          # seconds, the lag the flux is read at; positive = gas arrives after w
    noise_range=20,           # seconds, width of the far-tail noise windows
    lag_range=[-180, 180],    # seconds, how far out the covariance is computed
    lag_stepsize=10,          # records
    sampling_rate=10,         # Hz
)
fdl.run()
results = fdl.get_detection_limit()
fdl.flux_at_lag(2.0)          # the flux at any other lag on the grid, in seconds
```

At very large lags, wind and scalar can no longer be physically related, so the covariance found
there is noise. The scatter of those far-tail covariances gives an RMSE, and
the detection limit is three times it (Langford et al. 2015). [Flux detection limit](fdl.md) gives
the equations.

`default_lag=None` computes the detection limit only; `flux_signal_at_default_lag`,
`signal_to_noise` and `signal_to_detection_limit` are then `None`.
`flux_at_lag(lag_s)` reads the flux at any lag after `run()`, which is how `dyco.pwbfdl` reads it at
a lag that is only known later. The lag must fall on the grid set by `lag_stepsize`: with
`lag_stepsize=10` at 10 Hz, whole seconds only, and anything else raises a `ValueError`. Use
`lag_stepsize=1` to allow every record. `verbose=False` silences the run time, noise and detection
limit that `run()` otherwise prints.

`get_detection_limit()` returns a dict:

| Key | What it is |
|---|---|
| `flux_detection_limit` | The smallest flux distinguishable from noise: `3 * flux_noise_rmse`. |
| `flux_noise_rmse` | Scatter of the covariances in the far-lag windows, plus their offset from zero (Langford et al. 2015, eq. 9). |
| `flux_signal_at_default_lag` | The flux read at `default_lag`. This is the signal. |
| `flux_signal_at_cov_max_lag` | The flux at the largest absolute covariance anywhere in `lag_range` instead. |
| `cov_max_shift` / `cov_max_ix` | Where that peak sits, in records and as a row index. |
| `signal_to_noise` | `abs(signal) / flux_noise_rmse`. |
| `signal_to_detection_limit` | `abs(signal) / flux_detection_limit`. Below 1 means the flux for this period is not distinguishable from noise. |

**Units follow your scalar.** Feed it N₂O in nmol mol⁻¹ and the detection limit comes back in
nmol m⁻² s⁻¹; feed it CO₂ in µmol mol⁻¹ and you get µmol m⁻² s⁻¹. The conversion uses the ideal gas
law with the mean air temperature and dry-air pressure, so sonic temperature must be in K, H₂O in
mol mol⁻¹ and pressure in Pa.

`get_fig_cov()` returns the covariance figure after `run()`, which is the
quickest way to see whether the far tail really is flat. `examples/detectionlimit.py` runs the whole
thing on synthetic data.

## Time lag and detection limit together

`dyco.pwbfdl` runs PWB and the detection limit on the same averaging periods and reads each flux at
the PWBOPT lag. It is what `dyco fdl` runs. [Flux detection limit](fdl.md) is the guide: the method,
the units, a worked example, the output columns and how to read them.

```python
from dyco.pwbfdl import PwbFdl, PwbFdlBatch
```

`PwbFdlBatch(files, scalars, col_u, col_v, col_w, col_tsonic, col_h2o, col_press, ...)`
: Raw files on disk, given in time order. Cuts them into periods like `dyco detect-remove`, runs
  `PwbFdl` on each in parallel, applies PWBOPT across the sequence and reads the flux at the final
  lag. `run(on_progress=None)` returns one row per period as a DataFrame and, with `output_dir`
  set, writes `pwb_fdl_results.csv`. The result is also on `.results`. On Windows, guard the calling
  script with `if __name__ == '__main__':` when `n_workers` is not 1.

`PwbFdl(df, scalars, col_u, col_v, col_w, col_tsonic, col_h2o, col_press, ...)`
: One period already in memory, as raw, unrotated data. `run()` returns one row as a dict, with the
  flux read at that period's own PWB lag. `flux_at_lag(label, lag_s)` reads the flux at any other
  lag; `flux_curve(label)` returns it at every lag from `-fdl_lag_range_s` to `+fdl_lag_range_s`,
  one value per record. `.pwb[label]` and `.fdl[label]` hold the `PreWhiteningBootstrap` and
  `FluxDetectionLimit` objects.

The parameters both share:

`scalars`
: `{label: column}`. Result columns are prefixed with the label in lower case.

`scale`, `tsonic_unit`
: `{column: factor}`, multiplied in before the detection limit only, and `'K'` or `'degC'`. The
  detection limit needs sonic temperature in K, H₂O in mol mol⁻¹ and the ambient pressure in Pa.
  `PwbFdl.run()` raises `ValueError` on implausible means; `PwbFdlBatch` records the period as
  `error` unless `strict=True`.

`hz`, `lag_max_s`, `lws`, `uws`, `n_bootstrap`, `block_length_s`, `wdt`, `random_state`
: The PWB search, as for `PreWhiteningBootstrap`.

`gas_lag_overrides`
: `{label: {'lag_max_s': ..., 'block_length_s': ..., 'lws': ..., 'uws': ...}}`, a search window
  for one gas that differs from the global one. Unset keys take the global value.

`valid_ranges`
: `{label: (min, max)}`, in the column's own, unscaled units.

`fdl_lag_range_s`, `noise_range_s`
: The outer edge of the lag range (default 180 s) and the width of each noise window at its ends
  (default 20 s).

`PwbFdlBatch` adds the file layout (`skiprows`, `extra_rows`, `sep`, `na_values`), the chunking
(`chunk_seconds`, `min_chunk_seconds`, `start_time_regex`, `start_time_format`), PWBOPT
(`hdi_thresh`, `dev_thresh`, `hdi_prefilter`, `max_carry`, and `lag_fallback={label: donor}`), and
`n_workers`, `output_dir` and `strict`. Unlike the command line, `extra_rows` defaults to `0` and
`start_time_format` to `'%Y%m%d%H%M'`.

`examples/pwb_fdl_realdata.py` runs the batch on a bundled 6-hour CH-CHA file.

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
