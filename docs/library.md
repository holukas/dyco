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
```

<!-- TODO: a worked example, and the parameter list. Note that this path reads
     Parquet via dyco.files.read_raw_data, unlike the detect-remove pipeline,
     which is delimited text only. -->

## Wind rotation

`WindDoubleRotation` and `reynolds_decomposition`: double rotation for sonic anemometer tilt
correction, and turbulent departures `x' = x - mean(x)`.

```python
from dyco.rotation import WindDoubleRotation, reynolds_decomposition
```

The `detect-remove` pipeline uses this internally, in memory — rotated data never reaches disk.

## Flux detection limit

`FluxDetectionLimit` estimates the smallest flux distinguishable from noise, following Langford et
al. (2015). It reads the noise from the far tail of the same cross-covariance function used for lag
detection, which is why it lives here.

```python
from dyco.detectionlimit import FluxDetectionLimit
```

<!-- TODO: what it returns and how to read it. -->

## Covariance maximization

`MaxCovariance` is the covariance-maximization lag estimator, which `FluxDetectionLimit` builds on.

```python
from dyco.maxcov import MaxCovariance
```

:::{warning}
This is not the removed v2 *workflow*. The estimator survives because the detection limit needs it;
the daily-median lookup table and target-lag normalization built on top of it are gone. See
[Migrating from v2](migrating-from-v2.md).
:::
