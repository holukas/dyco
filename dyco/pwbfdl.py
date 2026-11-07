"""
PWBFDL: TIME LAG AND FLUX DETECTION LIMIT, TOGETHER
===================================================

Detect each gas's time lag with pre-whitening block-bootstrap (PWB) and compute
its flux detection limit (FDL, Langford et al. 2015) from the same averaging
period, then read the flux at the lag PWB found.

Two classes:

- ``PwbFdl`` works on one averaging period that is already in memory. It
  rotates the wind once, runs PWB for every gas, computes each gas's detection
  limit, and reads the flux at the lag PWB detected in that period.
- ``PwbFdlBatch`` works on raw files on disk, e.g. 6-hour files. It cuts them
  into averaging periods the way ``dyco detect-remove`` does, runs ``PwbFdl`` on
  each period in parallel, chooses the lag across the whole sequence with
  PWBOPT (S1/S2/S3, Vitale et al. 2024), and reads the flux at that lag.

On the command line, ``dyco fdl`` (or ``dyco-fdl``) runs ``PwbFdlBatch``; see
``dyco fdl --help``.

The two are computed together because the signal depends on the lag and the
detection limit does not. The detection limit is read from the far tail of the
cross-covariance (by default +/-160 to 180 s). Langford et al. recommend reading a low-SNR flux at a
prescribed lag rather than at the covariance maximum, which is biased away from
zero (their Fig. 6). The PWBOPT lag is that prescribed lag, decided period by
period. The covariance curve is computed once per period and kept, so the
signal can be read at the final PWBOPT lag after the whole sequence has been
seen, without reading the data twice.

Units. PWB works on correlations and does not care about units. The detection
limit does: it converts covariance to flux with the molar density of dry air,
so it needs sonic temperature in K, H2O as a mole fraction (mol mol-1) and
pressure in Pa. ``scale`` multiplies columns into those units before the FDL
step only. The detection limit and the flux come out in the gas's own scaled
unit times m-2 s-1: a gas scaled to nmol mol-1 gives nmol m-2 s-1.

References:

    Langford, B., Acton, W., Ammann, C., Valach, A., & Nemitz, E. (2015).
        Eddy-covariance data with low signal-to-noise ratio: time-lag
        determination, uncertainties and limit of detection. Atmospheric
        Measurement Techniques, 8(10), 4197-4213.
        https://doi.org/10.5194/amt-8-4197-2015

    Vitale, D., Fratini, G., Helfter, C., Hortnagl, L., Kohonen, K.-M.,
        Mammarella, I., Nemitz, E., Nicolini, G., Rebmann, C., Sabbatini, S.,
        & Papale, D. (2024). A pre-whitening with block-bootstrap
        cross-correlation procedure for temporal alignment of data sampled by
        eddy covariance systems. Environmental and Ecological Statistics, 31,
        219-244. https://doi.org/10.1007/s10651-024-00615-9

Part of the dyco package: https://github.com/holukas/dyco
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pandas import DataFrame

from dyco.detectionlimit import FluxDetectionLimit
from dyco.pipeline import (
    _chunk_row_slice,
    _chunk_start_time,
    _count_data_rows,
    _lag_reason,
    _parse_file_start_time,
    _resolve_gas_lag,
    apply_pwbopt_columns,
    parse_scalar_spec,
    resolve_lag_fallback,
)
from dyco.pwb import (
    _DEFAULT_NA_VALUES,
    _WHITESPACE_SEP,
    _finite_or_nan,
    _out_of_range,
    _read_engine_kwargs,
    format_valid_range,
    PreWhiteningBootstrap,
)
from dyco.rawio import read_preserved_lines, split_header_line, strip_compression
from dyco.rotation import WindDoubleRotation

# Plausible ranges after `scale`, checked once per period. A unit mistake here
# does not fail loudly downstream: H2O in mmol mol-1 instead of mol mol-1 makes
# the dry-air pressure negative, and pressure in hPa instead of Pa shrinks every
# detection limit a hundredfold. Both look like ordinary numbers in the results.
_PLAUSIBLE = {
    'sonic temperature': (200.0, 340.0, 'K'),
    'H2O': (0.0, 0.08, 'mol mol-1'),
    'pressure': (40_000.0, 110_000.0, 'Pa'),
}

_PWB_KEYS = ('tlag_s', 'hdi_lo_s', 'hdi_hi_s', 'hdi_range_s', 'tlag_pw_s',
             'cov_pwb', 'ar_order')


def _gas_row_template(label: str) -> dict:
    p = label.lower()
    row = {f'{p}_{k}': np.nan for k in _PWB_KEYS}
    row.update({
        f'{p}_n_valid': 0, f'{p}_n_out_of_range': 0,
        f'{p}_is_reliable': False, f'{p}_best_combination': '',
        f'{p}_fdl': np.nan, f'{p}_fdl_noise_rmse': np.nan,
        f'{p}_flux_at_tlag': np.nan, f'{p}_flux_to_fdl_at_tlag': np.nan,
        f'{p}_error': '',
    })
    return row


_GAS_LAG_KEYS = {'lag_max_s', 'block_length_s', 'lws', 'uws'}


def _check_gas_lag_overrides(overrides: dict, scalars: dict) -> dict:
    """Validate ``{label: {lag_max_s, block_length_s, lws, uws}}``, as the pipeline does."""
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(scalars)
    if unknown:
        raise ValueError(
            f"gas_lag_overrides references unknown gas label(s) {sorted(unknown)}; "
            f"known scalars are {sorted(scalars)}.")
    for label, o in overrides.items():
        bad = set(o) - _GAS_LAG_KEYS
        if bad:
            raise ValueError(
                f"gas_lag_overrides[{label!r}] has unknown key(s) {sorted(bad)}; "
                f"allowed: {sorted(_GAS_LAG_KEYS)}.")
    return overrides


def _check_plausible(what: str, values: pd.Series, column: str) -> None:
    lo, hi, unit = _PLAUSIBLE[what]
    mean = float(np.nanmean(values.to_numpy(dtype=float)))
    if not lo <= mean <= hi:
        raise ValueError(
            f"{what} column {column!r} averages {mean:g} after scaling, outside "
            f"the plausible {lo:g} to {hi:g} {unit}. The detection limit needs "
            f"{unit}: set `scale` for this column (and `tsonic_unit` for sonic "
            f"temperature).")


class PwbFdl:
    """PWB time lag and flux detection limit for one averaging period.

    The wind is double-rotated once; PWB runs on the rotated W for every gas,
    then the Langford et al. (2015) detection limit is computed from the
    cross-covariance of the rotated W and the gas over ``+/-fdl_lag_range_s``.
    The flux is read at the PWB lag. A lag detected in this period alone is
    only the first answer: for a sequence of periods, ``PwbFdlBatch`` replaces
    it with the PWBOPT lag and reads the flux there instead (``flux_at_lag``).

    Parameters
    ----------
    df : DataFrame
        One averaging period of raw, **unrotated** high-frequency data.
    scalars : dict
        ``{label: column}``, e.g. ``{'CH4': 'CH4_DRY_[LGR-A]', 'N2O': 'N2O_DRY_[LGR-A]'}``.
        Result columns are prefixed with the lower-cased label.
    col_u, col_v, col_w : str
        Wind components, m s-1.
    col_tsonic : str
        Sonic temperature; K, or degC with ``tsonic_unit='degC'``.
    col_h2o : str
        H2O mole fraction. Needed to turn sonic into air temperature and for
        the dry-air density. Scaled to mol mol-1 by ``scale``.
    col_press : str
        Ambient air pressure (not an analyser cell pressure). Scaled to Pa.
    hz : int
        Sampling rate.
    scale : dict, optional
        ``{column: factor}``, multiplied in before the detection limit only,
        e.g. ``{'N2O_DRY_[LGR-A]': 1e3, 'H2O_[LGR-A]': 1e-6, 'PRESS_BOX_[IRGA75-A]': 100}``
        for umol mol-1 gases, umol mol-1 H2O and hPa. A gas's valid range
        stays in the column's own, unscaled units.
    tsonic_unit : {'K', 'degC'}
    lag_max_s, lws, uws, n_bootstrap, block_length_s, wdt
        PWB search settings; see ``PreWhiteningBootstrap``. ``lws=0, uws=10``
        keeps the search to physical tube delays, which matters for weak
        gases: an unwindowed search can settle on a negative lag.
    gas_lag_overrides : dict, optional
        ``{label: {lag_max_s, block_length_s, lws, uws}}``: a search window for
        one gas that differs from the global one, resolved exactly as in
        ``dyco detect-remove``. Unset keys fall back to the global value, and
        overriding ``lag_max_s`` without ``block_length_s`` sets the block to
        ``2 * lag_max_s``.
    valid_ranges : dict, optional
        ``{label: (min, max)}`` in the column's own units. Values outside are
        missing, for detection and for the detection limit alike.
    random_state : int, optional
        Seed for the bootstrap; each gas gets ``random_state + its position``.
    fdl_lag_range_s : int
        Half-width of the cross-covariance for the detection limit, s.
    noise_range_s : int
        Width of each noise window at the ends of that range, s. The defaults
        give +/-160 to 180 s, as in Striednig et al. (2020).
    name : str
        Label for this period, used in plots and messages.

    Example
    -------
    ::

        p = PwbFdl(df, scalars={'N2O': 'N2O_DRY_[LGR-A]'}, col_u='U', col_v='V',
                   col_w='W', col_tsonic='T_SONIC', col_h2o='H2O', col_press='P',
                   scale={'N2O_DRY_[LGR-A]': 1e3, 'H2O': 1e-6, 'P': 100},
                   lws=0, uws=10, random_state=42)
        row = p.run()
        row['n2o_tlag_s'], row['n2o_fdl'], row['n2o_flux_at_tlag']
    """

    def __init__(
            self,
            df: DataFrame,
            scalars: dict,
            col_u: str,
            col_v: str,
            col_w: str,
            col_tsonic: str,
            col_h2o: str,
            col_press: str,
            hz: int = 20,
            scale: dict | None = None,
            tsonic_unit: str = 'K',
            lag_max_s: float = 10.0,
            lws: float | None = None,
            uws: float | None = None,
            n_bootstrap: int = 99,
            block_length_s: float | None = None,
            wdt: int = 5,
            gas_lag_overrides: dict | None = None,
            valid_ranges: dict | None = None,
            random_state: int | None = None,
            fdl_lag_range_s: int = 180,
            noise_range_s: int = 20,
            name: str = 'period',
    ):
        if tsonic_unit not in ('K', 'degC'):
            raise ValueError(f"tsonic_unit must be 'K' or 'degC', got {tsonic_unit!r}.")
        if not 0 < noise_range_s < fdl_lag_range_s:
            raise ValueError(
                f"noise_range_s ({noise_range_s}) must be positive and smaller than "
                f"fdl_lag_range_s ({fdl_lag_range_s}).")
        required = [col_u, col_v, col_w, col_tsonic, col_h2o, col_press, *scalars.values()]
        missing = [c for c in dict.fromkeys(required) if c not in df.columns]
        if missing:
            raise ValueError(f"columns missing from {name}: {missing}")

        self.df = df
        self.scalars = dict(scalars)
        self.col_u, self.col_v, self.col_w = col_u, col_v, col_w
        self.col_tsonic, self.col_h2o, self.col_press = col_tsonic, col_h2o, col_press
        self.hz = hz
        self.scale = dict(scale or {})
        self.tsonic_unit = tsonic_unit
        self.lag_max_s = lag_max_s
        self.lws, self.uws = lws, uws
        self.n_bootstrap = n_bootstrap
        self.block_length_s = block_length_s
        self.wdt = wdt
        self.gas_lag_overrides = _check_gas_lag_overrides(gas_lag_overrides, scalars)
        self.valid_ranges = dict(valid_ranges or {})
        self.random_state = random_state
        self.fdl_lag_range_s = int(fdl_lag_range_s)
        self.noise_range_s = int(noise_range_s)
        self.name = name

        self.pwb: dict = {}   # label -> PreWhiteningBootstrap, for plots
        self.fdl: dict = {}   # label -> FluxDetectionLimit, for plots and flux_at_lag
        self._results: dict | None = None

    @property
    def results(self) -> dict:
        """The row ``run()`` returned."""
        if self._results is None:
            raise RuntimeError("Call run() first.")
        return self._results

    def _scaled(self, column: str) -> pd.Series:
        return self.df[column].astype(float) * self.scale.get(column, 1.0)

    def run(self) -> dict:
        """Detect every gas's lag, compute its detection limit, read the flux.

        Returns one flat row: ``theta_deg``, ``phi_deg``, ``n_records``, and per
        gas (prefix = lower-cased label) the PWB fields (``tlag_s``, the HDI,
        ``is_reliable``, ...), ``n_valid``, ``n_out_of_range``, ``fdl``,
        ``fdl_noise_rmse``, ``flux_at_tlag`` and ``flux_to_fdl_at_tlag``. A gas
        with no usable record keeps NaN throughout; one whose PWB detection
        failed still gets its detection limit, but no flux.
        """
        df = self.df
        row: dict = {'n_records': len(df)}
        for label in self.scalars:
            row.update(_gas_row_template(label))

        u = df[self.col_u].astype(float)
        v = df[self.col_v].astype(float)
        w = df[self.col_w].astype(float)
        wr = WindDoubleRotation(u=u, v=v, w=w)
        row['theta_deg'] = float(np.degrees(wr.theta))
        row['phi_deg'] = float(np.degrees(wr.phi))

        ts = self._scaled(self.col_tsonic)
        if self.tsonic_unit == 'degC':
            ts = ts + 273.15
        h2o = self._scaled(self.col_h2o)
        press = self._scaled(self.col_press)
        _check_plausible('sonic temperature', ts, self.col_tsonic)
        _check_plausible('H2O', h2o, self.col_h2o)
        _check_plausible('pressure', press, self.col_press)

        for gi, (label, column) in enumerate(self.scalars.items()):
            p = label.lower()
            g_min, g_max = self.valid_ranges.get(label, (None, None))
            raw = df[column].to_numpy(dtype=float)
            x = _finite_or_nan(raw)
            oor = _out_of_range(x, g_min, g_max)
            x[oor] = np.nan
            n_valid = int(np.isfinite(x).sum())
            row[f'{p}_n_valid'] = n_valid
            row[f'{p}_n_out_of_range'] = int(oor.sum())
            if n_valid == 0:
                continue

            g_lag, g_block, g_lws, g_uws = _resolve_gas_lag(
                label, self.lag_max_s, self.block_length_s, self.lws, self.uws,
                self.gas_lag_overrides)
            try:
                pwb = PreWhiteningBootstrap(
                    df=pd.DataFrame({'W_rot': wr.w2.to_numpy(dtype=float),
                                     label: raw,
                                     'T_SONIC': df[self.col_tsonic].to_numpy(dtype=float)}),
                    var_w='W_rot', var_scalar=label, var_tsonic='T_SONIC',
                    hz=self.hz, lag_max_s=g_lag, n_bootstrap=self.n_bootstrap,
                    block_length_s=g_block, wdt=self.wdt,
                    random_state=None if self.random_state is None else int(self.random_state) + gi,
                    segment_name=self.name, lws=g_lws, uws=g_uws,
                    valid_min=g_min, valid_max=g_max)
                pwb.run()
                self.pwb[label] = pwb
                res = pwb.results
                for k in _PWB_KEYS:
                    row[f'{p}_{k}'] = float(res[k]) if res[k] is not None else np.nan
                row[f'{p}_is_reliable'] = bool(res['is_reliable'])
                row[f'{p}_best_combination'] = res['best_combination'] or ''
            except Exception as e:
                row[f'{p}_error'] = f'PWB: {type(e).__name__}: {e}'

            try:
                fdl = FluxDetectionLimit(
                    df=pd.DataFrame({'u': u.to_numpy(), 'v': v.to_numpy(), 'w': w.to_numpy(),
                                     'c': x * self.scale.get(column, 1.0),
                                     'ts': ts.to_numpy(), 'h2o': h2o.to_numpy(),
                                     'p': press.to_numpy()}),
                    u_col='u', v_col='v', w_col='w', c_col='c', ts_col='ts',
                    h2o_col='h2o', press_col='p', default_lag=None,
                    noise_range=self.noise_range_s,
                    lag_range=[-self.fdl_lag_range_s, self.fdl_lag_range_s],
                    lag_stepsize=1, sampling_rate=self.hz,
                    title_covariance_plot=f'{self.name} | {label}',
                    verbose=False)
                fdl.run()
                self.fdl[label] = fdl
                row[f'{p}_fdl'] = fdl.results['flux_detection_limit']
                row[f'{p}_fdl_noise_rmse'] = fdl.results['flux_noise_rmse']
                if np.isfinite(row[f'{p}_tlag_s']):
                    flux = fdl.flux_at_lag(row[f'{p}_tlag_s'])
                    row[f'{p}_flux_at_tlag'] = flux
                    row[f'{p}_flux_to_fdl_at_tlag'] = abs(flux) / row[f'{p}_fdl']
            except Exception as e:
                sep = '; ' if row[f'{p}_error'] else ''
                row[f'{p}_error'] += f'{sep}FDL: {type(e).__name__}: {e}'

        self._results = row
        return row

    def flux_at_lag(self, label: str, lag_s: float) -> float:
        """Flux of gas *label* at *lag_s* seconds (positive: gas arrives after w)."""
        if label not in self.fdl:
            return np.nan
        return self.fdl[label].flux_at_lag(lag_s)

    def flux_curve(self, label: str) -> np.ndarray:
        """Flux at every lag from ``-fdl_lag_range_s`` to ``+fdl_lag_range_s``, one per record.

        Index ``k`` is the lag ``k / hz - fdl_lag_range_s`` seconds. All NaN
        when the gas had no detection limit.
        """
        n = 2 * self.fdl_lag_range_s * self.hz + 1
        if label not in self.fdl:
            return np.full(n, np.nan)
        return self.fdl[label].cov_df['cov_flux'].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Batch over files
# ---------------------------------------------------------------------------

def _period_worker(task: dict) -> dict:
    """Read one period from disk and run PwbFdl on it. Pickle-safe.

    Returns ``{'key', 'row', 'curves'}``; ``curves`` holds each gas's flux at
    every lag, so the parent can read it at the PWBOPT lag later.
    """
    row = dict(task['row'])
    curves: dict = {}
    try:
        path = Path(task['path'])
        n_preserved = task['skiprows'] + 1 + task['extra_rows']
        header = read_preserved_lines(path, n_preserved)[task['skiprows']]
        columns = split_header_line(header.rstrip('\r\n'), task['sep'])
        df = pd.read_csv(path, skiprows=n_preserved + task['row_offset'],
                         nrows=task['n_rows'], header=None, sep=task['sep'],
                         na_values=task['na_values'], **_read_engine_kwargs(task['sep']))
        if df.shape[1] != len(columns):
            raise ValueError(
                f"header has {len(columns)} columns but the data has {df.shape[1]}; "
                f"check skiprows ({task['skiprows']}), extra_rows ({task['extra_rows']}) "
                f"and sep ({task['sep']!r}).")
        df.columns = columns
        p = PwbFdl(df=df, name=row['period'], **task['pwbfdl_kwargs'])
        row.update(p.run())
        # Only the lags a final lag can take are sent back: the full +/-180 s curve
        # is ~58 kB per gas at 20 Hz, ~2 GB for a year of half hours with two gases.
        mid, keep = p.fdl_lag_range_s * p.hz, task['keep_records']
        curves = {label: p.flux_curve(label)[mid - keep:mid + keep + 1] for label in p.scalars}
    except Exception as e:
        if task['strict']:
            raise
        for label in task['pwbfdl_kwargs']['scalars']:
            row.update(_gas_row_template(label))
        row['status'] = 'error'
        row['error'] = f'{type(e).__name__}: {e}'
        curves = {}
    return {'key': task['key'], 'row': row, 'curves': curves}


class PwbFdlBatch:
    """PWB lag, PWBOPT and flux detection limit over many raw files.

    Each file is cut into averaging periods like ``dyco detect-remove`` does
    (``chunk_seconds``; snapped to the :00/:30 grid when a start time can be
    read from the file name). Every period runs ``PwbFdl`` in parallel. PWBOPT
    then picks each gas's lag across the whole sequence, in the order the files
    are given, exactly as the pipeline does, and the flux is read at that lag.

    Give the files in time order: PWBOPT carries lags forward, so the order is
    part of the answer. ``sorted(Path(...).glob('*.csv.gz'))`` is right for
    names that start with a timestamp.

    All ``PwbFdl`` parameters are accepted and passed through, plus the file
    layout (``skiprows`` metadata rows before the header, ``extra_rows`` after
    it, ``sep``, ``na_values``), per-gas search windows
    (``gas_lag_overrides``, see ``PwbFdl``), the chunking (``chunk_seconds``,
    ``min_chunk_seconds``, ``start_time_regex``, ``start_time_format``), the
    PWBOPT settings (``hdi_thresh``, ``dev_thresh``, ``hdi_prefilter``,
    ``max_carry``, ``lag_fallback``; see ``dyco detect-remove --help``), and
    ``n_workers`` (1 runs in this process), ``output_dir`` (writes
    ``pwb_fdl_results.csv``) and ``strict`` (re-raise the first error instead
    of recording it).

    Result columns, per gas (prefix = lower-cased label), on top of the
    ``PwbFdl`` row and the PWBOPT columns:

    - ``{gas}_lag_s``: the lag the flux was read at, the PWBOPT
      ``tlag_final_pf_s`` rounded to a record. NaN where the period has no
      data for the gas or was skipped.
    - ``{gas}_lag_reason``: why the lag is what it is, in words.
    - ``{gas}_flux``: the flux at ``lag_s``, in the gas's scaled unit m-2 s-1.
    - ``{gas}_flux_to_fdl``: ``abs(flux) / fdl``.
    - ``{gas}_above_fdl``: ``flux_to_fdl > 1``, the flux is distinguishable
      from noise at 99% (Langford et al. 2015). A flux below it is still a
      measurement and belongs in any average; Langford et al. Eq. 10 gives the
      detection limit of an average of N periods as ``sqrt(sum(fdl**2)) / N``.

    Windows note: guard the calling script with ``if __name__ == '__main__':``,
    since the workers are spawned processes.
    """

    def __init__(
            self,
            files: list,
            scalars: dict,
            col_u: str,
            col_v: str,
            col_w: str,
            col_tsonic: str,
            col_h2o: str,
            col_press: str,
            hz: int = 20,
            scale: dict | None = None,
            tsonic_unit: str = 'K',
            skiprows: int = 0,
            extra_rows: int = 0,
            sep: str = ',',
            na_values: list | None = None,
            chunk_seconds: float = 1800,
            min_chunk_seconds: float = 300,
            start_time_regex: str | None = None,
            start_time_format: str = '%Y%m%d%H%M',
            lag_max_s: float = 10.0,
            lws: float | None = None,
            uws: float | None = None,
            n_bootstrap: int = 99,
            block_length_s: float | None = None,
            wdt: int = 5,
            gas_lag_overrides: dict | None = None,
            valid_ranges: dict | None = None,
            hdi_thresh: float = 0.5,
            dev_thresh: float = 0.5,
            hdi_prefilter: float = 1.0,
            max_carry: int | None = None,
            lag_fallback: dict | None = None,
            fdl_lag_range_s: int = 180,
            noise_range_s: int = 20,
            random_state: int | None = None,
            n_workers: int | None = None,
            output_dir: str | Path | None = None,
            strict: bool = False,
    ):
        self.files = [Path(f) for f in files]
        if not self.files:
            raise ValueError("no files given.")
        absent = [str(f) for f in self.files if not f.is_file()]
        if absent:
            raise FileNotFoundError(f"input file(s) not found: {absent}")
        self.scalars = dict(scalars)
        self.hz = hz
        self.skiprows, self.extra_rows = skiprows, extra_rows
        self.sep = sep
        self.na_values = list(na_values) if na_values is not None else list(_DEFAULT_NA_VALUES)
        self.chunk_seconds = chunk_seconds
        self.min_chunk_seconds = min_chunk_seconds
        self.start_time_regex = start_time_regex
        self.start_time_format = start_time_format
        self.hdi_thresh, self.dev_thresh = hdi_thresh, dev_thresh
        self.hdi_prefilter = hdi_prefilter
        self.max_carry = max_carry
        # Validated here, as in the pipeline, so a typo fails before any file is read.
        self.lag_fallback = dict(lag_fallback or {})
        resolve_lag_fallback(self.lag_fallback, self.scalars)
        gas_lag_overrides = _check_gas_lag_overrides(gas_lag_overrides, self.scalars)
        self.fdl_lag_range_s = int(fdl_lag_range_s)
        # Every final lag is some gas's detection (its own, a donor's, or the median
        # of detections), so it lies inside the widest search window of any gas.
        widest = max(
            max(abs(x) for x in (g_lag, g_lws, g_uws) if x is not None)
            for g_lag, _, g_lws, g_uws in (
                _resolve_gas_lag(label, lag_max_s, block_length_s, lws, uws, gas_lag_overrides)
                for label in self.scalars))
        self._keep_records = min(int(np.ceil(widest * hz)), self.fdl_lag_range_s * hz)
        self.random_state = random_state
        self.n_workers = n_workers or os.cpu_count() or 1
        self.output_dir = Path(output_dir) if output_dir else None
        self.strict = strict
        self._pwbfdl_kwargs = dict(
            scalars=self.scalars, col_u=col_u, col_v=col_v, col_w=col_w,
            col_tsonic=col_tsonic, col_h2o=col_h2o, col_press=col_press, hz=hz,
            scale=scale, tsonic_unit=tsonic_unit, lag_max_s=lag_max_s, lws=lws,
            uws=uws, n_bootstrap=n_bootstrap, block_length_s=block_length_s,
            wdt=wdt, gas_lag_overrides=gas_lag_overrides, valid_ranges=valid_ranges,
            fdl_lag_range_s=fdl_lag_range_s,
            noise_range_s=noise_range_s)
        self._results: DataFrame | None = None

    @property
    def results(self) -> DataFrame:
        """One row per period, in time order (available after ``run()``)."""
        if self._results is None:
            raise RuntimeError("Call run() first.")
        return self._results

    def _plan(self) -> list:
        """One task per averaging period, in file order then time order."""
        tasks = []
        n_preserved = self.skiprows + 1 + self.extra_rows
        min_rows = int(round(self.min_chunk_seconds * self.hz))
        for fi, path in enumerate(self.files):
            n_data = _count_data_rows(path, n_preserved)
            start = _parse_file_start_time(path.name, self.start_time_regex,
                                           self.start_time_format)
            stem = strip_compression(path.name).rsplit('.', 1)[0]
            ci = 0
            while True:
                offset, n_rows = _chunk_row_slice(start, ci, self.chunk_seconds, self.hz)
                if offset >= n_data:
                    break
                n_avail = min(n_rows, n_data - offset)
                t0 = _chunk_start_time(start, ci, self.chunk_seconds)
                period = f'{t0:%Y-%m-%d %H:%M}' if t0 else f'{stem}#{ci:02d}'
                row = {'period': period, 'file': path.name, 'chunk_index': ci,
                       'timestamp': t0.isoformat() if t0 else '', 'status': 'ok',
                       'error': ''}
                tasks.append({
                    'key': (fi, ci), 'path': str(path), 'row': row,
                    'row_offset': offset, 'n_rows': n_rows, 'skiprows': self.skiprows,
                    'extra_rows': self.extra_rows, 'sep': self.sep,
                    'na_values': self.na_values, 'strict': self.strict,
                    'short': n_avail < min_rows, 'n_avail': n_avail,
                    'keep_records': self._keep_records,
                })
                ci += 1
        # Seeds follow the period's place in the sequence, so a run is reproducible
        # whatever order the workers finish in. PwbFdl adds the gas's position (< 10).
        for n, t in enumerate(tasks):
            seed = None if self.random_state is None else int(self.random_state) + 10 * n
            t['pwbfdl_kwargs'] = {**self._pwbfdl_kwargs, 'random_state': seed}
        return tasks

    def run(self, on_progress: Callable | None = None) -> DataFrame:
        """Process every period, apply PWBOPT, read the flux at the final lag.

        Args:
            on_progress: Optional ``f(completed, total, row)``, called as each
                period finishes.

        Returns:
            The results DataFrame, also written to
            ``<output_dir>/pwb_fdl_results.csv`` when ``output_dir`` is set.
        """
        tasks = self._plan()
        done: dict = {}
        todo = []
        for t in tasks:
            if t['short']:
                row = {**t['row'], 'status': 'skipped:short', 'n_records': t['n_avail'],
                       'error': f"{t['n_avail']} records, fewer than min_chunk_seconds"}
                for label in self.scalars:
                    row.update(_gas_row_template(label))
                done[t['key']] = {'row': row, 'curves': {}}
            else:
                todo.append(t)

        total, completed = len(tasks), len(done)
        if self.n_workers == 1 or len(todo) <= 1:
            for t in todo:
                out = _period_worker(t)
                done[t['key']] = out
                completed += 1
                if on_progress:
                    on_progress(completed, total, out['row'])
        else:
            with ProcessPoolExecutor(max_workers=min(self.n_workers, len(todo))) as ex:
                futures = {ex.submit(_period_worker, t): t for t in todo}
                for fut in as_completed(futures):
                    t = futures[fut]
                    try:
                        out = fut.result()
                    except Exception as e:
                        # A worker that dies breaks the pool and every period still
                        # running in it. Record each as an error rather than lose it.
                        if self.strict:
                            raise
                        out = {'row': {**t['row'], 'status': 'error',
                                       'error': f'worker lost: {type(e).__name__}: {e}'},
                               'curves': {}}
                        for label in self.scalars:
                            out['row'].update(_gas_row_template(label))
                    done[t['key']] = out
                    completed += 1
                    if on_progress:
                        on_progress(completed, total, out['row'])

        keys = [t['key'] for t in tasks]
        summary = pd.DataFrame([done[k]['row'] for k in keys])
        not_ok = summary['status'].ne('ok').to_numpy()
        for label in self.scalars:
            p = label.lower()
            # A period that was skipped or failed has no detection, the same as one
            # with no data, so PWBOPT must not carry a lag into it.
            summary.loc[not_ok, f'{p}_n_valid'] = 0
        summary = apply_pwbopt_columns(
            summary, self.scalars, self.hdi_thresh, self.dev_thresh,
            self.hdi_prefilter, max_carry=self.max_carry, lag_fallback=self.lag_fallback)

        offset = self._keep_records
        for label in self.scalars:
            p = label.lower()
            # `no_data` means the analyser had nothing in this period. A skipped or
            # failed period was never looked at, which is `none`, as in detect-remove.
            summary.loc[not_ok, f'{p}_lag_source'] = 'none'
            final = summary[f'{p}_tlag_final_pf_s'].to_numpy(dtype=float)
            lag_s = np.full(len(summary), np.nan)
            flux = np.full(len(summary), np.nan)
            for i, k in enumerate(keys):
                curve = done[k]['curves'].get(label)
                if curve is None or not np.isfinite(final[i]):
                    continue
                shift = int(round(final[i] * self.hz))
                lag_s[i] = shift / self.hz
                if 0 <= shift + offset < len(curve):
                    flux[i] = curve[shift + offset]
            summary[f'{p}_lag_s'] = lag_s
            summary[f'{p}_flux'] = flux
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio = np.abs(flux) / summary[f'{p}_fdl'].to_numpy(dtype=float)
            summary[f'{p}_flux_to_fdl'] = ratio
            summary[f'{p}_above_fdl'] = np.where(np.isfinite(ratio), ratio > 1, pd.NA)
            summary[f'{p}_lag_reason'] = [
                (f"period {row_status}: no lag" if row_status != 'ok' else
                 _lag_reason(summary, i, p, self.hdi_thresh, self.dev_thresh,
                             self.hdi_prefilter, self.max_carry))
                for i, row_status in enumerate(summary['status'].astype(str))]

        self._results = summary
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            summary.to_csv(self.output_dir / 'pwb_fdl_results.csv', index=False)
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _scale_token(token: str) -> tuple:
    """Parse one ``--scale COLUMN=FACTOR`` into ``(column, factor)``.

    Split at the last ``=``, so a column name that itself contains one still
    works. argparse turns the ArgumentTypeError into a usage error.
    """
    import argparse
    column, eq, factor = token.rpartition('=')
    column = column.strip()
    if not eq or not column:
        raise argparse.ArgumentTypeError(f"expected COLUMN=FACTOR, got {token!r}")
    try:
        value = float(factor)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"the factor in {token!r} must be a number, got {factor.strip()!r}") from None
    if not np.isfinite(value):
        raise argparse.ArgumentTypeError(
            f"the factor in {token!r} must be a finite number, got {factor.strip()!r}")
    return column, value


def _build_parser():
    import argparse

    class _Formatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
        pass

    p = argparse.ArgumentParser(
        prog='dyco-fdl',
        description=(
            'Detect the time lag of every gas in every averaging period with\n'
            'pre-whitening block-bootstrap (PWB), choose the lag across the whole\n'
            'sequence of periods with PWBOPT, and compute the flux detection limit\n'
            '(Langford et al. 2015) of each period. The flux is read off the\n'
            "period's cross-covariance at the chosen lag and compared with the\n"
            'detection limit. The input files are cut into averaging periods as in\n'
            '`dyco detect-remove`, but no data files are written. The output, in\n'
            '--output-dir, is pwb_fdl_results.csv, with one row per period giving\n'
            'the lag, the flux, the detection limit and whether the flux is above\n'
            'it for each gas, and log.txt, a record of the run.\n'
            '\n'
            'The detection limit needs physical units: sonic temperature in K (or\n'
            'degC with --tsonic-unit), H2O as a mole fraction in mol mol-1, and the\n'
            'ambient air pressure in Pa. Use --scale for columns logged in other\n'
            "units. The flux and the detection limit come out in the gas's scaled\n"
            'unit times m-2 s-1: a gas scaled to nmol mol-1 gives nmol m-2 s-1.\n'
            'Lag detection itself does not depend on units.'
        ),
        epilog=(
            'example:\n'
            '  dyco fdl --input-dir raw --output-dir out \\\n'
            '      --col-u U --col-v V --col-w W --col-tsonic T_SONIC \\\n'
            '      --col-h2o H2O --col-press PRESS \\\n'
            '      --scalar "N2O:N2O_DRY" --scalar "CH4:CH4_DRY@lagfrom=N2O" \\\n'
            '      --scale "N2O_DRY=1e3" --scale "CH4_DRY=1e3" \\\n'
            '      --scale "H2O=1e-6" --scale "PRESS=100" \\\n'
            '      --lws 0 --uws 10 --max-carry 4'
        ),
        formatter_class=_Formatter,
    )
    # --- I/O ---
    p.add_argument('--input-dir', required=True,
                   help='Directory containing the raw (unrotated) EC files.')
    p.add_argument('--output-dir', required=True,
                   help='Directory for pwb_fdl_results.csv and log.txt. It is '
                        'created if it does not exist.')
    p.add_argument('--file-pattern', default='*.csv',
                   help='Glob pattern for the input files. The files are sorted '
                        'by name and that order is taken as time order, which '
                        'PWBOPT relies on, so name the files so that they sort '
                        'by time.')
    # --- Columns ---
    p.add_argument('--col-u', required=True,
                   help='Column name for U (horizontal x), m s-1.')
    p.add_argument('--col-v', required=True,
                   help='Column name for V (horizontal y), m s-1.')
    p.add_argument('--col-w', required=True,
                   help='Column name for W (vertical), m s-1.')
    p.add_argument('--col-tsonic', required=True,
                   help='Column name for sonic temperature, in K, or in degC '
                        'with --tsonic-unit degC.')
    p.add_argument('--col-h2o', required=True,
                   help='Column name for the H2O mole fraction. It turns sonic '
                        'temperature into air temperature and gives the dry-air '
                        'density. Scale it to mol mol-1 with --scale.')
    p.add_argument('--col-press', required=True,
                   help='Column name for the ambient air pressure, scaled to Pa '
                        'with --scale. Not an analyser cell pressure, which can '
                        'be far below ambient.')
    # --- Scalars ---
    p.add_argument('--scalar', dest='scalars', action='append', required=True,
                   metavar='LABEL:column[@key=value;...]',
                   help='Gas label and column name in the raw file, e.g. '
                        '"N2O:N2O_DRY_[LGR-A]". Repeat for each gas. Result '
                        'columns are prefixed with the lower-cased label. Append '
                        'per-gas options after "@" as ";"-separated key=value '
                        'pairs. lag (or lag_max), block, lws and uws, in seconds, '
                        'give this gas its own lag search window, e.g. '
                        '"N2O:n2o@lws=0;uws=5". "lagfrom=LABEL" makes this gas '
                        'borrow another gas\'s lag for periods it has none of its '
                        'own. Donate only between gases that behave alike: H2O '
                        'sticks to the tube wall and lags longer than the flow, so '
                        'it must not be the donor for CO2, CH4 or N2O even when it '
                        'is the gas that detects best. "min=" and "max=" give the '
                        'gas a valid range in its own, unscaled units, e.g. '
                        '"CO2:co2@min=200;max=900"; either may be left out. Values '
                        'outside it are treated as missing, for lag detection and '
                        'for the detection limit alike.')
    # --- Units ---
    p.add_argument('--scale', dest='scale', action='append', type=_scale_token,
                   metavar='COLUMN=FACTOR', default=None,
                   help='Multiply a column by FACTOR before the detection limit '
                        'is computed, e.g. "PRESS_BOX_[IRGA75-A]=100" for hPa to '
                        'Pa, "H2O_[LGR-A]=1e-6" for umol mol-1 to mol mol-1, or '
                        '"N2O_DRY_[LGR-A]=1e3" for umol mol-1 to nmol mol-1. '
                        'Repeat for each column. The text is split at its last '
                        '"=", so a column name may contain one. Lag detection and '
                        'the valid ranges use the unscaled values.')
    p.add_argument('--tsonic-unit', choices=('K', 'degC'), default='K',
                   help='Unit of the sonic temperature column after --scale.')
    # --- PWB parameters ---
    p.add_argument('--hz', type=int, default=20,
                   help='Sampling frequency in Hz.')
    p.add_argument('--lag-max', type=float, default=10.0,
                   help='CCF search half-width [s].')
    p.add_argument('--n-bootstrap', type=int, default=99,
                   help='Number of block-bootstrap replicates (paper: 99).')
    p.add_argument('--block-length', type=float, default=20.0,
                   help='Bootstrap block length [s] (paper: L = 20 s).')
    p.add_argument('--wdt', type=int, default=5,
                   help='Width [records] of the centred rolling mean applied to '
                        'each bootstrap CCF before its peak is taken. 5 follows '
                        'RFlux; the paper uses hz/2+1 (11 at 20 Hz, 6 at 10 Hz).')
    p.add_argument('--lws', type=float, default=None,
                   help='Optional lower limit [s] of an asymmetric lag search '
                        'window applied to all gases (per-gas "@lws=" overrides).')
    p.add_argument('--uws', type=float, default=None,
                   help='Optional upper limit [s] of the asymmetric lag search '
                        'window applied to all gases (per-gas "@uws=" overrides).')
    p.add_argument('--random-state', type=int, default=None,
                   help='Base seed for a reproducible bootstrap. Each period and '
                        'gas gets a derived seed, so the result does not depend '
                        'on the number of workers.')
    # --- Chunking ---
    p.add_argument('--chunk-seconds', type=float, default=1800.0,
                   help='Length of one averaging period [s]. Each input file is '
                        'cut into periods of this length.')
    p.add_argument('--min-chunk-seconds', type=float, default=300.0,
                   help='Periods shorter than this [s] are skipped. PWB needs '
                        'enough records to fit the block-bootstrap.')
    p.add_argument('--start-time-regex', default=None,
                   help='Regex extracting the start timestamp of the input file '
                        'from its name. Concatenated capture groups (or the whole '
                        'match) are parsed with --start-time-format. With a start '
                        'time, periods are named by their start time and aligned '
                        'to the clock; without one, by file name and index.')
    p.add_argument('--start-time-format', default='%Y%m%d-%H%M',
                   help='strptime format for the timestamp that '
                        '--start-time-regex extracts.')
    # --- File format ---
    p.add_argument('--skiprows', type=int, default=0,
                   help='Lines BEFORE the column-name row '
                        '(raw CSV with header on line 1: 0; '
                        'EddyPro rotated: 9).')
    p.add_argument('--extra-rows', type=int, default=2,
                   help='Extra rows AFTER the header but BEFORE data '
                        '(e.g. units + instrument-source rows: 2 for the '
                        'typical raw EC CSV).')
    p.add_argument('--sep', default=',',
                   help=r"Field separator. Default ',' (CSV). Use '\s+' for "
                        r"whitespace or '\t' for TSV.")
    p.add_argument('--na-values', nargs='+', default=list(_DEFAULT_NA_VALUES),
                   help='Strings to treat as NaN on read.')
    # --- PWBOPT post-processing (paper Section 2.3 defaults) ---
    p.add_argument('--hdi-thresh', type=float, default=0.5,
                   help='S1 HDI threshold [s]: periods with HDI range below '
                        'this are flagged S1_optimal (reliable).')
    p.add_argument('--dev-thresh', type=float, default=0.5,
                   help='S2 deviation threshold [s]: uncertain periods are '
                        'accepted if within this distance of the preceding '
                        'optimal lag.')
    p.add_argument('--hdi-prefilter', type=float, default=1.0,
                   help='Pre-filter [s]: lags with HDI range above this are '
                        'set to NaN before PWBOPT (pre-filtered variant, the one '
                        'the flux is read at). Set to 0 to disable.')
    p.add_argument('--max-carry', type=int, default=None,
                   help='Longest carry, in averaging periods, that PWBOPT S3 '
                        'may use: a period with no usable detection takes the '
                        'nearest optimal lag only if it is within this many '
                        'periods, otherwise the lag expires and falls through '
                        'to a donor gas or the median. Default: unlimited, '
                        'which is the published behaviour.')
    # --- Flux detection limit ---
    p.add_argument('--fdl-lag-range', type=int, default=180,
                   help='Half-width [s] of the cross-covariance used for the '
                        'detection limit. It must reach well beyond any real '
                        'lag, since the noise is read from its far ends.')
    p.add_argument('--noise-range', type=int, default=20,
                   help='Width [s] of each noise window at the two ends of the '
                        'cross-covariance. The detection limit is 3 times the '
                        'RMSE of the flux there. With the defaults the windows '
                        'run from -180 to -160 s and from 160 to 180 s.')
    # --- Execution ---
    p.add_argument('--n-workers', type=int, default=None,
                   help='Parallel worker processes. Default: os.cpu_count(). '
                        'Set to 1 for sequential in-process execution '
                        '(useful for debugging).')
    p.add_argument('--strict', action='store_true',
                   help='Re-raise exceptions on the first failure instead '
                        'of recording them per period.')
    return p


def _fmt_num(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return '--'
    return f'{x:.4g}' if np.isfinite(x) else '--'


def _fmt_lag(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return '--'
    return f'{x:.2f} s' if np.isfinite(x) else '--'


def _result_cell(row, label: str) -> str:
    """``N2O lag 1.80 s flux 2.12 fdl 0.829 ABOVE`` for one gas in one period."""
    p = label.lower()
    above = row.get(f'{p}_above_fdl')
    verdict = ''
    if isinstance(above, (bool, np.bool_)):
        verdict = ' ABOVE' if above else ' below'
    return (f'{label} lag {_fmt_lag(row.get(f"{p}_lag_s"))} '
            f'flux {_fmt_num(row.get(f"{p}_flux"))} '
            f'fdl {_fmt_num(row.get(f"{p}_fdl"))}{verdict}')


def _cli_main():
    import sys
    from datetime import datetime, timezone

    args = _build_parser().parse_args()
    log_lines: list = []

    def out(line: str = '') -> None:
        print(line, flush=True)
        log_lines.append(line)

    def fail(msg: str):
        print(f'ERROR: {msg}', file=sys.stderr)
        sys.exit(1)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        fail(f'--input-dir not found: {input_dir}')
    files = sorted(f for f in input_dir.glob(args.file_pattern) if f.is_file())
    if not files:
        fail(f'no files matched {args.file_pattern!r} in {input_dir}')

    scalars: dict = {}
    gas_lag_overrides: dict = {}
    lag_fallback: dict = {}
    valid_ranges: dict = {}
    for token in args.scalars:
        try:
            label, col, overrides = parse_scalar_spec(token)
        except ValueError as e:
            fail(str(e))
        scalars[label] = col
        donor = overrides.pop('lagfrom', None)
        if donor:
            lag_fallback[label] = donor
        lo = overrides.pop('valid_min', None)
        hi = overrides.pop('valid_max', None)
        if lo is not None or hi is not None:
            valid_ranges[label] = (lo, hi)
        if overrides:
            gas_lag_overrides[label] = overrides
    scale = dict(args.scale or [])

    if not 0 < args.noise_range < args.fdl_lag_range:
        fail(f'--noise-range ({args.noise_range}) must be positive and smaller '
             f'than --fdl-lag-range ({args.fdl_lag_range}).')

    # Same escapes as `dyco detect-remove`, so --sep "\t" means a tab.
    sep = (args.sep.replace('\\t', '\t').replace('\\r', '\r')
           .replace('\\n', '\n').replace('\\s+', _WHITESPACE_SEP))

    # Check the column names against the first file's header now: a typo would
    # otherwise fail every period, one error row at a time, and a --scale column
    # that is not in the file would be ignored without a word.
    try:
        header = read_preserved_lines(files[0], args.skiprows + 1)[args.skiprows]
        columns = set(split_header_line(header.rstrip('\r\n'), sep))
    except Exception as e:
        fail(f'could not read the header of {files[0].name}: {e}')
    named = [('--col-u', args.col_u), ('--col-v', args.col_v), ('--col-w', args.col_w),
             ('--col-tsonic', args.col_tsonic), ('--col-h2o', args.col_h2o),
             ('--col-press', args.col_press)]
    named += [(f'--scalar {label}', col) for label, col in scalars.items()]
    named += [('--scale', col) for col in scale]
    missing = [f'{flag} {col!r}' for flag, col in named if col not in columns]
    if missing:
        fail(f'column(s) not in the header of {files[0].name}: {", ".join(missing)}. '
             f'Check the names, --skiprows ({args.skiprows}) and --sep ({args.sep!r}).')

    output_dir = Path(args.output_dir)
    try:
        batch = PwbFdlBatch(
            files=files, scalars=scalars, col_u=args.col_u, col_v=args.col_v,
            col_w=args.col_w, col_tsonic=args.col_tsonic, col_h2o=args.col_h2o,
            col_press=args.col_press, hz=args.hz, scale=scale,
            tsonic_unit=args.tsonic_unit, skiprows=args.skiprows,
            extra_rows=args.extra_rows, sep=sep, na_values=args.na_values,
            chunk_seconds=args.chunk_seconds, min_chunk_seconds=args.min_chunk_seconds,
            start_time_regex=args.start_time_regex,
            start_time_format=args.start_time_format, lag_max_s=args.lag_max,
            lws=args.lws, uws=args.uws, n_bootstrap=args.n_bootstrap,
            block_length_s=args.block_length, wdt=args.wdt,
            gas_lag_overrides=gas_lag_overrides, valid_ranges=valid_ranges,
            hdi_thresh=args.hdi_thresh, dev_thresh=args.dev_thresh,
            hdi_prefilter=args.hdi_prefilter, max_carry=args.max_carry,
            lag_fallback=lag_fallback, fdl_lag_range_s=args.fdl_lag_range,
            noise_range_s=args.noise_range, random_state=args.random_state,
            n_workers=args.n_workers, output_dir=output_dir, strict=args.strict)
    except (ValueError, FileNotFoundError) as e:
        fail(str(e))

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / 'log.txt'

    def now() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')

    mode = 'sequential' if batch.n_workers == 1 else f'{batch.n_workers} workers'
    out('dyco fdl: PWB time lag and flux detection limit')
    out(f'started : {now()}')
    out(f'input   : {input_dir}  ({len(files)} files, pattern {args.file_pattern!r})')
    out(f'output  : {output_dir}')
    out(f'mode    : {mode}')
    out(f'PWB     : hz {args.hz}  lag-max {args.lag_max:g} s  n-boot {args.n_bootstrap}  '
        f'block {args.block_length:g} s  wdt {args.wdt}  '
        f'window {_fmt_num(args.lws)} to {_fmt_num(args.uws)} s')
    out(f'periods : {args.chunk_seconds:g} s  (min {args.min_chunk_seconds:g} s)')
    if args.start_time_regex:
        out(f"time    : regex '{args.start_time_regex}'  format '{args.start_time_format}'")
    out(f'wind    : U={args.col_u!r}  V={args.col_v!r}  W={args.col_w!r}  '
        f'T_SONIC={args.col_tsonic!r} ({args.tsonic_unit})')
    out(f'FDL     : H2O={args.col_h2o!r}  pressure={args.col_press!r}  '
        f'cross-covariance +/-{args.fdl_lag_range} s, noise from '
        f'{args.fdl_lag_range - args.noise_range} to {args.fdl_lag_range} s at both ends')
    for label, col in scalars.items():
        extra = ''
        if label in gas_lag_overrides:
            extra += '  window ' + ' '.join(
                f'{k}={v:g}' for k, v in gas_lag_overrides[label].items())
        if label in lag_fallback:
            extra += f'  lag from {lag_fallback[label]} where {label} has none'
        if label in valid_ranges:
            extra += f'  valid {format_valid_range(*valid_ranges[label])}'
        out(f'scalar  : {label} <- {col!r}{extra}')
    if scale:
        out('scale   : ' + '  '.join(f'{c!r} x {f:g}' for c, f in scale.items()))
    out(f'PWBOPT  : hdi-thresh {args.hdi_thresh:g} s  dev-thresh {args.dev_thresh:g} s  '
        f'hdi-prefilter {args.hdi_prefilter:g} s  max-carry '
        f'{args.max_carry if args.max_carry is not None else "unlimited"}')
    out()

    def on_progress(done: int, total: int, row: dict) -> None:
        status = row.get('status', '')
        if status == 'ok':
            detail = '  detected ' + ' | '.join(
                f'{g} {_fmt_lag(row.get(f"{g.lower()}_tlag_s"))}' for g in scalars)
        else:
            detail = f'  {str(row.get("error", ""))[:120]}'
        out(f'  {done:>{len(str(total))}}/{total} {row.get("period", "")}  {status}{detail}')

    try:
        res = batch.run(on_progress=on_progress)

        n = len(res)
        ok = res['status'].eq('ok')
        errors = res.loc[res['status'].eq('error'), 'error'].astype(str)
        if n and not ok.any() and len(errors):
            # Most often a unit mistake, which fails every period the same way.
            # Say it once, not as a wall of error rows that differ only in the
            # period's mean value.
            import re
            kinds: dict = {}
            for e in errors:
                kinds.setdefault(re.sub(r'\d+(\.\d+)?(e[-+]?\d+)?', '#', e), e)
            msg = ' | '.join(list(kinds.values())[:3])
            if 'plausible' in msg:
                msg += (' On the command line: --scale COLUMN=FACTOR, and '
                        '--tsonic-unit for sonic temperature.')
            log_lines.append(f'ERROR: every period failed: {msg}')
            print(f'\nERROR: every period failed: {msg}', file=sys.stderr, flush=True)
            raise SystemExit(1)

        out()
        out('Results, at the lag PWBOPT chose across the sequence:')
        width = len(str(n))
        for i, (_, r) in enumerate(res.iterrows(), start=1):
            if r['status'] == 'ok':
                detail = ' | '.join(_result_cell(r, g) for g in scalars)
            else:
                detail = str(r.get('error', ''))[:120]
            out(f'  {i:>{width}}/{n} {r["period"]}  {r["status"]}  {detail}')

        out()
        n_skip = int(res['status'].eq('skipped:short').sum())
        out(f'{int(ok.sum())} of {n} periods ok'
            f'{f", {n_skip} short" if n_skip else ""}'
            f'{f", {len(errors)} errors" if len(errors) else ""}.')
        for label in scalars:
            p = label.lower()
            above = res[f'{p}_above_fdl']
            has_flux = above.notna()
            n_above = int(above[has_flux].astype(bool).sum())
            flags = res.loc[ok, f'{p}_flag_pf'].astype(str).value_counts()
            sources = res.loc[ok, f'{p}_lag_source'].astype(str).value_counts()
            out(f'{label}: {n_above} of {int(has_flux.sum())} periods with a flux are '
                f'above the FDL; PWBOPT '
                + ', '.join(f'{s} {c}' for s, c in flags.items())
                + '; lag source '
                + ', '.join(f'{s} {c}' for s, c in sources.items()))
            if label in valid_ranges:
                n_oor = int(pd.to_numeric(res[f'{p}_n_out_of_range'],
                                          errors='coerce').fillna(0).sum())
                if n_oor:
                    out(f'{label}: {n_oor} records outside the valid range '
                        f'({format_valid_range(*valid_ranges[label])}), treated '
                        f'as missing')
        out(f'Results saved to: {output_dir / "pwb_fdl_results.csv"}')
    except BaseException as e:
        if not isinstance(e, SystemExit):
            out(f'ABORT {type(e).__name__}: {e}')
        raise
    finally:
        out(f'finished: {now()}')
        try:
            log_path.write_text('\n'.join(log_lines) + '\n', encoding='utf-8')
        except Exception:
            pass
        print(f'Log saved to: {log_path}')


if __name__ == '__main__':
    _cli_main()
