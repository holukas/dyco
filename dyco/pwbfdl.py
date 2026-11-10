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

# The shortest period, in multiples of fdl_lag_range_s, that gets a detection
# limit. The covariance at a far lag t only has the period minus t to pair up,
# so the noise windows use less data than the flux they are compared with and
# the limit comes out too high by about sqrt(L / (L - t)): 5 % for 30 min at
# +/-180 s, 11 % at 5 x 180 s = 15 min, 50 % for a 5-min period. Below this the
# limit is withheld rather than reported inflated. The flux is still read.
_FDL_MIN_PERIOD_FACTOR = 5

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
        Half-width of the cross-covariance for the detection limit, s. A period
        shorter than 5 times this (15 min at the default) gets no detection
        limit, only the flux: its far-lag covariances pair up too little data,
        which makes the limit too high. ``{gas}_error`` says so.
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

        min_fdl_s = _FDL_MIN_PERIOD_FACTOR * self.fdl_lag_range_s
        fdl_withheld = (
            '' if len(df) >= min_fdl_s * self.hz else
            f'FDL withheld: the period is {len(df) / self.hz:g} s, shorter than '
            f'{_FDL_MIN_PERIOD_FACTOR} x fdl_lag_range_s ({min_fdl_s} s); its limit '
            f'would come out too high')

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
                if fdl_withheld:
                    sep = '; ' if row[f'{p}_error'] else ''
                    row[f'{p}_error'] += f'{sep}{fdl_withheld}'
                else:
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


def _day_hours_token(token: str) -> tuple | None:
    """Parse ``--day-hours START-END`` (whole hours, e.g. ``6-18``), or ``none``."""
    import argparse
    if token.strip().lower() == 'none':
        return None
    start, dash, end = token.partition('-')
    try:
        start, end = int(start), int(end)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected START-END in whole hours, e.g. 6-18, or none; got {token!r}") from None
    if not dash or not 0 <= start < end <= 24:
        raise argparse.ArgumentTypeError(
            f"need 0 <= START < END <= 24, got {token!r}")
    return start, end


def _unit_token(token: str) -> tuple:
    """Parse one ``--flux-unit LABEL=TEXT`` into ``(label, text)``."""
    import argparse
    label, eq, text = token.partition('=')
    if not eq or not label.strip() or not text.strip():
        raise argparse.ArgumentTypeError(f"expected LABEL=TEXT, e.g. CH4='nmol m-2 s-1', "
                                         f"got {token!r}")
    return label.strip(), text.strip()


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
            'it for each gas; log.txt, a record of the run that ends with the\n'
            'numbers to report; and one figure per gas, fdl_{gas}.png and .pdf.\n'
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
                   help='Directory for pwb_fdl_results.csv, log.txt and the '
                        'figures. It is created if it does not exist.')
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
                        'lag, since the noise is read from its far ends. Periods '
                        'shorter than 5 times this get no detection limit.')
    p.add_argument('--noise-range', type=int, default=20,
                   help='Width [s] of each noise window at the two ends of the '
                        'cross-covariance. The detection limit is 3 times the '
                        'RMSE of the flux there. With the defaults the windows '
                        'run from -180 to -160 s and from 160 to 180 s.')
    # A string default goes through `type`, and the help shows it as typed.
    p.add_argument('--day-hours', type=_day_hours_token, default='6-18',
                   metavar='START-END',
                   help='Day for the day/night means in the summary, by the start '
                        'time of each period: START <= hour < END, in the time the '
                        'file names give. "none" leaves the split out.')
    p.add_argument('--flux-unit', type=_unit_token, action='append', default=None,
                   metavar='LABEL=TEXT', dest='flux_units',
                   help='The unit of a gas\'s flux and FDL, for the summary and the '
                        'figure, e.g. --flux-unit "CH4=nmol m-2 s-1". Repeat per gas. '
                        'Write it in ASCII; the figure sets "u" before mol as a '
                        'micro sign and exponents as superscripts. Without it the '
                        'unit is given as that of the scaled gas column, m-2 s-1.')
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


def _detected_cell(row, label: str) -> str:
    """``N2O 1.80 s (HDI 0.20 s)``: one gas's own detection in one period."""
    p = label.lower()
    hdi = row.get(f'{p}_hdi_range_s')
    hdi_txt = f' (HDI {_fmt_lag(hdi)})' if _fmt_lag(hdi) != '--' else ''
    return f'{label} {_fmt_lag(row.get(f"{p}_tlag_s"))}{hdi_txt}'


def _result_cell(row, label: str) -> str:
    """``N2O lag 1.80 s [own] flux 2.12 fdl 0.829 ABOVE`` for one gas in one period."""
    p = label.lower()
    above = row.get(f'{p}_above_fdl')
    verdict = ''
    if isinstance(above, (bool, np.bool_)):
        verdict = ' ABOVE' if above else ' below'
    source = row.get(f'{p}_lag_source')
    source_txt = f' [{source}]' if isinstance(source, str) and source else ''
    return (f'{label} lag {_fmt_lag(row.get(f"{p}_lag_s"))}{source_txt} '
            f'flux {_fmt_num(row.get(f"{p}_flux"))} '
            f'fdl {_fmt_num(row.get(f"{p}_fdl"))}{verdict}')


def _range_txt(values: pd.Series, seconds: bool = False) -> str:
    """``0.30 to 1.47, median 0.66`` over the finite values, or ``none``.

    Seconds get two decimals and an ``s``, as lags do everywhere else."""
    v = pd.to_numeric(values, errors='coerce')
    v = v[np.isfinite(v)]
    if v.empty:
        return 'none'
    fmt = _fmt_lag if seconds else _fmt_num
    return f'{fmt(v.min())} to {fmt(v.max())}, median {fmt(v.median())}'


def run_summary(res: DataFrame, scalars: dict, scale: dict | None = None,
                hdi_thresh: float = 0.5, hdi_prefilter: float = 1.0,
                min_chunk_seconds: float = 300, fdl_lag_range_s: int = 180,
                noise_range_s: int = 20, day_hours: tuple | None = (6, 18),
                units: dict | None = None) -> list:
    """The end-of-run summary of ``dyco fdl``, as lines of ASCII text.

    Per gas, everything a report of the detection limit needs: where the lag
    of each period came from and how wide the HDI of the gas's own detections
    was; how the limit was computed; its median, interquartile and 10-90 %
    range over the periods; the share of periods whose absolute flux is above
    it; and the mean flux of the run, of the day and of the night, each with N
    and the detection limit of that mean (Langford et al. 2015, eq. 10). The
    lag part comes first because it decides how far the flux can be trusted:
    a flux read at the median of the gas's raw detections is read at a lag
    that was not measured in its period.

    ``day_hours=(start, end)`` sets the day by the start time of the period,
    ``start <= hour < end``, in whatever time the file names give; None leaves
    the split out. ``units`` maps a gas label to its flux unit as text; a gas
    not in it is described by its scaled column.
    """
    scale = scale or {}
    lines = ['Summary']
    status = res['status'].astype(str)
    ok = status.eq('ok')
    n_short = int(status.eq('skipped:short').sum())
    n_err = int(status.eq('error').sum())
    span = ''
    if ok.any():
        periods = res.loc[ok, 'period'].astype(str)
        span = f', {periods.iloc[0]} to {periods.iloc[-1]}'
    n_files = res['file'].nunique() if 'file' in res.columns else 0
    lines.append(f'  periods : {len(res)} from {n_files} file(s){span}')
    lines.append(f'            {int(ok.sum())} ok, {n_short} short (under '
                 f'{min_chunk_seconds:g} s), {n_err} error(s)')

    for label, column in scalars.items():
        p = label.lower()
        r = res.loc[ok]
        factor = scale.get(column)
        if (units or {}).get(label):
            unit = units[label]
        else:
            unit = (f"(unit of {column!r} x {factor:g}) m-2 s-1" if factor is not None
                    else f"(unit of {column!r}) m-2 s-1")
        lines.append('')
        lines.append(f'  {label}: flux and FDL in {unit}')

        # Where each period's lag came from. 'own' splits further into a
        # detection accepted here, a carried one and a back-filled one.
        source = r[f'{p}_lag_source'].astype(str)
        flag = r[f'{p}_flag_pf'].astype(str)
        carry = pd.to_numeric(r[f'{p}_carry_periods'], errors='coerce')
        own = source.eq('own')
        n_s1 = int((own & flag.eq('S1_optimal')).sum())
        n_s2 = int((own & flag.eq('S2_optimal')).sum())
        n_carried = int((own & ~flag.isin(['S1_optimal', 'S2_optimal'])
                         & (carry > 0)).sum())
        n_back = int(own.sum()) - n_s1 - n_s2 - n_carried
        borrowed = source[source.str.startswith('from:')]
        n_median = int(source.eq('median').sum())
        n_none = int(source.eq('none').sum())
        n_nodata = int(source.eq('no_data').sum())
        parts = [f'detected here {n_s1 + n_s2} (S1 {n_s1}, S2 {n_s2})',
                 f'carried {n_carried}', f'back-filled {n_back}']
        parts += [f'borrowed {c} ({s})' for s, c in borrowed.value_counts().items()]
        parts += [f'median {n_median}', f'none {n_none}']
        if n_nodata:
            parts.append(f'no data {n_nodata}')
        lines.append(f'    lag   : used {_range_txt(r[f"{p}_lag_s"], seconds=True)}')
        lines.append(f'            source: {", ".join(parts)}')

        hdi = pd.to_numeric(r[f'{p}_hdi_range_s'], errors='coerce')
        hdi = hdi[np.isfinite(hdi)]
        if hdi.empty:
            lines.append('            own detections: none')
        else:
            lines.append(f'            own detections: {len(hdi)}, HDI '
                         f'{_range_txt(hdi, seconds=True)}')
            # The same comparisons PWBOPT makes: S1 needs HDI < hdi_thresh, and the
            # prefilter (off at 0) drops detections with HDI > hdi_prefilter.
            prefilter = (f', {int((hdi <= hdi_prefilter).sum())} within '
                         f'--hdi-prefilter {hdi_prefilter:g} s' if hdi_prefilter > 0 else '')
            lines.append(f'            {int((hdi < hdi_thresh).sum())} narrower than '
                         f'--hdi-thresh {hdi_thresh:g} s{prefilter}')
        n_lagged = int(len(source)) - n_nodata
        if n_median:
            lines.append(f'    NOTE  : {n_median} of {n_lagged} period(s) read the flux at '
                         f'the median of all raw detections, not at a lag measured '
                         f'in the period')
        if n_s1 + n_s2 == 0 and n_median < n_lagged:
            lines.append(f'    NOTE  : {label} has no reliable lag of its own in any period')

        errors = r[f'{p}_error'].astype(str).replace('nan', '')
        withheld = int(errors.str.contains('FDL withheld').sum())
        failed = int((errors.ne('') & ~errors.str.contains('FDL withheld')).sum())
        fdl = pd.to_numeric(r[f'{p}_fdl'], errors='coerce')
        rmse = pd.to_numeric(r[f'{p}_fdl_noise_rmse'], errors='coerce')
        flux = pd.to_numeric(r[f'{p}_flux'], errors='coerce')
        both = np.isfinite(flux) & np.isfinite(fdl)

        lines.append(f'    FDL   : 3 x RE_RMSE (Langford et al. 2015, eq. 9), noise from '
                     f'-{fdl_lag_range_s} to -{fdl_lag_range_s - noise_range_s} s and '
                     f'+{fdl_lag_range_s - noise_range_s} to +{fdl_lag_range_s} s')
        f = fdl[np.isfinite(fdl)]
        if f.empty:
            lines.append('            none computed')
        else:
            q10, q25, q50, q75, q90 = np.percentile(f, [10, 25, 50, 75, 90])
            lines.append(f'            per period: median {_fmt_num(q50)}, IQR {_fmt_num(q25)} '
                         f'to {_fmt_num(q75)}, 10-90 % {_fmt_num(q10)} to {_fmt_num(q90)}, '
                         f'range {_fmt_num(f.min())} to {_fmt_num(f.max())} ({len(f)} period(s)'
                         f'{f", {withheld} withheld as too short" if withheld else ""})')
            lines.append(f'            random error RE_RMSE (= FDL / 3): median '
                         f'{_fmt_num(rmse[np.isfinite(rmse)].median())}')
        lines.append(f'    flux  : {_range_txt(flux)} ({int(np.isfinite(flux).sum())} period(s))')
        above = r[f'{p}_above_fdl']
        has = above.notna()
        n_above, n_has = int(above[has].astype(bool).sum()), int(has.sum())
        share = f' ({100 * n_above / n_has:.1f} %)' if n_has else ''
        lines.append(f'    above : {n_above} of {n_has} period(s){share} with abs(flux) '
                     f'above their FDL')

        # The averages to report, each with the detection limit of that mean
        # (eq. 10). Every period enters both, including those below their limit:
        # leaving them out biases the mean away from zero.
        groups = [('all', both)]
        hours = None
        if day_hours is not None and 'timestamp' in r.columns:
            stamps = pd.to_datetime(r['timestamp'], errors='coerce')
            if stamps.notna().all():
                hours = stamps.dt.hour + stamps.dt.minute / 60
        if hours is not None:
            start, end = day_hours
            day = (hours >= start) & (hours < end)
            groups += [(f'day {start:02d}-{end:02d} h', both & day),
                       ('night', both & ~day)]
        if both.any():
            lines.append('    means : mean flux, N, FDL of the mean = sqrt(sum FDL**2) / N '
                         '(Langford et al. 2015, eq. 10)')
            for name, m in groups:
                if not m.any():
                    lines.append(f'            {name:<14}: no period')
                    continue
                mean = float(flux[m].mean())
                fdl_mean = float(np.sqrt((fdl[m] ** 2).sum()) / m.sum())
                verdict = 'above' if abs(mean) > fdl_mean else 'below'
                median_txt = (f', median FDL {_fmt_num(fdl[m].median())}'
                              if name != 'all' else '')
                lines.append(f'            {name:<14}: {_fmt_num(mean)}, N {int(m.sum())}, '
                             f'FDL of the mean {_fmt_num(fdl_mean)}: {verdict}{median_txt}')
            if hours is None and day_hours is not None:
                lines.append('            (no day/night split: the periods have no start time)')

            # With the right noise estimate and no real flux, flux / RE is
            # standard normal. A real flux widens it, and so does an RE that is
            # too small; the numbers cannot tell the two apart.
            z = (flux[both] / rmse[both]).to_numpy(dtype=float)
            z = z[np.isfinite(z)]
            if len(z) > 1:
                lines.append(f'    check : flux / RE has sd {np.std(z, ddof=1):.2f} and '
                             f'{100 * np.mean(np.abs(z) > 3):.1f} % beyond 3; pure noise '
                             f'gives 1.00 and 0.3 %, a real flux or a too small RE more')
        if failed:
            lines.append(f'    errors: {failed} period(s), see {p}_error')
    return lines


# Figure style: the reference palette of the dyco docs' chart guidance. Day and
# night are its first two categorical slots, which stay apart under every
# colour-vision deficiency; each also has its own marker, so print in grey and
# readers who see no hue still tell them apart. Everything else is ink.
_INK, _INK_2, _MUTED = '#0b0b0b', '#52514e', '#898781'
_GRID, _AXIS, _BAND = '#e1e0d9', '#c3c2b7', '#ecebe6'
_DAY, _NIGHT = '#eb6834', '#2a78d6'


def _fmt3(x) -> str:
    """Three significant digits, the precision a figure label needs."""
    return f'{x:.3g}' if np.isfinite(x) else '--'


def _gas_tex(label: str) -> str:
    """``CH4`` -> ``CH$_4$``, ``N2O`` -> ``N$_2$O``: chemical subscripts for labels."""
    import re
    return re.sub(r'(?<=[A-Za-z])(\d+)', r'$_{\1}$', label)


def _unit_tex(unit: str) -> str:
    """``umol m-2 s-1`` -> ``$\\mu$mol m$^{-2}$ s$^{-1}$`` for matplotlib labels.

    The CLI takes the unit as ASCII, since console text has to be; the figure
    can show the real symbols."""
    import re
    unit = re.sub(r'\bu(?=(mol|g)\b)', r'$\\mu$', unit)
    return re.sub(r'(?<=[A-Za-z])(-?\d+)\b', r'$^{\1}$', unit)


def plot_fdl_summary(res: DataFrame, label: str, unit: str | None = None,
                     day_hours: tuple | None = (6, 18), fdl_lag_range_s: int = 180,
                     noise_range_s: int = 20, title: str | None = None):
    """A four-panel figure of one gas's flux, lag and detection limit, for publication.

    (a) The flux of every period against its detection limit: the grey band is
    +/-FDL, so a point outside it is distinguishable from zero. Filled points
    are above their FDL, open ones below it.
    (b) The lag the flux was read at, and the gas's own PWB detections with
    their 95 % HDI. Filled detections were accepted (S1/S2), open ones
    rejected; where none was accepted, the lag is not a measured one.
    (c) The detection limit of every period, all, by day and by night, with
    the median and the interquartile range, on a log axis.
    (d) The mean flux of the run, the day and the night, each with the
    detection limit of that mean (Langford et al. 2015, eq. 10) as its bar.

    A footer gives the numbers a report needs. *res* is a ``PwbFdlBatch``
    result; *unit* is the flux unit as text (e.g. ``'nmol m-2 s-1'``), written
    with superscripts. 180 mm wide, the full width of a two-column journal
    page. Returns the matplotlib Figure.
    """
    import textwrap

    from dyco.pipeline import _ensure_headless_backend
    _ensure_headless_backend()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    p = label.lower()
    r = res.loc[res['status'].astype(str).eq('ok')].copy()
    flux = pd.to_numeric(r[f'{p}_flux'], errors='coerce').to_numpy(dtype=float)
    fdl = pd.to_numeric(r[f'{p}_fdl'], errors='coerce').to_numpy(dtype=float)
    stamps = pd.to_datetime(r['timestamp'], errors='coerce') if 'timestamp' in r else None
    has_time = stamps is not None and len(stamps) > 0 and bool(stamps.notna().all())
    x = stamps.to_numpy() if has_time else np.arange(len(r))
    split = has_time and day_hours is not None
    if split:
        hours = (stamps.dt.hour + stamps.dt.minute / 60).to_numpy()
        day = (hours >= day_hours[0]) & (hours < day_hours[1])
    else:
        day = np.ones(len(r), dtype=bool)
    unit_txt = _unit_tex(unit) if unit else 'flux units'
    gas = _gas_tex(label)
    both = np.isfinite(flux) & np.isfinite(fdl)

    style_rc = {
        'font.family': 'sans-serif', 'font.size': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.edgecolor': _AXIS, 'axes.linewidth': 0.6, 'xtick.color': _INK_2,
        'ytick.color': _INK_2, 'axes.labelcolor': _INK, 'text.color': _INK,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 3,
        'ytick.major.size': 3, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    }
    with plt.rc_context(style_rc):
        fig = plt.figure(figsize=(7.09, 7.6), facecolor='white')
        # Two grids: the time series share one left edge, and the bottom row
        # gets a wider one for its two-line category labels.
        top = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.62, left=0.10,
                               right=0.98, top=0.885, bottom=0.445)
        bottom = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.62, left=0.155,
                                  right=0.98, top=0.335, bottom=0.165)
        ax_a = fig.add_subplot(top[0])
        ax_b = fig.add_subplot(top[1], sharex=ax_a)
        ax_c = fig.add_subplot(bottom[0])
        ax_d = fig.add_subplot(bottom[1])

        def style(ax, letter, heading, title_y=1.04):
            # Panel titles are placed by hand, so a legend can sit between the
            # title and the plot.
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color=_GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.text(0, title_y, f'({letter})  {heading}', transform=ax.transAxes,
                    fontweight='bold', ha='left', va='bottom')

        def legend_above(ax, handles):
            ax.legend(handles=handles, ncol=len(handles), frameon=False, loc='lower left',
                      bbox_to_anchor=(0, 1.0), borderaxespad=0.2, handletextpad=0.4,
                      columnspacing=1.2)

        groups = ([('day', day, _DAY, 'o'), ('night', ~day, _NIGHT, 's')] if split
                  else [('all', day, _INK_2, 'o')])
        xs = np.asarray(x)

        # (a) every period's flux against its own detection limit
        style(ax_a, 'a', 'Flux and its detection limit, per period', title_y=1.17)
        ax_a.fill_between(x, -fdl, fdl, step='mid', color=_BAND, linewidth=0, zorder=1)
        ax_a.axhline(0, color=_AXIS, linewidth=0.8, zorder=2)
        above = np.abs(flux) > fdl
        for _, m, color, marker in groups:
            for sel, face in ((m & above, color), (m & ~above, 'white')):
                ax_a.scatter(xs[sel], flux[sel], s=20, marker=marker, facecolor=face,
                             edgecolor=color, linewidth=1.0, zorder=3)
        ax_a.set_ylabel(f'{gas} flux\n({unit_txt})')
        # Scale to the fluxes, not to the widest band: a few noisy periods would
        # otherwise flatten every point. A band cut by the axis runs off it.
        reach = np.nanmax(np.r_[np.abs(flux), 2 * np.nanmedian(fdl)]) if both.any() else np.nan
        if np.isfinite(reach) and reach > 0:
            ax_a.set_ylim(-1.25 * reach, 1.25 * reach)
        handles = [Patch(facecolor=_BAND, label=r'$\pm$FDL (3 $\times$ RE$_{\mathrm{RMSE}}$)')]
        handles += [Line2D([], [], linestyle='', marker=mk, markersize=5, markerfacecolor=c,
                           markeredgecolor=c, label=n) for n, _, c, mk in groups if split]
        handles += [Line2D([], [], linestyle='', marker='o', markersize=5,
                           markerfacecolor=_INK_2, markeredgecolor=_INK_2, label='above FDL'),
                    Line2D([], [], linestyle='', marker='o', markersize=5,
                           markerfacecolor='white', markeredgecolor=_INK_2, label='below FDL')]
        legend_above(ax_a, handles)
        plt.setp(ax_a.get_xticklabels(), visible=False)
        ax_a.xaxis.get_offset_text().set_visible(False)

        # (b) the lag used, and the gas's own detections with their HDI
        style(ax_b, 'b', 'Time lag', title_y=1.17)
        tlag = pd.to_numeric(r[f'{p}_tlag_s'], errors='coerce').to_numpy(dtype=float)
        lo = pd.to_numeric(r[f'{p}_hdi_lo_s'], errors='coerce').to_numpy(dtype=float)
        hi = pd.to_numeric(r[f'{p}_hdi_hi_s'], errors='coerce').to_numpy(dtype=float)
        used = pd.to_numeric(r[f'{p}_lag_s'], errors='coerce').to_numpy(dtype=float)
        source = r[f'{p}_lag_source'].astype(str).to_numpy()
        flag = r[f'{p}_flag_pf'].astype(str).to_numpy()
        accepted = (source == 'own') & np.isin(flag, ['S1_optimal', 'S2_optimal'])
        ok_hdi = np.isfinite(tlag) & np.isfinite(lo) & np.isfinite(hi)
        ax_b.vlines(xs[ok_hdi], lo[ok_hdi], hi[ok_hdi], color=_GRID, linewidth=1.6, zorder=1)
        ax_b.scatter(xs[accepted], tlag[accepted], s=14, color=_INK_2, zorder=3)
        rejected = np.isfinite(tlag) & ~accepted
        ax_b.scatter(xs[rejected], tlag[rejected], s=14, facecolor='white', edgecolor=_MUTED,
                     linewidth=0.8, zorder=3)
        ax_b.step(x, used, where='mid', color=_INK, linewidth=1.4, zorder=4)
        ax_b.set_ylabel('Lag (s)')
        legend_above(ax_b, [
            Line2D([], [], color=_INK, linewidth=1.4, label='lag used for the flux'),
            Line2D([], [], linestyle='', marker='o', markersize=4, markerfacecolor=_INK_2,
                   markeredgecolor=_INK_2, label='own detection, accepted'),
            Line2D([], [], linestyle='', marker='o', markersize=4, markerfacecolor='white',
                   markeredgecolor=_MUTED, label='own detection, rejected'),
            Line2D([], [], color=_GRID, linewidth=1.6, label='95 % HDI')])
        if has_time:
            locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
            ax_b.xaxis.set_major_locator(locator)
            ax_b.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        else:
            ax_b.set_xlabel('Period')

        # (c) the detection limit of each period, one dot strip per group
        style(ax_c, 'c', 'Detection limit per period')
        ax_c.grid(axis='y', visible=False)
        ax_c.grid(axis='x', color=_GRID, linewidth=0.5, which='major')
        rows = [('all', np.isfinite(fdl), _INK_2, 'o')]
        if split:
            rows += [(n, m & np.isfinite(fdl), c, mk) for n, m, c, mk in groups]
        rng = np.random.default_rng(0)  # jitter only, fixed so the figure is reproducible
        labels_c = []
        for i, (name, m, color, marker) in enumerate(rows):
            yv = len(rows) - 1 - i
            v = fdl[m]
            if not len(v):
                labels_c.append(f'{name} (n 0)')
                continue
            ax_c.scatter(v, yv + rng.uniform(-0.18, 0.18, len(v)), s=10, marker=marker,
                         facecolor='white', edgecolor=color, linewidth=0.7, zorder=2)
            q25, q50, q75 = np.percentile(v, [25, 50, 75])
            ax_c.hlines(yv, q25, q75, color=color, linewidth=3.0, zorder=3)
            ax_c.vlines(q50, yv - 0.3, yv + 0.3, color=_INK, linewidth=1.4, zorder=4)
            labels_c.append(f'{name} (n {len(v)})\n{_fmt3(q50)} [{_fmt3(q25)}, {_fmt3(q75)}]')
        ax_c.set_xscale('log')
        # Plain numbers on the log axis, at 1-2-5 steps: 3 x 10^0 reads badly.
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        lo_x = hi_x = np.nan
        if np.isfinite(fdl).any():
            lo_x, hi_x = np.nanmin(fdl) / 1.3, np.nanmax(fdl) * 1.3
            steps = [m * 10.0 ** e for e in range(-4, 6) for m in (1, 2, 5)]
            ax_c.xaxis.set_major_locator(FixedLocator([t for t in steps if lo_x <= t <= hi_x]))
        ax_c.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax_c.xaxis.set_minor_formatter(NullFormatter())
        ax_c.set_yticks(range(len(rows)))
        ax_c.set_yticklabels(labels_c[::-1], linespacing=1.3)
        ax_c.set_ylim(-0.6, len(rows) - 0.4)
        if np.isfinite(lo_x):
            ax_c.set_xlim(lo_x, hi_x)
        ax_c.set_xlabel(f'FDL ({unit_txt}), log scale\nlabels: median [IQR]')
        ax_c.tick_params(axis='y', length=0)
        ax_c.spines['left'].set_visible(False)

        # (d) the means, each with the detection limit of that mean as its bar
        style(ax_d, 'd', 'Means and their FDL')
        ax_d.grid(axis='y', visible=False)
        ax_d.grid(axis='x', color=_GRID, linewidth=0.5)
        means = [('all', both, _INK_2, 'o')]
        if split:
            means += [(n, m & both, c, mk) for n, m, c, mk in groups]
        ax_d.axvline(0, color=_AXIS, linewidth=0.8)
        labels_d = []
        for i, (name, m, color, marker) in enumerate(means):
            yv = len(means) - 1 - i
            if not m.any():
                labels_d.append(f'{name} (N 0)')
                continue
            mean = flux[m].mean()
            fdl_mean = np.sqrt((fdl[m] ** 2).sum()) / m.sum()
            hit = abs(mean) > fdl_mean
            ax_d.errorbar(mean, yv, xerr=fdl_mean, fmt=marker, markersize=5, color=color,
                          markerfacecolor=color if hit else 'white', elinewidth=1.4, capsize=0)
            labels_d.append(f'{name} (N {int(m.sum())})\n{_fmt3(mean)} $\\pm$ {_fmt3(fdl_mean)}')
        ax_d.set_yticks(range(len(means)))
        ax_d.set_yticklabels(labels_d[::-1], linespacing=1.3)
        ax_d.set_ylim(-0.6, len(means) - 0.4)
        ax_d.set_xlabel(f'Mean $\\pm$ FDL of the mean\n({unit_txt})')
        ax_d.tick_params(axis='y', length=0)
        ax_d.spines['left'].set_visible(False)

        # The numbers to report, under the figure.
        f_ok = fdl[np.isfinite(fdl)]
        q25, q50, q75 = np.percentile(f_ok, [25, 50, 75]) if len(f_ok) else (np.nan,) * 3
        mean_all = flux[both].mean() if both.any() else np.nan
        fdl_all = np.sqrt((fdl[both] ** 2).sum()) / both.sum() if both.any() else np.nan
        n_median = int((source == 'median').sum())
        lag_txt = (f'Lag: {int(accepted.sum())} of {len(r)} periods with an accepted own '
                   f'detection' + (f'; {n_median} read at the median of all raw detections, '
                                   f'not at a lag measured in the period.' if n_median
                                   else '.'))
        if split:
            lag_txt += f' Day: {day_hours[0]:02d}-{day_hours[1]:02d} h by period start.'
        foot = [
            f'FDL = 3 $\\times$ RE$_{{\\mathrm{{RMSE}}}}$ of the cross-covariance at '
            f'$\\pm${fdl_lag_range_s - noise_range_s}-{fdl_lag_range_s} s (Langford et al. '
            f'2015, eq. 9); a flux is above its FDL when |flux| > FDL.',
            f'FDL of a mean of N periods = $\\sqrt{{\\Sigma\\, \\mathrm{{FDL}}_i^2}}\\,/\\,N$ '
            f'(Langford et al. 2015, eq. 10).',
            f'{len(f_ok)} periods. Median FDL {_fmt3(q50)} (IQR {_fmt3(q25)}-{_fmt3(q75)}) '
            f'{unit_txt}; {int(above[both].sum())} of {int(both.sum())} periods above their '
            f'FDL; mean flux {_fmt3(mean_all)}, FDL of the mean {_fmt3(fdl_all)} {unit_txt}.',
            lag_txt,
        ]
        # Wrap plain text only: a break inside $...$ would break the mathtext.
        wrapped = []
        for para in foot:
            if '$' in para:
                wrapped.append(para)
            else:
                wrapped += textwrap.wrap(para, width=160, break_on_hyphens=False)
        fig.text(0.02, 0.012, '\n'.join(wrapped), ha='left', va='bottom', fontsize=6.3,
                 color=_INK_2, linespacing=1.6)
        fig.suptitle(title or f'{gas}: flux detection limit', x=0.02, y=0.985, ha='left',
                     va='top', fontsize=10, fontweight='bold')
    return fig


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
    flux_units = dict(args.flux_units or [])
    unknown_units = sorted(set(flux_units) - set(scalars))
    if unknown_units:
        fail(f'--flux-unit names gas label(s) {unknown_units} that no --scalar has; '
             f'known labels are {sorted(scalars)}.')

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
            detail = '  detected ' + ' | '.join(_detected_cell(row, g) for g in scalars)
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
        for line in run_summary(res, scalars, scale=scale, hdi_thresh=args.hdi_thresh,
                                hdi_prefilter=args.hdi_prefilter,
                                min_chunk_seconds=args.min_chunk_seconds,
                                fdl_lag_range_s=args.fdl_lag_range,
                                noise_range_s=args.noise_range,
                                day_hours=args.day_hours, units=flux_units):
            out(line)
        out()
        # One figure per gas, PNG for a look and PDF (vector, fonts embedded)
        # for a manuscript. A figure that fails must not cost the results.
        for label in scalars:
            stem = output_dir / f'fdl_{label.lower()}'
            try:
                fig = plot_fdl_summary(res, label, unit=flux_units.get(label),
                                       day_hours=args.day_hours,
                                       fdl_lag_range_s=args.fdl_lag_range,
                                       noise_range_s=args.noise_range)
                fig.savefig(stem.with_suffix('.png'), dpi=300)
                fig.savefig(stem.with_suffix('.pdf'))
                import matplotlib.pyplot as plt
                plt.close(fig)
                out(f'Figure saved to: {stem}.png and .pdf')
            except Exception as e:
                out(f'WARNING: no figure for {label}: {type(e).__name__}: {e}')
        for label in scalars:
            p = label.lower()
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
