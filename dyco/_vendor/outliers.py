"""
OUTLIERS: ROLLING Z-SCORE OUTLIER FLAGGING
===========================================

Flag outliers by the z-score computed within a centred rolling window,
iterating until no further outliers are found.

Ported from diive's `zScoreRolling` (`preprocessing/outlier_detection/zscore.py`)
together with the iteration driver it inherited from `FlagBase.repeat`. The
detection maths and the flag semantics are reproduced exactly; only the class
scaffolding is dropped, because carrying it over meant importing roughly 936
lines across five diive modules to serve a single call site.

[DELIBERATE DEVIATION] diive's `FlagBase.__init__` regularizes the input by
calling `series.asfreq(detected_freq)` when the index carries no frequency. That
is not done here. dyco applies this to a series of high-quality covariance peaks
indexed by segment start time, which is inherently irregular - segments are
missing wherever a raw file was missing or its peak was low quality.
Regularizing would insert rows the caller never had, and the returned flag would
then not align with the caller's own series. Results may therefore differ from
dyco v2 on data where diive's frequency detection previously succeeded; verify
against a v2 reference run before trusting the numbers.

Part of the dyco package: https://github.com/holukas/dyco
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series

from dyco._vendor.plotstyle import default_format

FLAG_OK = 0
FLAG_REJECTED = 2


def rolling_zscore_flag(series: Series,
                        thres_zscore: float = 4,
                        winsize: int = None,
                        repeat: bool = True,
                        plot: bool = False,
                        plottitle: str = None):
    """Flag outliers using an iterated centred rolling z-score.

    Each iteration computes a rolling mean and standard deviation centred on
    every record, forms `|(x - mean) / sd|`, and rejects records above
    *thres_zscore*. Rejected records are set to missing and the next iteration
    runs on what remains, so a single extreme value cannot mask its neighbours.

    Args:
        series: Values to screen. The index is used as-is (see the module
            docstring on regularization).
        thres_zscore: Rejection threshold. Records with a z-score above this
            are flagged. Must be positive.
        winsize: Rolling window in records. If None, `len(series) / 20` (5% of
            the data) is used, computed once on the first iteration and held
            fixed for the rest. Must be at least 3 if given explicitly.
        repeat: If True, iterate until an iteration finds no further outliers.
            If False, run a single pass.
        plot: If True, also build and return a diagnostic figure.
        plottitle: Optional title for that figure.

    Returns:
        `(flag, fig)` where *flag* is a Series over `series.index` carrying 0
        for kept and 2 for rejected, and *fig* is a matplotlib Figure or None.

    Notes:
        Records whose z-score is undefined (NaN, e.g. where the rolling SD is
        zero or the window has too few values) are neither accepted nor
        rejected during an iteration, and end up flagged 0. This matches diive.
    """
    if thres_zscore <= 0:
        raise ValueError('thres_zscore must be positive.')
    if winsize is not None and winsize < 3:
        raise ValueError('winsize must be at least 3 records.')

    filtered = series.copy().astype(float)
    iteration_flags = []
    n_outliers = 9999
    iteration = 0

    while n_outliers > 0:
        iteration += 1

        s = filtered.dropna()

        if not winsize:
            # Computed once, on the first iteration, then held fixed - matching
            # diive, where this assigns to self.winsize and so persists.
            winsize = int(len(s) / 20)

        rmean = s.rolling(window=winsize, center=True, min_periods=3).mean()
        rsd = s.rolling(window=winsize, center=True, min_periods=3).std()
        rzscore = np.abs((s - rmean) / rsd)

        ok = rzscore <= thres_zscore
        ok = ok[ok].index
        rejected = rzscore > thres_zscore
        rejected = rejected[rejected].index

        flag = pd.Series(index=filtered.index, data=np.nan)
        flag.loc[ok] = FLAG_OK
        flag.loc[rejected] = FLAG_REJECTED
        iteration_flags.append(flag)

        filtered.loc[rejected] = np.nan
        n_outliers = len(rejected)

        if not repeat:
            break

    # Sum only the 2s across iterations: a record rejected in any iteration ends
    # up 2, everything else (including never-tested NaN z-scores) ends up 0.
    iteration_flags_df = pd.concat(iteration_flags, axis=1)
    overall_flag = iteration_flags_df[iteration_flags_df == FLAG_REJECTED].sum(axis=1)
    overall_flag.name = f"FLAG_{series.name}_OUTLIER_ZSCOREROLLING_TEST"

    fig = None
    if plot:
        fig = _plot(series=series, flag=overall_flag, n_iterations=len(iteration_flags),
                    thres_zscore=thres_zscore, winsize=winsize, plottitle=plottitle)

    return overall_flag, fig


def _plot(series: Series, flag: Series, n_iterations: int,
          thres_zscore: float, winsize: int, plottitle: str = None):
    """Build the diagnostic figure: input with rejects marked, and the result."""
    ok = flag == FLAG_OK
    rejected = flag == FLAG_REJECTED

    fig = plt.Figure(facecolor='white', figsize=(16, 9))
    gs = gridspec.GridSpec(2, 1)
    gs.update(wspace=0.3, hspace=0.3, left=0.06, right=0.97, top=0.92, bottom=0.06)
    ax_before = fig.add_subplot(gs[0, 0])
    ax_after = fig.add_subplot(gs[1, 0], sharex=ax_before)

    ax_before.plot(series.index, series, marker='o', ms=4, ls='none',
                   color='#455A64', label='input', zorder=98)
    if rejected.sum():
        ax_before.plot(series.index[rejected], series[rejected], marker='X', ms=10, ls='none',
                       color='#F44336', label=f'rejected ({int(rejected.sum())})', zorder=99)
    ax_before.legend(frameon=False, loc='upper right')
    default_format(ax=ax_before, ax_ylabel_txt=str(series.name),
                   ax_labels_fontsize=12, ticks_labels_fontsize=12)

    ax_after.plot(series.index[ok], series[ok], marker='o', ms=4, ls='none',
                  color='#2196F3', label=f'kept ({int(ok.sum())})', zorder=98)
    ax_after.legend(frameon=False, loc='upper right')
    default_format(ax=ax_after, ax_ylabel_txt=str(series.name),
                   ax_labels_fontsize=12, ticks_labels_fontsize=12)

    title = plottitle if plottitle else 'Rolling z-score outlier removal'
    fig.suptitle(f"{title}\n"
                 f"threshold {thres_zscore}, window {winsize} records, "
                 f"{n_iterations} iteration(s)",
                 fontsize=14, fontweight='bold')
    return fig
