import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

import files


def prepare_df(df):
    df['filename'] = df.index
    df.reset_index(drop=True, inplace=True)
    df['file_idx'] = df.index
    df['cov_max_shift'] = df['cov_max_shift'].astype(int)
    return df


def timeseries_segment_lagtimes(df, outdir, iteration, show_all=False):
    _df = df.copy()
    if not show_all:
        _df = _df.loc[_df['iteration'] == iteration, :]
        txt_info = f"Found lag times in iteration {iteration}"
        outfile = f"{iteration}_TIMESERIES-PLOT_found_segment_lag_times_iteration-{iteration}"
    else:
        txt_info = f"Found lag times from ALL iterations"
        outfile = f"TIMESERIES-PLOT_found_segment_lag_times_iteration-ALL"

    gs, fig, ax = setup_fig_ax()

    cmap = plt.cm.get_cmap('RdBu_r', iteration)
    colors = cmap(np.linspace(0, 1, iteration))

    alpha = 0.5
    iteration_grouped = _df.groupby('iteration')
    for idx, group_df in iteration_grouped:
        # alpha+=0.1
        ax.plot_date(pd.to_datetime(group_df['start']), group_df['shift_peak_cov_abs_max'],
                     alpha=alpha, marker='o', ms=6, color=colors[int(idx - 1)], lw=1, ls='-')

    default_format(ax=ax, label_color='black', fontsize=12,
                   txt_xlabel='segment', txt_ylabel='found lag time', txt_ylabel_units='[records]')

    ax.text(0.02, 0.98, txt_info,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes,
            size=12, color='black', backgroundcolor='none', zorder=100)

    outpath = outdir / outfile
    print(f"Saving time series of found segment lag times in {outpath} ...")
    fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w',
                transparent=True, dpi=150)

    # years = mdates.YearLocator()  # every year
    # months = mdates.MonthLocator()  # every month
    # years_fmt = mdates.DateFormatter('%Y')
    # ax.xaxis.set_major_locator(years)
    # ax.xaxis.set_major_formatter(years_fmt)
    # ax.xaxis.set_minor_locator(months)
    # myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
    # ax.xaxis.set_major_formatter(myFmt)


def results(df):
    # todo
    cols = ['shift_median', 'cov_max_shift', 'shift_P25', 'shift_P75', 'search_win_upper', 'search_win_lower']
    _df = df[cols].copy()

    _df[cols].plot()
    plt.show()
    return None


def limit_data_range_percentiles(df, col, perc_limits):
    p_lower = df[col].quantile(perc_limits[0])
    p_upper = df[col].quantile(perc_limits[1])
    p_filter = (df[col] >= p_lower) & (df[col] <= p_upper)
    df = df[p_filter]
    return df


def default_format(ax, fontsize=12, label_color='black',
                   txt_xlabel='', txt_ylabel='', txt_ylabel_units='',
                   width=1, length=5, direction='in', colors='black', facecolor='white'):
    """Apply default format to plot."""
    ax.set_facecolor(facecolor)
    ax.tick_params(axis='x', width=width, length=length, direction=direction, colors=colors, labelsize=fontsize)
    ax.tick_params(axis='y', width=width, length=length, direction=direction, colors=colors, labelsize=fontsize)
    format_spines(ax=ax, color=colors, lw=1)
    if txt_xlabel:
        ax.set_xlabel(txt_xlabel, color=label_color, fontsize=fontsize, fontweight='bold')
    if txt_ylabel and txt_ylabel_units:
        ax.set_ylabel(f'{txt_ylabel}  {txt_ylabel_units}', color=label_color, fontsize=fontsize, fontweight='bold')
    if txt_ylabel and not txt_ylabel_units:
        ax.set_ylabel(f'{txt_ylabel}', color=label_color, fontsize=fontsize, fontweight='bold')
    return None


def format_spines(ax, color, lw):
    spines = ['top', 'bottom', 'left', 'right']
    for spine in spines:
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(lw)
    return None


def setup_fig_ax():
    """Setup grid with one figure and one axis."""
    gs = gridspec.GridSpec(1, 1)  # rows, cols
    gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
    fig = plt.Figure(facecolor='white', figsize=(16, 9))
    ax = fig.add_subplot(gs[0, 0])
    return gs, fig, ax


def make_scatter_cov(df, idx_peak_cov_abs_max, idx_peak_auto, iteration, win_lagsearch,
                     segment_name, segment_start, segment_end, filename,
                     file_idx, shift_stepsize, props_peak_auto):
    """Make scatter plot with z-values as colors and display found max covariance."""

    gs, fig, ax = setup_fig_ax()

    # Time series of covariances per shift
    x_shift = df.loc[:, 'shift']
    y_cov = df.loc[:, 'cov']
    z_cov_abs = df.loc[:, 'cov_abs']
    ax.scatter(x_shift, y_cov, c=z_cov_abs,
               alpha=0.9, edgecolors='none',
               marker='o', s=24, cmap='coolwarm', zorder=98)

    # z values as colors
    # From: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x_shift, y_cov]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(z_cov_abs.min(), z_cov_abs.max())
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    # Set the values used for colormapping
    lc.set_array(z_cov_abs)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('absolute covariance', rotation=90)

    txt_info = \
        f"Iteration: {iteration}\n" \
        f"Time lag search window: from {win_lagsearch[0]} to {win_lagsearch[1]} records\n" \
        f"Segment name: {segment_name}\n" \
        f"Segment start: {segment_start}\n" \
        f"Segment end: {segment_end}\n" \
        f"File: {filename} - File date: {file_idx}\n" \
        f"Lag search step size: {shift_stepsize} records\n"

    # Mark max cov abs peak
    if idx_peak_cov_abs_max:
        ax.scatter(df.iloc[idx_peak_cov_abs_max]['shift'],
                   df.iloc[idx_peak_cov_abs_max]['cov'],
                   alpha=1, edgecolors='red', marker='o', s=72, c='red',
                   label='maximum absolute covariance', zorder=99)

        txt_info += \
            f"\nFOUND PEAK MAX ABS COV\n" \
            f"    cov {df.iloc[idx_peak_cov_abs_max]['cov']:.3f}\n" \
            f"    record {df.iloc[idx_peak_cov_abs_max]['shift']}\n"

    else:
        txt_info += \
            f"\n(!)NO PEAK MAX ABS COV FOUND\n"

    # Mark auto-detected peak
    if idx_peak_auto:
        ax.scatter(df.iloc[idx_peak_auto]['shift'],
                   df.iloc[idx_peak_auto]['cov'],
                   alpha=1, edgecolors='black', marker='o', s=200, c='None',
                   label='auto-detected peak', zorder=90)

        txt_info += \
            f"\nFOUND AUTO-PEAK\n" \
            f"    cov {df.iloc[idx_peak_auto]['cov']:.3f}\n" \
            f"    record {df.iloc[idx_peak_auto]['shift']}\n" \
            f"    prominence {props_peak_auto['prominences']}\n" \
            f"    width {props_peak_auto['widths']}\n" \
            f"    width_height {props_peak_auto['width_heights']}\n"

    else:
        txt_info += \
            f"\n(!)NO AUTO-PEAK FOUND\n"

    ax.text(0.02, 0.98, txt_info,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, size=10, color='black', backgroundcolor='none', zorder=100)

    default_format(ax=ax, label_color='black', fontsize=12,
                   txt_xlabel='shift [records]', txt_ylabel='covariance', txt_ylabel_units='-')

    ax.legend(frameon=False, loc='upper right').set_zorder(100)

    return fig


# def lagsearch_collection(lagsearch_df_collection, outdir):
#     fig = plt.Figure(facecolor='white', figsize=(16, 9))
#     gs = gridspec.GridSpec(1, 1)  # rows, cols
#     gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
#     ax = fig.add_subplot(gs[0, 0])
#
#     grouped = lagsearch_df_collection.groupby('segment_name')
#     for segment_key,segment_df in grouped:
#         ax.plot(segment_df['shift'], segment_df['cov_abs'],
#                    alpha=0.1, c='black',
#                    marker='None', zorder=98)
#
#         # ax.scatter(lagsearch_df_collection['shift'], lagsearch_df_collection['cov_abs'],
#         #            alpha=0.2, edgecolors='none', c='black',
#         #            marker='o', s=12, zorder=98)
#
#     outfile = '1_lagsearch_cov_abs_collection'
#     outpath = outdir / outfile
#     fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w',
#                 transparent=True, dpi=150)
#     return None


def cov_collection(indir, outdir):
    """
    Read and plot segment covariance files.

    Parameters
    ----------
    indir: Path
    outdir: Path

    Returns
    -------
    None

    """
    print("Plotting covariance collection ...")

    gs = gridspec.GridSpec(3, 1)  # rows, cols
    gs.update(wspace=0.3, hspace=0.2, left=0.03, right=0.97, top=0.95, bottom=0.03)
    fig = plt.Figure(facecolor='white', figsize=(16, 12))
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    # Read results from last iteration
    collection_df = pd.DataFrame()
    filelist = os.listdir(indir)
    num_files = len(filelist)
    for idx, filename in enumerate(filelist):
        print(f"Reading segment covariance file {idx + 1} of {num_files}: {filename}")
        filepath = os.path.join(indir, filename)
        segment_cov_df = files.read_segments_file(filepath=filepath)
        collection_df = collection_df.append(segment_cov_df)

        ax1.plot(segment_cov_df['shift'], segment_cov_df['cov'],
                 alpha=0.1, c='black', lw=0.5,
                 marker='None', zorder=98)

        ax2.plot(segment_cov_df['shift'], segment_cov_df['cov_abs'],
                 alpha=0.1, c='black', lw=0.5,
                 marker='None', zorder=98)

        ax3.plot(segment_cov_df['shift'], segment_cov_df['cov_abs'].divide(segment_cov_df['cov_abs'].max()),
                 alpha=0.1, c='black', lw=0.5,
                 marker='None', zorder=98)

    # todo to this proper
    # Median lines
    _df = collection_df[collection_df['segment_name'].str.contains('iter')]
    _df = _df.groupby('shift').agg('median')
    args = dict(alpha=1, c='red', lw=1, marker='None', zorder=98)
    ax1.plot(_df.index, _df['cov'], label='median', **args)
    ax2.plot(_df.index, _df['cov_abs'], **args)
    ax3.plot(_df.index, _df['cov_abs'].divide(_df['cov_abs'].max()), **args)

    fig.suptitle("Results for all segments and from all iterations")
    text_args = dict(horizontalalignment='left', verticalalignment='top',
                     size=12, color='black', backgroundcolor='none', zorder=100)
    ax1.text(0.02, 0.97, "Covariances",
             transform=ax1.transAxes, **text_args)
    ax2.text(0.02, 0.97, "Absolute covariances",
             transform=ax2.transAxes, **text_args)
    ax3.text(0.02, 0.97, "Normalized absolute covariances\n(normalized to max of median line)",
             transform=ax3.transAxes, **text_args)

    ax1.legend(frameon=False, loc='upper right')
    default_format(ax=ax1, txt_xlabel='', txt_ylabel='covariance', txt_ylabel_units='')
    default_format(ax=ax2, txt_xlabel='', txt_ylabel='absolute covariance', txt_ylabel_units='')
    default_format(ax=ax3, txt_xlabel='shift [records]', txt_ylabel='normalized absolute covariance',
                   txt_ylabel_units='')

    outfile = '1_covariance_collection_all_segments'
    outpath = outdir / outfile
    fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w',
                transparent=True, dpi=150)
