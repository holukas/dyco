import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import files
import os
def prepare_df(df):
    df['filename'] = df.index
    df.reset_index(drop=True, inplace=True)
    df['file_idx'] = df.index
    df['cov_max_shift'] = df['cov_max_shift'].astype(int)
    return df


def segment_lagtimes(df, outdir, iteration, show_all=False):
    print(f"Saving plot in HTML file: _found_lag_times.html ...")

    _df = df.copy()
    if not show_all:
        _df = _df.loc[_df['iteration'] == iteration, :]
        txt_info=f"Found lag times in iteration {iteration}"
        outfile = f"{iteration}_TIMESERIES-PLOT_found_segment_lag_times_iteration-{iteration}"
    else:
        txt_info = f"Found lag times from ALL iterations"
        outfile = f"TIMESERIES-PLOT_found_segment_lag_times_iteration-ALL"

    #todo hier weiter
    fig = timeseries_lagtimes(x=pd.to_datetime(_df['start']),
                              y=_df['cov_max_shift'],
                              txt_info=txt_info,
                              color='#42A5F5')

    outpath = outdir / outfile
    fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w',
                transparent=True, dpi=150)


def results(df):
    # todo
    cols = ['shift_median', 'cov_max_shift', 'shift_P25', 'shift_P75', 'search_win_upper', 'search_win_lower']
    df[cols].plot()
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


def make_scatter_cov(x, y, txt_info, cov_max_shift, cov_max, z_color):
    """Make scatter plot with z-values as colors and display found max covariance."""

    # import plotly.graph_objects as go
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=x, y=y,
    #                          mode='markers',
    #                          marker=dict(color=z_color, opacity=1, size=10,
    #                                      showscale=True, colorscale=[[0, '#2196F3'], [1, '#f44336']])
    #                          ))
    #
    # # Edit the layout
    # fig.update_layout(title='Covariance',
    #                   xaxis=dict(title='shift [records]',
    #                              gridcolor='#eee', gridwidth=1,
    #                              linewidth=1, linecolor='black', mirror=True,
    #                              tickmode='auto', ticks='inside', ),
    #                   yaxis=dict(title='covariance',
    #                              gridcolor='#eee', gridwidth=1,
    #                              linewidth=1, linecolor='black', mirror=True,
    #                              zeroline=True, zerolinewidth=1, zerolinecolor='black',
    #                              tickmode='auto', ticks='inside', ),
    #                   plot_bgcolor='#fff',
    #                   )

    # fig.show()
    # from plotly.io import to_html
    # xxx = to_html(fig)

    gs = gridspec.GridSpec(1, 1)  # rows, cols
    gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)

    fig = plt.Figure(facecolor='white', figsize=(16, 9))
    ax = fig.add_subplot(gs[0, 0])

    if cov_max:
        ax.scatter(x, y, alpha=0.9, edgecolors='none',
                   marker='o', s=24, c=z_color, cmap='coolwarm', zorder=98)

        # From: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html
        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be (numlines) x (points per line) x 2 (for x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(z_color.min(), z_color.max())
        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        # Set the values used for colormapping
        lc.set_array(z_color)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax)



        ax.scatter(cov_max_shift, cov_max, alpha=1, edgecolors='red',
                   marker='o', s=64, c='red', zorder=99)

        from scipy.signal import find_peaks
        import lag
        # p=int(lag.FindHistogramPeaks.search_bin_most_prominent(z_color))
        p,_=find_peaks(z_color, width=10, prominence=2)#todo hier weiter

        # p,_=find_peaks(z_color, prominence=1, distance=1, height=1, threshold=1, width=1)

        ax.scatter(x[p], y.iloc[p], alpha=1, edgecolors='black',
                   marker='o', s=128, c='None', zorder=90)

    default_format(ax=ax, label_color='black', fontsize=12,
                   txt_xlabel='shift [records]', txt_ylabel='covariance', txt_ylabel_units='-')

    ax.text(0.02, 0.98, txt_info,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes,
            size=12, color='black', backgroundcolor='none', zorder=100)

    if not cov_max:
        ax.text(0.5, 0.5, "Not enough values",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes,
                size=20, color='#ef5350', backgroundcolor='none', zorder=100)
        #todo

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


def timeseries_lagtimes(x, y, txt_info, color):
    """Make time series plot of found lag times."""

    gs = gridspec.GridSpec(1, 1)  # rows, cols
    gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)

    fig = plt.Figure(facecolor='white', figsize=(16, 9))
    ax = fig.add_subplot(gs[0, 0])

    ax.plot_date(x, y, alpha=.9, marker='o', ms=6, color=color, lw=1, ls='-')

    default_format(ax=ax, label_color='black', fontsize=12,
                   txt_xlabel='segment', txt_ylabel='found lag time', txt_ylabel_units='[records]')

    # years = mdates.YearLocator()  # every year
    # months = mdates.MonthLocator()  # every month
    # years_fmt = mdates.DateFormatter('%Y')
    # ax.xaxis.set_major_locator(years)
    # ax.xaxis.set_major_formatter(years_fmt)
    # ax.xaxis.set_minor_locator(months)
    # myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
    # ax.xaxis.set_major_formatter(myFmt)

    ax.text(0.02, 0.98, txt_info,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes,
            size=12, color='black', backgroundcolor='none', zorder=100)

    return fig

def cov_collection(indir, outdir):
    print("Plotting covariance collection ...")
    fig = plt.Figure(facecolor='white', figsize=(16, 10))
    gs = gridspec.GridSpec(2, 1)  # rows, cols
    gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])


    # Read results from last iteration
    collection_df = pd.DataFrame()
    for filename in os.listdir(indir):
        filepath = os.path.join(indir, filename)
        segment_cov_df = files.read_segments_file(filepath=filepath)
        collection_df = collection_df.append(segment_cov_df)

        ax.plot(segment_cov_df['shift'], segment_cov_df['cov_abs'],
                alpha=0.1, c='black', lw=0.5,
                marker='None', zorder=98)

        ax2.plot(segment_cov_df['shift'], segment_cov_df['cov_abs'].divide(segment_cov_df['cov_abs'].max()),
                alpha=0.1, c='black', lw=0.5,
                marker='None', zorder=98)

    # todo to this proper
    _df = collection_df[collection_df['segment_name'].str.contains('iter')]
    _df = _df.groupby('shift').agg('median')
    ax.plot(_df.index, _df['cov_abs'],
            alpha=1, c='red', lw=1,
            marker='None', zorder=98)
    ax2.plot(_df.index, _df['cov_abs'].divide(_df['cov_abs'].max()),
            alpha=1, c='red', lw=1,
            marker='None', zorder=98)

    outfile = '1_covariance_collection_all_segments'
    outpath = outdir / outfile
    fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w',
                transparent=True, dpi=150)