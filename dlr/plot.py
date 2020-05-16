import altair as alt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def prepare_df(df):
    df['filename'] = df.index
    df.reset_index(drop=True, inplace=True)
    df['file_idx'] = df.index
    df['cov_max_shift'] = df['cov_max_shift'].astype(int)
    return df


def segment_lagtimes(df, dir_output):
    print(f"Saving plot in HTML file: _found_lag_times.html ...")

    fig = make_scatter_lagtimes(x=df.index,
                                y=df['cov_max_shift'],
                                txt_info='Found lag times')

    outpath = dir_output / '3_found_segment_lag_times'
    fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w',
                transparent=True, dpi=150)


def results(df):
    # todo
    cols = ['shift_median', 'cov_max_shift', 'shift_P25', 'shift_P75', 'search_win_upper', 'search_win_lower']
    df[cols].plot()
    # plt.show()
    return None


def limit_data_range_percentiles(df, col, perc_limits):
    p_lower = df[col].quantile(perc_limits[0])
    p_upper = df[col].quantile(perc_limits[1])
    p_filter = (df[col] >= p_lower) & (df[col] <= p_upper)
    df = df[p_filter]
    return df


def plot_results_hist(df, dir_output):
    _df = df.copy()
    # _df['cov_max_shift_perclimited'] = np.nan

    _df = limit_data_range_percentiles(df=_df, col='cov_max_shift', perc_limits=[0.25, 0.75])

    num_bins = len(_df['cov_max_shift'].unique())
    chart = alt.Chart(_df).mark_bar().encode(
        alt.X('cov_max_shift:Q', bin=alt.Bin(maxbins=num_bins)),
        y='count()',
    ).properties(
        width=1600,
        height=900,
        title='Found Lag Times'
    )

    outfile = dir_output / '_plot_results_hist'
    chart.save(f"{outfile}.html")
    return None


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

    gs = gridspec.GridSpec(1, 1)  # rows, cols
    gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)

    fig = plt.Figure(facecolor='white', figsize=(16, 9))
    ax = fig.add_subplot(gs[0, 0])

    ax.scatter(x, y, alpha=1, edgecolors='none',
               marker='o', s=16, c=z_color)

    ax.scatter(cov_max_shift, cov_max, alpha=1, edgecolors='black',
               marker='o', s=64, c='red')

    default_format(ax=ax, label_color='black', fontsize=12,
                   txt_xlabel='shift [records]', txt_ylabel='covariance', txt_ylabel_units='-')

    ax.text(0.02, 0.98, txt_info,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes,
            size=12, color='black', backgroundcolor='none', zorder=100)

    return fig


def make_scatter_lagtimes(x, y, txt_info):
    """Make scatter plot with z-values as colors and display found max covariance."""

    gs = gridspec.GridSpec(1, 1)  # rows, cols
    gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)

    fig = plt.Figure(facecolor='white', figsize=(16, 9))
    ax = fig.add_subplot(gs[0, 0])

    ax.scatter(x, y, alpha=1, edgecolors='none',
               marker='o', s=16)

    default_format(ax=ax, label_color='black', fontsize=12,
                   txt_xlabel='segment', txt_ylabel='found lag time', txt_ylabel_units='-')

    ax.text(0.02, 0.98, txt_info,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes,
            size=12, color='black', backgroundcolor='none', zorder=100)

    return fig
