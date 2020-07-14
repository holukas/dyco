import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


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


def format_spines(ax, color, lw):
    spines = ['top', 'bottom', 'left', 'right']
    for spine in spines:
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(lw)


def setup_fig_ax():
    """Setup grid with one figure and one axis."""
    gs = gridspec.GridSpec(1, 1)  # rows, cols
    gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
    fig = plt.Figure(facecolor='white', figsize=(16, 9))
    ax = fig.add_subplot(gs[0, 0])
    return gs, fig, ax
