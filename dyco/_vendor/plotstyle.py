"""
PLOTSTYLE: DEFAULT MATPLOTLIB AXIS FORMATTING
==============================================

Chrome formatting for matplotlib axes, used by the covariance plots.

Copied from diive (`core/plotting/plotfuncs.py` and
`core/plotting/styles/LightTheme.py`). Only the constants actually read by
`default_format` are carried over; the rest of diive's theme is not.

Note dyco also has an older `plot.default_format` with a different signature
(`txt_xlabel=` rather than `ax_xlabel_txt=`). That one is used by dyco's own
v2 modules. This one exists so the covariance plotting code copied from diive
works unchanged. The two collapse into one during the v3 restructure.

Part of the dyco package: https://github.com/holukas/dyco
"""

# Theme constants (diive LightTheme)
LINEWIDTH_SPINES = 1
COLOR_LINE_GRID = '#B0BEC5'  # Material Blue Grey 200
AX_LABELS_FONTSIZE = 16
AX_LABELS_FONTCOLOR = '#000000'
AX_LABELS_FONTWEIGHT = 'normal'
TICKS_WIDTH = 1
TICKS_LENGTH = 4
TICKS_DIRECTION = 'in'
TICKS_LABELS_FONTSIZE = 16


def show_ticks_on_all_spines(ax, left=True, right=True, top=True, bottom=True):
    """Enable tick marks on the selected spines."""
    ax.tick_params(left=left, right=right, top=top, bottom=bottom)


def format_ticks(ax, width, length, direction, color, labelsize):
    """Apply consistent tick width/length/direction/colour/label size on both axes."""
    ax.tick_params(axis='x', width=width, length=length, direction=direction,
                   colors=color, labelsize=labelsize)
    ax.tick_params(axis='y', width=width, length=length, direction=direction,
                   colors=color, labelsize=labelsize)
    show_ticks_on_all_spines(ax)


def format_spines(ax, color, lw):
    """Set the colour and line width of all four spines."""
    lw = LINEWIDTH_SPINES if not lw else lw
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(lw)


def default_grid(ax):
    """Draw the default dashed grid."""
    ax.grid(True, ls='--', color=COLOR_LINE_GRID, lw=LINEWIDTH_SPINES, zorder=0)


def default_format(ax,
                   ax_labels_fontsize: float = AX_LABELS_FONTSIZE,
                   ax_labels_fontcolor: str = AX_LABELS_FONTCOLOR,
                   ax_labels_fontweight=AX_LABELS_FONTWEIGHT,
                   ax_xlabel_txt=False,
                   ax_ylabel_txt=False,
                   spines_lw: float = None,
                   txt_ylabel_units=False,
                   ticks_width=TICKS_WIDTH,
                   ticks_length=TICKS_LENGTH,
                   ticks_direction=TICKS_DIRECTION,
                   ticks_labels_fontsize=TICKS_LABELS_FONTSIZE,
                   color='black',
                   facecolor='white',
                   showgrid: bool = True) -> None:
    """Apply default format to ax."""
    ax.set_facecolor(facecolor)

    format_ticks(ax=ax, width=ticks_width, length=ticks_length,
                 direction=ticks_direction, color=color,
                 labelsize=ticks_labels_fontsize)

    format_spines(ax=ax, color=color, lw=spines_lw)

    # The "no label" default for these parameters is False, which matplotlib
    # would render literally as the string "False" - pass an empty label.
    ax.set_xlabel(ax_xlabel_txt if ax_xlabel_txt else '', color=ax_labels_fontcolor,
                  fontsize=ax_labels_fontsize, fontweight=ax_labels_fontweight)

    if ax_ylabel_txt and txt_ylabel_units:
        _ax_ylabel_txt = f"{ax_ylabel_txt}  {txt_ylabel_units}"
    elif ax_ylabel_txt and not txt_ylabel_units:
        _ax_ylabel_txt = f"{ax_ylabel_txt}"
    else:
        _ax_ylabel_txt = ''
    ax.set_ylabel(_ax_ylabel_txt, color=ax_labels_fontcolor, fontsize=ax_labels_fontsize,
                  fontweight=ax_labels_fontweight)

    if showgrid:
        default_grid(ax=ax)
    else:
        ax.grid(False)
