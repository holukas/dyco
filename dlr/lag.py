import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import files
import plot


class LagSearch:
    def __init__(self, wind_rot_df, segment_name, ref_sig, lagged_sig, dir_out, data_timewin_lag,
                 file_idx, segment_start, segment_end, filename, iteration):
        self.wind_rot_df = wind_rot_df
        self.segment_name = segment_name
        self.segment_start = segment_start
        self.segment_end = segment_end
        self.dir_out = dir_out
        self.w_rot_turb_col = ref_sig
        self.scalar_turb_col = lagged_sig
        self.data_timewin_lag = data_timewin_lag
        self.file_idx = file_idx
        self.filename = filename
        self.iteration = iteration

        self.run()

    def run(self):
        self.cov_max_shift, self.cov_max, self.cov_max_timestamp, self.lag_search_df = \
            self.find_max_cov(df=self.wind_rot_df)

        self.save_cov_plot(x=self.lag_search_df['shift'],
                           y=self.lag_search_df['cov'],
                           z_color=self.lag_search_df['cov_abs'])

    def get(self):
        return self.cov_max_shift, self.cov_max, self.cov_max_timestamp

    def save_cov_plot(self, x, y, z_color):
        """Plot and save covariance plot for segment."""

        txt_info = \
            f"Iteration: {self.iteration}\n" \
            f"Time lag search window: from {self.data_timewin_lag[0]} to {self.data_timewin_lag[1]} records\n" \
            f"Segment name: {self.segment_name}\n" \
            f"Segment start: {self.segment_start}\n" \
            f"Segment end: {self.segment_end}\n" \
            f"File: {self.filename} - File date: {self.file_idx}\n" \
            f"Max absolute covariance {self.cov_max:.3f} found @ record {self.cov_max_shift}"

        fig = plot.make_scatter_cov(x=x, y=y, z_color=z_color,
                                    cov_max_shift=self.cov_max_shift,
                                    cov_max=self.cov_max,
                                    txt_info=txt_info)

        dir_out = self.dir_out / '2_plots_segment_covariances'
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)
        outpath = dir_out / f'{self.segment_name}_iteration-{self.iteration}'

        # Save
        print(f"Saving plot in PNG file: {outpath}.png ...")
        fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w',
                    transparent=True, dpi=150)

    def find_max_cov(self, df):
        """Find maximum absolute covariance between turbulent wind data
        and turbulent scalar data.
        """

        print("Searching maximum covariance ...")

        _df = df.copy()
        _df['index'] = _df.index

        lag_search_df = pd.DataFrame()
        lagwin_start = self.data_timewin_lag[0]
        lagwin_end = self.data_timewin_lag[1]
        lag_search_df['shift'] = range(lagwin_start, lagwin_end)  # Negative moves lagged values "upwards" in column
        lag_search_df['cov'] = np.nan
        lag_search_df['index'] = np.nan

        for ix, row in lag_search_df.iterrows():
            shift = int(row['shift'])
            # print(shift)

            try:
                if shift < 0:
                    index_shifted = str(_df['index'][-shift])  # Note the negative sign
                else:
                    index_shifted = pd.NaT
            except:
                print("S")

            scalar_data_shifted = _df[self.scalar_turb_col].shift(shift)
            cov = _df[self.w_rot_turb_col].cov(scalar_data_shifted)
            lag_search_df.loc[lag_search_df['shift'] == row['shift'], 'cov'] = cov
            lag_search_df.loc[lag_search_df['shift'] == row['shift'], 'index'] = index_shifted

        lag_search_df['cov_abs'] = lag_search_df['cov'].abs()
        cov_max_ix = lag_search_df['cov_abs'].idxmax()
        cov_max_shift = lag_search_df.iloc[cov_max_ix]['shift']
        cov_max = lag_search_df.iloc[cov_max_ix]['cov']
        cov_max_timestamp = lag_search_df.iloc[cov_max_ix]['index']

        return cov_max_shift, cov_max, cov_max_timestamp, lag_search_df


def calc_quantiles(df):
    args = dict(window=5, min_periods=1, center=True)
    df['shift_median'] = df['cov_max_shift'].rolling(**args).median()
    df['search_win_upper'] = df['shift_median'] + 100
    df['search_win_lower'] = df['shift_median'] - 100
    df['shift_P25'] = df['cov_max_shift'].rolling(**args).quantile(0.25)
    df['shift_P75'] = df['cov_max_shift'].rolling(**args).quantile(0.75)

    return df



class FindHistogramPeaks():
    def __init__(self, series, dir_output, iteration, plot=True):
        self.series = series
        self.dir_output = dir_output
        self.iteration = iteration
        self.plot = plot

        self.run()

    def run(self):
        self.timewin_lag, self.peak_idx, self.start_idx, self.end_idx, \
        self.counts, self.divisions = \
            self.find_hist_peaks()
        self.plot_results_hist(xtick_labels=self.divisions)

    def get(self):
        return self.timewin_lag

    def find_hist_peaks(self):
        """Find single clearest peak in histogram of found lag times."""
        peak_idx = []
        prom = 0  # Prominence for peak finding

        # Make histogram distribution
        counts, divisions = np.histogram(self.series, bins=10)

        # Remove first and last group from histogram. In case of unclear lag times
        # data tend to accumulate in these edge regions of the search window.
        counts = counts[1:-1]
        divisions = divisions[1:-1]  # Start values of bins

        # kudos: https://www.kaggle.com/simongrest/finding-peaks-in-the-histograms-of-the-variables
        # Increase prominence until only one single peak is found
        while (len(peak_idx) == 0) or (len(peak_idx) > 1):
            prom += 1
            if prom > 20:
                break
            peak_idx, props = find_peaks(counts, prominence=prom)
            print(f"Prominence: {prom}    Peaks at: {peak_idx}")

        if peak_idx:
            start_idx = peak_idx - 2
            start_idx = start_idx if start_idx >= 0 else 0

            end_idx = peak_idx + 2 + 1
            max_idx = len(divisions) - 1
            end_idx = end_idx if end_idx <= max_idx else max_idx

        else:
            start_idx = 0
            end_idx = len(divisions) - 1

        timewin_lag = [divisions[start_idx], divisions[end_idx]]
        timewin_lag = [int(x) for x in timewin_lag]  # Convert elements in array to integers, needed for indexing

        return timewin_lag, peak_idx, start_idx, end_idx, counts, divisions

    def plot_results_hist(self, xtick_labels):
        # label_bin_start = plot_df_agg['cov_max_shift']['min'].to_list()
        # Histogram bars
        # xtick_labels = xtick_labels[0:-1]

        gs = gridspec.GridSpec(1, 1)  # rows, cols
        gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)

        fig = plt.Figure(facecolor='white', figsize=(16, 9))
        ax = fig.add_subplot(gs[0, 0])

        # bar_positions = plot_df_agg.index + 0.5  # Shift by half position
        ax.bar(x=xtick_labels[0:-1], height=self.counts, width=0.9,
               label='counts', zorder=98, color='#78909c')

        if self.peak_idx:
            ax.bar(x=xtick_labels[self.peak_idx], height=self.counts[self.peak_idx], width=0.9,
                   label='counts', zorder=99, color='#ef5350', alpha=0.9)

        ax.axvline(x=xtick_labels[self.start_idx], ls='--', c='#ef5350')
        ax.axvline(x=xtick_labels[self.end_idx], ls='--', c='#ef5350')

        txt_info = \
            f"Histogram of found lag times in iteration {self.iteration}"

        ax.text(0.02, 0.98, txt_info,
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes,
                size=12, color='black', backgroundcolor='none', zorder=100)

        xtick_labels_int = [int(l) for l in xtick_labels]
        ax.set_xticks(xtick_labels_int)  # Position of ticks

        outpath = self.dir_output / f'2_plot_results_hist_iteration-{self.iteration}'
        fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w',
                    transparent=True, dpi=150)

        # ax.set_xticklabels(division)  # Labels of ticks, shown in plot
        # if len(label_bin_start) > 30:
        #     ax.tick_params(rotation=45)
        # else:
        #     ax.tick_params(rotation=0)
        # default_format(ax=ax, txt_xlabel=f'bin {col[1]}', txt_ylabel=f'{col[0]} counts',
        #                              txt_ylabel_units='[#]')
        # gui.plotfuncs.default_grid(ax=ax)

        return None

    # import matplotlib.pyplot as plt
    #
    # plt.subplot()
    # plt.plot(peaks, count[peaks], "ob")
    # plt.plot(count)
    # plt.legend(['prominence'])
    # plt.show()

    # peaks, _ = find_peaks(x, distance=20)
    # peaks2, _ = find_peaks(x, prominence=1)  # BEST!
    # peaks3, _ = find_peaks(x, width=20)
    # peaks4, _ = find_peaks(x,
    #                        threshold=0.4)  # Required vertical distance to its direct neighbouring samples, pretty useless
    # plt.subplot(2, 2, 1)
    # plt.plot(peaks, x[peaks], "xr")
    # plt.plot(x)
    # plt.legend(['distance'])
    #
    # plt.subplot(2, 2, 2)
    # plt.plot(peaks2, x[peaks2], "ob")
    # plt.plot(x)
    # plt.legend(['prominence'])
    #
    # plt.subplot(2, 2, 3)
    # plt.plot(peaks3, x[peaks3], "vg")
    # plt.plot(x)
    # plt.legend(['width'])
    # plt.subplot(2, 2, 4)
    # plt.plot(peaks4, x[peaks4], "xk")
    # plt.plot(x)
    # plt.legend(['threshold'])

    # return counts, divisions, peak_idx, props
