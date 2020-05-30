import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import plot


class LagSearch:
    def __init__(self, wind_rot_df, segment_name, ref_sig, lagged_sig, dir_out, win_lagsearch,
                 file_idx, segment_start, segment_end, filename, iteration):
        self.wind_rot_df = wind_rot_df
        self.segment_name = segment_name
        self.segment_start = segment_start
        self.segment_end = segment_end
        self.dir_out = dir_out
        self.w_rot_turb_col = ref_sig
        self.scalar_turb_col = lagged_sig
        self.win_lagsearch = win_lagsearch
        self.file_idx = file_idx
        self.filename = filename
        self.iteration = iteration

        self.run()

    def run(self):
        self.cov_max_shift, self.cov_max, self.cov_max_timestamp, self.lagsearch_df = \
            self.find_max_cov(df=self.wind_rot_df)

        self.save_cov_plot(x=self.lagsearch_df['shift'],
                           y=self.lagsearch_df['cov'],
                           z_color=self.lagsearch_df['cov_abs'])

    def get(self):
        return self.cov_max_shift, self.cov_max, self.cov_max_timestamp

    def save_cov_plot(self, x, y, z_color):
        """Plot and save covariance plot for segment."""

        txt_info = \
            f"Iteration: {self.iteration}\n" \
            f"Time lag search window: from {self.win_lagsearch[0]} to {self.win_lagsearch[1]} records\n" \
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
        # fig.write_html(f"{outpath}.html")  # plotly
        fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w',
                    transparent=True, dpi=100)

    def find_max_cov(self, df):
        """Find maximum absolute covariance between turbulent wind data
        and turbulent scalar data.
        """

        print("Searching maximum covariance ...")

        lagwin_start = self.win_lagsearch[0]
        lagwin_end = self.win_lagsearch[1]

        _df = df.copy()
        _df['index'] = _df.index
        lagsearch_df = pd.DataFrame()
        lagsearch_df['shift'] = range(lagwin_start, lagwin_end)  # Negative moves lagged values "upwards" in column
        lagsearch_df['cov'] = np.nan
        lagsearch_df['index'] = np.nan

        # Check if data column is empty
        if _df[self.scalar_turb_col].dropna().empty:
            cov_max_shift = False
            cov_max = False
            cov_max_timestamp = False
            lagsearch_df['cov_abs'] = np.nan

        else:
            for ix, row in lagsearch_df.iterrows():
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
                lagsearch_df.loc[lagsearch_df['shift'] == row['shift'], 'cov'] = cov
                lagsearch_df.loc[lagsearch_df['shift'] == row['shift'], 'index'] = index_shifted

            lagsearch_df['cov_abs'] = lagsearch_df['cov'].abs()
            cov_max_ix = lagsearch_df['cov_abs'].idxmax()
            cov_max_shift = lagsearch_df.iloc[cov_max_ix]['shift']
            cov_max = lagsearch_df.iloc[cov_max_ix]['cov']
            cov_max_timestamp = lagsearch_df.iloc[cov_max_ix]['index']

        return cov_max_shift, cov_max, cov_max_timestamp, lagsearch_df


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
        self.series = series.dropna()  # NaNs yield error in histogram
        self.numvals_series = self.series.size
        self.dir_output = dir_output
        self.iteration = iteration
        self.plot = plot

        self.run()

    def run(self):
        self.win_lagsearch_adj, self.peak_max_count_idx, self.start_idx, self.end_idx, \
        self.counts, self.divisions, self.peak_most_prom_idx = \
            self.find_hist_peaks()
        self.plot_results_hist(hist_bins=self.divisions)

    def find_hist_peaks(self):
        """Find peak in histogram of found lag times."""

        # Make histogram of found lag times, remove fringe bins at start and end
        counts, divisions = self.calc_hist(series=self.series, bins=20, remove_fringe_bins=True)

        # Search bin with most found lag times
        peak_max_count_idx = self.search_bin_max_counts(counts=counts)

        # Search most prominent bin
        peak_most_prom_idx = self.search_bin_most_prominent(counts=counts)

        # Adjust lag search time window for next iteration
        win_lagsearch_adj, start_idx, end_idx = self.adjust_win_lagsearch(counts=counts, divisions=divisions,
                                                                          perc_threshold=0.9,
                                                                          peak_max_count_idx=peak_max_count_idx,
                                                                          peak_most_prom_idx=peak_most_prom_idx)

        # Check if most prominent peak is also the max peak
        if peak_most_prom_idx in peak_max_count_idx:  # todo hier weiter
            clear_peak_idx = np.where(peak_max_count_idx == peak_most_prom_idx)
        else:
            clear_peak_idx = False

        win_lagsearch_adj = [divisions[start_idx], divisions[end_idx]]
        win_lagsearch_adj = [int(x) for x in
                             win_lagsearch_adj]  # Convert elements in array to integers, needed for indexing

        return win_lagsearch_adj, peak_max_count_idx, start_idx, end_idx, counts, divisions, peak_most_prom_idx

    def get(self):
        return self.win_lagsearch_adj

    def search_bin_max_counts(self, counts):
        print("Searching peak of maximum counts ...")
        max_count = np.amax(counts)
        peak_max_count_idx = np.where(counts == np.amax(max_count))  # Returns array in tuple
        if len(peak_max_count_idx) == 1:
            peak_max_count_idx = peak_max_count_idx[0]  # Yields array of index or multiple indices
        return peak_max_count_idx

    def calc_hist(self, series=False, bins=20, remove_fringe_bins=False):
        """Calculate histogram of found lag times."""
        counts, divisions = np.histogram(series, bins=bins)
        # Remove first and last bins from histogram. In case of unclear lag times
        # data tend to accumulate in these edge regions of the search window.
        if remove_fringe_bins:
            counts = counts[1:-1]
            divisions = divisions[1:-1]  # Contains start values of bins
        return counts, divisions

    def search_bin_most_prominent(self, counts):
        # kudos: https://www.kaggle.com/simongrest/finding-peaks-in-the-histograms-of-the-variables
        # Increase prominence until only one single peak is found
        print("Searching most prominent peak ...")
        peak_most_prom_idx = []
        prom = 0  # Prominence for peak finding
        while (len(peak_most_prom_idx) == 0) or (len(peak_most_prom_idx) > 1):
            prom += 1
            if prom > 40:
                peak_most_prom_idx = False
                break
            peak_most_prom_idx, props = find_peaks(counts, prominence=prom)
            print(f"Prominence: {prom}    Peaks at: {peak_most_prom_idx}")
        if peak_most_prom_idx:
            peak_most_prom_idx = int(peak_most_prom_idx)
        return peak_most_prom_idx

    def adjust_win_lagsearch(self, counts, divisions, perc_threshold, peak_max_count_idx, peak_most_prom_idx):
        """Set new time window for next lag search, based on previous results.

        Includes more and more bins around the bin where most lag times were found
        until a threshold is reached.
        """
        start_idx, end_idx = self.include_bins_next_to_peak(peak_max_count_idx=peak_max_count_idx,
                                                            peak_most_prom_idx=peak_most_prom_idx)

        counts_total = np.sum(counts)
        perc = 0
        while perc < perc_threshold:
            start_idx = start_idx - 1 if start_idx > 0 else start_idx
            end_idx = end_idx + 1 if end_idx < len(counts) else end_idx
            c = counts[start_idx:end_idx]
            perc = np.sum(c) / counts_total
            print(perc, start_idx, end_idx)
            if (start_idx == 0) and (end_idx == len(counts)):
                break
        win_lagsearch_adj = [divisions[start_idx], divisions[end_idx]]
        win_lagsearch_adj = [int(x) for x in
                             win_lagsearch_adj]  # Convert elements in array to integers, needed for indexing
        return win_lagsearch_adj, start_idx, end_idx

    def include_bins_next_to_peak(self, peak_max_count_idx, peak_most_prom_idx):
        """Include histogram bins next to the bin for which max was found and the
        most prominent bin.
        """
        start_idx = peak_max_count_idx
        end_idx = peak_max_count_idx + 1
        start_end_idx = np.unique(np.concatenate([start_idx, end_idx, [peak_most_prom_idx]]))
        start_idx = np.min(start_end_idx)
        end_idx = np.max(start_end_idx[-1])
        return start_idx, end_idx

    def plot_results_hist(self, hist_bins):
        # label_bin_start = plot_df_agg['cov_max_shift']['min'].to_list()
        # Histogram bars
        # xtick_labels = xtick_labels[0:-1]

        gs = gridspec.GridSpec(1, 1)  # rows, cols
        gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)

        fig = plt.Figure(facecolor='white', figsize=(16, 9))
        ax = fig.add_subplot(gs[0, 0])

        # bar_positions = plot_df_agg.index + 0.5  # Shift by half position

        bar_width = (hist_bins[1] - hist_bins[0]) * 0.9  # Calculate bar width
        args = dict(width=bar_width, align='edge')
        ax.bar(x=hist_bins[0:-1], height=self.counts, label='counts', zorder=90, color='#78909c', **args)

        ax.set_xlim(hist_bins[0], hist_bins[-1])
        for i, v in enumerate(self.counts):
            ax.text(hist_bins[0:-1][i] + (bar_width / 2), 1, str(v), zorder=99)

        ax.bar(x=hist_bins[self.peak_max_count_idx], height=self.counts[self.peak_max_count_idx],
               label='max peak', zorder=98, edgecolor='#ef5350', linewidth=5,
               color='None', alpha=0.9, linestyle='-', **args)

        if self.peak_most_prom_idx:
            ax.bar(x=hist_bins[self.peak_most_prom_idx], height=self.counts[self.peak_most_prom_idx],
                   label='most prominent peak', zorder=99, edgecolor='#FFA726', linewidth=2,
                   color='None', alpha=0.9, linestyle='--', **args)

        # xtick_labels_int = [int(l) for l in xtick_labels]
        # ax.set_xticks(xtick_labels_int)  # Position of ticks

        ax.axvline(x=hist_bins[self.start_idx], ls='--', c='#42A5F5',
                   label='lag search window start for next iteration')
        ax.axvline(x=hist_bins[self.end_idx], ls='--', c='#AB47BC',
                   label='lag search window end for next iteration')

        txt_info = \
            f"Histogram of found lag times in iteration {self.iteration}\n" \
            f"Number of found lag times: {self.numvals_series}"

        ax.text(0.02, 0.98, txt_info,
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes,
                size=12, color='black', backgroundcolor='none', zorder=100)

        # ax.legend()

        outpath = self.dir_output / f'{self.iteration}_HISTOGRAM_found_lag_times_iteration-{self.iteration}'
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
