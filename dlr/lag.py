import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import plot


class LagSearch:
    def __init__(self, wind_rot_df, segment_name, ref_sig, lagged_sig, outdir_plots, win_lagsearch,
                 file_idx, segment_start, segment_end, filename, iteration, shift_stepsize,
                 outdir_data):
        self.wind_rot_df = wind_rot_df
        self.segment_name = segment_name
        self.segment_start = segment_start
        self.segment_end = segment_end
        self.outdir_data = outdir_data
        self.outdir_plots = outdir_plots
        self.w_rot_turb_col = ref_sig
        self.scalar_turb_col = lagged_sig
        self.file_idx = file_idx
        self.filename = filename
        self.iteration = iteration
        self.win_lagsearch = win_lagsearch
        # self.shift_stepsize = int(np.sum(np.abs(self.win_lagsearch)) / 100)
        self.shift_stepsize = shift_stepsize  # Negative moves lagged values "upwards" in column

        self.lagsearch_df = self.setup_lagsearch_df()

        self.run()

    def setup_lagsearch_df(self):
        df = pd.DataFrame(columns=['index', 'segment_name', 'shift', 'cov', 'cov_abs',
                                   'flag_peak_max_cov_abs', 'flag_peak_auto'])
        df['shift'] = range(self.win_lagsearch[0], self.win_lagsearch[1] + self.shift_stepsize, self.shift_stepsize)
        df['index'] = np.nan
        df['segment_name'] = self.segment_name
        df['cov'] = np.nan
        df['cov_abs'] = np.nan
        df['flag_peak_max_cov_abs'] = False  # Flag True = found peak
        df['flag_peak_auto'] = False
        return df

    def run(self):
        self.lagsearch_df = self.setup_lagsearch_df()

        self.lagsearch_df = \
            self.find_max_cov_peak(wind_rot_df=self.wind_rot_df, lagsearch_df=self.lagsearch_df)

        self.lagsearch_df, self.props_peak_auto = \
            self.find_peak_auto(df=self.lagsearch_df)

        self.save_cov_data(df=self.lagsearch_df)
        self.save_cov_plot()
        return None

    def find_peak_auto(self, df):
        """
        Automatically find peak in covariance time series.

        The found peak is flagged TRUE in df.

        Parameters
        ----------
        df: pandas DataFrame
            Contains covariances for each shift.

        Returns
        -------


        """
        found_peaks_idx, found_peaks_dict = find_peaks(df['cov_abs'],
                                                       width=1, prominence=5)  # todo hier weiter PEAK QUALITY?
        found_peaks_props_df = pd.DataFrame.from_dict(found_peaks_dict)

        props_peak_df = pd.DataFrame()  # Props for one peak

        if len(found_peaks_idx) > 0:
            max_width_height_idx = found_peaks_dict['width_heights'].argmax()  # Good metric to find peaks
            most_prominent_idx = found_peaks_dict['prominences'].argmax()

            if max_width_height_idx == most_prominent_idx:
                peak_idx = found_peaks_idx[max_width_height_idx]
                df.loc[peak_idx, 'flag_peak_auto'] = True
                props_peak_df = found_peaks_props_df.iloc[max_width_height_idx]

        return df, props_peak_df

    def get(self):
        return self.lagsearch_df, self.props_peak_auto

    def find_max_cov_peak(self, wind_rot_df, lagsearch_df):
        """Find maximum absolute covariance between turbulent wind data
        and turbulent scalar data.
        """

        print(f"Searching maximum covariance in range from {self.win_lagsearch[0]} to "
              f"{self.win_lagsearch[1]} records ...")

        _wind_rot_df = wind_rot_df.copy()
        _wind_rot_df['index'] = _wind_rot_df.index

        # cov_max_shift = False
        # cov_max = False
        # cov_max_timestamp = False

        # Check if data column is empty
        if _wind_rot_df[self.scalar_turb_col].dropna().empty:
            pass

        else:
            for ix, row in lagsearch_df.iterrows():
                shift = int(row['shift'])
                try:
                    if shift < 0:
                        index_shifted = str(_wind_rot_df['index'][-shift])  # Note the negative sign
                    else:
                        index_shifted = pd.NaT
                    scalar_data_shifted = _wind_rot_df[self.scalar_turb_col].shift(shift)
                    cov = _wind_rot_df[self.w_rot_turb_col].cov(scalar_data_shifted)
                    lagsearch_df.loc[lagsearch_df['shift'] == row['shift'], 'cov'] = cov
                    lagsearch_df.loc[lagsearch_df['shift'] == row['shift'], 'index'] = index_shifted

                except IndexError:
                    # If not enough data in the file to perform the shift, continue
                    # to the next shift and try again. This can happen for the last
                    # segments in each file, when there is no more data available
                    # at the end.
                    continue

            # Results
            lagsearch_df['cov_abs'] = lagsearch_df['cov'].abs()
            cov_max_ix = lagsearch_df['cov_abs'].idxmax()
            # cov_max_shift = lagsearch_df.iloc[cov_max_ix]['shift']
            # cov_max = lagsearch_df.iloc[cov_max_ix]['cov']
            # cov_max_timestamp = lagsearch_df.iloc[cov_max_ix]['index']
            lagsearch_df.loc[cov_max_ix, 'flag_peak_max_cov_abs'] = True

        return lagsearch_df

    def save_cov_data(self, df):
        outpath = self.outdir_data / f'{self.segment_name}_segment_covariance_iteration-{self.iteration}'
        df.to_csv(f"{outpath}.csv")
        return None

    @staticmethod
    def get_peak_idx(df):
        """
        Check data rows for peak flags.

        Parameters
        ----------
        df

        Returns
        -------

        """
        if True in df['flag_peak_auto'].values:
            idx_peak_auto = df.loc[df['flag_peak_auto'] == True, :].index.values[0]
            # shift_peak_auto = df.iloc[idx_peak_auto]['shift']
            # cov_peak_auto = df.iloc[idx_peak_auto]['cov']
        else:
            idx_peak_auto = False

        if True in df['flag_peak_max_cov_abs'].values:
            idx_peak_cov_abs_max = df.loc[df['flag_peak_max_cov_abs'] == True, :].index.values[0]
            # shift_peak_cov_abs_max = df.iloc[idx_peak_cov_abs_max]['shift']
            # cov_peak_cov_abs_max = df.iloc[idx_peak_cov_abs_max]['cov']
        else:
            idx_peak_cov_abs_max=False

        return idx_peak_cov_abs_max, idx_peak_auto

    def prepare_info_txt(self, shift_peak_auto, cov_abs_peak_auto, shift_peak_cov_abs_max, cov_peak_cov_abs_max):
        txt_info = \
            f"Iteration: {self.iteration}\n" \
            f"Time lag search window: from {self.win_lagsearch[0]} to {self.win_lagsearch[1]} records\n" \
            f"Segment name: {self.segment_name}\n" \
            f"Segment start: {self.segment_start}\n" \
            f"Segment end: {self.segment_end}\n" \
            f"File: {self.filename} - File date: {self.file_idx}\n" \
            f"Max absolute covariance {cov_peak_cov_abs_max:.3f} found @ record {shift_peak_cov_abs_max}\n" \
            f"Lag search step size: {self.shift_stepsize} records"

        # Check if automatically detected peak available
        if not self.props_peak_auto.empty:
            txt_auto_peak_props = \
                f"Peak auto-detected found @ record {shift_peak_auto}\n" \
                f"    cov: {cov_abs_peak_auto}\n" \
                f"    prominence: {self.props_peak_auto['prominences']}\n" \
                f"    width: {self.props_peak_auto['widths']}\n" \
                f"    width height: {self.props_peak_auto['width_heights']}\n"
        else:
            txt_auto_peak_props = "No auto-detected peak"

        return txt_info, txt_auto_peak_props

    def save_cov_plot(self):
        """Plot and save covariance plot for segment."""

        idx_peak_cov_abs_max, idx_peak_auto = \
            self.get_peak_idx(df=self.lagsearch_df)

        fig = plot.make_scatter_cov(df=self.lagsearch_df,
                                    idx_peak_cov_abs_max=idx_peak_cov_abs_max,
                                    idx_peak_auto=idx_peak_auto,
                                    props_peak_auto=self.props_peak_auto,
                                    iteration=self.iteration,
                                    win_lagsearch=self.win_lagsearch,
                                    segment_name=self.segment_name,
                                    segment_start=self.segment_start,
                                    segment_end=self.segment_end,
                                    filename=self.filename,
                                    file_idx=self.file_idx,
                                    shift_stepsize=self.shift_stepsize)

        outpath = self.outdir_plots / f'{self.segment_name}_segment_covariance_iteration-{self.iteration}'

        # Save
        print(f"Saving plot in PNG file: {outpath}.png ...")
        fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w',
                    transparent=True, dpi=100)
        return None


class AdjustLagsearchWindow():
    def __init__(self, series, outdir, iteration, plot=True, hist_num_bins=30, remove_fringe_bins=True,
                 perc_threshold=0.9):
        self.series = series.dropna()  # NaNs yield error in histogram
        self.numvals_series = self.series.size
        self.outdir = outdir
        self.iteration = iteration
        self.plot = plot
        self.hist_num_bins = hist_num_bins
        self.remove_fringe_bins = remove_fringe_bins
        self.perc_threshold = perc_threshold

        self.run()

    def run(self):
        self.win_lagsearch_adj, self.peak_max_count_idx, self.start_idx, self.end_idx, \
        self.counts, self.divisions, self.peak_most_prom_idx = \
            self.find_hist_peaks()

        if self.plot:
            self.plot_results_hist(hist_bins=self.divisions)

    def find_hist_peaks(self):
        """Find peak in histogram of found lag times."""

        # Make histogram of found lag times, remove fringe bins at start and end
        counts, divisions = self.calc_hist(series=self.series, bins=self.hist_num_bins,
                                           remove_fringe_bins=self.remove_fringe_bins)

        # Search bin with most found lag times
        peak_max_count_idx = self.search_bin_max_counts(counts=counts)

        # Search most prominent bin
        peak_most_prom_idx = self.search_bin_most_prominent(counts=counts)

        # Adjust lag search time window for next iteration
        win_lagsearch_adj, start_idx, end_idx = self.adjust_win_lagsearch(counts=counts, divisions=divisions,
                                                                          perc_threshold=self.perc_threshold,
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
        if remove_fringe_bins and len(counts) >=5:
            counts = counts[1:-1]
            divisions = divisions[1:-1]  # Contains start values of bins
        return counts, divisions

    @staticmethod
    def search_bin_most_prominent(counts):
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
            print(f"Expanding lag window: {perc}  from record: {start_idx}  to record: {end_idx}")
            if (start_idx == 0) and (end_idx == len(counts)):
                break
        win_lagsearch_adj = [divisions[start_idx], divisions[end_idx]]
        win_lagsearch_adj = [int(x) for x in
                             win_lagsearch_adj]  # Convert elements in array to integers, needed for indexing
        return win_lagsearch_adj, start_idx, end_idx

    def include_bins_next_to_peak(self, peak_max_count_idx, peak_most_prom_idx):
        """Include histogram bins next to the bin for which max was found and the
        most prominent bin.

        Since multiple max count peaks can be detected in the histogram, all found
        peaks are considered and all bins before and after each detected peak are
        included to calculate the adjusted start and end indices.

        For example:
            Three peaks were with max count were found in the histogram. The peaks
            were found in bins 5, 9 and 14:
                peak_max_count_index = [5,9,14]
            The most prominent peak was detected in bin 2:
                peak_most_prom_idx = 2
            Then the bins before the max count peaks are included:
                start_idx = [4,5,8,9,13,14]
            Then the bins after the max count peaks are included:
                end_idx = [4,5,6,8,9,10,13,14,15]
            Then the max count peaks are combined with the most prominent peak,
            using np.unique() in case of overlapping bins:
                start_end_idx = [2,4,5,6,8,9,10,13,14,15]
            The start_idx is the min of this collection:
                start_idx = 2
            The end_idx is the max of this collection:
                end_idx = 15
            The adjusted time window for lag search starts with the starting time
            of bin 2 and ends with the end time with bin 15 (starting time is added
            in next steps).
        """
        start_idx = np.subtract(peak_max_count_idx, 1)  # Include bins before each peak
        start_idx[start_idx < 0] = 0  # Negative index not possible
        end_idx = np.add(peak_max_count_idx, 1)  # Include bins after each peak
        start_end_idx = np.unique(np.concatenate([start_idx, end_idx, [peak_most_prom_idx]]))  # Combine peaks
        start_idx = np.min(start_end_idx)
        end_idx = np.max(start_end_idx[-1])
        return start_idx, end_idx

    def plot_results_hist(self, hist_bins):

        gs, fig, ax = plot.setup_fig_ax()

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

        outpath = self.outdir / f'{self.iteration}_HISTOGRAM_found_lag_times_iteration-{self.iteration}'
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


def calc_quantiles(df):
    _df = df.copy()
    args = dict(window=72, min_periods=37, center=True)
    _df['shift_median'] = _df['cov_max_shift'].rolling(**args).median()
    _df['search_win_upper'] = _df['shift_median'] + 100
    _df['search_win_lower'] = _df['shift_median'] - 100
    _df['shift_P25'] = _df['cov_max_shift'].rolling(**args).quantile(0.25)
    _df['shift_P75'] = _df['cov_max_shift'].rolling(**args).quantile(0.75)

    return _df


def search_settings(win_lagsearch):
    """
    Set step size of shift during lag search and calculate number of
    histogram bins.

    Parameters
    ----------
    win_lagsearch: list
        Contains start record for lag search in [0] and end record
        in [1], e.g. [-1000, 1000] searches between records -1000 and
        +1000.


    Returns
    -------
    shift_stepsize: int
        Step-size (number of records) for lag search.
    hist_num_bins: range
        Bin range for the histogram of found lag times. The histogram
        is used to narrow down win_lagsearch.

    Examples
    --------
    >> shift_stepsize, hist_num_bins = lagsearch_settings(win_lagsearch=-1000, 1000])
    >> print(shift_stepsize)
    10
    >> print(hist_num_bins)
    range(-1000, 1000, 50)

    """
    shift_stepsize = int(np.sum(np.abs(win_lagsearch)) / 100 / 2)
    hist_num_bins = range(win_lagsearch[0], win_lagsearch[1], int(shift_stepsize * 5))
    return shift_stepsize, hist_num_bins
