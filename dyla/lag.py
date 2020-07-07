from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.signal import find_peaks

import plot
from _setup import create_logger


class LagSearch:
    def __init__(self,
                 segment_df: pd.DataFrame,
                 segment_name: str,
                 ref_sig: str,
                 lagged_sig: str,
                 win_lagsearch: list,
                 file_idx: pd.Timestamp,
                 segment_start: pd.Timestamp,
                 segment_end: pd.Timestamp,
                 filename: str,
                 iteration: int,
                 shift_stepsize: int,
                 outdir_data: Path = None,
                 outdir_plots: Path = None,
                 logfile_path: Path = None):

        self.segment_df = segment_df
        self.segment_name = segment_name
        self.segment_start = segment_start
        self.segment_end = segment_end
        self.outdir_data = outdir_data
        self.outdir_plots = outdir_plots
        self.ref_sig = ref_sig
        self.lagged_sig = lagged_sig
        self.file_idx = file_idx
        self.filename = filename
        self.iteration = iteration
        self.win_lagsearch = win_lagsearch
        self.shift_stepsize = shift_stepsize  # Negative moves lagged values "upwards" in column
        self.logfile_path = logfile_path

        self.cov_df = self.setup_lagsearch_df(win_lagsearch=self.win_lagsearch,
                                              shift_stepsize=self.shift_stepsize,
                                              segment_name=self.segment_name)

        if self.logfile_path:
            self.logger = create_logger(logfile_path=self.logfile_path, name=__name__)

        self.run()

    def run(self):
        self.cov_df = self.setup_lagsearch_df(win_lagsearch=self.win_lagsearch,
                                              shift_stepsize=self.shift_stepsize,
                                              segment_name=self.segment_name)

        # Max covariance peak detection
        self.cov_df = self.find_max_cov_peak(segment_df=self.segment_df,
                                             cov_df=self.cov_df,
                                             ref_sig=self.ref_sig,
                                             lagged_sig=self.lagged_sig)

        # Automatic peak detection
        self.cov_df, self.props_peak_auto = self.find_peak_auto(cov_df=self.cov_df)

        if self.outdir_data:
            self.save_cov_data(cov_df=self.cov_df)

        # Plot covariance
        idx_peak_cov_abs_max, idx_peak_auto = self.get_peak_idx(df=self.cov_df)
        fig = self.make_scatter_cov(df=self.cov_df,
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

        if self.outdir_plots:
            self.save_cov_plot(fig=fig)
        return None

    @staticmethod
    def setup_lagsearch_df(win_lagsearch: list, shift_stepsize: int, segment_name):
        df = pd.DataFrame(columns=['index', 'segment_name', 'shift', 'cov', 'cov_abs',
                                   'flag_peak_max_cov_abs', 'flag_peak_auto'])
        df['shift'] = range(int(win_lagsearch[0]),
                            int(win_lagsearch[1]) + shift_stepsize,
                            shift_stepsize)
        df['index'] = np.nan
        df['segment_name'] = segment_name
        df['cov'] = np.nan
        df['cov_abs'] = np.nan
        df['flag_peak_max_cov_abs'] = False  # Flag True = found peak
        df['flag_peak_auto'] = False
        return df

    @staticmethod
    def find_peak_auto(cov_df: pd.DataFrame):
        """
        Automatically find peaks in covariance time series

        The found peak is flagged TRUE in cov_df.

        Peaks are searched automatically using scipy's .find_peaks method.
        The peak_score is calculated for each peak, the top-scoring peaks
        are retained. If the previously calculated max covariance peak is
        part of the top-scoring peaks, the record at which these two peaks
        were found is flagged in cov_df. Basically, this method works as a
        validation step to detect high-quality covariance peaks that are
        later used to calculate default lag times.


        Parameters
        ----------
        cov_df: pandas DataFrame
            Contains covariances for each shift.

        Returns
        -------
        cov_df
        props_peak_df

        """

        found_peaks_idx, found_peaks_dict = find_peaks(cov_df['cov_abs'],
                                                       height=0, width=0, prominence=0)
        found_peaks_props_df = pd.DataFrame.from_dict(found_peaks_dict)
        found_peaks_props_df['idx_in_cov_df'] = found_peaks_idx

        # Calculate peak score, a combination of peak_height, prominences and width_heights
        found_peaks_props_df['peak_score'] = found_peaks_props_df['prominences'] \
                                             * found_peaks_props_df['width_heights'] \
                                             * found_peaks_props_df['peak_heights']
        found_peaks_props_df['peak_score'] = found_peaks_props_df['peak_score'] ** .5  # Make numbers smaller
        found_peaks_props_df['peak_rank'] = found_peaks_props_df['peak_score'].rank(ascending=False)

        score_threshold = found_peaks_props_df['peak_score'].quantile(0.9)
        top_scoring_peaks_df = found_peaks_props_df.loc[found_peaks_props_df['peak_score'] >= score_threshold]
        top_scoring_peaks_df = top_scoring_peaks_df.sort_values(by=['peak_score', 'prominences', 'width_heights'],
                                                                ascending=False)

        idx_peak_cov_abs_max, _ = LagSearch.get_peak_idx(df=cov_df)

        # Check if peak of max absolute covariance is also in auto-detected peaks
        if idx_peak_cov_abs_max in top_scoring_peaks_df['idx_in_cov_df'].values:
            props_peak_df = top_scoring_peaks_df.iloc[
                top_scoring_peaks_df['idx_in_cov_df'].values == idx_peak_cov_abs_max]
            props_peak_df = props_peak_df.iloc[0]
            peak_idx = int(props_peak_df['idx_in_cov_df'])
            cov_df.loc[peak_idx, 'flag_peak_auto'] = True
        else:
            props_peak_df = pd.DataFrame()

        # props_peak_df = pd.DataFrame()  # Props for one peak
        # if len(found_peaks_idx) > 0:
        #     max_width_height_idx = found_peaks_dict['width_heights'].argmax()  # Good metric to find peaks
        #     most_prominent_idx = found_peaks_dict['prominences'].argmax()
        #
        #     if max_width_height_idx == most_prominent_idx:
        #         peak_idx = found_peaks_idx[max_width_height_idx]
        #         cov_df.loc[peak_idx, 'flag_peak_auto'] = True
        #         props_peak_df = found_peaks_props_df.iloc[max_width_height_idx]

        return cov_df, props_peak_df

    def get(self):
        return self.cov_df, self.props_peak_auto

    @staticmethod
    def find_max_cov_peak(segment_df, cov_df, lagged_sig, ref_sig):
        """
        Find maximum absolute covariance
        """

        _segment_df = segment_df.copy()
        _segment_df['index'] = _segment_df.index

        # Check if data column is empty
        if _segment_df[lagged_sig].dropna().empty:
            pass

        else:
            for ix, row in cov_df.iterrows():
                shift = int(row['shift'])
                try:
                    if shift < 0:
                        index_shifted = str(_segment_df['index'][-shift])  # Note the negative sign
                    else:
                        index_shifted = pd.NaT
                    scalar_data_shifted = _segment_df[lagged_sig].shift(shift)
                    cov = _segment_df[ref_sig].cov(scalar_data_shifted)
                    cov_df.loc[cov_df['shift'] == row['shift'], 'cov'] = cov
                    cov_df.loc[cov_df['shift'] == row['shift'], 'index'] = index_shifted

                except IndexError:
                    # If not enough data in the file to perform the shift, continue
                    # to the next shift and try again. This can happen for the last
                    # segments in each file, when there is no more data available
                    # at the end.
                    continue

            # Results
            cov_df['cov_abs'] = cov_df['cov'].abs()
            cov_max_ix = cov_df['cov_abs'].idxmax()
            # cov_max_shift = lagsearch_df.iloc[cov_max_ix]['shift']
            # cov_max = lagsearch_df.iloc[cov_max_ix]['cov']
            # cov_max_timestamp = lagsearch_df.iloc[cov_max_ix]['index']
            cov_df.loc[cov_max_ix, 'flag_peak_max_cov_abs'] = True

        return cov_df

    def save_cov_data(self, cov_df):
        outfile = f'{self.segment_name}_segment_covariance_iteration-{self.iteration}.csv'
        outpath = self.outdir_data / outfile
        cov_df.to_csv(f"{outpath}")
        # self.logger.info(f"Saved covariance data in {outfile}")
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
            idx_peak_cov_abs_max = False

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

    def save_cov_plot(self, fig):
        """Save covariance plot for segment"""
        outfile = f'{self.segment_name}_segment_covariance_iteration-{self.iteration}.png'
        outpath = self.outdir_plots / outfile
        fig.savefig(f"{outpath}", format='png', bbox_inches='tight', facecolor='w',
                    transparent=True, dpi=100)
        # self.logger.info(f"Saved covariance plot in {outfile}")
        return

    def make_scatter_cov(self, df, idx_peak_cov_abs_max, idx_peak_auto, iteration, win_lagsearch,
                         segment_name, segment_start, segment_end, filename,
                         file_idx, shift_stepsize, props_peak_auto):
        """Make scatter plot with z-values as colors and display found max covariance."""

        gs, fig, ax = plot.setup_fig_ax()

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
                f"    peak_score {props_peak_auto['peak_score']:.0f}\n" \
                f"    peak_rank {props_peak_auto['peak_rank']:.0f}\n" \
                f"    peak_height {props_peak_auto['peak_heights']:.0f}\n" \
                f"    prominence {props_peak_auto['prominences']:.0f}\n" \
                f"    width {props_peak_auto['widths']:.0f}\n" \
                f"    width_height {props_peak_auto['width_heights']:.0f}\n"

        else:
            txt_info += \
                f"\n(!)NO AUTO-PEAK FOUND\n"

        ax.text(0.02, 0.98, txt_info,
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, size=10, color='black', backgroundcolor='none', zorder=100)

        plot.default_format(ax=ax, label_color='black', fontsize=12,
                            txt_xlabel='lag [records]', txt_ylabel='covariance', txt_ylabel_units='-')

        ax.legend(frameon=False, loc='upper right').set_zorder(100)

        return fig


class AdjustLagsearchWindow():
    def __init__(self, series, iteration, plot=True, hist_num_bins=30, remove_fringe_bins=True,
                 perc_threshold=0.9, outdir=None):
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
        if peak_most_prom_idx in peak_max_count_idx:  # todo
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
        # print("Searching peak of maximum counts ...")
        max_count = np.amax(counts)
        peak_max_count_idx = np.where(counts == np.amax(max_count))  # Returns array in tuple
        if len(peak_max_count_idx) == 1:
            peak_max_count_idx = peak_max_count_idx[0]  # Yields array of index or multiple indices
        return peak_max_count_idx

    @staticmethod
    def calc_hist(series=False, bins=20, remove_fringe_bins=False):
        """Calculate histogram of found lag times."""
        counts, divisions = np.histogram(series, bins=bins)
        # Remove first and last bins from histogram. In case of unclear lag times
        # data tend to accumulate in these edge regions of the search window.
        if remove_fringe_bins and len(counts) >= 5:
            counts = counts[1:-1]
            divisions = divisions[1:-1]  # Contains start values of bins
        return counts, divisions

    @staticmethod
    def search_bin_most_prominent(counts):
        # kudos: https://www.kaggle.com/simongrest/finding-peaks-in-the-histograms-of-the-variables
        # Increase prominence until only one single peak is found
        # print("Searching most prominent peak ...")
        peak_most_prom_idx = []
        prom = 0  # Prominence for peak finding
        while (len(peak_most_prom_idx) == 0) or (len(peak_most_prom_idx) > 1):
            prom += 1
            if prom > 40:
                peak_most_prom_idx = False
                break
            peak_most_prom_idx, props = find_peaks(counts, prominence=prom)
            # print(f"Prominence: {prom}    Peaks at: {peak_most_prom_idx}")
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
            # print(f"Expanding lag window: {perc}  from record: {start_idx}  to record: {end_idx}")
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

        For examples:
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

        # Counts
        # bar_positions = plot_df_agg.index + 0.5  # Shift by half position
        bar_width = (hist_bins[1] - hist_bins[0]) * 0.9  # Calculate bar width
        args = dict(width=bar_width, align='edge')
        ax.bar(x=hist_bins[0:-1], height=self.counts, label='counts', zorder=90, color='#78909c', **args)

        # Text counts
        ax.set_xlim(hist_bins[0], hist_bins[-1])
        for i, v in enumerate(self.counts):
            ax.text(hist_bins[0:-1][i] + (bar_width / 2), 1, str(v), zorder=99)

        ax.bar(x=hist_bins[self.peak_max_count_idx], height=self.counts[self.peak_max_count_idx],
               label='most counts', zorder=98, edgecolor='#ef5350', linewidth=4,
               color='None', alpha=0.9, linestyle='-', **args)

        if self.peak_most_prom_idx:
            ax.bar(x=hist_bins[self.peak_most_prom_idx], height=self.counts[self.peak_most_prom_idx],
                   label='most prominent counts peak', zorder=99, edgecolor='#FFA726', linewidth=2,
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
                transform=ax.transAxes, size=14, color='black',
                backgroundcolor='none', zorder=100)

        ax.legend()

        plot.default_format(ax=ax, label_color='black', fontsize=12,
                            txt_xlabel='lag [records]', txt_ylabel='counts', txt_ylabel_units='[#]')

        if self.outdir:
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
