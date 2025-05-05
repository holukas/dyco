"""
    DYCO Dynamic Lag Compensation
    Copyright (C) 2020-2025 Lukas Hörtnagl

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from dyco import plot


class AdjustLagsearchWindow:
    """

    Adjust the time window for lag search that is used in the next iteration

    """

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
        """Start adjustment of lagsearch time window"""
        self.lgs_winsize_adj, self.peak_max_count_idx, self.start_idx, self.end_idx, \
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
        lgs_winsize_adj, start_idx, end_idx = self.adjust_lgs_winsize(counts=counts, divisions=divisions,
                                                                      perc_threshold=self.perc_threshold,
                                                                      peak_max_count_idx=peak_max_count_idx,
                                                                      peak_most_prom_idx=peak_most_prom_idx)

        # todo (maybe) Check if most prominent peak is also the max peak
        if peak_most_prom_idx in peak_max_count_idx:
            clear_peak_idx = np.where(peak_max_count_idx == peak_most_prom_idx)
        else:
            clear_peak_idx = False

        lgs_winsize_adj = [divisions[start_idx], divisions[end_idx]]
        # Convert elements in array to integers, needed for indexing
        lgs_winsize_adj = [int(x) for x in lgs_winsize_adj]

        # Expand lag search window size
        # Set minimum size of lag search time window to min. 20 records
        # Zero is also valid, or in case of two positive numbers, the values are both used, +1
        # [+1, +8] --> allowed values are [1, 2, 3, 4, 5, 6, 7, 8] --> range=8-1=7 -> +1=8
        # [-4, +2] --> allowed values are [-4, -3, -2, -1, 0, +1, +2] --> range=2-(-4)=6 -> +1=7
        _range = (lgs_winsize_adj[1] - lgs_winsize_adj[0]) + 1
        while _range < 20:
            # [-8, -3] --> [-9, -2] ... --> finally [-15, 4]
            # [-4, +2] --> [-5, +3] ...
            # [+2, +7] --> [+1, +8] ...
            lgs_winsize_adj[0] = lgs_winsize_adj[0] - 1
            lgs_winsize_adj[1] = lgs_winsize_adj[1] + 1
            _range = (lgs_winsize_adj[1] - lgs_winsize_adj[0]) + 1

        return lgs_winsize_adj, peak_max_count_idx, start_idx, end_idx, counts, divisions, peak_most_prom_idx

    def get(self):
        """
        Adjusted time window for lag search

        Returns
        -------
        lgs_winsize_adj: list
            Contains two elements: start index and end index for lag search, i.e. the
            "search from/to" range for the lag search

        """
        return self.lgs_winsize_adj

    def search_bin_max_counts(self, counts):
        """
        Search histogram peak of maximum counts

        Parameters
        ----------
        counts: array
            Counts in histogram bins

        Returns
        -------
        Index of the histogram peak with maximum counts
        """
        max_count = np.amax(counts)
        peak_max_count_idx = np.where(counts == np.amax(max_count))  # Returns array in tuple
        if len(peak_max_count_idx) == 1:
            peak_max_count_idx = peak_max_count_idx[0]  # Yields array of index or multiple indices
        return peak_max_count_idx

    @staticmethod
    def calc_hist(series=False, bins=20, remove_fringe_bins=False):
        """
        Calculate histogram of found lag times

        Done after each iteration.

        Parameters
        ----------
        series: pandas Series
            Found absolute covariance peaks for each segment
        bins: int or range
            Number of bins for the histogram, can also be given as range
        remove_fringe_bins: bool
            If True, the fringe bins in the histogram will be removed. Important for
            constraining the lag during lag search.

        Returns
        -------
        counts: counts per histogram division
        divisions: histogram divisions
        """
        counts, divisions = np.histogram(series, bins=bins)
        # Remove first and last bins from histogram. In case of unclear lag times
        # data tend to accumulate in these edge regions of the search window.
        if remove_fringe_bins and len(counts) >= 5:
            counts = counts[1:-1]
            divisions = divisions[1:-1]  # Contains start values of bins
        return counts, divisions

    @staticmethod
    def search_bin_most_prominent(counts):
        """
        Search most prominent histogram peak

        Increase prominence until only one single peak is found
        kudos: https://www.kaggle.com/simongrest/finding-peaks-in-the-histograms-of-the-variables

        Parameters
        ----------
        counts: array
            Counts per histogram bin/division

        Returns
        -------
        Index of the most prominent peak
        """
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

    def adjust_lgs_winsize(self, counts, divisions, perc_threshold, peak_max_count_idx, peak_most_prom_idx):
        """
        Set new time window for next lag search, based on previous results

        Includes more and more bins around the bin where most lag times were found
        until a threshold is reached.

        Parameters
        ----------
        counts: numpy array
            Counts per histogram division
        divisions: numpy array
            Histogram divisions
        perc_threshold: float
            Cumulative percentage threshold in histogram of found lag times. Using the
            histogram, lagsearch time windows are narrowed down by including bins around
            the peak until a certain percentage of the total values is reached.
        peak_max_count_idx:
            Index of histogram division with max counts
        peak_most_prom_idx
            Index of most prominent histogram peak

        Returns
        -------
        lgs_winsize_adj: Adjusted (narrowed down) time window for lag search
        start_idx: Index of histogram division where the adjusted time window start
        end_idx: Index of histogram division where the adjusted time window ends
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
        lgs_winsize_adj = [divisions[start_idx], divisions[end_idx]]
        lgs_winsize_adj = [int(x) for x in
                           lgs_winsize_adj]  # Convert elements in array to integers, needed for indexing

        return lgs_winsize_adj, start_idx, end_idx

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
        """
        Plot histogram of found lag times

        Parameters
        ----------
        hist_bins: numpy array
            Histogram bins

        Returns
        -------
        None
        """

        gs, fig, ax = plot.setup_fig_ax()

        # Counts
        # bar_positions = plot_df_agg.index + 0.5  # Shift by half position
        bar_width = (hist_bins[1] - hist_bins[0]) * 0.9  # Calculate bar width
        args = dict(width=bar_width, align='edge')
        ax.bar(x=hist_bins[0:-1], height=self.counts, label='counts', zorder=90, color='#78909c', **args)
        ax.set_xlim(hist_bins[0], hist_bins[-1])

        # Text counts
        for i, v in enumerate(self.counts):
            if v > 0:
                ax.text(hist_bins[0:-1][i] + (bar_width / 2), v, str(v), zorder=99, size=6)

        ax.bar(x=hist_bins[self.peak_max_count_idx], height=self.counts[self.peak_max_count_idx],
               label='most counts', zorder=98, edgecolor='#ef5350', linewidth=4,
               color='None', alpha=0.9, linestyle='-', **args)

        if self.peak_most_prom_idx:
            ax.bar(x=hist_bins[self.peak_most_prom_idx], height=self.counts[self.peak_most_prom_idx],
                   label='most prominent counts peak', zorder=99, edgecolor='#FFA726', linewidth=2,
                   color='None', alpha=0.9, linestyle='--', **args)

        ax.axvline(x=hist_bins[self.start_idx], ls='--', c='#42A5F5',
                   label='lag search window start for next iteration')
        ax.axvline(x=hist_bins[self.end_idx], ls='--', c='#AB47BC',
                   label='lag search window end for next iteration')

        txt_info = \
            f"Histogram of found lag times in iteration {self.iteration}\n" \
            f"Number of found lag times: {self.numvals_series}"

        if self.remove_fringe_bins:
            txt_info += "\nFringe bins removed: yes"
        else:
            txt_info += "\nFringe bins removed: no"

        ax.text(0.02, 0.98, txt_info,
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, size=14, color='black',
                backgroundcolor='none', zorder=100)

        ax.legend(loc='upper right', prop={'size': 10}).set_zorder(100)

        plot.default_format(ax=ax, label_color='black', fontsize=12,
                            txt_xlabel='lag [records]', txt_ylabel='counts', txt_ylabel_units='[#]')

        if self.outdir:

            if self.remove_fringe_bins:
                outfile = f'{self.iteration}_HISTOGRAM_segment_lag_times_iteration-{self.iteration}.png'
            else:
                outfile = f'HISTOGRAM_segment_lag_times_FINAL.png'

            outpath = self.outdir / outfile
            fig.savefig(f"{outpath}", format='png', bbox_inches='tight', facecolor='w',
                        transparent=True, dpi=150)
            plt.close(fig)

        return None
