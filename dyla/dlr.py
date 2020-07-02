"""

DYNAMIC LAG REMOVER - DLR
-------------------------
A Python package to detect and compensate for shifting lag times in ecosystem time series

LARE


File:       Data file
Segment:    Segment in file, e.g. one 6-hour file may consist of 12 half-hour segments.

Step (1)
FilesDetector (files.FilesDetector)

"""
import pandas as pd

import loop
from analyze import AnalyzeLoopResults
from correction import NormalizeLags
from dlr_setup import Setup

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)


class DynamicLagRemover:
    """
    DLR - Dynamic lag remover
    """

    files_overview_df = pd.DataFrame()

    def __init__(self, lgs_refsig, lgs_lagsig, dir_root,
                 fnm_date_format='%Y%m%d%H%M%S',
                 del_previous_results=False,
                 fnm_pattern='*.csv',
                 dat_recs_timestamp_format=False,
                 files_how_many=False,
                 file_generation_res='30T',
                 file_duration='30T',
                 lgs_segment_dur=False,
                 lgs_hist_perc_thres=0.9,
                 lgs_hist_remove_fringe_bins=True,
                 dat_recs_nominal_timeres=0.05,
                 lgs_winsize=1000,
                 lgs_num_iter=3,
                 dir_input_ext=False,
                 dir_output_ext=False,
                 dir_input='input',
                 dir_output='output',
                 lag_target=-100,
                 normalize_lag_for_cols=None):
        """

        Parameters
        ----------
        lgs_refsig: str
            Column name of the reference signal in the data. Lags are
            determined in relation to this signal.

        lgs_lagsig: str
            Column name of the lagged signal  for which the lag time in
            relation to the reference signal is determined.

        dir_root todo

        fnm_date_format: str
            Date format in data filenames.

        del_previous_results: bool
            If True, delete all previous results in the current folder
            settings.
            If False, search for previously calculated results and
            continue.

        fnm_pattern: str, accepts regex
            Filename pattern for file search.

        dat_recs_timestamp_format: str
            Timestamp format for each row record.

        files_how_many: int
            Limits number of found files that are be used.

        file_generation_res: pandas DateOffset
            Frequency at which new files were generated. This does not
            relate to the data records but to the file creation time.
            Examples:
                * '30T' means a new file was generated every 30 minutes.
                * '1H' means a new file was generated every hour.
                * '6H' means a new file was generated every six hours.

        file_duration: pandas DateOffset
            Duration of one data file.
            Example:
                * '30T': data file contains data from 30 minutes.

        lgs_segment_dur: pandas DateOffset
            Segment duration for lag determination. If it is the same
            as file_duration, the lag time for the complete file is
            calculated from all file data. If it is shorter than
            file_duration, then the file data is split into segments
            and the lag time is calculated for each segment separately.
            Examples:
                * '10T': calculates lag times for 10-minute segments.
                * With the settings
                    file_duration = '30T' and
                    lgs_segments_dur = '10T'
                    the 30-minute data file is split into three 10-minute
                    segments and the lag time is determined in each of the
                    segments, yielding three lag times.

        lgs_hist_perc_thres: float between 0 and 1 (percentage)
            Cumulative percentage threshold in histogram of found lag times.
            The time window for lag search during each iteration (i) is
            narrowed down based on the histogram of all found lag times
            from the previous iteration (i-1).

            During each iteration and after lag times were determined for
            all files and segments, a histogram of found lag times is created
            to identify the histogram bin in which most lag times (most counts)
            were found (peak bin). To narrow down the time window for lag search
            during the next iteration (i+1), the bins around the peak bin are
            included until a certain percentage of the total values is reached
            over the expanded bin range, centered on the peak bin.

            Example:
                * 0.9: include all bins to each site of the peak bin until 90%
                    of the total found lag times are included. The time window
                    for the lag search during the next iteration (i+1) is
                    determined by checking the left side (start) of the first
                    included bin and the right side (end) of the last included
                    bin.

        lgs_hist_remove_fringe_bins: bool
            Remove fringe bins in histogram of found lag times. In case of low
            signal-to-noise ratios the lag search yields less clear results and
            found lag times tend to accumulate in the fringe bins of the histogram,
            i.e. in the very first and last bins, potentially creating non-desirable
            peak bins. In other words, if True the first and last bins of the
            histogram are removed before the time window for lag search is adjusted.

        dat_recs_nominal_timeres: float
            Nominal (expected) time resolution of data records.
            Example:
                * 0.05: one record every 0.05 seconds (20Hz)

        lgs_winsize: int
            Starting time window size for lag search +/-, given as number of records.
            Example:
                * 1000: Lag search during the first iteration is done in a time window
                    from -1000 records to +1000 records.

        lgs_num_iter: int
            Number of lag search interations. Before each iteration, the time window
            for the lag search is narrowed down, taking into account results from the
            previous iteration. Exception is the first iteration for which the time
            window as given in lgs_winsize is used.
            Example:
                * lgs_num_iter = 3: lag search in iteration 1 (i1) uses lgs_winsize
                    to search for lag times, then the lag window is narrowed down using
                    results from i1. The adjusted search window is the used in i2 to
                    again search lag times for the same data. Likewise, i3 uses the
                    adjusted search window based on results from i2.

        dir_input_ext: str or Path or False
            Source folder that contains the data files. If False, a folder named 'input'
            is searched in the current working directory.

        dir_output_ext: str or Path or False
            Output folder for results. If False, a folder named 'output'
            is created in the current working directory.

        dir_input: str
            Name of source folder that is used in the current working directory in
            case dir_input_ext is set to False.

        dir_output: str
            Name of output folder that is created in the current working directory in
            case dir_output_ext is set to False.

        lag_target: int
            The target lag given in records to which lag times of all files are
            normalized.
            Example:
                * -100: The default lag time for all files is set to -100 records.
                    This means that if a lag search is performed on these date, the
                    lag time should be consistently be found around -100 records.

        normalize_lag_for_cols: list of strings
            Column names of the time series the normalized lag should be applied to.

        Links
        -----
        * Overview of pandas DateOffsets:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        """

        # Input and output directories
        self.dir_root = dir_root
        self.dir_input, self.dir_output = self.set_dirs(dir_input=dir_input, dir_input_ext=dir_input_ext,
                                                        dir_output=dir_output, dir_output_ext=dir_output_ext)

        # Measurements
        self.lgs_refsig = lgs_refsig
        self.lgs_lagsig = lgs_lagsig

        # File settings
        self.fnm_date_format = fnm_date_format
        self.file_generation_res = file_generation_res
        self.fnm_pattern = fnm_pattern
        self.files_how_many = files_how_many
        self.del_previous_results = del_previous_results

        self.dat_recs_nominal_timeres = dat_recs_nominal_timeres
        self.dat_recs_timestamp_format = dat_recs_timestamp_format
        self.file_duration = file_duration
        # self.data_segment_overhang = data_segment_overhang

        self.lgs_segment_dur = lgs_segment_dur
        self.lgs_winsize = [lgs_winsize * -1, lgs_winsize]
        self.lgs_num_iter = lgs_num_iter
        self.lgs_hist_remove_fringe_bins = lgs_hist_remove_fringe_bins
        self.lgs_hist_perc_thres = lgs_hist_perc_thres

        self.lag_target = lag_target
        self.normalize_lag_for_cols = normalize_lag_for_cols

        self.run()

    def run(self):

        # SETUP
        # =====
        setup = Setup(outdir=self.dir_output,
                      del_previous_results=self.del_previous_results,
                      dir_input=self.dir_input,
                      fnm_pattern=self.fnm_pattern,
                      fnm_date_format=self.fnm_date_format,
                      file_generation_res=self.file_generation_res,
                      dat_recs_nominal_timeres=self.dat_recs_nominal_timeres,
                      files_how_many=self.files_how_many)
        setup.run()
        self.outdirs, self.files_overview_df = setup.get()

        # FILE LOOPS, LAG SEARCH
        # ======================
        # Loop through files and their segments
        for iteration in range(1, 1 + self.lgs_num_iter):
            loop_iter = loop.Loop(dat_recs_timestamp_format=self.dat_recs_timestamp_format,
                                  dat_recs_nominal_timeres=self.dat_recs_nominal_timeres,
                                  iteration=iteration,
                                  lgs_hist_remove_fringe_bins=self.lgs_hist_remove_fringe_bins,
                                  lgs_hist_perc_thres=self.lgs_hist_perc_thres,
                                  outdirs=self.outdirs,
                                  lgs_segment_dur=self.lgs_segment_dur,
                                  lgs_refsig=self.lgs_refsig,
                                  lgs_lagsig=self.lgs_lagsig,
                                  lgs_num_iter=self.lgs_num_iter,
                                  lgs_winsize=self.lgs_winsize,
                                  files_overview_df=self.files_overview_df)
            loop_iter.run()

        # PLOT LOOP RESULTS
        # =================
        loop_plots = loop.PlotLoopResults(outdirs=self.outdirs,
                                          lgs_num_iter=self.lgs_num_iter,
                                          lgs_hist_perc_thres=self.lgs_hist_perc_thres)
        loop_plots.run()

        # ANALYZE RESULTS
        # ===============
        analyze = AnalyzeLoopResults(lgs_num_iter=self.lgs_num_iter,
                                     outdirs=self.outdirs,
                                     lag_target=self.lag_target)
        analyze.run()
        # lut_default_lag_times_df = analyze.get()

        # NORMALIZE LAGS
        # ==============
        if analyze.lut_success:
            normalize_lags = NormalizeLags(files_overview_df=self.files_overview_df,
                                           dat_recs_timestamp_format=self.dat_recs_timestamp_format,
                                           outdirs=self.outdirs,
                                           normalize_lag_for_cols=self.normalize_lag_for_cols)

    def check_if_file_exists(self, file_idx, filepath):
        print(f"\n-------------------\n{file_idx}")
        file_exists = True

        # The last listed file does not exist, only used for timestamp endpoint.
        if file_idx == self.files_overview_df.index[-1]:
            print(f"(!) Last listed file does not exist ({filepath})")
            file_exists = False
            return file_exists

        # Found files have a filepath, expected but missing files not.
        if filepath:
            print(f"Data found in {filepath}")
            file_exists = True
        else:
            print(f"(!) No data found ({filepath})")
            file_exists = False

        return file_exists

    def collect_file_data(self, data_df, file_idx, data_collection_df):
        if file_idx == self.files_overview_df.index[0]:
            data_collection_df = data_df.copy()
        else:
            data_collection_df = data_collection_df.append(data_df)
        return data_collection_df

    def set_dirs(self, dir_input, dir_input_ext, dir_output, dir_output_ext):
        """
        In case external dir was selected for input, set it as source dir.

        Parameters
        ----------
        dir_input
        dir_input_ext
        dir_output
        dir_output_ext

        Returns
        -------

        """
        if dir_input_ext:
            dir_input = dir_input_ext  # Override
        else:
            dir_input = self.dir_root / dir_input
        if dir_output_ext:
            dir_output = dir_output_ext  # Override
        else:
            dir_output = self.dir_root / dir_output
        return dir_input, dir_output

if __name__ == "__main__":
    DynamicLagRemover()
