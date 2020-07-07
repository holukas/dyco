"""

DYLA - DYNAMIC LAG REMOVER
--------------------------
A Python package to detect and compensate for shifting lag times in ecosystem time series



File:       Data file
Segment:    Segment in file, e.g. one 6-hour file may consist of 12 half-hour segments.

Step (1)
FilesDetector (files.FilesDetector)

"""

from pathlib import Path

import pandas as pd

import _setup
import loop
from analyze import AnalyzeLoopResults
from correction import NormalizeLags

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)


class DynamicLagRemover:
    """
    DLR - Dynamic lag remover
    """

    files_overview_df = pd.DataFrame()

    def __init__(self, lgs_refsig, lgs_lagsig,
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
                 indir=False,
                 outdir=False,
                 target_lag=-100,
                 target_cols=None):
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
            settings. If False, search for previously calculated results and
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

        indir: str or Path or False
            Source folder that contains the data files. If False, a folder named 'input'
            is searched in the current working directory.

        outdir: str or Path or False
            Output folder for results. If False, a folder named 'output'
            is created in the current working directory.

        target_lag: int
            The target lag given in records to which lag times of all files are
            normalized.
            Example:
                * -100: The default lag time for all files is set to -100 records.
                    This means that if a lag search is performed on these date, the
                    lag time should be consistently be found around -100 records.

        target_cols: list of strings
            Column names of the time series the normalized lag should be applied to.

        Links
        -----
        * Overview of pandas DateOffsets:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        """

        self.run_id, self.script_start_time = _setup.generate_run_id()

        # Input and output directories
        self.indir, self.outdir = _setup.set_dirs(indir=indir, outdir=outdir)

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

        self.lag_target = target_lag
        self.normalize_lag_for_cols = target_cols

        self.run()

    def run(self):
        self.logfile_path, self.files_overview_df = self.setup()
        self.calc_lagtimes()
        lut_success = self.analyze_lagtimes()
        self.normalize_lagtimes(lut_success=lut_success)

    def setup(self):
        """Create output folders, start logger and search for files"""
        # Create folders
        self.outdirs = _setup.CreateOutputDirs(root_dir=self.outdir,
                                               del_previous_results=self.del_previous_results).setup_output_dirs()

        # Start logging
        logfile_path = _setup.create_logfile(run_id=self.run_id, outdir=self.outdirs['_log'])
        logger = _setup.create_logger(logfile_path=logfile_path, name=__name__)
        logger.info(f"Run ID: {self.run_id}")

        # Search files
        fd = _setup.FilesDetector(dir_input=self.indir,
                                  outdir=self.outdirs['0-0_Found_Files'],
                                  file_pattern=self.fnm_pattern,
                                  file_date_format=self.fnm_date_format,
                                  file_generation_res=self.file_generation_res,
                                  data_res=self.dat_recs_nominal_timeres,
                                  files_how_many=self.files_how_many,
                                  logfile_path=logfile_path)
        fd.run()
        files_overview_df = fd.get()
        return logfile_path, files_overview_df

    def calc_lagtimes(self):
        """Calculate covariances and detect covariance peaks to determine lags"""

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
                                  files_overview_df=self.files_overview_df,
                                  logfile_path=self.logfile_path)
            loop_iter.run()

        # Plot loop results
        loop_plots = loop.PlotLoopResults(outdirs=self.outdirs,
                                          lgs_num_iter=self.lgs_num_iter,
                                          lgs_hist_perc_thres=self.lgs_hist_perc_thres,
                                          logfile_path=self.logfile_path)
        loop_plots.run()
        return

    def analyze_lagtimes(self):
        """Analyze lag search results and create look-up table for lag-time normalization"""
        # Analyze results
        analyze = AnalyzeLoopResults(lgs_num_iter=self.lgs_num_iter,
                                     outdirs=self.outdirs,
                                     lag_target=self.lag_target,
                                     logfile_path=self.logfile_path)
        analyze.run()
        return analyze.lut_success

    def normalize_lagtimes(self, lut_success):
        """Apply look-up table to normalize lag for each file"""
        if lut_success:
            normalize_lags = NormalizeLags(files_overview_df=self.files_overview_df,
                                           dat_recs_timestamp_format=self.dat_recs_timestamp_format,
                                           outdirs=self.outdirs,
                                           normalize_lag_for_cols=self.normalize_lag_for_cols,
                                           logfile_path=self.logfile_path)
        return

# if __name__ == "__main__":
#     DynamicLagRemover()
