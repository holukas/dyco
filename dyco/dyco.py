"""

DYCO - DYNAMIC LAG REMOVER
--------------------------
Dynamic lag detection and compensation
A Python package to detect and compensate for shifting lag times in
ecosystem time series



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


def dyco(cls):
    class Wrapper:
        def __init__(self, **args):
            args['outdir'] = args['outdir'] / 'results_input_files'
            self.run_input_files = cls(**args)

            # TODO hier weiter
            args['indir'] = self.run_input_files.outdir
            args['outdir'] = self.run_input_files.outdir / 'results_normalized_files'
            args['lgs_refsig'] = f"{args['lgs_refsig']}_DYCO"
            args['lgs_lagsig'] = f"{args['lgs_lagsig']}_DYCO"
            self.run_output_files = cls(**args)

    return Wrapper


@dyco
class DynamicLagCompensation:
    """
    DYCO - Dynamic lag compensation
    """

    files_overview_df = pd.DataFrame()

    def __init__(self,
                 lgs_refsig: str,
                 lgs_lagsig: str,
                 fnm_date_format: str = '%Y%m%d%H%M%S',
                 del_previous_results: bool = False,
                 fnm_pattern: str = '*.csv',
                 dat_recs_timestamp_format: False or str = False,
                 files_how_many: False or int = False,
                 file_generation_res: str = '30T',
                 file_duration: str = '30T',
                 lgs_segment_dur: False or pd.DateOffset = False,
                 lgs_hist_perc_thres: float = 0.9,
                 lgs_hist_remove_fringe_bins: bool = True,
                 dat_recs_nominal_timeres: float = 0.05,
                 lgs_winsize: int = 1000,
                 lgs_num_iter: int = 3,
                 indir: Path = False,
                 outdir: Path = False,
                 target_lag: int = -100,
                 target_cols: list = None):
        """

        Parameters
        ----------
        lgs_refsig: str
            Column name of the reference signal in the data. Lags are
            determined in relation to this signal.

        lgs_lagsig: str
            Column name of the lagged signal  for which the lag time in
            relation to the reference signal is determined.

        fnm_pattern: str, accepts regex
            Filename pattern for data file search.
            Example:
                - With data files following the naming structure '20161015123000.csv'
                the corresponding setting is: fnm_pattern='2016*.csv'

        fnm_date_format: str
            Date format in data filenames. Is used to parse the date and
            time info from the filename of found files. Only files found
            with *fnm_pattern* will be parsed.
            Example:
                - With a data file named '20161015123000.csv' the
                 corresponding setting is: fnm_date_format='%Y%m%d%H%M%S'


        del_previous_results: bool
            If *True*, delete all previous results in *indir*. If *False*,
            search for previously calculated results and continue.

        dat_recs_timestamp_format: str
            Timestamp format for each row record.

        files_how_many: int
            Limits number of found files that are used.

        file_generation_res: str (pandas DateOffset)
            Frequency at which new files were generated. This does not
            relate to the data records but to the file creation time.
            Examples:
                * '30T' means a new file was generated every 30 minutes.
                * '1H' means a new file was generated every hour.
                * '6H' means a new file was generated every six hours.
            For pandas DateOffset options see:
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        file_duration: str (pandas DateOffset)
            Duration of one data file.
            Example:
                * '30T': data file contains data from 30 minutes.
            For pandas DateOffset options see:
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        lgs_segment_dur: str (pandas DateOffset)
            Segment duration for lag determination. If it is the same
            as *file_duration*, the lag time for the complete file is
            calculated from all file data. If it is shorter than
            *file_duration*, then the file data is split into segments
            and the lag time is calculated for each segment separately.
            Examples:
                * '10T': calculates lag times for 10-minute segments.
                * With the settings
                    file_duration = '30T' and
                    lgs_segments_dur = '10T'
                    the 30-minute data file is split into three 10-minute
                    segments and the lag time is determined in each of the
                    segments, yielding three lag times.
            For pandas DateOffset options see:
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        lgs_hist_perc_thres: float between 0.1 and 1 (percentage)
            Cumulative percentage threshold in histogram of found lag times.
            The time window for lag search during each iteration (i) is
            narrowed down based on the histogram of all found lag times
            from the previous iteration (i-1). Is set to 1 if > 1 or set
            to 0.1 if < 0.1.

            During each iteration and after lag times were determined for
            all files and segments, a histogram of found lag times is created
            to identify the histogram bin in which most lag times (most counts)
            were found (peak bin). To narrow down the time window for lag search
            during the next iteration (i+1), the bins around the peak bin are
            included until a certain percentage of the total values is reached
            over the expanded bin range, centered on the peak bin.

            Example:
                * 0.9: include all bins to each site of the peak bin until 90%
                    of the total found lag times (counts) are included. The time
                    window for the lag search during the next iteration (i+1) is
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
            If negative, the absolute value will be used.
            Example:
                * 1000: Lag search during the first iteration is done in a time window
                    from -1000 records to +1000 records.

        lgs_num_iter: int
            Number of lag search interations. Before each iteration, the time window
            for the lag search is narrowed down, taking into account results from the
            previous iteration. Exception is the first iteration for which the time
            window as given in *lgs_winsize* is used.
            Example:
                * *lgs_num_iter* = 3: lag search in iteration 1 (i1) uses *lgs_winsize*
                    to search for lag times, then the lag window is narrowed down using
                    results from i1. The adjusted search window is the used in i2 to
                    again search lag times for the same data. Likewise, i3 uses the
                    adjusted search window based on results from i2.

        indir: Path or False
            Source folder that contains the data files. If *False*, a folder named 'input'
            is searched in the current working directory.

        outdir: Path or False
            Output folder for results. If *False*, a folder named 'output'
            is created in the current working directory.

        target_lag: int
            The target lag given in records to which lag times of all files are
            normalized. A negative number means that *lgs_lagsig* lags x records
            behind *lgs_refsig*.
            Example:
                * -100: The default lag time for all files is set to -100 records.
                    This means that if a lag search is performed on these date, the
                    lag time should consistently be found around -100 records.

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

        # File settings
        self.fnm_date_format = fnm_date_format
        self.file_generation_res = file_generation_res
        self.fnm_pattern = fnm_pattern
        self.files_how_many = files_how_many
        self.file_duration = file_duration
        self.del_previous_results = del_previous_results

        # Data records
        self.dat_recs_nominal_timeres = dat_recs_nominal_timeres
        self.dat_recs_timestamp_format = dat_recs_timestamp_format
        # self.data_segment_overhang = data_segment_overhang

        # Lag search and normalization
        self.lgs_refsig = lgs_refsig
        self.lgs_lagsig = lgs_lagsig
        self.lgs_segment_dur = lgs_segment_dur
        self.lgs_winsize = [abs(lgs_winsize) * -1, abs(lgs_winsize)]
        self.lgs_winsize_initial = self.lgs_winsize
        self.lgs_num_iter = lgs_num_iter
        self.lgs_hist_remove_fringe_bins = lgs_hist_remove_fringe_bins
        if lgs_hist_perc_thres > 1:
            self.lgs_hist_perc_thres = 1
        elif lgs_hist_perc_thres < 0.1:
            self.lgs_hist_perc_thres = 0.1  # Minimum 10% since less would not be useful
        else:
            self.lgs_hist_perc_thres = lgs_hist_perc_thres
        self.lag_target = target_lag
        self.normalize_lag_for_cols = target_cols

        # Start scripts
        self.run()

    def run(self):
        """
        Run setup, calculations, analyses and correction of files

        Full processing consists of 4 steps:
            * Step 1: Setup
            * Step 2: Calculate lag times for each file segment
            * Step 3: Analyze results and create default-lag lookup-table (LUT)
            * Step 4: Use look-up table to normalize time lags across files

        Each step uses results from the previous step.

        """
        # Step 1: Setup
        self.logfile_path, self.files_overview_df = self.setup()

        # # Step 2: Calculation of lag times for each file segment in input files
        self.calc_lagtimes()

        # Step 3: Analyses of results, create LUT
        lut_success = self.analyze_lagtimes()

        # Step 4: Lag-time normalization for each file
        self.normalize_lagtimes(lut_success=lut_success)

    def setup(self):
        """Create output folders, start logger and search for files"""
        # Create folders
        self.outdirs = _setup.CreateOutputDirs(root_dir=self.outdir,
                                               del_previous_results=self.del_previous_results).setup_output_dirs()

        # Start logging
        logfile_path = _setup.set_logfile_path(run_id=self.run_id, outdir=self.outdirs['_log'])
        logger = _setup.create_logger(logfile_path=logfile_path, name=__name__)
        logger.info(f"Run ID: {self.run_id}")

        # Search files
        fd = _setup.FilesDetector(dyco_instance=self,
                                  outdir=self.outdirs['1-0_input_files_overview'],
                                  logfile_path=logfile_path)
        fd.run()
        files_overview_df = fd.get()
        return logfile_path, files_overview_df

    def calc_lagtimes(self):
        """
        Calculate covariances and detect covariance peaks to determine lags
        for each file segment
        """

        for iteration in range(1, 1 + self.lgs_num_iter):
            loop_iter = loop.Loop(dyco_instance=self,
                                  iteration=iteration)
            loop_iter.run()
            self.lgs_winsize = loop_iter.get()  # Update search window for next iteration

        # Plot loop results after all iterations finished
        loop_plots = loop.PlotLoopResults(dyco_instance=self,
                                          plot_cov_collection=True,
                                          plot_hist=True,
                                          plot_timeseries_segment_lagtimes=True)
        loop_plots.run()
        return

    def analyze_lagtimes(self):
        """Analyze lag search results and create look-up table for lag-time normalization"""
        analyze = AnalyzeLoopResults(dyco_instance=self)
        analyze.run()
        return analyze.lut_success

    def normalize_lagtimes(self, lut_success):
        """
        Apply look-up table to normalize lag for each file
        """
        if lut_success:
            NormalizeLags(dyco_instance=self)
        return

# if __name__ == "__main__":
#     DynamicLagRemover()
