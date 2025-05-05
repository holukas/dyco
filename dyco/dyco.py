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
from pathlib import Path

import pandas as pd
from diive.core.io.filedetector import FileDetector
from diive.core.io.filereader import search_files

import dyco.setup as setup
from dyco import cli, loop
from dyco.analyze import AnalyzeLags
from dyco.correction import RemoveLags
from dyco.files import read_segment_lagtimes_file

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)


class Dyco:
    """
    DYCO - Dynamic lag compensation
    """



    def __init__(self,
                 var_reference: str,
                 var_lagged: str,
                 var_target: list,
                 indir: str = None,
                 outdir: str = None,
                 filename_date_format: str = '%Y%m%d%H%M%S',
                 filename_pattern: str = '*.csv',
                 files_how_many: None or int = None,
                 file_generation_res: str = '30min',
                 file_duration: str = '30min',
                 data_timestamp_format: None or str = None,
                 data_nominal_timeres: float = 0.05,
                 lag_segment_dur: str = '30min',
                 lag_winsize: list or int = 1000,
                 lag_n_iter: int = 1,
                 lag_hist_remove_fringe_bins: bool = True,
                 lag_hist_perc_thres: float = 0.9,
                 lag_shift_stepsize: int = None,
                 target_lag: int = 0,
                 del_previous_results: bool = False
                 ):
        """

        Parameters
        ----------
        phase: int
            Phase in the processing chain, automatically filled attributed during processing.
            * Phase 1 works on input files and applies the first normalization.
            * Phase 2 works on normalized files from Phase 1 and refines normalization.

        var_reference: str
            Column name of the reference signal in the data. Lags are
            determined in relation to this signal.

        var_lagged: str
            Column name of the lagged signal  for which the lag time in
            relation to the reference signal is determined.

        filename_pattern: str, accepts regex
            Filename pattern for data file search.
            Example:
                - With data files following the naming structure '20161015123000.csv'
                the corresponding setting is: fnm_pattern='2016*.csv'

        filename_date_format: str
            Date format in data filenames. Is used to parse the date and
            time info from the filename of found files. Only files found
            with *fnm_pattern* will be parsed.
            Example:
                - With a data file named '20161015123000.csv' the
                 corresponding setting is: fnm_date_format='%Y%m%d%H%M%S'
            For format codes see here:
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes


        del_previous_results: bool
            If *True*, delete all previous results in *indir*. If *False*,
            search for previously calculated results and continue.

        data_timestamp_format: str
            Timestamp format for each row record.

        files_how_many: int
            Limits number of found files that are used.

        file_generation_res: str (pandas DateOffset)
            Frequency at which new files were generated. This does not
            relate to the data records but to the file creation time.
            Examples:
                * '30min' means a new file was generated every 30 minutes.
                * '1h' means a new file was generated every hour.
                * '6h' means a new file was generated every six hours.
            For pandas DateOffset options see:
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        file_duration: str (pandas DateOffset)
            Duration of one data file.
            Example:
                * '30min': data file contains data from 30 minutes.
            For pandas DateOffset options see:
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        lag_segment_dur: str (pandas DateOffset)
            Segment duration for lag determination. If it is the same
            as *file_duration*, the lag time for the complete file is
            calculated from all file data. If it is shorter than
            *file_duration*, then the file data is split into segments
            and the lag time is calculated for each segment separately.
            Examples:
                * '10min': calculates lag times for 10-minute segments.
                * With the settings
                    file_duration = '30min' and
                    lgs_segments_dur = '10min'
                    the 30-minute data file is split into three 10-minute
                    segments and the lag time is determined in each of the
                    segments, yielding three lag times.
            For pandas DateOffset options see:
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        lag_hist_perc_thres: float between 0.1 and 1 (percentage)
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

        lag_hist_remove_fringe_bins: bool
            Remove fringe bins in histogram of found lag times. In case of low
            signal-to-noise ratios the lag search yields less clear results and
            found lag times tend to accumulate in the fringe bins of the histogram,
            i.e. in the very first and last bins, potentially creating non-desirable
            peak bins. In other words, if True the first and last bins of the
            histogram are removed before the time window for lag search is adjusted.

        data_nominal_timeres: float
            Nominal (expected) time resolution of data records.
            Example:
                * 0.05: one record every 0.05 seconds (20Hz)

        lag_winsize: int
            Starting time window size for lag search +/-, given as number of records.
            If negative, the absolute value will be used.
            Example:
                * 1000: Lag search during the first iteration is done in a time window
                    from -1000 records to +1000 records.

        lag_n_iter: int
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
            normalized. A negative number means that *var_lagged* lags x records
            behind *var_reference*.
            Example:
                * 0: The default lag time for all files is set to 0 records.
                    This means that if a lag search is performed on these date, the
                    lag time should consistently be found around 0 records.

        var_target: list of strings
            Column names of the time series the normalized lag should be applied to.


        Links
        -----
        * Overview of pandas DateOffsets:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        """

        # Variable settings (parameters)
        self.var_reference = var_reference
        self.var_lagged = var_lagged
        self.var_target = var_target

        # Folder settings (parameters)
        self.indir = indir
        self.outdir = outdir

        # File settings (parameters)
        self.filename_date_format = filename_date_format
        self.filename_pattern = filename_pattern
        self.file_generation_res = file_generation_res
        self.file_duration = file_duration
        self.files_how_many = files_how_many

        # Data settings (parameters)
        self.data_timestamp_format = data_timestamp_format
        self.data_nominal_timeres = data_nominal_timeres
        self.del_previous_results = del_previous_results

        # Lag settings (parameters)
        self.lag_segment_dur = lag_segment_dur
        if isinstance(lag_winsize, list):
            self.lag_winsize = lag_winsize
        elif isinstance(lag_winsize, int):
            self.lag_winsize = [abs(lag_winsize) * -1, abs(lag_winsize)]
        elif isinstance(lag_winsize, float):
            self.lag_winsize = [abs(int(lag_winsize)) * -1, abs(int(lag_winsize))]
        else:
            raise ValueError("lgs_winsize must be a list or int")
        self.lag_n_iter = lag_n_iter
        self.lag_hist_remove_fringe_bins = lag_hist_remove_fringe_bins
        self.lag_hist_perc_thres = \
            self._set_lgs_hist_perc_thres(lgs_hist_perc_thres=lag_hist_perc_thres)
        self.lag_shift_stepsize = lag_shift_stepsize
        self.target_lag = target_lag

        # Setup
        self.run_id, self.script_start_time = setup.generate_run_id()
        self.lag_winsize_initial = self.lag_winsize

        # Newly generated
        self.outdirs = None
        self.logfile_path = None
        self.files_overview_df = pd.DataFrame()
        self.files_overview_df = pd.DataFrame()
        self.segment_lagtimes_df = pd.DataFrame()
        self.lut_lag_times_df = pd.DataFrame()

        # Run setup
        self._setup()

    def _setup(self):
        # Input and output directories
        self.indir, self.outdir = setup.set_dirs(indir=self.indir, outdir=self.outdir)

        # Setup
        self.outdirs, self.logfile_path, self.logger = self._setup_dirs()

        # Search files
        self.files_overview_df = self._search_files()

    def detect_lags(self):
        """Calculate covariances and detect covariance peaks to determine lags
        for each file segment."""

        for iteration in range(1, 1 + self.lag_n_iter):
            loop_iter = loop.Loop(
                dat_recs_timestamp_format=self.data_timestamp_format,
                dat_recs_nominal_timeres=self.data_nominal_timeres,
                lgs_hist_remove_fringe_bins=self.lag_hist_remove_fringe_bins,
                lgs_hist_perc_thres=self.lag_hist_perc_thres,
                outdirs=self.outdirs,
                lgs_segment_dur=self.lag_segment_dur,
                var_reference=self.var_reference,
                var_lagged=self.var_lagged,
                lgs_num_iter=self.lag_n_iter,
                files_overview_df=self.files_overview_df,
                logfile_path=self.logfile_path,
                lgs_winsize=self.lag_winsize,
                fnm_date_format=self.filename_date_format,
                iteration=iteration,
                logger=self.logger,
                shift_stepsize=self.lag_shift_stepsize,
                segment_lagtimes_df=self.segment_lagtimes_df)
            loop_iter.run()
            self.lag_winsize, self.segment_lagtimes_df = loop_iter.get()  # Update search window for next iteration

        # Plot loop results after all iterations finished
        loop_plots = loop.PlotLoopResults(outdirs=self.outdirs,
                                          lag_n_iter=self.lag_n_iter,
                                          histogram_percentage_threshold=self.lag_hist_perc_thres,
                                          plot_cov_collection=True,
                                          plot_hist=True,
                                          plot_timeseries_segment_lagtimes=True,
                                          logger=self.logger,
                                          segment_lagtimes_df=self.segment_lagtimes_df)
        loop_plots.run()
        return

    def _run(self):
        """
        Run setup, calculations, analyses and correction of files

        Processing consists of 4 steps:
            * Step 1: Setup
            * Step 2: Calculate time lags: lag times for each file segment
            * Step 3: Analyze time lags: analyze results and create default-lag lookup-table (LUT)
            * Step 4: Remove time lags: use look-up table to normalize time lags across files

        Each step uses results from the previous step.

        """

        # # Step 1: Setup
        # self.outdirs, self.logfile_path = self._setup_dirs()
        #
        # # Step 2: Search files
        # self.files_overview_df = self._search_files()

        # # Step 3: Calculation of lag times for each file segment in input files
        # self.detect_lags()
        #
        # # Step 4: Analyses of results, create LUT
        # lut_success = self.analyze_lags()
        #
        # # Step 5: Lag-time normalization for each file
        # self.remove_lags(lut_success=lut_success)

    @staticmethod
    def _set_lgs_hist_perc_thres(lgs_hist_perc_thres):
        if lgs_hist_perc_thres > 1:
            lgs_hist_perc_thres = 1
        elif lgs_hist_perc_thres < 0.1:
            lgs_hist_perc_thres = 0.1  # Minimum 10% since less would not be useful
        else:
            lgs_hist_perc_thres = lgs_hist_perc_thres
        return lgs_hist_perc_thres

    def _search_files(self):
        # Search files with PATTERN
        print(f"\nSearching files with pattern {self.filename_pattern} in dir {self.indir} ...")
        filelist = search_files(searchdirs=str(self.indir), pattern=self.filename_pattern)
        for filepath in filelist:
            print(f"    --> Found file: {filepath.name} in {filepath}.")

        fide = FileDetector(filelist=filelist,
                            file_date_format=self.filename_date_format,
                            file_generation_res=self.file_generation_res,
                            data_res=self.data_nominal_timeres,
                            files_how_many=self.files_how_many)
        fide.run()
        files_overview_df = fide.get_results()

        # Export dataframe to csv
        outpath = self.outdirs['1_overview'] / '1_files_overview.csv'
        files_overview_df.to_csv(outpath)
        return files_overview_df

    def _setup_dirs(self):
        """Create output folders, start logger and search for files"""
        # Create folders
        outdirs = setup.CreateOutputDirs(dyco_instance=self).setup_output_dirs()

        # Start logging
        logfile_path = setup.set_logfile_path(run_id=self.run_id, outdir=outdirs['0_log'])
        logger = setup.create_logger(logfile_path=logfile_path, name=__name__)
        logger.info(f"Run ID: {self.run_id}")
        return outdirs, logfile_path, logger

    def analyze_lags(self,
                     filepath_found_lag_times: str = None,
                     outlier_thres_zscore: float = 1.4,
                     outlier_winsize: int = None):
        """Analyze lag search results and create look-up table for lag-time normalization"""

        if filepath_found_lag_times:
            segment_lagtimes_df = read_segment_lagtimes_file(filepath=filepath_found_lag_times)
        else:
            segment_lagtimes_df = self.segment_lagtimes_df

        analyze = AnalyzeLags(lgs_num_iter=self.lag_n_iter,
                              outdirs=self.outdirs,
                              target_lag=self.target_lag,
                              logger=self.logger,
                              lags=segment_lagtimes_df,
                              outlier_winsize=outlier_winsize,
                              outlier_thres_zscore=outlier_thres_zscore)

        self.lut_lag_times_df = analyze.get_lut()

        return

    def remove_lags(self, filepath_lut: str = None):
        """
        Apply look-up table to normalize lag for each file

        Parameters
        ----------
        filepath_lut:

        Returns
        -------
        None
        """
        if filepath_lut:
            lut_df = read_segment_lagtimes_file(filepath=filepath_lut)
        else:
            lut_df = self.segment_lagtimes_df

        lut_df.index = pd.to_datetime(lut_df.index)

        RemoveLags(lut=lut_df,
                   files_overview_df=self.files_overview_df,
                   data_timestamp_format=self.data_timestamp_format,
                   outdirs=self.outdirs,
                   var_target=self.var_target,
                   lag_n_iter=self.lag_n_iter,
                   logger=self.logger)

        return


def main(args):
    """Main function that is called with the given args when the script
     is executed from the command line."""

    Dyco(var_reference=args.var_reference,
         var_lagged=args.var_lagged,
         var_target=args.var_target,
         indir=args.indir,
         outdir=args.outdir,
         filename_date_format=args.filenamedateformat,
         filename_pattern=args.filenamepattern,
         files_how_many=args.limitnumfiles,
         file_generation_res=args.filegenres,
         file_duration=args.fileduration,
         data_timestamp_format=args.datatimestampformat,
         data_nominal_timeres=args.datanominaltimeres,
         lag_segment_dur=args.lssegmentduration,
         lag_winsize=args.lswinsize,
         lag_n_iter=args.lsnumiter,
         lag_hist_remove_fringe_bins=args.lsremovefringebins,
         lag_hist_perc_thres=args.lspercthres,
         target_lag=args.targetlag,
         del_previous_results=args.delprevresults)


if __name__ == '__main__':
    args = cli.get_args()
    args = cli.validate_args(args)
    main(args)
