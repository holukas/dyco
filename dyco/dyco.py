"""

DYCO - DYNAMIC LAG COMPENSATION
-------------------------------
Dynamic lag detection and compensation
A Python package to detect and compensate for shifting lag times in
ecosystem time series

"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

import _setup
import files
import loop
import plot
from analyze import AnalyzeLags
from correction import RemoveLags

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)


def dyco(cls):
    """
    Wrapper function for DYCO processing chain

    Parameters
    ----------
    cls: class
        The class that is wrapped.

    Returns
    -------
    Wrapper

    """

    class ProcessingChain:

        def __init__(self, **args):
            # PHASE 1 - First normalization to default lag
            # ============================================
            args['phase'] = 1
            self.run_phase_1_input_files = cls(**args)

            # PHASE 2 - Second normalization to default lag
            # =============================================
            args['phase'] = 2
            args = self._update_args(args=args,
                                     prev_phase=self.run_phase_1_input_files.phase,
                                     prev_phase_files=self.run_phase_1_input_files.phase_files,
                                     prev_outdir_files=self.run_phase_1_input_files.outdir,
                                     prev_last_iteration=self.run_phase_1_input_files.lgs_num_iter,
                                     prev_outdirs=self.run_phase_1_input_files.outdirs)
            self.run_phase_2_normalized_files = cls(**args)

            # PHASE 3 - Correction for instantaneous lag
            # ==========================================
            args['phase'] = 3
            args = self._update_args(args=args,
                                     prev_phase=self.run_phase_2_normalized_files.phase,
                                     prev_phase_files=self.run_phase_2_normalized_files.phase_files,
                                     prev_outdir_files=self.run_phase_2_normalized_files.outdir,
                                     prev_last_iteration=self.run_phase_2_normalized_files.lgs_num_iter,
                                     prev_outdirs=self.run_phase_2_normalized_files.outdirs)
            self.run_phase_3_finalize = cls(**args)

            # FINALIZE - Make some more plots summarizing Phases 1-3
            # ======================================================
            plot.SummaryPlots(instance_phase_1=self.run_phase_1_input_files,
                              instance_phase_2=self.run_phase_2_normalized_files,
                              instance_phase_3=self.run_phase_3_finalize)

        def _update_args(self, args, prev_phase, prev_phase_files, prev_outdir_files, prev_last_iteration,
                         prev_outdirs):
            """Update args for running Phases 2 and 3: use results from Phase 1 and 2, respectively"""
            if args['phase'] == 2:
                args['lgs_winsize'] = self._update_winsize(prev_phase=prev_phase,
                                                           prev_phase_files=prev_phase_files,
                                                           prev_last_iteration=prev_last_iteration,
                                                           prev_outdirs=prev_outdirs)
            else:
                args['lgs_winsize'] = 100  # Small window for instantaneous search in Phase 3
                args['lgs_num_iter'] = 1

            args['indir'] = prev_outdir_files / f"{prev_phase}-7_{prev_phase_files}_normalized"
            args['var_lagged'] = f"{args['var_lagged']}_DYCO"  # Use normalized signal
            filename, file_extension = os.path.splitext(args['fnm_pattern'])
            args['fnm_pattern'] = f"{filename}_DYCO{file_extension}"  # Search normalized files
            args['fnm_date_format'] = f"{args['fnm_date_format']}_DYCO"  # Parse file names of normalized files
            var_target = [var_target + '_DYCO' for var_target in args['var_target']]  # Use normalized target cols
            args['var_target'] = var_target
            return args

        def _update_winsize(self, prev_phase, prev_phase_files, prev_last_iteration, prev_outdirs):
            """
            Calculate the range of the lag search window from the last iteration and use
            it for lag search in normalized files

            During the last iteration, this window was detected as the *next* window for
            the *next* iteration, i.e. it was not yet used for lag detection.

            Returns
            -------
            Time window for lag search in Phase 2 iteration 1 and in Phase 3
            """
            filepath_last_iteration = \
                prev_outdirs[
                    f"{prev_phase}-3_{prev_phase_files}_time_lags_overview"] \
                / f'{prev_last_iteration}_segments_found_lag_times_after_iteration-{prev_last_iteration}.csv'

            segment_lagtimes_last_iteration_df = \
                files.read_segment_lagtimes_file(filepath=filepath_last_iteration)
            lgs_winsize = \
                [segment_lagtimes_last_iteration_df['lagsearch_next_start'].unique()[0],
                 segment_lagtimes_last_iteration_df['lagsearch_next_end'].unique()[0]]

            lgs_winsize_normalized = np.abs(lgs_winsize[0] - lgs_winsize[1])  # Range
            lgs_winsize_normalized = lgs_winsize_normalized / 2  # Normalized search window +/- around zero
            return lgs_winsize_normalized

    return ProcessingChain


@dyco
class DynamicLagCompensation:
    """
    DYCO - Dynamic lag compensation
    """

    files_overview_df = pd.DataFrame()

    def __init__(self,

                 var_reference: str,
                 var_lagged: str,
                 phase: int = 1,
                 phase_files: str = 'input_files',
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
                 target_lag: int = 0,
                 var_target: list = None):
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

        self.run_id, self.script_start_time = _setup.generate_run_id()

        # Setup for Phases 1-3
        self.phase = phase
        if self.phase == 1:
            self.phase_files = '_input_files_'
        elif self.phase == 2:
            self.phase_files = '_normalized_files_'
        else:
            self.phase_files = '_refined_normalized_files_'

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
        self.var_reference = var_reference
        self.var_lagged = var_lagged
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
        self.target_lag = target_lag
        self.var_target = var_target

        # Indicate if any new iteration data was created, by default *False*.
        # Will be set to *True* automatically when a new iteration is run.
        self.new_iteration_data = False

        # Start scripts
        self.run()

    def run(self):
        """
        Run setup, calculations, analyses and correction of files

        Processing consists of 4 steps:
            * Step 1: Setup
            * Step 2: Calculate time lags: lag times for each file segment
            * Step 3: Analyze time lags: analyze results and create default-lag lookup-table (LUT)
            * Step 4: Remove time lags: use look-up table to normalize time lags across files

        Each step uses results from the previous step.

        """
        # Step 1: Setup
        self.logfile_path, self.files_overview_df = self.setup()

        # Step 2: Calculation of lag times for each file segment in input files
        self.calculate_lags()

        # Step 3: Analyses of results, create LUT
        lut_success = self.analyze_lags()

        # Step 4: Lag-time normalization for each file
        self.remove_lags(lut_success=lut_success)

    def setup(self):
        """Create output folders, start logger and search for files"""
        # Create folders
        self.outdirs = _setup.CreateOutputDirs(dyco_instance=self).setup_output_dirs()

        # Start logging
        logfile_path = _setup.set_logfile_path(run_id=self.run_id,
                                               outdir=self.outdirs['0-0_log'],
                                               phase=self.phase)
        logger = _setup.create_logger(logfile_path=logfile_path, name=__name__)
        logger.info(f"Run ID: {self.run_id}")

        # Search files
        fd = _setup.FilesDetector(dyco_instance=self,
                                  outdir=self.outdirs[f'{self.phase}-0_{self.phase_files}_overview'],
                                  logfile_path=logfile_path)
        fd.run()
        files_overview_df = fd.get()
        return logfile_path, files_overview_df

    def calculate_lags(self):
        """
        Calculate covariances and detect covariance peaks to determine lags
        for each file segment
        """
        for iteration in range(1, 1 + self.lgs_num_iter):
            loop_iter = loop.Loop(dyco_instance=self,
                                  iteration=iteration)
            loop_iter.run()
            self.lgs_winsize, self.new_iteration_data = loop_iter.get()  # Update search window for next iteration

        # Plot loop results after all iterations finished TODO ACT
        loop_plots = loop.PlotLoopResults(dyco_instance=self,
                                          plot_cov_collection=True,
                                          plot_hist=True,
                                          plot_timeseries_segment_lagtimes=True)
        loop_plots.run()
        return

    def analyze_lags(self):
        """Analyze lag search results and create look-up table for lag-time normalization"""
        analyze = AnalyzeLags(dyco_instance=self)
        return analyze.lut_available

    def remove_lags(self, lut_success):
        """
        Apply look-up table to normalize lag for each file
        """
        if lut_success:
            RemoveLags(dyco_instance=self)
        return

# if __name__ == "__main__":
#     DynamicLagRemover()
