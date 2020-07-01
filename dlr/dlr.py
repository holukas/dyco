"""

DYNAMIC LAG REMOVER - DLR
-------------------------
A Python package to detect and compensate for shifting lag times in ecosystem time series



File:       Data file
Segment:    Segment in file, e.g. one 6-hour file may consist of 12 half-hour segments.

Step (1)
FilesDetector (files.FilesDetector)

"""
import pandas as pd

import loop
from analyze import AnalyzeLoopResults
from correction import NormalizeLags
from setup import Setup

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)


class DynamicLagRemover:
    # Dataframes for results collection

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

# if __name__ == "__main__":
#     DynamicLagRemover()
