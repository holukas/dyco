import datetime as dt
import unittest
from pathlib import Path

import pandas as pd

from dyla import _setup, analyze, files, loop, lag


class Tests(unittest.TestCase):

    def test_read_segment_lagtimes_file(self):
        filepath = 'test_data/test_segment_lagtimes_file/1_segments_found_lag_times_after_iteration-1.csv'
        segment_lagtimes_df = files.read_segment_lagtimes_file(filepath=filepath)
        self.assertEqual(len(segment_lagtimes_df), 828)
        self.assertEqual(len(segment_lagtimes_df.columns), 15)
        self.assertEqual(len(segment_lagtimes_df['lagsearch_next_start'].unique()), 1)

    def test_read_raw_data(self):
        filepath = 'test_data/test_raw_data/20161020113000.csv'
        raw_data_df = files.read_raw_data(filepath=filepath, data_timestamp_format='%Y-%m-%d %H:%M:%S.%f')
        self.assertEqual(len(raw_data_df), 36000)
        self.assertEqual(len(raw_data_df.columns), 30)
        self.assertEqual(raw_data_df.index[0], pd.Timestamp('2016-10-20 11:30:00.024999'))
        self.assertEqual(raw_data_df['co2_ppb_qcl'].mean(), 369599.2060149097)

    def test_files_detector(self):
        dir_input = Path('test_data/test_raw_data/')

        fd = _setup.FilesDetector(dir_input=dir_input,
                                  outdir=False,
                                  file_pattern='2016*.csv',
                                  file_date_format='%Y%m%d%H%M%S',
                                  file_generation_res='30T',
                                  data_res=0.05,
                                  files_how_many=False,
                                  logfile_path=None)
        fd.run()
        self.files_overview_df = fd.get()

        self.assertEqual(len(self.files_overview_df), 4)
        self.assertEqual(len(self.files_overview_df.columns), 9)
        self.assertEqual(self.files_overview_df['file_available'].sum(), 2)
        self.assertEqual(self.files_overview_df.loc[self.files_overview_df['file_available'] == 1, :].index.to_list(),
                         [dt.datetime(2016, 10, 20, 11, 30, 00), dt.datetime(2016, 10, 20, 12, 30, 00)])

    def test_lagsearch_class(self):
        """
        Test lag search and covariance peak detection
        """
        filepath = 'test_data/test_raw_data/20161020113000.csv'
        segment_df = files.read_raw_data(filepath=filepath, data_timestamp_format='%Y-%m-%d %H:%M:%S.%f')

        lagsearch_df, props_peak_auto = lag.LagSearch(segment_df=segment_df,
                                                      segment_name='20161020113000_iter1',
                                                      ref_sig='w_ms-1_rot_turb',
                                                      lagged_sig='co2_ppb_qcl_turb',
                                                      win_lagsearch=[-1000, 1000],
                                                      file_idx=dt.datetime(2016, 10, 20, 11, 30, 00),
                                                      segment_start=dt.datetime(2016, 10, 20, 11, 30, 00, 24999),
                                                      segment_end=dt.datetime(2016, 10, 20, 11, 59, 59, 966666),
                                                      filename='20161020113000',
                                                      iteration=1,
                                                      shift_stepsize=10).get()

        self.assertEqual(lagsearch_df.loc[lagsearch_df['flag_peak_max_cov_abs'] == 1, 'shift'].values[0], -290)
        self.assertEqual(lagsearch_df.loc[lagsearch_df['flag_peak_auto'] == 1, 'shift'].values[0], -290)
        self.assertEqual(lagsearch_df.loc[lagsearch_df['flag_peak_max_cov_abs'] == 1, 'cov_abs'].values[0],
                         223.13887346667508)

    def test_adjust_lag_search_window(self):
        """
        Test setting time window for next lag search iteration, based on results
        from previous iteration.
        """
        filepath = 'test_data/test_segment_lagtimes_file/1_segments_found_lag_times_after_iteration-1.csv'
        segment_lagtimes_df = files.read_segment_lagtimes_file(filepath=filepath)
        hist_series = loop.Loop.filter_series(filter_col='iteration', filter_equal_to=1,
                                              df=segment_lagtimes_df, series_col='shift_peak_cov_abs_max')
        win_lagsearch_adj = lag.AdjustLagsearchWindow(series=hist_series,
                                                      outdir=None,
                                                      iteration=1,
                                                      plot=True,
                                                      hist_num_bins=30,
                                                      remove_fringe_bins=True,
                                                      perc_threshold=0.9).get()
        self.assertEqual(win_lagsearch_adj, [-923, 668])

    def test_analyze_loop_results(self):
        """
        Test creation of lookup-table containing default lag times
        """
        filepath = 'test_data/test_segment_lagtimes_file/1_segments_found_lag_times_after_iteration-1.csv'
        # segment_lagtimes_df = files.read_segment_lagtimes_file(filepath=filepath)
        obj = analyze.AnalyzeLoopResults(lgs_num_iter=1,
                                         direct_path_to_segment_lagtimes_file=filepath)
        obj.run()
        lut_default_lag_times_df, lut_success = obj.get()

        self.assertEqual(lut_success, True)
        self.assertEqual(lut_default_lag_times_df.index[6], dt.date(2016, 10, 20))
        self.assertEqual(len(lut_default_lag_times_df.loc[:, 'median'].dropna()), 18)
        self.assertEqual(len(lut_default_lag_times_df.loc[:, 'correction'].dropna()), 19)
        self.assertEqual(lut_default_lag_times_df.loc[lut_default_lag_times_df.index[-1], 'correction'], -320)

    # def test_detect_covariance_peaks(self):
    #     """Test peak detection only"""
    #     filepath = 'test_data/test_raw_data/20161020113000.csv'
    #     segment_df = files.read_raw_data(filepath=filepath, data_timestamp_format='%Y-%m-%d %H:%M:%S.%f')
    #     lagsearch_df = lag.LagSearch.setup_lagsearch_df(win_lagsearch=[-1000, 1000],
    #                                                     shift_stepsize=10,
    #                                                     segment_name='20161031230000_iter1')
    #     lagsearch_df = \
    #         lag.LagSearch.find_max_cov_peak(segment_df=segment_df,
    #                                         lagsearch_df=lagsearch_df,
    #                                         ref_sig='w_ms-1_rot_turb',
    #                                         lagged_sig='co2_ppb_qcl_turb')
    #     lagsearch_df, props_peak_auto = lag.LagSearch.find_peak_auto(df=lagsearch_df)
    #
    #     self.assertEqual(lagsearch_df.loc[lagsearch_df['flag_peak_max_cov_abs'] == 1, 'shift'].values[0], -290)
    #     self.assertEqual(lagsearch_df.loc[lagsearch_df['flag_peak_auto'] == 1, 'shift'].values[0], -290)
    #     self.assertEqual(lagsearch_df.loc[lagsearch_df['flag_peak_max_cov_abs'] == 1, 'cov_abs'].values[0],
    #                      223.13887346667508)


if __name__ == '__main__':
    unittest.main()
