import datetime as dt
import unittest
from pathlib import Path

import pandas as pd

from dyco import setup_dyco, analyze, files, loop, lag


class Tests(unittest.TestCase):

    def test_read_segment_lagtimes_file(self):
        filepath = 'test_data/test_segment_lagtimes_file/1_segments_found_lag_times_after_iteration-1.csv'
        segment_lagtimes_df = files.read_segment_lagtimes_file(filepath=filepath)
        self.assertEqual(len(segment_lagtimes_df), 9)
        self.assertEqual(len(segment_lagtimes_df.columns), 19)
        self.assertEqual(len(segment_lagtimes_df['lagsearch_next_start'].unique()), 1)

    def test_read_raw_data(self):
        filepath = 'test_data/test_raw_data/20161020113000.csv'
        raw_data_df = files.read_raw_data(filepath=filepath, data_timestamp_format='%Y-%m-%d %H:%M:%S.%f')
        self.assertEqual(len(raw_data_df), 36000)
        self.assertEqual(len(raw_data_df.columns), 30)
        self.assertEqual(raw_data_df.index[0], pd.Timestamp('2016-10-20 11:30:00.024999'))
        self.assertEqual(raw_data_df['co2_ppb_qcl'].mean(), 369599.2060149097)

    def test_files_detector(self):
        indir = Path('test_data/test_raw_data/')

        class TestClass(object):
            def __init__(self):
                self.indir = indir
                self.fnm_pattern = '2016*.csv'
                self.fnm_date_format = '%Y%m%d%H%M%S'
                self.file_generation_res = '30T'
                self.dat_recs_nominal_timeres = 0.05
                self.files_how_many = False

        test_class_instance = TestClass()

        fd = setup_dyco.FilesDetector(dyco_instance=test_class_instance,
                                      outdir=False,
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

        class TestClass(object):
            def __init__(self):
                self.var_reference = 'w_ms-1_rot_turb'
                self.var_lagged = 'co2_ppb_qcl_turb'
                self.lgs_winsize = [-1000, 1000]
                self.iteration = 1
                self.shift_stepsize = 10
                self.logfile_path = None
                self.phase = 0
                self.phase_files = '__testing__'

        test_class_instance = TestClass()

        lagsearch_df, props_peak_auto = lag.LagSearch(loop_instance=test_class_instance,
                                                      segment_df=segment_df,
                                                      segment_name='20161020113000_iter1',
                                                      file_idx=dt.datetime(2016, 10, 20, 11, 30, 00),
                                                      segment_start=dt.datetime(2016, 10, 20, 11, 30, 00, 24999),
                                                      segment_end=dt.datetime(2016, 10, 20, 11, 59, 59, 966666),
                                                      filename='20161020113000').get()

        self.assertEqual(lagsearch_df.loc[lagsearch_df['flag_peak_max_cov_abs'] == 1, 'shift'].values[0], -290)
        self.assertEqual(lagsearch_df.loc[lagsearch_df['flag_peak_auto'] == 1, 'shift'].values[0], -290)
        # self.assertEqual(lagsearch_df.loc[lagsearch_df['flag_peak_max_cov_abs'] == 1, 'cov_abs'].values[0],
        #                  223.13887346667508)

    def test_adjust_lag_search_window(self):
        """
        Test setting time window for next lag search iteration, based on results
        from previous iteration.
        """
        filepath = 'test_data/test_segment_lagtimes_file/1_segments_found_lag_times_after_iteration-1.csv'
        segment_lagtimes_df = files.read_segment_lagtimes_file(filepath=filepath)
        hist_series = loop.Loop.filter_series(filter_col='iteration', filter_equal_to=1,
                                              df=segment_lagtimes_df, series_col='PEAK-COVABSMAX_SHIFT')
        lgs_winsize_adj = lag.AdjustLagsearchWindow(series=hist_series,
                                                    outdir=None,
                                                    iteration=1,
                                                    plot=True,
                                                    hist_num_bins=30,
                                                    remove_fringe_bins=True,
                                                    perc_threshold=0.9).get()
        self.assertEqual(lgs_winsize_adj, [-755, -152])

    def test_analyze_loop_results(self):
        """
        Test creation of lookup-table containing default lag times
        """
        filepath = 'test_data/test_segment_lagtimes_file/1_segments_found_lag_times_after_iteration-1.csv'

        # segment_lagtimes_df = files.read_segment_lagtimes_file(filepath=filepath)

        class TestClass(object):
            def __init__(self):
                self.var_reference = 'w_ms-1_rot_turb'
                self.var_lagged = 'co2_ppb_qcl_turb'
                self.lgs_winsize = [-1000, 1000]
                self.iteration = 1
                self.shift_stepsize = 10
                self.logfile_path = None
                self.lgs_num_iter = 1
                self.outdirs = None
                self.target_lag = 0
                self.phase = 0
                self.phase_files = '__testing__'

        test_class_instance = TestClass()

        _analyze = analyze.AnalyzeLags(dyco_instance=test_class_instance,
                                       direct_path_to_segment_lagtimes_file=filepath)
        # obj.run()
        # lut_default_lag_times_df, lut_success = obj.get()

        self.assertEqual(_analyze.lut_available, True)
        self.assertEqual(_analyze.lut_lag_times_df.index[0], dt.date(2016, 10, 15))
        self.assertEqual(len(_analyze.lut_lag_times_df.loc[:, 'median'].dropna()), 1)
        self.assertEqual(len(_analyze.lut_lag_times_df.loc[:, 'correction'].dropna()), 1)
        self.assertEqual(_analyze.lut_lag_times_df.loc[_analyze.lut_lag_times_df.index[-1], 'correction'], -290)

    def test_detect_covariance_peaks(self):
        """Test peak detection only"""
        filepath = 'test_data/test_raw_data/20161020113000.csv'
        segment_df = files.read_raw_data(filepath=filepath, data_timestamp_format='%Y-%m-%d %H:%M:%S.%f')
        lagsearch_df = lag.LagSearch.setup_lagsearch_df(lgs_winsize=[-1000, 1000],
                                                        shift_stepsize=10,
                                                        segment_name='20161031230000_iter1')
        lagsearch_df = \
            lag.LagSearch.find_max_cov_peak(segment_df=segment_df,
                                            cov_df=lagsearch_df,
                                            var_reference='w_ms-1_rot_turb',
                                            var_lagged='co2_ppb_qcl_turb')
        lagsearch_df, props_peak_auto = lag.LagSearch.find_auto_peak(cov_df=lagsearch_df)

        self.assertEqual(lagsearch_df.loc[lagsearch_df['flag_peak_max_cov_abs'] == 1, 'shift'].values[0], -290)
        self.assertEqual(lagsearch_df.loc[lagsearch_df['flag_peak_auto'] == 1, 'shift'].values[0], -290)
        self.assertEqual(int(lagsearch_df.loc[lagsearch_df['flag_peak_max_cov_abs'] == 1, 'cov_abs'].values[0]),
                         223)


if __name__ == '__main__':
    unittest.main()
