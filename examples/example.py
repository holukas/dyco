import os
from pathlib import Path

from dyla.dlr import DynamicLagRemover as dlr

# # Default parameters
# dir_input = 'input'
# dir_output = 'output'
# data_segment_duration = '30T'  # 30min segments
# data_segment_overhang = '1T'  # 1min todo
# data_nominal_res = 0.05  # Measurement every 0.05s
# file_generation_res = '6H'  # One file expected every x hour(s)
# files_how_many = False  # Can be int or False
# win_lagsearch = 1000  # Initial time window for lag search, number of records from/to
# num_iterations = 5
# hist_remove_fringe_bins = True
# hist_perc_threshold = 0.9  # Include bins to both sides of peak bin until x% of total counts is reached

# Directories for input/output
dir_input_ext = Path(
    r'A:\FLUXES\CH-DAV\ms - CH-DAV - non-CO2\tests_dynamic_lag\JOSS paper\1_FISP_output\splits')  # Can be False
dir_output_ext = Path(
    r'A:\FLUXES\CH-DAV\ms - CH-DAV - non-CO2\tests_dynamic_lag\JOSS paper\2_DLR_output')  # Can be False
dir_input = 'input'
dir_output = 'output'

# Source files
fnm_pattern = '2016*.csv'  # Filename pattern for file search, accepts regex
fnm_date_format = '%Y%m%d%H%M%S'  # Date format in filename
files_how_many = False  # Limit number of files to use
file_generation_res = '30T'  # One file generated every x mins
file_duration = '30T'  # Duration of one data file

# Lag search
lgs_segment_dur = '30T'  # Segment duration, e.g. calc lag for 10min segments
lgs_refsig = 'w_ms-1_rot_turb'  # Reference signal
lgs_lagsig = 'co2_ppb_qcl_turb'  # Lagged signal
lgs_hist_perc_thres = 0.9  # Cumulative percentage threshold in histogram of found lag times, expand bins around peak
lgs_hist_remove_fringe_bins = True  # Remove fringe bins in histogram of found lag times
lgs_winsize = 1000  # Window size +/-, given as number of records
lgs_num_iter = 5  # Number of iterations

lag_target = -100  # Set all files to target lag
normalize_lag_for_cols= ['co2_ppb_qcl_turb', 'n2o_ppb_qcl', 'co2_ppb_qcl', 'h2o_ppb_qcl', 'ch4_ppb_qcl']

# Data records
# Set to False if not in data, uses the available timestamp, otherwise
# a timestamp is created.
dat_recs_timestamp_format = '%Y-%m-%d %H:%M:%S.%f'  # Timestamp format for each row record
dat_recs_nominal_timeres = 0.05  # Nominal (expected) time resolution, one record every x seconds

del_previous_results = False
# del_previous_results = True  # Danger zone! True deletes all output in output folder

# # TEST DLR FILES (files w/ normalized lag times)
# dir_input_ext = Path(
#     r'A:\FLUXES\CH-DAV\ms - CH-DAV - non-CO2\tests_dynamic_lag\JOSS paper\2_DLR_output\3-1_____Normalized_Files')  # Can be False
# dir_output_ext = Path(
#     r'A:\FLUXES\CH-DAV\ms - CH-DAV - non-CO2\tests_dynamic_lag\JOSS paper\3_DLR_output_test')  # Can be False
# lgs_lagsig = 'co2_ppb_qcl_turb_DLR'  # Lagged signal
# fnm_date_format = '%Y%m%d%H%M%S_DLR'  # Date format in filename


dlr(dir_root=Path(os.path.dirname(os.path.abspath(__file__))),
    dir_input=dir_input,
    dir_output=dir_output,
    dir_input_ext=dir_input_ext,
    dir_output_ext=dir_output_ext,
    fnm_date_format=fnm_date_format,
    fnm_pattern=fnm_pattern,
    dat_recs_timestamp_format=dat_recs_timestamp_format,
    dat_recs_nominal_timeres=dat_recs_nominal_timeres,
    files_how_many=files_how_many,
    file_generation_res=file_generation_res,
    file_duration=file_duration,
    lgs_num_iter=lgs_num_iter,
    lgs_winsize=lgs_winsize,
    lgs_hist_remove_fringe_bins=lgs_hist_remove_fringe_bins,
    lgs_hist_perc_thres=lgs_hist_perc_thres,
    lgs_refsig=lgs_refsig,
    lgs_lagsig=lgs_lagsig,
    lgs_segment_dur=lgs_segment_dur,
    del_previous_results=del_previous_results,
    lag_target=lag_target,
    normalize_lag_for_cols=normalize_lag_for_cols)
