import os
from pathlib import Path

# from dyla.main import DynamicLagRemover as dyla
import dyla

# data_segment_overhang = '1T'  # 1min todo

# # TEST DLR FILES (files w/ normalized lag times)
# dir_input_ext = Path(
#     r'A:\FLUXES\CH-DAV\ms - CH-DAV - non-CO2\tests_dynamic_lag\JOSS paper\2_DLR_output\3-1_____Normalized_Files')  # Can be False
# dir_output_ext = Path(
#     r'A:\FLUXES\CH-DAV\ms - CH-DAV - non-CO2\tests_dynamic_lag\JOSS paper\3_DLR_output_test')  # Can be False
# lgs_lagsig = 'co2_ppb_qcl_turb_DLR'  # Lagged signal
# fnm_date_format = '%Y%m%d%H%M%S_DLR'  # Date format in filename

# Example with full arguments
dyla.DynamicLagRemover(
    indir=Path(r'A:\FLUXES\CH-DAV\ms - CH-DAV - non-CO2\tests_dynamic_lag\JOSS paper\1_FISP_output\splits'),
    outdir=Path(r'A:\FLUXES\CH-DAV\ms - CH-DAV - non-CO2\tests_dynamic_lag\JOSS paper\2_DYLA_output'),
    fnm_date_format='%Y%m%d%H%M%S',  # Date format in filename
    fnm_pattern='2016*.csv',  # Filename pattern for file search, accepts regex
    dat_recs_timestamp_format='%Y-%m-%d %H:%M:%S.%f',  # Timestamp format for each row record, will be created if False
    dat_recs_nominal_timeres=0.05,  # Nominal (expected) time resolution, one record every x seconds
    files_how_many=False,  # Limit number of files to use
    file_generation_res='30T',  # One file generated every x mins
    file_duration='30T',  # Duration of one data file
    lgs_num_iter=5,  # Number of lag search iterations
    lgs_winsize=1000,  # Window size +/-, given as number of records
    lgs_hist_remove_fringe_bins=True,  # Remove fringe bins in histogram of found lag times
    lgs_hist_perc_thres=0.9,  # Cumulative % threshold in histogram of found lag times, expand bins around peak
    lgs_refsig='w_ms-1_rot_turb',  # Reference signal
    lgs_lagsig='co2_ppb_qcl_turb',  # Lagged signal
    lgs_segment_dur='30T',  # Segment duration, e.g. calc lag for 10min segments
    del_previous_results=False,  # Danger zone! True deletes all previous output in outdir
    target_lag=-100,  # Normalize lag time for target_cols to target_lag, in records
    target_cols=['co2_ppb_qcl_turb', 'n2o_ppb_qcl', 'co2_ppb_qcl', 'h2o_ppb_qcl', 'ch4_ppb_qcl'])  # Target columns
