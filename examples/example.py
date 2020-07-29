"""
Example settings
"""

from pathlib import Path

import dyco

# data_segment_overhang = '1T'  # 1min todo

# Example with full arguments
dyco.run(
    var_reference='w_ms-1_rot_turb',  # Reference signal
    var_lagged='co2_ppb_qcl_turb',  # Lagged signal
    var_target=['co2_ppb_qcl_turb', 'n2o_ppb_qcl', 'co2_ppb_qcl', 'ch4_ppb_qcl'],  # Target columns
    indir=Path(r'A:\FLUXES\CH-DAV\ms_CH-DAV_non-CO2\tests_dynamic_lag\JOSS_paper\1_FISP_output\splits'),
    outdir=Path(r'A:\FLUXES\CH-DAV\ms_CH-DAV_non-CO2\tests_dynamic_lag\JOSS_paper\2_DYCO_output'),
    fnm_date_format='%Y%m%d%H%M%S',  # Date format in filename
    fnm_pattern='2016*.csv',  # Filename pattern for file search, accepts regex
    dat_recs_timestamp_format='%Y-%m-%d %H:%M:%S.%f',  # Timestamp format for each row record, will be created if False
    dat_recs_nominal_timeres=0.05,  # Nominal (expected) time resolution, one record every x seconds
    files_how_many=False,  # Limit number of files to use
    file_generation_res='30T',  # One file generated every x mins
    file_duration='30T',  # Duration of one data file
    lgs_num_iter=3,  # Number of lag search iterations in Phase 1 and 2 (Phase 3 is only 1 iteration)
    lgs_winsize=1000,  # Initial window size +/-, given as number of records, used in iteration 1
    lgs_hist_remove_fringe_bins=True,  # Remove fringe bins in histogram of found lag times for winsize adjustments
    lgs_hist_perc_thres=0.9,  # Cumulative % threshold in histogram of found lag times, expand bins around peak
    lgs_segment_dur='30T',  # Segment duration, e.g. calc lag for 10min segments
    del_previous_results=False,  # Danger zone! True deletes all previous output in outdir
    target_lag=0)  # Normalize lag time for *var_target* to target_lag, in records
