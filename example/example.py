import os
from pathlib import Path

from dlr import DynamicLagRemover as dlr

u_col = 'u_ms-1'
v_col = 'v_ms-1'
w_col = 'w_ms-1'
scalar_col = 'co2_ppb_qcl'


dir_input_ext = Path(r'A:\FLUXES\CH-DAV\ms - CH-DAV - non-CO2\tests_dynamic_lag\2016_2\converted_raw_data_csv')
dir_input = 'input'
dir_output = 'output'
file_pattern = '201610061300.csv'

file_date_format = '%Y%m%d%H%M'  # Date format in filename
file_generation_res = '6H'  # One file expected every x hour(s)

data_nominal_res = 0.05  # Measurement every 0.05s
data_segment_duration = '30T'  # 30min segments
data_segment_overhang = '1T'  # 1min todo

win_lagsearch = [-1000, 1000]  # Initial time window for lag search, number of records from/to
shift_stepsize = 10  # Shift by x number of records during lag search #todo variable based on window?
num_iterations = 10

hist_num_bins = 30  # Number of equal-width bins in the histogram
hist_remove_fringe_bins = False
hist_perc_threshold = 0.9  # Include bins to both sides of peak bin until x% of total counts is reached

# del_previous_results = False
del_previous_results = True

testing = False
# timewin_noise = [2600, 3000]  # todo number of records from/to
# calculate_noise = True  # todo

dlr(u_col=u_col,
    v_col=v_col,
    w_col=w_col,
    scalar_col=scalar_col,
    dir_root=Path(os.path.dirname(os.path.abspath(__file__))),
    dir_input=dir_input,
    dir_input_ext=dir_input_ext,
    dir_output=dir_output,
    file_date_format=file_date_format,
    file_generation_res=file_generation_res,
    data_nominal_res=data_nominal_res,
    data_segment_duration=data_segment_duration,
    data_segment_overhang=data_segment_overhang,
    win_lagsearch=win_lagsearch,
    testing=testing,
    num_iterations=num_iterations,
    hist_num_bins=hist_num_bins,
    hist_remove_fringe_bins=hist_remove_fringe_bins,
    del_previous_results=del_previous_results,
    shift_stepsize=shift_stepsize,
    hist_perc_threshold=hist_perc_threshold,
    file_pattern=file_pattern)
