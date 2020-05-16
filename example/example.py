import os
from pathlib import Path

from dlr import DynamicLagRemover as dlr

u_col = 'u_ms-1'
v_col = 'v_ms-1'
w_col = 'w_ms-1'
scalar_col = 'co2_ppb_qcl'

dir_input = 'input'
dir_output = 'output'

file_date_format = '%Y%m%d%H%M'  # Date format in filename
file_generation_res = '1H'  # One file expected every x hour(s)

data_res = 0.05  # Measurement every 0.05s
data_segment_duration = '15T'  # 15min segments
data_segment_overhang = '1T'  # 1min todo
data_timewin_lag = [-300, 300]  # number of records from/to, time window for lag search

testing = False
# timewin_noise = [2600, 3000]  # todo number of records from/to
# calculate_noise = True  # todo

dlr(u_col=u_col,
    v_col=v_col,
    w_col=w_col,
    scalar_col=scalar_col,
    dir_root=Path(os.path.dirname(os.path.abspath(__file__))),
    dir_input=dir_input,
    dir_output=dir_output,
    file_date_format=file_date_format,
    file_generation_res=file_generation_res,
    data_res=data_res,
    data_segment_duration=data_segment_duration,
    data_segment_overhang=data_segment_overhang,
    data_timewin_lag=data_timewin_lag,
    testing=testing)
