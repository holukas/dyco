"""
Example settings

DYCO can be run using keyword arguments.

For an overview of arguments see here:
https://github.com/holukas/dyco/wiki/Usage
"""

from dyco.dyco import DynamicLagCompensation


def example():
    """Main function that is called with the given args when the script
     is executed from the command line."""
    DynamicLagCompensation(var_reference='w_ms-1_rot_turb',
                           var_lagged='co2_ppb_qcl_turb',
                           var_target=['co2_ppb_qcl_turb', 'n2o_ppb_qcl', 'ch4_ppb_qcl'],
                           indir=r'F:\TMP\das\input_data',
                           outdir=r'F:\TMP\das',
                           fnm_date_format='%Y%m%d%H%M%S',
                           fnm_pattern='*.csv',
                           files_how_many=0,
                           file_generation_res='30min',
                           file_duration='30min',
                           dat_recs_timestamp_format="%Y-%m-%d %H:%M:%S.%f",
                           dat_recs_nominal_timeres=0.05,
                           lgs_segment_dur='30min',
                           lgs_winsize=1000,
                           # lgs_winsize=400,
                           # lgs_num_iter=1,
                           lgs_num_iter=3,
                           lgs_hist_remove_fringe_bins=True,
                           lgs_hist_perc_thres=0.9,
                           target_lag=0,
                           del_previous_results=True)


if __name__ == '__main__':
    example()
