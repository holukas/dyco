"""
Example settings

DYCO can be run using keyword arguments.

For an overview of arguments see here:
https://github.com/holukas/dyco/wiki/Usage
"""

import os

from dyco.dyco import DynamicLagCompensation

cli = r'python "F:\Sync\luhk_work\20 - CODING\27 - VARIOUS\dyco\dyco\dyco.py" ' \
      r'w_ms-1_rot_turb co2_ppb_qcl_turb ' \
      r'co2_ppb_qcl_turb n2o_ppb_qcl ch4_ppb_qcl ' \
      r'-i F:\TMP\das\input_data ' \
      r'-o F:\TMP\das ' \
      r'-fnd %Y%m%d%H%M%S ' \
      r'-fnp *.csv ' \
      r'-flim 0 ' \
      r'-fgr 30T ' \
      r'-fdur 30T ' \
      r'-dtf "%Y-%m-%d %H:%M:%S.%f" ' \
      r'-dres 0.05 ' \
      r'-lss 30T ' \
      r'-lsw 1000 ' \
      r'-lsi 3 ' \
      r'-lsf 1 ' \
      r'-lsp 0.9 ' \
      r'-lt 0 ' \
      r'-del 0'
print(cli)
os.system(cli)


def main(args):
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
                           file_generation_res='30T',
                           file_duration='30T',
                           dat_recs_timestamp_format="%Y-%m-%d %H:%M:%S.%f",
                           dat_recs_nominal_timeres=0.05,
                           lgs_segment_dur='30T',
                           lgs_winsize=1000,
                           lgs_num_iter=3,
                           lgs_hist_remove_fringe_bins=True,
                           lgs_hist_perc_thres=0.9,
                           target_lag=0,
                           del_previous_results=True)


if __name__ == '__main__':
    main()
