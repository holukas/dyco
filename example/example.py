"""
Example settings

DYCO is run from the command line interface (CLI).

The example code below calls the CLI from within Python.

For an overview of arguments see here:
https://github.com/holukas/dyco/wiki/Usage
"""

import os

cli = 'python L:\Dropbox\luhk_work\programming\DYCO_Dynamic_Lag_Compensation\dyco\dyco.py ' \
      'w_ms-1_rot_turb co2_ppb_qcl_turb ' \
      'co2_ppb_qcl_turb n2o_ppb_qcl ch4_ppb_qcl ' \
      '-i L:\Dropbox\luhk_work\programming\DYCO_Dynamic_Lag_Compensation\example\input_data ' \
      '-o L:\Dropbox\luhk_work\programming\DYCO_Dynamic_Lag_Compensation\example\output ' \
      '-fnd %Y%m%d%H%M%S ' \
      '-fnp *.csv ' \
      '-flim 0 ' \
      '-fgr 30T ' \
      '-fdur 30T ' \
      '-dtf "%Y-%m-%d %H:%M:%S.%f" ' \
      '-dres 0.05 ' \
      '-lss 30T ' \
      '-lsw 1000 ' \
      '-lsi 3 ' \
      '-lsf 1 ' \
      '-lsp 0.9 ' \
      '-lt 0 ' \
      '-del 0'
print(cli)
os.system(cli)
