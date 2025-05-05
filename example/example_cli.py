"""
Example settings

dyco can be run from the command line interface (CLI).

The example code below calls the CLI from within Python.

"""

import os

cli = r'python L:\Dropbox\luhk_work\programming\DYCO_Dynamic_Lag_Compensation\dyco\dyco.py ' \
      r'w_ms-1_rot_turb co2_ppb_qcl_turb ' \
      r'co2_ppb_qcl_turb n2o_ppb_qcl ch4_ppb_qcl ' \
      r'-i A:\FLUXES\CH-DAV\0___ms_CH-DAV_non-CO2\tests_dynamic_lag\JOSS_paper\1_FISP_output\splits ' \
      r'-o A:\FLUXES\CH-DAV\0___ms_CH-DAV_non-CO2\tests_dynamic_lag\JOSS_paper\2_DYCO_output_2_noTimestampInPhase3 ' \
      r'-fnd %Y%m%d%H%M%S ' \
      r'-fnp *.csv ' \
      r'-flim 0 ' \
      r'-fgr 30T ' \
      r'-fdur 30T ' \
      r'-dtf "%Y-%m-%d %H:%M:%S.%f" ' \
      r'-dres 0.05 ' \
      r'-lss 30min ' \
      r'-lsw 1000 ' \
      r'-lsi 3 ' \
      r'-lsf 1 ' \
      r'-lsp 0.9 ' \
      r'-lt 0 ' \
      r'-del 0'
print(cli)
os.system(cli)
