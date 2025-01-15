"""
Example settings

DYCO can be run using keyword arguments.

For an overview of arguments see here:
https://github.com/holukas/dyco/wiki/Usage
"""

from dyco.dyco import Dyco


def example():
    """Main function that is called with the given args."""

    # Settings
    OUTDIR = r'F:\TMP\das_filesplitter'
    SEARCHDIRS = [r'L:\Sync\luhk_work\CURRENT\testdata_dyco\0-raw_data_ascii']
    PATTERN = 'CH-DAS_*.csv.gz'
    FILEDATEFORMAT = 'CH-DAS_%Y%m%d%H%M.csv.gz'
    FILE_GENERATION_RES = '6h'
    DATA_NOMINAL_RES = 0.05
    FILES_HOW_MANY = 3
    FILETYPE = 'ETH-SONICREAD-BICO-CSVGZ-20HZ'
    DATA_SPLIT_DURATION = '30min'
    DATA_SPLIT_OUTFILE_PREFIX = 'CH-DAS_'
    DATA_SPLIT_OUTFILE_SUFFIX = '_30MIN-SPLIT'
    ROTATION = True
    U = 'U_[R350-B]'
    V = 'V_[R350-B]'
    W = 'W_[R350-B]'
    # C = 'CO2_DRY_[IRGA72-A]'
    C = 'CH4_DRY_[QCL-C2]'
    W_TURB = f"{W}_TURB"
    C_TURB = f"{C}_TURB"

    # from diive.core.io.filesplitter import FileSplitterMulti
    # fsm = FileSplitterMulti(
    #     outdir=OUTDIR,
    #     searchdirs=SEARCHDIRS,
    #     pattern=PATTERN,
    #     file_date_format=FILEDATEFORMAT,
    #     file_generation_res=FILE_GENERATION_RES,
    #     data_res=DATA_NOMINAL_RES,
    #     files_how_many=FILES_HOW_MANY,
    #     filetype=FILETYPE,
    #     data_split_duration=DATA_SPLIT_DURATION,
    #     data_split_outfile_prefix=DATA_SPLIT_OUTFILE_PREFIX,
    #     data_split_outfile_suffix=DATA_SPLIT_OUTFILE_SUFFIX,
    #     rotation=ROTATION,
    #     u=U,
    #     v=V,
    #     w=W,
    #     c=C,
    #     compress_splits=True
    # )
    # fsm.run()

    Dyco(var_reference=W_TURB,
         var_lagged=C_TURB,
         var_target=[C_TURB],
         # var_target=[C_TURB, 'n2o_ppb_qcl', 'ch4_ppb_qcl'],
         indir=r'F:\CURRENT\DAS\2-filtered_CH4_ROT_TRIM_1-4',
         outdir=r'F:\CURRENT\DAS\3-dyco',
         filename_date_format='CH-DAS_%Y%m%d%H%M%S_30MIN-SPLIT_ROT_TRIM.csv',
         filename_pattern='CH-DAS_*_30MIN-SPLIT_ROT_TRIM.csv',
         files_how_many=None,
         file_generation_res='30min',
         file_duration='30min',
         data_timestamp_format="%Y-%m-%d %H:%M:%S.%f",
         data_nominal_timeres=0.05,
         lag_segment_dur=DATA_SPLIT_DURATION,
         lag_winsize=1000,
         lag_n_iter=3,
         remove_histogram_fringe_bins=True,
         histogram_percentage_threshold=0.7,
         target_lag=0,
         del_previous_results=True)


if __name__ == '__main__':
    example()
