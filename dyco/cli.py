"""
    DYCO Dynamic Lag Compensation
    Copyright (C) 2020-2025 Lukas Hörtnagl

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import argparse
from pathlib import Path


def validate_args(args):
    """Check validity of optional args"""
    if args.limitnumfiles < 0:
        raise argparse.ArgumentTypeError("LIMITNUMFILES must be 0 or a positive integer.")
    if args.lsnumiter < 1:
        raise argparse.ArgumentTypeError("LSNUMITER must be > 1.")
    if (args.lspercthres < 0.1) | (args.lspercthres > 1):
        raise argparse.ArgumentTypeError("LSPERCTHRES must be between 0.1 and 1.")
    if args.lssegmentduration > args.fileduration:  # todo
        raise argparse.ArgumentTypeError("LSSEGMENTDURATION must be shorter or equal to FILEDURATION.")
    if not args.lssegmentduration:
        # If not specified, then lag times are determined using all of the file data
        args.lssegmentduration = args.fileduration
    if args.lsnumiter <= 0:
        raise argparse.ArgumentTypeError("LSNUMITER must be a positive integer.")
    args.lsremovefringebins = True if args.lsremovefringebins == 1 else False  # Translate settings to bool
    args.delprevresults = True if args.delprevresults == 1 else False  # Translate settings to bool
    return args


def get_args():
    """Get args from CLI input"""
    parser = argparse.ArgumentParser(description="dyco - dynamic lag compensation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Positional args
    parser.add_argument('var_reference', type=str,
                        help="Column name of the unlagged reference variable in the data files (one-row header). "
                             "Lags are determined in relation to this signal.")
    parser.add_argument('var_lagged', type=str,
                        help="Column name of the lagged variable in the data files (one-row header). "
                             "The time lag of this signal is determined in relation to the reference "
                             "signal var_reference.")
    parser.add_argument('var_target', nargs='+',
                        help="Column name(s) of the target variable(s). "
                             "Column names of the variables the lag that was found between "
                             "var_reference and var_lagged should be applied to. "
                             "Example: var1 var2 var3")

    # Optional args
    parser.add_argument('-i', '--indir', type=Path,
                        help="Path to the source folder that contains the data files, e.g. 'C:/dyco/input'")
    parser.add_argument('-o', '--outdir', type=Path,
                        help="Path to output folder, e.g. C:/bico/output")
    parser.add_argument('-fnd', '--filenamedateformat', type=str, default='%Y%m%d%H%M%S',
                        help="Filename date format as datetime format strings. Is used to parse the date and "
                             "time info from the filename of found files. The filename(s) of the files found in "
                             "INDIR must contain datetime information. Example for data files named like "
                             "20161015123000.csv: %%Y%%m%%d%%H%%M%%S")
    parser.add_argument('-fnp', '--filenamepattern', type=str, default='*.csv',
                        help="Filename pattern for raw data file search, e.g. *.csv")
    parser.add_argument('-flim', '--limitnumfiles', type=int, default=0,
                        help="Defines how many of the found files should be used. Must be 0 or a positive "
                             "integer. If set to 0, all found files will be used. ")
    parser.add_argument('-fgr', '--filegenres', type=str, default='30T',
                        help="File generation resolution. Example for data files that were generated "
                             "every 30 minutes: 30min")
    parser.add_argument('-fdur', '--fileduration', type=str, default='30T',
                        help="Duration of one data file. Example for data files containing 30 minutes "
                             "of data: 30T")
    parser.add_argument('-dtf', '--datatimestampformat', type=str, default='%Y-%m-%d %H:%M:%S.%f',
                        help="Timestamp format for each row record in the data files. Example for "
                             "high-resolution timestamps like 2016-10-24 10:00:00.024999: "
                             "%%Y-%%m-%%d %%H:%%M:%%S.%%f")
    parser.add_argument('-dres', '--datanominaltimeres', type=float, default=0.05,
                        help="Nominal (expected) time resolution of data records in the files, given as "
                             "one record every x seconds. Example for files recorded at 20Hz: 0.05")
    parser.add_argument('-lss', '--lssegmentduration', type=str, default='30T',
                        help="Segment duration for lag determination. Can be the same as or shorter "
                             "than FILEDURATION.")
    parser.add_argument('-lsw', '--lswinsize', type=int, default=1000,
                        help="Initial size of the time window in which the lag is searched given as "
                             "number of records.")
    parser.add_argument('-lsi', '--lsnumiter', type=int, default=3,
                        help="Number of lag search iterations in Phase 1 and Phase 2. Must be larger than 0.")
    parser.add_argument('-lsf', '--lsremovefringebins', type=int, choices=[0, 1], default=1,
                        help="Remove fringe bins in histogram of found lag times. "
                             "Set to 1 if fringe bins should be removed.")
    parser.add_argument('-lsp', '--lspercthres', type=float, default=0.9,
                        help="Cumulative percentage threshold in histogram of found lag times.")
    parser.add_argument('-lt', '--targetlag', type=int, default=0,
                        help="The target lag given in records to which lag times of all variables "
                             "in var_target are normalized.")
    parser.add_argument('-del', '--delprevresults', type=int, choices=[0, 1], default=0,
                        help="If set to 1, delete all previous results in INDIR. "
                             "If set to 0, search for previously calculated results in "
                             "INDIR and continue.")

    args = parser.parse_args()
    return args
