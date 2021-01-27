In this example, we install DYCO from pip and then use it in a small example script to find the time lag between two signals. We then remove the found time lag also from three other signals.

If you want to follow along with this example and run it on your computer, you can download the example data and example script [here](https://github.com/holukas/dyco/tree/main/example).

# Installation

You can install `DYCO` using pip.
`pip install dyco`

Alternatively, you can download the source code and create a Python conda environment to run the code in. The respective `environment.yml` is provided.  
Or, after downloading the source code, use the `requirements.txt` file to install all needed depencies via pip.

# Example Data

This example uses 20Hz raw data of turbulent wind and the greenhouse gases CO2 (carbon dioxide), N2O (nitrous oxide) and CH4 (methane). Data were measured in 2016 at the research station Davos in Switzerland ([more info](https://www.swissfluxnet.ethz.ch/index.php/sites/ch-dav-davos/site-info-ch-dav/))

We will use 10 raw data files with data from 24 Oct 2016, recorded between 11:00 and 15:30. Each of these files contains 30 minutes of raw data:   
`20161024100000.csv  20161024110000.csv  20161024120000.csv  20161024130000.csv  20161024140000.csv`  
`20161024103000.csv  20161024113000.csv  20161024123000.csv  20161024133000.csv  20161024143000.csv`

Note that in practice, considerably more than 10 files would be used with `DYCO`. Normally the script should be run on a dataset that consists of at least several days of data, but for the purpose of this example 10 files are sufficient. For the research site Davos, `DYCO` is used on multi-year datasets.

Each of these files contains more data than the variables we are interested in.

Data in the files look like this (20161024110000.csv):
* Header with variable names in 1st row
* Timestamp in 1st column
* Data starting in 2nd column

In this example, the imporant variables are: 
* wind (`w_ms-1_rot_turb`)
* CO2 (`co2_ppb_qcl_turb`)
* N2O (`n2o_ppb_q`
* `cl`)
* CH4 (`ch4_ppb_qcl`)

# Goal

The goal is to prepare the N2O and CH4 data in the raw data files for an easier and more accurate calculation of the respective ecosystem fluxes.

First we want to remove the time lag between the wind variable and the CO2 variable. Then, we will use this information to remove the time shift also from the other signals (N2O, CH4). Finding the lag between wind and CO2 is relatively easy because the CO2 exchange at the site is strong. In contrast, finding the lag between wind and N2O or CH4 is challenging due to low N2O and CH4 exchange.

# Running The Code

DYCO is run from the command line interface (CLI).

## Example with full arguments

`python L:\Dropbox\luhk_work\programming\DYCO_Dynamic_Lag_Compensation\dyco\dyco.py w_ms-1_rot_turb co2_ppb_qcl_turb n2o_ppb_qcl ch4_ppb_qcl -i L:\Dropbox\luhk_work\programming\DYCO_Dynamic_Lag_Compensation\example\input_data -o L:\Dropbox\luhk_work\programming\DYCO_Dynamic_Lag_Compensation\example\output -fnd %Y%m%d%H%M%S -fnp *.csv -flim 0 -fgr 30T -fdur 30T -dtf "%Y-%m-%d %H:%M:%S.%f" -dres 0.05 -lss 30T -lsw 1000 -lsi 3 -lsf 1 -lsp 0.9 -lt 0 -del 0`

## Usage

In the following, we will look at all parameter in the context of this example.

usage: dyco.py [-h] [-i INDIR] [-o OUTDIR] [-fnd FILENAMEDATEFORMAT]
               [-fnp FILENAMEPATTERN] [-flim LIMITNUMFILES] [-fgr FILEGENRES]
               [-fdur FILEDURATION] [-dtf DATATIMESTAMPFORMAT]
               [-dres DATANOMINALTIMERES] [-lss LSSEGMENTDURATION]
               [-lsw LSWINSIZE] [-lsi LSNUMITER] [-lsf {0,1}]
               [-lsp LSPERCTHRES] [-lt TARGETLAG] [-del {0,1}]
               var_reference var_lagged var_target [var_target ...]

BICO - Conversion of ETH binary files to ASCII

### positional arguments:

####  var_reference

Column name of the unlagged reference variable in the data files (one-row header). Lags are determined in relation to this signal.

-  *Example*: `w_ms-1_rot_turb` is the name of the column that contains turbulent wind data.

#### var_lagged

Column name of the lagged variable in the data files (one-row header). The time lag of this signal is determined in relation to the reference signal var_reference.

- *Example*: `co2_ppb_qcl_turb` is the name of the column that contains turbulent CO2 data.

####  var_target

Column name(s) of the target variable(s). Column names of the variables the lag that was found between `var_reference` and `var_lagged` should be applied to.

- *Example*: `n2o_ppb_qcl ch4_ppb_qcl` are the names of the two columns that contain the N2O and CH4 data, respectively. Data of these columns will be shifted in a way so that the lag in relation to `var_reference` is removed.

### optional arguments:

#### -h, --help

show this help message and exit

#### -i INDIR, --indir INDIR

Path to the source folder that contains the data files. (default: `None`)

- *Example*: `-i C:/bico/input` 

#### -o OUTDIR, --outdir OUTDIR

Path to output folder. Results are stored to this folder. For an overview of created results folders see here: [Results Output Folders](https://github.com/holukas/dyco/wiki/Results-Output-Folders). (default: `None`)

- *Example*: `-o C:/bico/output` 

#### -fnd FILENAMEDATEFORMAT, --filenamedateformat FILENAMEDATEFORMAT

Filename date format as datetime format strings. Is used to parse the date and time info from the filename of found files. Only files found with `FILENAMEPATTERN` will be parsed. The filename(s) of the files found in `INDIR` must contain datetime information. For datetime format codes see e.g. [here](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes). (default: `%Y%m%d%H%M%S`)

- *Example*: To parse the datetime info from data files named like *20161015123000.csv*: `-fnd %Y%m%d%H%M%S`. In this case there is a 4-digit year, 2-digit month, 2-digit day, 2-digit hours, 2-digit minutes and 2-digit seconds. 

#### -fnp FILENAMEPATTERN, --filenamepattern FILENAMEPATTERN

Filename pattern for raw data file search, accepts regex. (default: `*.csv*`)

- *Example*: To search for all files named like *20161024100000.csv*: `-fnp 2016*.csv`. Here, all files are csv files that start with `2016`.

#### -flim LIMITNUMFILES, --limitnumfiles LIMITNUMFILES

Limit the number of files used in processing. Must be 0 or a positive integer. If set to 0, all found files with `FILENAMEPATTERN`will be used. (default: `0`)

- *Example*s:
  - `-flim 0` to use all files that were found with `FILENAMEPATTERN`.
  - `-flim 5` to use the first five files that were found with `FILENAMEPATTERN`.

#### -fgr FILEGENRES, --filegenres FILEGENRES

File generation resolution. Frequency at which new files were generated, e.g. every 30 minutes one file was created.. This does not relate to the data records but to the file creation time. Must be given as *pandas DateOffset* string, for options see: [offset aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases) (default: `30T`)

- *Examples*:
     - `-fgr 30T` means a new file was generated every 30 minutes.
     - `-fgr 1H` means a new file was generated every hour.
     - `-fgr 6H` means a new file was generated every six hours.

#### -fdur FILEDURATION, --fileduration FILEDURATION

Duration of one data file. Must be given as *pandas DateOffset* string, for options see: [offset aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases) (default: `30T`)

- *Example*s:
  - `-fdur 30T` means each file contains 30 minutes of data.
  - `-fdur 10T` means each file contains 10 minutes of data.
  - `-fdur 1T` means each file contains 1 minute of data.
  - `-fdur 60S` means each file contains 60 seconds of data.
  - `-fdur 1H` means each file contains 1 hour of data.

#### -dtf DATATIMESTAMPFORMAT, --datatimestampformat DATATIMESTAMPFORMAT

Timestamp format for each row record in the data files. If set to `None`, a timestamp will be created for each row. For datetime format codes see e.g. [here](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes). (default: `%Y-%m-%d %H:%M:%S.%f`)

- *Examples*:
  - `-dtf %Y-%m-%d %H:%M:%S.%f ` to correctly parse *2016-10-24 10:00:00.024999*
  - `-dtf %Y/%m/%d %H%M%S ` to correctly parse *2016/10/24 100000*
  - `-dtf %Y%m%d%H%M ` to correctly parse *201610241000*

#### -dres DATANOMINALTIMERES, --datanominaltimeres DATANOMINALTIMERES

Nominal (expected) time resolution of data records in the files, given as one record every x seconds. Example for files recorded at 20Hz: 0.05 (default: `0.05`)

- *Example:* `-dres 0.05` for files recorded at 20Hz, i.e. one data point every 0.05 seconds.

#### -lss LSSEGMENTDURATION, --lssegmentduration LSSEGMENTDURATION

Segment duration for lag determination. Can be the same as or shorter than `FILEDURATION`. If it is the same as `FILEDURATION`, the lag time for the complete file is calculated from all file data. If it is shorter than `FILEDURATION`, then the file data is split into segments and the lag time is calculated for each segment separately.  Must be given as *pandas DateOffset* string, for options see: [offset aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases) (default: same as `FILEDURATION`)

-  *Example*:
  - `-lss 30T` to calculate lag times for 30-minute segments.
  - `-lss 10T` to calculate lag times for 10-minute segments. If the `FILEDURATION` was set to `30T`, then three time lags are calculated for the file, i.e. the 30-minute file is split into three 10-minute segments and the time lag is calculated for each of the three segments.
  - `-lss 60S` to calculate lag times for 60-second segments. If the `FILEDURATION` was set to `30T`, then 30 time lags are calculated for the file, i.e. the 30-minute file is split into 30 60-second segments and the time lag is calculated for each of the 30 segments.

#### -lsw LSWINSIZE, --lswinsize LSWINSIZE

Initial lag search window size. Initial size of the time window in which the lag is searched given as number of records. Generally, it is better to set this parameter to a relatively high number, e.g. `1000`, because the time window is automatically narrowed down after this initial time window. Used in Phase 1 iteration 1 only, since Phase 2 uses results from Phase 1, and Phase 3 uses results from Phase 2. See [here](https://github.com/holukas/dyco/wiki/Processing-Chain) for an overview of the different Phases in the processing chain. (default: `1000`)

- *Example*: `-lsw 1000` corresponds to a lag search window of +/- 1000 records. For data recorded at 20 Hz, this translated to a time window of +/- 50 seconds.

#### -lsi LSNUMITER, --lsnumiter LSNUMITER

Number of lag search iterations. Must be a positive integer. Used in Phase 1 and 2 (Phase 3 is only one iteration). See [here](https://github.com/holukas/dyco/wiki/Processing-Chain) for an overview of the different Phases in the processing chain. Before each iteration, the time window for the lag search is narrowed down, taking into account results from the previous iteration. Exception is the first iteration in Phase 1 for which the time window as given in `LSWINSIZE` is used. Generally, it is advisable to keep the number of iterations as low as possible. (default: `3`)

- *Example*: `-lsi 3` means three iterations in Phase 1 and Phase 2. During Phase 1 iteration 1 (P1i1), the time lag is searched in a time window with the size as defined in `LSWINSIZE`. Then, the lag window is narrowed down using results from P1i1. This adjusted time window is then used in P1i2 to again search lag times for the same data. Likewise, P1i3 uses an adjusted search window based on results from i2. During Phase 2, the steps are repeated but the initial time window size for lag search is automatically set based on results from P1i3. After the Phase 2 processing (three iterations: P2i1, P2i2, P2i3) a narrow time window is automatically determined based on results from P2i3 and used for the final lag search in Phase 3. In this example three iterations were found to produce good results for this dataset.

#### -lsf {0,1}, --lsremovefringebins {0,1}

Remove fringe bins in histogram of found lag times. Must be `0` or `1`. During the lag search iterations, the time window in which the lag is searched is continuously narrowed down and gets smaller and smaller. To narrow down the time window, the histogram of found lag times is checked for the peak distribution. However, in case of low signal-to-noise ratios the lag search yields less clear results and found lag times tend to accumulate in the fringe bins of the histogram, i.e. in the very first and last bins. This accumulation sometimes creates non-desirable peak bins in the histogram. Therefore, the first and last bin of the histogram can be removed before the lag search time window is narrowed down. If `1`, the first and last bins of the histogram are removed before the time window for lag search is adjusted. (default: `1`)

- *Example*: 
  - `-lsf 1` removes the fringe bins in the histogram of found lag times.
  - `-lsf 0` does not remove the fringe bins in the histogram of found lag times. 

#### -lsp LSPERCTHRES, --lspercthres LSPERCTHRES

Cumulative percentage threshold in histogram of found lag times. Must be float between `0.1` and `1`. Is automatically set to `1` if > 1 or set to `0.1` if < 0.1. The time window for lag search during each iteration (i) is narrowed down based on the histogram of all found lag times from the previous iteration (i-1).  During each iteration and after lag times were determined for all files and segments, a histogram of found lag times is created to identify the histogram bin in which most lag times (most counts) were found (peak bin). To narrow down the time window for lag search during the next iteration (i+1), the bins around the peak bin are included until a certain percentage of the total values is reached over the expanded bin range, centered on the peak bin. (default: `0.9`)

- *Example*: `-lsp 0.9`: include all bins to each site of the peak bin until 90% of the total found lag times (counts) are included. The time window for the lag search during the next iteration (i+1) is determined by checking the left side (start) of the first included bin and the right side (end) of the last included bin.

#### -lt TARGETLAG, --targetlag TARGETLAG

The target lag given in records to which lag times of all files are normalized. A negative number means that `var_lagged` lags x records behind `var_reference`. The raw data files will be processed and new raw data files, where `var_target` variables were shifted to `target_lag`, are produced. (default: `0`)

- *Example*: `-lt 0`: Time lags between `var_reference` and `var_target` will be (close to) zero after processing.  

#### -del {0,1}, --delprevresults {0,1}

If set to `1`, delete all previous results in `INDIR`. If set to `0`, search for previously calculated results in `INDIR` and continue. (default: `0`)

