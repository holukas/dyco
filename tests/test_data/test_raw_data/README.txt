# Additional information about test data

## ORIGIN

	The files contain 30-minute eddy covariance raw data recorded
	at 20 Hz at the forest site Davos in Switzerland.

	For more information about Davos see here:
	https://www.swissfluxnet.ethz.ch/index.php/sites/ch-dav-davos/site-info-ch-dav/


## FILES

	"20161020113000.csv"
	"20161020123000.csv"

	The filename gives information about the start of the respective
	raw data file, e.g. "20161020113000.csv" contains data between
	11:30h and 12:00h on 20 Oct 2016.


## FILE FORMAT

	- one-row header in first row, showing variable name and units
	- timestamp in first column
	  Due to the high measurement frequency of 20Hz, the timestamp
	  contains time information up to microseconds, with the format:
	  "2016-10-20 11:30:00.024999"		
	- recorded variables start in second column


## VARIABLES
	
	Variables in the files contain parameters typical for eddy covariance
	raw data, such as information about wind speeds, scalar mixing ratios
	and status data of the sensors.
	
	In the context of the DYCO tests, only a few are relevant, namely:
		- "w_ms-1_rot_turb": turbulent departures of the vertical wind
		- "co2_ppb_qcl_turb": turbulent departures of the CO2 mixing ratios
		- "co2_ppb_qcl": CO2 mixing ratios
