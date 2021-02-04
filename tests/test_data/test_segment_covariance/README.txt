# Additional information about test data

## ORIGIN

	Output file from DYCO, produced in Phase 1, Step 2, iteration 1.
	* For information about the DYCO processing chain see here:
	https://github.com/holukas/dyco/wiki/Processing-Chain

	DYCO output folder: "1-1__input_files__covariances"
	* For information about output folders produced by DYCO see here:
	https://github.com/holukas/dyco/wiki/Results-Output-Folders
		
	Original raw data was recorded at 20 Hz at the forest site
	Davos in Switzerland.
	* For more information about Davos see here:
	https://www.swissfluxnet.ethz.ch/index.php/sites/ch-dav-davos/site-info-ch-dav/


## FILES

	"20161020113000_iter1_segment_covariance_iteration-1.csv"
	
	The file gives an overview of the found covariances between
	turbulent wind and turbulent CO2 data at different time steps
	for the raw data file "20161020113000.csv".


## FILE FORMAT

	- one-row header in first row, showing variable name
	- 8 columns	


## VARIABLES
	
	- column #1: "record number"
	- column #2: "index":
	  timestamp in relation to shift
	- column #3: "segment_name":
	  date information from raw data file and iteration identifier
	- column #4: "shift":
	  given as number of records, shows by how much the time series of
	  the reference variable (turbulent CO2 mixing ratios) was shifted
	  in relation to the turbulent wind
	- column #5: "cov":
	  covariance between the reference variable and the wind when applying
	  the respective "shift"
	- column #6: "cov_abs":
	  absolute value of "cov"
	- column #7: "flag_peak_max_cov_abs":
	  denotes whether "cov_abs" found for the respective "shift" was the
	  maximum absolute covariance for this file
	- column #8: "flag_peak_auto":
	  denotes whether the respective "cov" was part of the automatically-
	  detected most-prominent peak during covariance search
