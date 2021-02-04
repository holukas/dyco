# Additional information about test data

## ORIGIN

	Output file from DYCO, produced in Phase 1, Step 2, iteration 1.
	* For information about the DYCO processing chain see here:
	https://github.com/holukas/dyco/wiki/Processing-Chain

	DYCO output folder: "1-3__input_files__time_lags_overview"
	* For information about output folders produced by DYCO see here:
	https://github.com/holukas/dyco/wiki/Results-Output-Folders

	Original raw data was recorded at 20 Hz at the forest site
	Davos in Switzerland.
	* For more information about Davos see here:
	https://www.swissfluxnet.ethz.ch/index.php/sites/ch-dav-davos/site-info-ch-dav/


## FILES

	"1_segments_found_lag_times_after_iteration-1.csv"
	
	The file gives an overview of lag search results for each of the
	raw data files.


## FILE FORMAT

	- one-row header in first row, showing variable name
	- 20 columns


## VARIABLES
	
	- column #1: original raw data date and iteration identifier
	- column #2: "file_date": timestamp, date info from original raw data file
	- column #3: "start": timestamp, start of respective file
	- column #4: "end": timestamp, end of respective file
	- column #5: "numvals_w_ms-1_rot_turb":
	  number of values for turbulent wind variable
	- column #6: "numvals_co2_ppb_qcl_turb"
	  number of values for turbulent CO2 mixing ratios (the reference variable)
	- column #7: "lagsearch_start"
	  start of time window for lag search for current iteration, given as number of records
	- column #8: "lagsearch_end"
	  end of time window for lag search for current iteration, given as number of records
	- column #9: "iteration": current iteration number
	- column #10: "PEAK-COVABSMAX_SHIFT":
	  given as number of records, shows by how much the reference variable needs to be shifted
	  in relation to the wind data to reach maximum absolute covariance
	- column #11: "PEAK-COVABSMAX_COV": covariance at "PEAK-COVABSMAX_SHIFT"
	- column #12: "PEAK-COVABSMAX_TIMESTAMP":
	  the same as "PEAK-COVABSMAX_SHIFT", but expressed as a timestamp instead of number of records
	- column #13: "PEAK-AUTO_SHIFT":
	  given as number of records, shows by how much the reference variable needs to be shifted
	  in relation to the wind data so that the found covariance is the peak of the automatically-
	  detected peak during lag search; most often the same as "PEAK-COVABSMAX_SHIFT"
	- column #14: "PEAK-AUTO_COV": covariance at "PEAK-AUTO_SHIFT"
	- column #15: "PEAK-AUTO_TIMESTAMP":
	  the same as "PEAK-AUTO_SHIFT", but expressed as a timestamp instead of number of records
	- column #16: "DEFAULT-LAG_SHIFT": (-left empty, not used in tests-)
	  given as number of records, shows by how much the reference variable needs to be shifted
	  in relation to the wind data to correct for the default (nominal) time lag
	- column #17: "DEFAULT-LAG_COV": (-left empty, not used in tests-)
	  covariance at "DEFAULT-LAG_SHIFT"
	- column #18: "DEFAULT-LAG_TIMESTAMP": (-left empty, not used in tests-)
	  the same as "DEFAULT-LAG_SHIFT", but expressed as a timestamp instead of number of records
	- column #19: "lagsearch_next_start":
	  start of time window for lag search for next iteration, given as number of records
	- column #20: "lagsearch_next_end"
	  end of time window for lag search for next iteration, given as number of records
