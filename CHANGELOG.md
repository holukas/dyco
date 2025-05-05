# CHANGELOG

## v2.0.0 | XX XXX 2025

`dyco` assists in removing time lags from time series data. Version `2.0.0` changes the previous workflow.

`dyco` identifies time delays between two time series and applies these findings to other related variables.





- todo limit deviation from median in lookup table?

At one of our research sites both of these issues occured at the same time: very low fluxes in combination with
drifting/inconsistent time lags.

The main motivation for new udpates to `dyco` is the observation of drifting time lags at some of our stations.
These lag drifts are rare, but when they occur they hinder our ability to calculate reliable ecosystem fluxes
using the eddy covariance (EC) method.

Another challenging aspect when calculating fluxes with the EC method is the occurrence of *very low (close to zero)
fluxes*. These fluxes are characterized by a very low signal-to-noise ratio, which in turn makes it challenging to find
a "maximum covariance", leading to a **mirroring** effect (see Fig. 8 in Langford et al., 2005). `dyco` can assist
in setting an adequate default time lag and a lag window as narrow as possible.

This update also implements [diive](https://github.com/holukas/diive) as a required dependency. The advantage
of this implementation is that existing (and better tested) code in `diive` does not have to be duplicated for `dyco`
(although the copy-paste approach has its merits), the drawback is that yes there is another dependency. I will try
not to break things.

### Usage example

### Changes

- The minimum time window for lag search is now min. 20 records, which corresponds to +/- 0.5s for 20 Hz
  data. If the automatically detected window is smaller than 20 records it is automatically expanded.
  For example:
    - if the lag is searched between -8 and -3 records: `[-8, -3]` is expanded incrementally until the range between
      the two values is >= 20, `[-16, 5]`. Since the increase is always done on both sides of the search window,
      the resulting range in this example is 21 records.
    - Another example: `[-5, 5]` is expanded to `[-10, 10]`
      (`lag.AdjustLagsearchWindow.adjust_lgs_winsize`)
- When creating the look-up table for daily median lags, missing median values are now filled with the
  rolling median in a 5-day window, centered around the missing value. (`analyze.AnalyzeLags.make_lut_agg`)
- Added `diive` library to dependencies
- Several functions are now handled
  by `diive`: `calc_true_resolution`, `create_timestamp`, `search_files`, `FileDetector`, `MaxCovariance`
- Plots are now explicitely closed after export to avoid memory issues
- Parquet files can now be used for data input. If found files have the extension `.parquet`, the correct function to
  read the file is used.

### References

- Langford, B., Acton, W., Ammann, C., Valach, A. & Nemitz, E. Eddy-covariance data with low signal-to-noise ratio:
  time-lag determination, uncertainties and limit of detection. Atmos. Meas. Tech. 8, 4197–4213 (2015).

## v1.2.0 | 6 Mar 2024

- Refactored code to work with newest package versions
- Several small bugfixes
- Now using `poetry` for dependency management
- Now using Python `3.9.18`
- All dependencies were updated to their newest possible versions
- Added example for using the class `DynamicLagCompensation` to run `dyco` directly from
  code (`example.example_kwargs.example`)

## v1.1.2 | 16 Jun 2021

### Release version for publication in JOSS

- JOSS: https://joss.theoj.org/
- DYCO open review: openjournals/joss-reviews#2575
