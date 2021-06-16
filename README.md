![Normalization example](images/logo_dyco1_256px.png)

[![DOI](status.svg)](https://doi.org/10.21105/joss.02575)

[![DOI](https://zenodo.org/badge/311300577.svg)](https://zenodo.org/badge/latestdoi/311300577)



# DYCO - Dynamic Lag Compensation

The lag detection between the turbulent departures of measured wind and the scalar of interest is a central step in the calculation of eddy covariance (EC) ecosystem fluxes. In case the covariance maximization fails to detect a clear peak in the covariance function between the wind and the scalar, current flux calculation software can apply a constant default (nominal) time lag to the respective scalar. However, both the detection of clear covariance peaks in a pre-defined time window and the definition of a reliable default time lag is challenging for compounds which are often characterized by low SNR, such as N<sub>2</sub>O. In addition, the application of one static default time lag may produce inaccurate results in case systematic time shifts are present in the raw data.

 `DYCO` is meant to assist current flux processing software in the calculation of fluxes for compounds with low SNR. In the context of current flux processing schemes, the unique features offered as part of the `DYCO` package include:

- (i) the dynamic application of progressively smaller time windows during lag search for a *reference* compound (e.g. CO<sub>2</sub>),
- (ii) the calculation of default time lags on a daily scale for the *reference* compound,
- (iii) the application of daily default *reference* time lags to one or more *target* compounds (e.g. N<sub>2</sub>O)
- (iv) the dynamic normalization of time lags across raw data files,
- (v) the automatic correction of systematic time shifts in *target* raw data time series, e.g. due to failed synchronization of instrument clocks, and
- (vi) the application of instantaneous *reference* time lags, calculated from lag-normalized files, to one or more *target* compounds.

As `DYCO` aims to complement current flux processing schemes, final lag-removed files are produced that can be directly used in current flux calculation software.



## Scientific background

In ecosystem research, the EC method is widely used to quantify the biosphere-atmosphere exchange of greenhouse gases (GHGs) and energy (Aubinet et al., 2012; Baldocchi et al., 1988). The raw ecosystem flux (i.e. net exchange) is calculated by the covariance between the turbulent vertical wind component measured by a sonic anemometer and the entity of interest, e.g. CO<sub>2</sub>, measured by a gas analyzer. Due to the application of two different instruments, wind and gas are not recorded at exactly the same time, resulting in a time lag between the two time series. For the calculation of ecosystem fluxes this time delay has to be quantified and corrected for, otherwise fluxes are systematically biased. Time lags for each averaging interval can be estimated by finding the maximum absolute covariance between the two turbulent time series at different time steps in a pre-defined time window of physically possible time-lags  (e.g., McMillen, 1988; Moncrieff et al., 1997). Lag detection works well when processing fluxes for compounds with high signal-to-noise ratio (SNR), which is typically the case for e.g. CO<sub>2</sub>. In contrast, for compounds with low SNR (e.g., N<sub>2</sub>O, CH<sub>4</sub>) the cross-covariance function with the turbulent wind component yields noisier results and calculated fluxes are biased towards larger absolute flux values (Langford et al., 2015), which in turn renders the accurate calculation of yearly ecosystem GHG budgets more difficult and results may be inaccurate.

One suggestion to adequately calculate fluxes for compounds with low SNR is to first calculate the time lag for a *reference* compound with high SNR (e.g. CO<sub>2</sub>) and then apply the same time lag to the *target* compound of interest (e.g. N<sub>2</sub>O), with both compounds being recorded by the same analyzer (Nemitz et al., 2018). `DYCO` follows up on this suggestion by facilitating the dynamic lag-detection between the turbulent wind data and a *reference* compound and the subsequent application of found *reference* time lags to one or more *target* compounds. 



## Processing chain

![DYCO processing chain](images/dyco_processing_chain.png)

**Figure 1.** *The DYCO processing chain.*

`DYCO` uses eddy covariance raw data files as input and produces lag-compensated raw data files as output.

The full `DYCO` processing chain comprises four phases and several iterations during which *reference* lags are refined iteratively in progressively smaller search windows (Figure 1). Generally, the *reference* lag search is facilitated by prior normalization of default (nominal) time lags across files. This is achieved by compensating *reference* and *target* time series data for daily default *reference* lags, calculated from high-quality *reference* lags available around the respective date (Figure 2). Due to this normalization, *reference* lags fall into a specific, pre-defined and therefore known time range, which in turn allows the application of increasingly narrower time windows during lag search. This approach has the advantage that *reference* lags can be calculated from a *reference* signal that shows clear peaks in the cross-covariance analysis with the wind data and thus yields unambiguous time lags due to its high SNR. In case the lag search failed to detect a clear time delay for the *reference* variable (e.g. during the night), the respective daily default *reference* lag is used instead. *Reference* lags can then be used to compensate *target* variables with low SNR for detected *reference* time delays. 

A description of the different Phases along with output examples can be found in the Wiki: [Processing Chain](https://github.com/holukas/dyco/wiki/Processing-Chain).

![Normalization example](images/fig_PHASE-1_ITERATION-3_TIMESERIES-PLOT_segment_lag_times_iteration.png)

**Figure 2**. *Example showing the normalization of default reference time lags across files. Shown are found instantaneous time lags (red) between turbulent wind data and turbulent CO<sub>2</sub> mixing ratios, calculated daily reference default lags (yellow bars), normalization correction (blue arrows) and the daily default reference lag after normalization (green bar). Negative lag values mean that the CO<sub>2</sub> signal was lagged behind the wind data, e.g. -400 means that the instantaneous CO<sub>2</sub> records arrived 400 records (corresponds to 20s in this example) later at the analyzer than the wind data. Daily default reference lags were calculated as the 3-day median time lag from a selection of high-quality time lags, i.e. when cross-covariance analyses yielded a clear covariance peak. The normalization correction is applied dynamically to shift the CO<sub>2</sub> data so that the default time lag is found close to zero across files. Note the systematic shift in time lags starting after 27 Oct 2016.*



## Installation

`DYCO` can be installed via pip:

`pip install dyco`



## Usage

`DYCO` is run from the command line interface (CLI).

`usage: dyco.py [-h] [-i INDIR] [-o OUTDIR] [-fnd FILENAMEDATEFORMAT]`
               `[-fnp FILENAMEPATTERN] [-flim LIMITNUMFILES] [-fgr FILEGENRES]`
               `[-fdur FILEDURATION] [-dtf DATATIMESTAMPFORMAT]`
               `[-dres DATANOMINALTIMERES] [-lss LSSEGMENTDURATION]`
               `[-lsw LSWINSIZE] [-lsi LSNUMITER] [-lsf {0,1}]`
               `[-lsp LSPERCTHRES] [-lt TARGETLAG] [-del {0,1}]`
               `var_reference var_lagged var_target [var_target ...]`

- Example usage with full arguments can be found here: [Example](https://github.com/holukas/dyco/wiki/Example)
- For an overview of arguments see here: [Usage](https://github.com/holukas/dyco/wiki/Usage) 
- `DYCO` creates a range of output folders which are described here: [Results Output Folders](https://github.com/holukas/dyco/wiki/Results-Output-Folders)



## Documentation

Please refer to the [Wiki](https://github.com/holukas/dyco/wiki) for documentation and additional examples.



## Real-world examples

The [ICOS](https://www.icos-cp.eu/) Class 1 site [Davos](https://www.swissfluxnet.ethz.ch/index.php/sites/ch-dav-davos/site-info-ch-dav/) (CH-Dav), a subalpine forest ecosystem station in the east of Switzerland, provides one of the longest continuous time series (24 years and running) of ecosystem fluxes globally. Since 2016, measurements of the strong GHG N<sub>2</sub>O are recorded by a closed-path gas analyzer that also records CO<sub>2</sub>. To calculate fluxes using the EC method, wind data from the sonic anemometer is combined with instantaneous gas measurements from the gas analyzer. However, the air sampled by the gas analyzer needs a certain amount of time to travel from the tube inlet to the measurement cell in the analyzer and is thus lagged behind the wind signal. The lag between the two signals needs to be compensated for by detecting and then removing the time lag at which the cross-covariance between the turbulent wind and the turbulent gas signal reaches the maximum absolute value. This works generally well when using CO<sub>2</sub> (high SNR) but is challenging for N<sub>2</sub>O (low SNR). Using covariance maximization to search for the lag between wind and N<sub>2</sub>O mostly fails to accurately detect time lags between the two variables (noisy cross-correlation function), resulting in relatively noisy fluxes. However, since N<sub>2</sub>O has similar adsorption / desorption characteristics as CO<sub>2</sub> it is valid to assume that both compounds need approx. the same time to travel through the tube to the analyzer, i.e. the time lag for both compounds in relation to the wind is similar. Therefore, `DYCO` can be applied (i) to calculate time lags across files for CO<sub>2</sub> (*reference* compound), and then (ii) to remove found CO<sub>2</sub> time delays from the N<sub>2</sub>O signal (*target* compound). The lag-compensated files produced by `DYCO` can then be used to calculate N<sub>2</sub>O fluxes. Since `DYCO` normalizes time lags across files and compensated the N<sub>2</sub>O signal for instantaneous CO<sub>2</sub> lags, the *true* lag between wind and N<sub>2</sub>O can be found close to zero, which in turn facilitates the application of a small time window for the final lag search during flux calculations. 

Another application example are managed grasslands where the biosphere-atmosphere exchange of N<sub>2</sub>O is often characterized by sporadic high-emission events (e.g., Hörtnagl et al., 2018; Merbold et al., 2014). While high N<sub>2</sub>O quantities can be emitted during and after management events such as fertilizer application and ploughing, fluxes in between those events typically remain low and often below the limit-of-detection of the applied analyzer. In this case, calculating N<sub>2</sub>O fluxes works well during the high-emission periods (high SNR) but is challenging during the rest of the year (low SNR). Here, `DYCO` can be used to first calculate time lags for a *reference* gas measured in the same analyzer (e.g. CO<sub>2</sub>, CO, CH<sub>4</sub>)  and then remove *reference* time lags from the N<sub>2</sub>O data.



## Contributing

All contributions in the form of code, bug reports, comments or general feedback are always welcome and greatly appreciated! Credit will always be given.

- **Pull requests**: If you added new functionality or made the `DYCO` code run faster (always welcome), please create a fork in GitHub, make the contribution public in your repo and then issue a [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork). Please include tests in your pull requests.
- **Issues**:  If you experience any issue, please use the [issue tracker](https://github.com/holukas/dyco/issues) to submit it as an issue ticket with the label 'bug'. Please also include a minimal code example that produces the issue. 
- **Feature request**:  If there is a feature that you would like to see in a later version, please use the [issue tracker](https://github.com/holukas/dyco/issues) to submit it as an issue ticket with the label 'feature request'.
- **Contact details**: For direct questions or enquiries the maintainer of this repository can be contacted directly by writing an email with the title "DYCO" to: holukas@ethz.ch



## Acknowledgements

This work was supported by the Swiss National Science Foundation SNF (ICOS CH, grant nos. 20FI21_148992, 20FI20_173691) and the EU project Readiness of ICOS for Necessities of integrated Global Observations RINGO (grant no. 730944).



## References

Aubinet, M., Vesala, T., Papale, D. (Eds.), 2012. Eddy Covariance: A Practical Guide to Measurement and Data Analysis. Springer Netherlands, Dordrecht. https://doi.org/10.1007/978-94-007-2351-1

Baldocchi, D.D., Hincks, B.B., Meyers, T.P., 1988. Measuring Biosphere-Atmosphere Exchanges of Biologically Related Gases with Micrometeorological Methods. Ecology 69, 1331–1340. https://doi.org/10.2307/1941631

Hörtnagl, L., Barthel, M., Buchmann, N., Eugster, W., Butterbach-Bahl, K., Díaz-Pinés, E., Zeeman, M., Klumpp, K., Kiese, R., Bahn, M., Hammerle, A., Lu, H., Ladreiter-Knauss, T., Burri, S., Merbold, L., 2018. Greenhouse gas fluxes over managed grasslands in Central Europe. Glob. Change Biol. 24, 1843–1872. https://doi.org/10.1111/gcb.14079

Langford, B., Acton, W., Ammann, C., Valach, A., Nemitz, E., 2015. Eddy-covariance data with low signal-to-noise ratio: time-lag determination, uncertainties and limit of detection. Atmospheric Meas. Tech. 8, 4197–4213. https://doi.org/10.5194/amt-8-4197-2015

McMillen, R.T., 1988. An eddy correlation technique with extended applicability to non-simple terrain. Bound.-Layer Meteorol. 43, 231–245. https://doi.org/10.1007/BF00128405

Merbold, L., Eugster, W., Stieger, J., Zahniser, M., Nelson, D., Buchmann, N., 2014. Greenhouse gas budget (CO<sub>2</sub> , CH<sub>4</sub> and N<sub>2</sub>O) of intensively managed grassland following restoration. Glob. Change Biol. 20, 1913–1928. https://doi.org/10.1111/gcb.12518

Moncrieff, J.B., Massheder, J.M., de Bruin, H., Elbers, J., Friborg, T., Heusinkveld, B., Kabat, P., Scott, S., Soegaard, H., Verhoef, A., 1997. A system to measure surface fluxes of momentum, sensible heat, water vapour and carbon dioxide. J. Hydrol. 188–189, 589–611. https://doi.org/10.1016/S0022-1694(96)03194-0

Nemitz, E., Mammarella, I., Ibrom, A., Aurela, M., Burba, G.G., Dengel, S., Gielen, B., Grelle, A., Heinesch, B., Herbst, M., Hörtnagl, L., Klemedtsson, L., Lindroth, A., Lohila, A., McDermitt, D.K., Meier, P., Merbold, L., Nelson, D., Nicolini, G., Nilsson, M.B., Peltola, O., Rinne, J., Zahniser, M., 2018. Standardisation of eddy-covariance flux measurements of methane and nitrous oxide. Int. Agrophysics 32, 517–549. https://doi.org/10.1515/intag-2017-0042

