---
title: 'DYCO: A Python package to dynamically detect and compensate for time lags in ecosystem time series'
tags:
  - Python
  - eddy covariance
  - flux
authors:
  - name: Lukas Hörtnagl
    orcid: 0000-0002-5569-0761
    affiliation: "1"
affiliations:
 - name: ETH Zurich, Department of Environmental Systems Science, ETH Zurich, CH-8092, Switzerland
   index: 1
date: 31 Jul 2020
bibliography: paper.bib
---

# Summary

`DYCO` facilitates the detection of time lags between two time series using covariance maximization. The time delay for each averaging interval is estimated by incrementally shifting one signal (the reference signal) and repeatedly calculating the time-lagged cross-covariance between both variables. The time lag that produces the highest covariance is selected as the *reference lag*, which can then be applied to selected *target* *variables*. The full processing chain comprises several phases and iterations during which *reference lags* are refined iteratively in progressively smaller search windows. The lag search is facilitated by prior normalization of default (nominal) time lags across files. This is achieved by compensating data for daily default lags, calculated from high-quality *reference lags* available around the respective date (Figure 1). Due to this normalization, *reference lags* fall into a specific, pre-defined and therefore known time range, which in turn allows the application of increasingly narrow time windows during lag search.

This approach has the advantage that *reference lags* can be calculated from signals that show clear peaks in the cross-covariance analysis and thus yield unambiguous time lags due to their high signal-to-noise ratio (SNR). In case the lag search failed to detect a clear time delay for the *reference variable*, the respective daily default lag is used instead.  *Reference lags* can then be used to compensate *target variables* with low SNR for detected *reference* time delays. 



![Normalization example](figure.png)

**Figure 1**. *Example showing the normalization of default reference time lags across files. Shown are found instantaneous time lags (red) between turbulent wind data and CO<sub>2</sub> mixing ratios, calculated daily reference default lags (yellow bars), normalization correction (blue arrows) and the default lag after normalization (green bar). Negative lag values mean that the CO<sub>2</sub> signal was lagged behind the wind data, e.g. -400 means that the instantaneous CO<sub>2</sub> records arrived 400 records (corresponds to 20s in this example) later at the analyzer than the wind data. Daily default lags were calculated as the 3-day median time lag from a selection of high-quality time lags, i.e. when cross-covariance analyses yielded a clear covariance peak. The normalization correction is applied dynamically to shift the CO<sub>2</sub> data so that the default time lag is found close to zero across files. Note the systematic shift in time lags starting after 27 Oct 2016.*

# Statement of need

In ecosystem research, the eddy covariance (EC) method is widely used to quantify the biosphere-atmosphere exchange of greenhouse gases (GHGs) and energy [@Aubinet2012; @Baldocchi1988]. The raw ecosystem flux (i.e. net exchange) is calculated by the covariance between the turbulent vertical wind component measured by a sonic anemometer and the entity of interest, e.g. CO<sub>2</sub>, measured by a gas analyzer. Due to the application of two different instruments, wind and gas are not recorded at exactly the same time, resulting in a time lag between the two time series. For the calculation of ecosystem fluxes this time delay has to be quantified and corrected for, otherwise fluxes are systematically biased.

Time lags for each averaging interval can be estimated by finding the maximum absolute covariance between the two time series at different time steps in a pre-defined time window of physically possible time-lags  [e.g., @McMillen1988; @Moncrieff1997]. Lag detection works well when processing fluxes for compounds with high SNR, which is typically the case for e.g. CO<sub>2</sub>. In contrast, for compounds with low SNR the cross-covariance function yields noisier results and biases the flux towards larger absolute flux values [@Langford2015]. This can be the case for compounds that are characterized by sporadic high-emission events, while fluxes in between those events remain low and often below the limit-of-detection of the applied analyzer. A typical example are fluxes of the strong GHG nitrous oxide (N<sub>2</sub>O) over managed grasslands: fluxes are typically low throughout the year, but high N<sub>2</sub>O quantities can be emitted during and after management events such as fertilizer application and ploughing [e.g., @Hoertnagl2018; @Merbold2014]. In this case, calculating fluxes works well during the high-emission periods (high SNR) but is challenging during the rest of the year (low SNR), which exacerbates subsequent data analyses and the calculation of a yearly GHG budget for the ecosystem.

One suggestion to adequately calculate fluxes for compounds with low SNR is to first calculate the time lag for a reference compound with high SNR (e.g. CO<sub>2</sub>) and then apply the same time lag to the target compound of interest (e.g. N<sub>2</sub>O), with both compounds being recorded by the same analyzer [@Nemitz2018]. `DYCO` follows up on this suggestion and facilitates dynamic lag-detection for a reference compound and the application of found reference time lags to one or more target compounds.

# Real-world example

The [ICOS](https://www.icos-cp.eu/) Class 1 site Davos (CH-Dav), a subalpine forest ecosystem station in the east of Switzerland, provides one of the longest continuous time series (23 years and running) of ecosystem fluxes globally. Since 2016, measurements of the strong GHG N<sub>2</sub>O are recorded by a closed-path gas analyzer that also records CO<sub>2</sub>. To calculate fluxes using the EC method, wind data from the sonic anemometer is combined with instantaneous gas measurements from the gas analyzer. However, the air sampled by the gas analyzer needs a certain amount of time to travel from the tube inlet to the measurement cell in the analyzer and is thus lagged behind the wind signal. The lag between the two signals needs to be compensated for by detecting and then removing the time lag at which the cross-covariance between the wind and the gas signal reaches the maximum absolute value. This works generally well when using CO<sub>2</sub> (high SNR) but is challenging for N<sub>2</sub>O (low SNR). Performing the lag detection on wind and N<sub>2</sub>O data yields noisy time lags and the true lag remains unknown. However, since N<sub>2</sub>O has similar adsorption / desorption characteristics as CO<sub>2</sub> it is valid to assume that both compounds need approx. the same time to travel through the tube to the analyzer, i.e. the time lag for both compounds in relation to the wind is similar. Therefore, `DYCO` can be applied (i) to calculate time lags across files for CO<sub>2</sub> (reference compound), and then (ii) to remove found CO<sub>2</sub> time delays from the N<sub>2</sub>O signal (target compound). After lag compensation, wind records are correctly matched with instantaneous N<sub>2</sub>O records and the ecosystem flux can be calculated more accurately.

# Acknowledgements

This work was supported by the Swiss National Science Foundation SNF (ICOS CH, grant nos. 20FI21_148992, 20FI20_173691) and the EU project Readiness of ICOS for Necessities of integrated Global Observations RINGO (grant no. 730944).

# References
