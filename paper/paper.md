---
title: 'DYCO: Dynamic lag compensation'
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
date: 29 Jul 2020
bibliography: paper.bib
---

## Summary

A map of the binary orbit can then be constructed by observing the phase of the pulsations over time, which can be converted into time delays -- a measure of the relative time taken for the light to reach Earth (as shown in Fig. 1). This method, phase modulation, is uniquely suited to observing intermediate period binaries [@Murphy2015Deriving].

![Phase modulation of the binary system, KIC 4471379, which is composed of two $\delta$ Scuti pulsating stars. The top panel shows the observed flux of the system (the light curve). The middle panel shows the amplitude spectrum of the light curve, which is the superposition of each star's pulsation spectrum. The bottom panel shows the time delay derived from the pulsations. Blue and orange points correspond to the first and second stars in the system respectively. As they orbit each other, the time taken for the light to reach us changes over time. Since both stars are pulsating, we can identify which pulsations belong to which star.](PB2_KIC_4471379_JOSS.png)

We have developed a Python package, ``Maelstrom``, which implements this
technique. ``Maelstrom`` is written using the popular Bayesian inference
framework, ``PyMC3``, allowing for the use of gradient based samplers such as
No-U-Turn [@Hoffman2011NoUTurn] and Hamiltonian Monte Carlo [@Duane1987Hybrid].
``Maelstrom`` features a series of pre-defined ``PyMC3`` models for analysing
binary motion within stellar pulsations. These are powered by the ``orbit``
module, which returns a light curve given the frequencies of pulsation and the
classical orbital parameters. Using this light curve, one can compare with
photometric data from the *Kepler* and *TESS* space missions to fit for binary
motion. For more complex systems outside the pre-defined scope, the ``orbit``
module can be used to construct custom models with different priors, and
combine them with other ``PyMC3`` codes, such as exoplanet
[@DanForeman-Mackey2019Dfm]. To the best of our knowledge, ``Maelstrom`` is
currently the only available open code for analysing time delay signals.

The documentation of `maelstrom` consists of pages describing the various
available functions, as well as tutorial notebooks.

# Statement of need

# Acknowledgements

ICOS, RINGO

# References



