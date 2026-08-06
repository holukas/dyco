# dyco — dynamic lag compensation

`dyco` takes eddy covariance raw data files as input and produces lag-compensated raw data files as
output, ready for flux calculation software such as EddyPro.

The method is **pre-whitening with block-bootstrap (PWB)**, following
[Vitale et al. (2024)](https://doi.org/10.1007/s10651-024-00615-9). An AR(p) filter strips the serial
autocorrelation out of both series before the cross-correlation is computed, which sharpens a peak
that turbulence would otherwise smear. The lag is then re-estimated on block-bootstrap resamples, so
each detection carries a 95% uncertainty interval instead of a bare number. The PWBOPT decision rule
reads that interval, discards the detections it cannot trust, and puts a reliable neighbouring lag in
their place. This is what makes low-SNR gases such as N₂O and CH₄ workable.

Start with [the terminal UI](tui.md), or run `dyco detect-remove` on the
[command line](cli/index.md).

## One detection method

v3 has a single way of finding the time lag between the vertical wind `W` and a scalar `S`: PWB. The
covariance-maximization method that `dyco` shipped up to v2 was removed. `CHANGELOG.md` records what
went and why.

```{toctree}
:maxdepth: 2
:caption: Getting started

install
tui
example-irga-20hz
example-qcl-10hz
```

```{toctree}
:maxdepth: 2
:caption: Reference

cli/index
output
library
```

```{toctree}
:maxdepth: 2
:caption: Background

method
background
migrating-from-v2
```

## Citing dyco

Cite the software by its Zenodo DOI, [10.5281/zenodo.4964067](https://doi.org/10.5281/zenodo.4964067).
That is the concept DOI: it resolves to the latest version and is the one to use for all versions.
Zenodo also mints a DOI for each individual release, if the work depends on a particular one.
`CITATION.cff` in the repository carries the full metadata.

`dyco` was first published in 2021 and has kept developing since. The detection method was replaced
in v3, so the paper from that first release describes an algorithm the current version does not
implement — see [Migrating from v2](migrating-from-v2.md).

For the method itself, cite Vitale et al. (2024); [the PWB method](method.md) carries the
reference.

## Acknowledgements

This work was supported by the Swiss National Science Foundation SNF (ICOS CH, grant nos.
20FI21_148992, 20FI20_173691) and the EU project Readiness of ICOS for Necessities of integrated
Global Observations RINGO (grant no. 730944).
