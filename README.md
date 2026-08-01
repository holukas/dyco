![Logo](https://raw.githubusercontent.com/holukas/dyco/refs/heads/main/images/logo_dyco1_256px.png)

# **dyco** - dynamic lag compensation

`dyco` takes eddy covariance raw data files as input and produces lag-compensated raw data files as
output, ready for flux calculation software such as EddyPro.

> **Version 3 is in development.** The working tree carries the v3 layout described below; the last
> released version on PyPI is `2.0.3`, which has a different API and depends on
> [diive](https://github.com/holukas/diive). v3 is standalone. See `CHANGELOG.md` for release status.

## Two detection methods

v3 ships two ways of finding the time lag between the vertical wind `W` and a scalar `S`.

| | **PWB** (primary) | **CM** (retained) |
|---|---|---|
| Method | Pre-whitening with block-bootstrap, Vitale et al. (2024) | Iterative covariance-maximization window narrowing |
| Best for | Low-SNR gases: N<sub>2</sub>O, CH<sub>4</sub> | High-SNR reference gases, and long records where a daily default lag is wanted |
| Lag decision | PWBOPT per averaging period, with an uncertainty interval | Daily median look-up table across all files |
| Uncertainty | 95% highest-density interval per detection | none |
| Entry point | `dyco detect-remove` | `dyco cm` / `Dyco` class |
| Lag unit | seconds | number of records |

**Use PWB unless you have a reason not to.** It reports how confident each detection is, which the
covariance-maximization approach cannot, and that is exactly the problem low-SNR gases present. The
CM method remains available and unchanged for existing workflows.

## Installation

Requires Python 3.12 or 3.13.

```bash
git clone https://github.com/holukas/dyco.git
cd dyco
uv sync
```

`dyco` has no dependency on `diive`. Everything it needs is bundled.

## Command-line tools

Everything is reachable through one command:

```bash
dyco                    # list the workflows
dyco <command> --help   # options for one of them
```

| Command | Does |
|---|---|
| `dyco detect-remove` | **Main entry point.** Split long raw files into averaging-period chunks, rotate, detect the lag per chunk, then remove it. One pass. |
| `dyco tui` | The same pipeline behind a terminal UI, with live validation and a preflight check. Run with `--demo` to explore it without data. |
| `dyco pwb-batch` | Detect lags only, across many already-split files. Writes `tlag_results.csv`. |
| `dyco apply-batch` | Remove lags listed in an existing `tlag_results.csv`. |

| `dyco cm` | The covariance-maximization workflow — see below. |

Each also exists standalone: `dyco-detect-remove`, `dyco-detect-remove-tui`,
`dyco-pwb-batch`, `dyco-apply-batch`.

### Try it

A real 1-hour raw file from CH-LAE (Lägeren) ships with the repository, so the
primary pipeline can be run without supplying data:

```bash
uv run python examples/detect_remove_tlag_realdata.py
```

That detects and removes the tube delay for CO<sub>2</sub> and H<sub>2</sub>O
across two 30-minute chunks, and prints what was detected against what was
actually applied. Set `DYCO_OUT` to a path to keep the results. Expect about a
minute. `examples/detect_remove_tlag.py` is the synthetic counterpart, where the
lag is known in advance and can be checked.

## The PWB workflow

### What pre-whitening and block-bootstrap do

Turbulent wind and trace-gas series are strongly autocorrelated, which broadens and distorts the peak
of the cross-correlation function and blurs the lag estimate. **Pre-whitening** fits an AR(p) filter
(order chosen by AIC) to each series so the residuals are approximately white noise, sharpening the
peak.

**Block-bootstrap** then answers a question conventional lag detection cannot: *how sure are we?*
Rather than one cross-correlation over the whole averaging period, the method draws resampled series
using overlapping blocks that preserve local autocorrelation, finds the peak lag in each, and
summarises the resulting distribution as a mode (the estimate) plus a 95% highest-density interval
(HDI). A narrow HDI means repeated resampling keeps finding the same lag.

Four cross-correlation combinations are evaluated per period, using either `W` or sonic temperature
`T_SONIC` as the reference and applying the AR filter to either the scalar or the reference. Because
`T_SONIC` and `W` are coupled through buoyant turbulence, the `T_SONIC` combinations often expose a
cleaner peak for gases whose direct scalar × `W` signal is weak. The combination with the highest
smoothed peak wins. **`T_SONIC` is required.**

### Running it

`dyco-detect-remove` works in two phases over fixed-length chunks, because a multi-hour raw file is
the wrong granularity for lag detection: both the rotation angles and the tube delay drift over hours.

**Phase 1 — detect.** Each chunk is read, double-rotated in memory, and passed to the PWB detector.
Nothing is written yet. Rotation happens in memory only; the rotated data never reach disk.

**PWBOPT.** With every chunk's raw detection in hand, the S1/S2/S3 decision rule (Vitale et al. 2024,
Section 2.3) runs across the whole sequence in temporal order. A chunk with a wide HDI has an
untrustworthy mode lag, and PWBOPT replaces it with a neighbouring reliable one rather than accepting
a spurious value. This is why detection and removal are separate phases — the rule needs the full
sequence before it can decide anything.

**Phase 2 — remove.** Each scalar in the **unrotated** chunk is shifted by `round(tlag * hz)` rows,
and the chunk is written out as its own file with the original header rows and column order intact.
A 6-hour input file yields up to twelve 30-minute output files. Gzipped input
is read and written transparently: a `.csv.gz` in gives `.gz` chunks out.

Chunk boundaries snap to the wall-clock grid (:00 / :30) when the file start time can be parsed, so
downstream software bins them correctly. A file starting off-grid produces a shorter leading chunk.

> **Important:** downstream flux processing must run with time-lag maximization **disabled**. The lag
> has already been removed.

### Per-gas search windows

Gases with different inlet geometry need different search windows. Each gas can have its own:

```bash
dyco-detect-remove --scalar "CH4:ch4" --scalar "H2O:h2o@lag=30;uws=25" --lws 0 --uws 5
```

A positive-only window keeps only physical tube delays (a closed-path delay is always > 0). A
long-inlet gas such as H<sub>2</sub>O can use a wider window than the dry gases in the same run —
necessary because EddyPro applies a single lag setting to all gases downstream. Keep the expected lag
near the middle of the window; detections pinned to a boundary are unreliable and are discarded.

### Input requirements

PWB detection needs **wind-rotation-corrected** high-frequency data. `dyco-detect-remove` handles
this itself. If you use `dyco-pwb-batch` on pre-split files, they must already be rotated (double
rotation or planar fit, e.g. EddyPro "Advanced" rotated output) — a non-zero mean `W` corrupts the
cross-correlation.

## The CM workflow (retained)

The original `dyco` method. It searches for the lag in a broad window, narrows that window based on
where lags actually cluster, repeats, then reduces everything to a daily look-up table.

**Lag is expressed in number of records here, not seconds.** At 20 Hz, 1000 records is 50 seconds. A
negative lag means `S` arrived after `W`.

### Step 1: Detect time lags across all files

Detection starts in a broad window, e.g. `[-1000, +1000]` records. This is iteration 1. Lag search can
run on segments within a file: a 30-minute file searched in 10-minute segments yields three lags.

![Covariance](https://raw.githubusercontent.com/holukas/dyco/refs/heads/main/images/dyco_v2_fig_covariance_20230517102000_segment3_iter1_segment_3_iteration-1.png)
**Figure 1**. _Covariance between turbulent vertical wind and turbulent CH4 mixing ratios from the
subcanopy station [CH-DAS](https://www.swissfluxnet.ethz.ch/index.php/sites/site-info-ch-das/) on 17
May 2023. Searched between `-500` and `0` records in a 10MIN segment. Peak absolute covariance at lag
`-246`: CH4 arrived 246 records after the wind._

### Step 2: Analyze found time lags

The distribution of found lags is examined, the histogram peak identified, and a narrower window built
outward from it until a set percentage of detections is enclosed.

![Histogram](https://raw.githubusercontent.com/holukas/dyco/refs/heads/main/images/dyco_v2_fig_HISTOGRAM_segment_lag_times_iteration-1.png)
**Figure 2**. _Histogram of found lags (iteration 1), window `[-500, 0]`, from 6919 files between 12
May and 31 Dec 2023 searched in 10MIN segments. The clear peak just below `-200` sets the next
iteration's window._

### Step 3: Repeat

Steps 1 and 2 repeat with the narrower window. There is no iteration limit, but watch that the window
stays wide enough to be meaningful.

### Step 4: Collect lags across all iterations

Lags found for `S` across all iterations are pooled. A lag that survives progressively narrower windows
indicates high covariance between `W` and `S`.

![Histogram](https://raw.githubusercontent.com/holukas/dyco/refs/heads/main/images/dyco_v2_fig_HISTOGRAM_segment_lag_times_iteration-3.png)
**Figure 3**. _Distribution after the third iteration, window `[-482, -26]`. Little narrowing was
needed since the initial `[-500, 0]` was well chosen. The count includes lags from earlier iterations._

![Time series plot](https://raw.githubusercontent.com/holukas/dyco/refs/heads/main/images/dyco_v2_TIMESERIES-PLOT_segment_lag_times_FINAL.png)
**Figure 4**. _All found lags across files and iterations. Lags accumulate near -200 but are not
constant — there is a clear drift._

### Step 5: Remove outlier lags

A rolling z-score filter removes outliers so only consistent lags feed the look-up table.

![Outlier removal](https://raw.githubusercontent.com/holukas/dyco/refs/heads/main/images/dyco_v2_TIMESERIES-PLOT_segment_lag_times_FINAL_outlierRemoved.png)
**Figure 5**. _Outlier removal. The lower left panel shows the retained lags._

### Step 6: Create look-up table and remove lags

The filtered lags become a daily look-up table, which is then applied to shift one or more variables in
each file. `S` is used for detection, but the correction can be applied to any variable — so a strong
`S` signal can drive lag detection even when `S` is not the compound of interest.

![Time series](https://raw.githubusercontent.com/holukas/dyco/refs/heads/main/images/dyco_v2_TIMESERIES-PLOT_segment_lag_times2_FINAL.png)
**Figure 6**. _The 5-day median of high-quality lags after outlier removal, used to shift each scalar.
All files from a given day are shifted by the same number of records. Afterwards the lag between wind
and scalar is at or near zero._

### Step 7: Use the lag-compensated files

The output files go straight into flux calculation.

### Using it

```python
from dyco.dyco import Dyco

Dyco(var_reference="W_[R350-B]_TURB",  # Turbulent departures of vertical wind
     var_lagged="CH4_DRY_[QCL-C2]_TURB",  # Turbulent departures of CH4
     var_target=["CH4_DRY_[QCL-C2]_TURB", "N2O_DRY_[QCL-C2]_TURB"],
     indir=r"F:\example\input_files",
     outdir=r"F:\example\output",
     filename_date_format="CH-DAS_%Y%m%d%H%M%S_30MIN-SPLIT_ROT_TRIM.csv",
     filename_pattern="CH-DAS_*_30MIN-SPLIT_ROT_TRIM.csv",
     files_how_many=None,
     file_generation_res="30min",
     file_duration="30min",
     data_timestamp_format="%Y-%m-%d %H:%M:%S.%f",
     data_nominal_timeres=0.05,
     lag_segment_dur="10min",
     lag_winsize=1000,
     lag_n_iter=3,
     lag_hist_remove_fringe_bins=True,
     lag_hist_perc_thres=0.7,
     target_lag=0,
     del_previous_results=False)
```

This method needs input files that are **already rotated**. `FileSplitterMulti` (below) does the
splitting and rotation in one step.

## Other tools

These arrived in v3 and support the lag work, but are useful on their own.

### Splitting and rotating raw files

`FileSplitter` / `FileSplitterMulti` divide long raw files into shorter time-based parts, optionally
applying double rotation and writing the turbulent departures alongside. Output as CSV (optionally
gzipped) or Parquet.

```python
from dyco.split import FileSplitterMulti
```

### Wind rotation

`WindDoubleRotation` and `reynolds_decomposition` — double rotation for sonic anemometer tilt
correction, and turbulent departures `x' = x - mean(x)`.

```python
from dyco.rotation import WindDoubleRotation, reynolds_decomposition
```

### Flux detection limit

`FluxDetectionLimit` estimates the smallest flux distinguishable from noise, following Langford et al.
(2015). It reads the noise from the far tail of the same cross-covariance function used for lag
detection, which is why it lives here.

```python
from dyco.detectionlimit import FluxDetectionLimit
```

## Motivation

Detecting the lag between the turbulent departures of measured wind and the scalar of interest is a
central step in calculating eddy covariance ecosystem fluxes. When covariance maximization fails to
find a clear peak, flux software falls back to a constant nominal lag. But both finding a clear peak
and choosing a reliable default are hard for compounds with low signal-to-noise ratio such as
N<sub>2</sub>O — and one static default produces poor results when the raw data contain systematic
time shifts.

`dyco` assists flux processing software for exactly these compounds. It offers:

- **PWB**: a lag estimate with an explicit uncertainty interval, so unreliable detections can be
  identified rather than silently accepted, and a decision rule that substitutes a trustworthy
  neighbouring lag when a period's own detection cannot be trusted
- **CM**: progressively narrower search windows for a *reference* compound (e.g. CO<sub>2</sub>, which
  usually shows a clear peak), daily default lags derived from it, and application of those lags to one
  or more *target* compounds
- Dynamic compensation across raw files, and automatic correction of systematic time drifts, e.g. from
  unsynchronized instrument clocks

Both produce lag-removed files usable directly in flux calculation software.

## Scientific background

In ecosystem research the EC method is widely used to quantify biosphere-atmosphere exchange of
greenhouse gases and energy (Aubinet et al., 2012; Baldocchi et al., 1988). The raw flux is the
covariance between the turbulent vertical wind measured by a sonic anemometer and the entity of
interest measured by a gas analyzer. Because two instruments are involved, wind and gas are not
recorded at the same instant, producing a time lag that must be quantified and corrected or fluxes are
systematically biased. Lags are conventionally estimated by finding maximum absolute covariance within
a window of physically possible lags (e.g., McMillen, 1988; Moncrieff et al., 1997).

This works for compounds with high SNR such as CO<sub>2</sub>. For low-SNR compounds such as
N<sub>2</sub>O and CH<sub>4</sub> the cross-covariance function is noisy, and fluxes are biased toward
larger absolute values (Langford et al., 2015), making annual GHG budgets harder to calculate
accurately.

Two responses are implemented here. One is to detect the lag for a high-SNR *reference* compound and
apply it to the low-SNR *target* measured by the same analyzer (Nemitz et al., 2018) — the CM method.
The other is to improve the estimate itself: pre-whitening sharpens the cross-correlation peak by
removing serial autocorrelation, and block-bootstrap resampling quantifies how reproducible the
resulting lag is (Vitale et al., 2024) — the PWB method.

## Real-world examples

The [ICOS](https://www.icos-cp.eu/) Class 1
site [Davos](https://www.swissfluxnet.ethz.ch/index.php/sites/ch-dav-davos/site-info-ch-dav/) (CH-Dav),
a subalpine forest in eastern Switzerland, holds one of the longest continuous flux records globally
(24 years and running). Since 2016 N<sub>2</sub>O has been measured by a closed-path analyzer that also
records CO<sub>2</sub>. Air sampled by the analyzer takes time to travel from the tube inlet to the
measurement cell, so the gas signal lags the wind. Covariance maximization handles CO<sub>2</sub> well
but mostly fails for N<sub>2</sub>O, whose cross-correlation function is noisy, giving noisy fluxes.
Since N<sub>2</sub>O has adsorption/desorption characteristics similar to CO<sub>2</sub>, both need
roughly the same travel time — so `dyco` can detect lags on CO<sub>2</sub> and remove them from
N<sub>2</sub>O. Normalizing lags across files leaves the *true* wind-to-N<sub>2</sub>O lag near zero,
which makes a small window or a constant lag viable during flux calculation.

Another case is managed grassland, where N<sub>2</sub>O exchange is dominated by sporadic
high-emission events (e.g., Hörtnagl et al., 2018; Merbold et al., 2014). Large quantities are emitted
during and after fertilizer application and ploughing, but between those events fluxes stay low, often
below the analyzer's detection limit. Flux calculation works during high-emission periods (high SNR)
and struggles the rest of the year. Here too, lags from a *reference* gas in the same analyzer
(CO<sub>2</sub>, CO, CH<sub>4</sub>) can be removed from the N<sub>2</sub>O data.

## Contributing

All contributions in the form of code, bug reports, comments or general feedback are always welcome and
greatly appreciated! Credit will always be given.

- **Pull requests**: If you added new functionality or made the `dyco` code run faster (always
  welcome), please create a fork in GitHub, make the contribution public in your repo and then issue
  a [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork).
  Please include tests in your pull requests.
- **Issues**: If you experience any issue, please use
  the [issue tracker](https://github.com/holukas/dyco/issues) to submit it as an issue ticket with the
  label 'bug'. Please also include a minimal code example that produces the issue.
- **Feature request**: If there is a feature that you would like to see in a later version, please use
  the [issue tracker](https://github.com/holukas/dyco/issues) to submit it as an issue ticket with the
  label 'feature request'.
- **Contact details**: For direct questions or enquiries the maintainer of this repository can be
  contacted directly by writing an email with the title "dyco" to: holukas@ethz.ch

## Acknowledgements

This work was supported by the Swiss National Science Foundation SNF (ICOS CH, grant nos.
20FI21_148992, 20FI20_173691) and the EU project Readiness of ICOS for Necessities of integrated Global
Observations RINGO (grant no. 730944).

## Notes

A previous version of `dyco` was used in a publication in JOSS.
[![DOI](status.svg)](https://doi.org/10.21105/joss.02575) [![DOI](https://zenodo.org/badge/311300577.svg)](https://zenodo.org/badge/latestdoi/311300577)

## References

Aubinet, M., Vesala, T., Papale, D. (Eds.), 2012. Eddy Covariance: A Practical Guide to Measurement and
Data Analysis. Springer Netherlands, Dordrecht. https://doi.org/10.1007/978-94-007-2351-1

Baldocchi, D.D., Hincks, B.B., Meyers, T.P., 1988. Measuring Biosphere-Atmosphere Exchanges of
Biologically Related Gases with Micrometeorological Methods. Ecology 69,
1331–1340. https://doi.org/10.2307/1941631

Hörtnagl, L., Barthel, M., Buchmann, N., Eugster, W., Butterbach-Bahl, K., Díaz-Pinés, E., Zeeman, M.,
Klumpp, K., Kiese, R., Bahn, M., Hammerle, A., Lu, H., Ladreiter-Knauss, T., Burri, S., Merbold, L.,
2018. Greenhouse gas fluxes over managed grasslands in Central Europe. Glob. Change Biol. 24,
1843–1872. https://doi.org/10.1111/gcb.14079

Langford, B., Acton, W., Ammann, C., Valach, A., Nemitz, E., 2015. Eddy-covariance data with low
signal-to-noise ratio: time-lag determination, uncertainties and limit of detection. Atmospheric Meas.
Tech. 8, 4197–4213. https://doi.org/10.5194/amt-8-4197-2015

McMillen, R.T., 1988. An eddy correlation technique with extended applicability to non-simple terrain.
Bound.-Layer Meteorol. 43, 231–245. https://doi.org/10.1007/BF00128405

Merbold, L., Eugster, W., Stieger, J., Zahniser, M., Nelson, D., Buchmann, N., 2014. Greenhouse gas
budget (CO<sub>2</sub>, CH<sub>4</sub> and N<sub>2</sub>O) of intensively managed grassland following
restoration. Glob. Change Biol. 20, 1913–1928. https://doi.org/10.1111/gcb.12518

Moncrieff, J.B., Massheder, J.M., de Bruin, H., Elbers, J., Friborg, T., Heusinkveld, B., Kabat, P.,
Scott, S., Soegaard, H., Verhoef, A., 1997. A system to measure surface fluxes of momentum, sensible
heat, water vapour and carbon dioxide. J. Hydrol. 188–189,
589–611. https://doi.org/10.1016/S0022-1694(96)03194-0

Nemitz, E., Mammarella, I., Ibrom, A., Aurela, M., Burba, G.G., Dengel, S., Gielen, B., Grelle, A.,
Heinesch, B., Herbst, M., Hörtnagl, L., Klemedtsson, L., Lindroth, A., Lohila, A., McDermitt, D.K.,
Meier, P., Merbold, L., Nelson, D., Nicolini, G., Nilsson, M.B., Peltola, O., Rinne, J., Zahniser, M.,
2018. Standardisation of eddy-covariance flux measurements of methane and nitrous oxide. Int.
Agrophysics 32, 517–549. https://doi.org/10.1515/intag-2017-0042

Vitale, D., Fratini, G., Helfter, C., Hörtnagl, L., et al., 2024. A pre-whitening with block-bootstrap
cross-correlation procedure for temporal alignment of data sampled by eddy covariance systems. Environ.
Ecol. Stat. 31, 219–244. https://doi.org/10.1007/s10651-024-00615-9
