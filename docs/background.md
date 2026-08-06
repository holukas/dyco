# Motivation and background

## Why lag compensation needs its own tool

Detecting the lag between the turbulent departures of measured wind and the scalar of interest is a
central step in calculating eddy covariance ecosystem fluxes. When covariance maximization fails to
find a clear peak, flux software falls back to a constant nominal lag. But both finding a clear peak
and choosing a reliable default are hard for compounds with low signal-to-noise ratio such as N₂O.
One static default also produces poor results when the raw data contain systematic time shifts.

`dyco` assists flux processing software for exactly these compounds. It offers:

- a lag estimate with an explicit uncertainty interval, so unreliable detections can be identified
  rather than silently accepted, and a decision rule that substitutes a trustworthy neighbouring lag
  when a period's own detection cannot be trusted
- lags detected on a high-SNR *reference* gas used for a low-SNR *target* measured by the same
  analyzer, where the target cannot determine its own — see
  [Taking a gas's lag from another gas](cli/index.md#taking-a-gass-lag-from-another-gas)
- dynamic compensation across raw files, so a lag that drifts (from an unsynchronized instrument
  clock, say) is followed period by period instead of averaged away

The output is lag-removed files usable directly in flux calculation software.

## Scientific background

In ecosystem research the EC method is widely used to quantify biosphere-atmosphere exchange of
greenhouse gases and energy (Aubinet et al., 2012; Baldocchi et al., 1988). The raw flux is the
covariance between the turbulent vertical wind measured by a sonic anemometer and the entity of
interest measured by a gas analyzer. Because two instruments are involved, wind and gas are not
recorded at the same instant, producing a time lag that must be quantified and corrected or fluxes
are systematically biased. Lags are conventionally estimated by finding maximum absolute covariance
within a window of physically possible lags (e.g., McMillen, 1988; Moncrieff et al., 1997).

This works for compounds with high SNR such as CO₂. For low-SNR compounds such as N₂O and CH₄ the
cross-covariance function is noisy, and fluxes are biased toward larger absolute values (Langford et
al., 2015), making annual GHG budgets harder to calculate accurately.

There are two responses to this. One is to detect the lag for a high-SNR *reference* compound and
apply it to the low-SNR *target* measured by the same analyzer (Nemitz et al., 2018), which `dyco`
supports by pairing one gas's lag with another gas's column at the removal step. The other is to
improve the estimate itself: pre-whitening sharpens the cross-correlation peak by removing serial
autocorrelation, and block-bootstrap resampling quantifies how reproducible the resulting lag is
(Vitale et al., 2024). The second is [the method](method.md) `dyco` implements as of v3.

## Real-world examples

The [ICOS](https://www.icos-cp.eu/) Class 1 site
[Davos](https://www.swissfluxnet.ethz.ch/index.php/sites/ch-dav-davos/site-info-ch-dav/) (CH-Dav), a
subalpine forest in eastern Switzerland, holds one of the longest continuous flux records globally
(24 years and running). Since 2016 N₂O has been measured by a closed-path analyzer that also records
CO₂. Air sampled by the analyzer takes time to travel from the tube inlet to the measurement cell, so
the gas signal lags the wind. Covariance maximization handles CO₂ well but mostly fails for N₂O,
whose cross-correlation function is noisy, giving noisy fluxes. Since N₂O has adsorption/desorption
characteristics similar to CO₂, both need roughly the same travel time, so `dyco` can detect lags on
CO₂ and remove them from N₂O. Once the tube delay is out of the files, the remaining wind-to-N₂O lag
sits near zero, which makes a small window or a constant lag viable during flux calculation.

Another case is managed grassland, where N₂O exchange is dominated by sporadic high-emission events
(e.g., Hörtnagl et al., 2018; Merbold et al., 2014). Large quantities are emitted during and after
fertilizer application and ploughing, but between those events fluxes stay low, often below the
analyzer's detection limit. Flux calculation works during high-emission periods (high SNR) and
struggles the rest of the year. Here too, lags from a *reference* gas in the same analyzer (CO₂, CO,
CH₄) can be removed from the N₂O data.

## References

Aubinet, M., Vesala, T., Papale, D. (Eds.), 2012. *Eddy Covariance: A Practical Guide to Measurement
and Data Analysis.* Springer Netherlands, Dordrecht.
<https://doi.org/10.1007/978-94-007-2351-1>

Baldocchi, D.D., Hincks, B.B., Meyers, T.P., 1988. Measuring Biosphere-Atmosphere Exchanges of
Biologically Related Gases with Micrometeorological Methods. *Ecology* 69, 1331–1340.
<https://doi.org/10.2307/1941631>

Hörtnagl, L., Barthel, M., Buchmann, N., Eugster, W., Butterbach-Bahl, K., Díaz-Pinés, E., Zeeman,
M., Klumpp, K., Kiese, R., Bahn, M., Hammerle, A., Lu, H., Ladreiter-Knauss, T., Burri, S., Merbold,
L., 2018. Greenhouse gas fluxes over managed grasslands in Central Europe. *Global Change Biology*
24, 1843–1872. <https://doi.org/10.1111/gcb.14079>

Langford, B., Acton, W., Ammann, C., Valach, A., Nemitz, E., 2015. Eddy-covariance data with low
signal-to-noise ratio: time-lag determination, uncertainties and limit of detection. *Atmospheric
Measurement Techniques* 8, 4197–4213. <https://doi.org/10.5194/amt-8-4197-2015>

McMillen, R.T., 1988. An eddy correlation technique with extended applicability to non-simple
terrain. *Boundary-Layer Meteorology* 43, 231–245. <https://doi.org/10.1007/BF00128405>

Merbold, L., Eugster, W., Stieger, J., Zahniser, M., Nelson, D., Buchmann, N., 2014. Greenhouse gas
budget (CO₂, CH₄ and N₂O) of intensively managed grassland following restoration. *Global Change
Biology* 20, 1913–1928. <https://doi.org/10.1111/gcb.12518>

Moncrieff, J.B., Massheder, J.M., de Bruin, H., Elbers, J., Friborg, T., Heusinkveld, B., Kabat, P.,
Scott, S., Soegaard, H., Verhoef, A., 1997. A system to measure surface fluxes of momentum, sensible
heat, water vapour and carbon dioxide. *Journal of Hydrology* 188–189, 589–611.
<https://doi.org/10.1016/S0022-1694(96)03194-0>

Nemitz, E., Mammarella, I., Ibrom, A., Aurela, M., Burba, G.G., Dengel, S., Gielen, B., Grelle, A.,
Heinesch, B., Herbst, M., Hörtnagl, L., Klemedtsson, L., Lindroth, A., Lohila, A., McDermitt, D.K.,
Meier, P., Merbold, L., Nelson, D., Nicolini, G., Nilsson, M.B., Peltola, O., Rinne, J., Zahniser,
M., 2018. Standardisation of eddy-covariance flux measurements of methane and nitrous oxide.
*International Agrophysics* 32, 517–549. <https://doi.org/10.1515/intag-2017-0042>

Vitale, D., Fratini, G., Helfter, C., Hörtnagl, L., et al., 2024. A pre-whitening with
block-bootstrap cross-correlation procedure for temporal alignment of data sampled by eddy covariance
systems. *Environmental and Ecological Statistics* 31, 219–244.
<https://doi.org/10.1007/s10651-024-00615-9>
