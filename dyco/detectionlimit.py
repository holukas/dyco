"""
FLUX DETECTION LIMIT: MEASUREMENT SENSITIVITY ASSESSMENT
=========================================================

Quantify the minimum detectable flux signal based on noise characteristics.

Moved here from diive (`flux/hires/fluxdetectionlimit.py`). The detection limit
is derived from the cross-covariance function in the far tail of the lag search,
so it is the same computation as lag detection read a different way, and it
shares `MaxCovariance` and the wind rotation with it.

Part of the dyco package: https://github.com/holukas/dyco

Based on Langford et al. (2015). Parts of this code are based on Striednig et al. (2020).

References:

    (CAM98) Campbell, G. S., & Norman, J. M. (1998). An Introduction to Environmental Biophysics.
        Springer New York. https://doi.org/10.1007/978-1-4612-1626-1

    (LAN15) Langford, B., Acton, W., Ammann, C., Valach, A., & Nemitz, E. (2015). Eddy-covariance
        data with low signal-to-noise ratio: Time-lag determination, uncertainties and limit of
        detection. Atmospheric Measurement Techniques, 8(10), 4197–4213.
        https://doi.org/10.5194/amt-8-4197-2015

    (MAM16) Mammarella, I., Peltola, O., Nordbo, A., Järvi, L., & Rannik, Ü. (2016). Quantifying
        the uncertainty of eddy covariance fluxes due to the use ofdifferent software packages
        and combinations of processing steps in two contrasting ecosystems. Atmospheric Measurement
        Techniques, 9(10), 4915–4933. https://doi.org/10.5194/amt-9-4915-2016

    (NEM18) Nemitz, E., Mammarella, I., Ibrom, A., Aurela, M., Burba, G. G., Dengel, S., Gielen, B., Grelle,
        A., Heinesch, B., Herbst, M., Hörtnagl, L., Klemedtsson, L., Lindroth, A., Lohila, A., McDermitt,
        D. K., Meier, P., Merbold, L., Nelson, D., Nicolini, G., … Zahniser, M. (2018). Standardisation of
        eddy-covariance flux measurements of methane and nitrous oxide. International Agrophysics, 32(4),
        517–549. https://doi.org/10.1515/intag-2017-0042

    (SAB18) Sabbatini, S., Mammarella, I., Arriga, N., Fratini, G., Graf, A., Hörtnagl, L., Ibrom, A.,
        Longdoz, B., Mauder, M., Merbold, L., Metzger, S., Montagnani, L., Pitacco, A., Rebmann, C.,
        Sedlák, P., Šigut, L., Vitale, D., & Papale, D. (2018). Eddy covariance raw data processing
        for CO2 and energy fluxes calculation at ICOS ecosystem stations. International Agrophysics,
        32(4), 495–515. https://doi.org/10.1515/intag-2017-0043

    (STR20) Striednig, M., Graus, M., Märk, T. D., & Karl, T. G. (2020). InnFLUX – an open-source
        code for conventional and disjunct eddy covariance analysis of trace gas measurements:
        An urban test case. Atmospheric Measurement Techniques, 13(3), 1447–1465.
        https://doi.org/10.5194/amt-13-1447-2020
        Source code: https://www.atm-phys-chem.at/innflux/
        Source code: https://git.uibk.ac.at/acinn/apc/innflux


"""
import math
import numbers
import time

import matplotlib
import pandas as pd
from pandas import DataFrame

from dyco._vendor.console import info
from dyco.maxcov import MaxCovariance
from dyco.rotation import WindDoubleRotation, reynolds_decomposition


def air_temp_from_sonic_temp(sonic_temp: pd.Series, h2o: pd.Series) -> pd.Series:
    """Calculate air temperature from sonic temperature and water vapor concentration.

    Inlined from diive's `dv.variables.air_temp_from_sonic_temp` rather than
    depended on - it is a single formula, and it remains public API in diive
    where it has its own test and example.

    Based on the code in:
    Striednig, M., Graus, M., Maerk, T. D., & Karl, T. G. (2020). InnFLUX - an
        open-source code for conventional and disjunct eddy covariance analysis
        of trace gas measurements: An urban test case. Atmospheric Measurement
        Techniques, 13(3), 1447-1465. https://doi.org/10.5194/amt-13-1447-2020

    Args:
        sonic_temp: Sonic temperature data in Kelvin.
        h2o: Water vapor concentration in the air in mol mol-1.

    Returns:
        Air temperature data in Kelvin.
    """
    ta = sonic_temp / (1 + 0.32 * h2o)
    ta.name = "TA_SONIC"
    return ta


class FluxDetectionLimit:
    """Calculates the flux detection limit from high-resolution eddy covariance data.

    This class implements the method described by Langford et al. (2015) to
    determine the flux detection limit (FDL) and signal-to-noise ratio for
    a given scalar flux (e.g., N2O, CH4).

    The FDL is calculated based on the noise in the cross-covariance function
    at large time lags, which are assumed to be uncorrelated with the turbulent
    flux. The flux noise is quantified as the Root Mean Square Error (RMSE)
    of the covariances in these "noise" windows (e.g., +/- 160-180s). The FDL
    is then defined as 3 * RMSE.

    The implementation follows the methodology used in Striednig et al. (2020)
    for the noise RMSE calculation (based on LAN15, eq. 9) and Sabbatini et al.
    (2018) for the flux conversion factor.

    The class processes a pandas DataFrame of high-resolution (e.g., 10Hz or
    20Hz) eddy covariance data.

    The main workflow executed by the `run()` method is:
    1.  Convert sonic temperature to air temperature.
    2.  Calculate turbulent fluctuations (primes) via 2D wind rotation.
    3.  Compute the full cross-covariance function between vertical wind (w')
        and the scalar (c') over the specified `lag_range`.
    4.  Convert covariance to physical flux units using the ideal gas law
        and the mean air temperature and dry air pressure.
    5.  Calculate the noise RMSE from the covariance function at the large
        lag windows defined by `lag_range` and `noise_range`.
    6.  Determine the FDL (3 * RMSE).
    7.  Find the lag of maximum absolute covariance, and read the flux at
        `default_lag` as the signal for the signal-to-noise ratios.

    Parameters
    ----------
    df : pd.DataFrame
        High-resolution time series data, e.g., a half-hourly eddy covariance
        raw data file.
    u_col : str
        Column name for the u component of wind speed (units: m s-1).
    v_col : str
        Column name for the v component of wind speed (units: m s-1).
    w_col : str
        Column name for the w component of wind speed (units: m s-1).
    c_col : str
        Column name for the scalar concentration. Examples: N2O in nmol mol-1,
        CO2 in umol mol-1, CH4 in nmol mol-1. The scalar units directly affect
        the output flux units. For example:
        - If c_col is N2O in nmol mol-1, FDL will be in nmol m-2 s-1
        - If c_col is CO2 in umol mol-1, FDL will be in umol m-2 s-1
    ts_col : str
        Column name for the sonic temperature (units: K).
        Note: Internally converted to air temperature using H2O content.
    h2o_col : str
        Column name for the H2O mole fraction (units: mol mol-1).
    press_col : str
        Column name for the air pressure (units: Pa).
    default_lag : float or None
        Default time lag (units: seconds) for calculating the flux "signal"
        in the signal-to-noise ratio. Positive lag means the scalar arrives
        after the vertical wind (e.g., due to inlet tube transit time).
        None computes the detection limit only; the signal keys are then None,
        and `flux_at_lag()` reads the flux at any lag afterwards.
    noise_range : int
        Width of the time window (units: seconds) at the edges of lag_range
        used to calculate noise. Example: if lag_range=[-180, 180] and
        noise_range=20, then noise is calculated from -180 to -160s and
        +160 to +180s.
    lag_range : list
        Two-element list [min, max] specifying the time lag range
        (units: seconds) for the covariance calculation.
        Typical default: [-180, 180] (calculates all covariances from -180s to +180s).
        The noise windows for FDL calculation are defined at the edges of this range.
        For example, with [-180, 180] and noise_range=20, noise is calculated from
        -180 to -160s and +160 to +180s (LAN15). The range must run from a
        negative to a positive lag, otherwise one noise window would sit on the
        flux; a ValueError says so.
    lag_stepsize : int
        Step size (units: records) for iterating through the covariance lag
        search. Important: This parameter is in RECORDS, not seconds.
        The relationship between lag_range (seconds) and lag_stepsize (records)
        depends on sampling_rate. For example:
        - If sampling_rate=10 Hz and lag_stepsize=10 records, then each step = 1 second
        - If sampling_rate=20 Hz and lag_stepsize=20 records, then each step = 1 second
    sampling_rate : int
        Data sampling rate (units: Hz), e.g., 10 for 10Hz or 20 for 20Hz.
    title_covariance_plot : str or None
        Text appended to the title of the figure from `get_fig_cov()`.
    verbose : bool
        Print the run time, noise RMSE and detection limit. Default True.

    Attributes
    ----------
    hires_df : pd.DataFrame
        A copy of the input DataFrame with added calculated columns including:
        'e' (partial pressure of water vapor in Pa),
        'pd' (dry air partial pressure in Pa),
        '{w_col}_TURB' (turbulent fluctuation of w),
        '{c_col}_TURB' (turbulent fluctuation of scalar c).
    cov_df : pd.DataFrame
        DataFrame holding cross-covariance calculation results, including the
        columns 'shift' (lag in records, positive when the scalar arrives after
        the wind), 'cov' (covariance), 'cov_abs' (its absolute value) and
        'cov_flux' (converted to physical flux units using ideal gas law).
    results : dict
        Dictionary containing final results after `run()` is called. Keys:
        - 'flux_detection_limit' : Detection limit (3 * RMSE), units depend
          on scalar units (e.g., nmol m-2 s-1 for N2O in nmol mol-1)
        - 'flux_noise_rmse' : Root mean square of noise, same units as flux
        - 'cov_max_ix' : Index of maximum covariance in cov_df
        - 'cov_max_shift' : Lag in records where max covariance occurs
        - 'flux_signal_at_default_lag' : Flux at the specified default_lag,
          same units as flux_detection_limit
        - 'flux_signal_at_cov_max_lag' : Flux at the lag with max covariance,
          same units as flux_detection_limit
        - 'signal_to_noise' : Ratio (unitless) of signal to noise RMSE
        - 'signal_to_detection_limit' : Ratio (unitless) of signal to FDL

    See Also
    --------
    MaxCovariance : Detect time lag between wind and scalars using covariance.
    WindDoubleRotation : Coordinate rotation and tilt correction for eddy covariance measurements.
    reynolds_decomposition : Compute turbulent fluctuation x' = x - mean(x).

    Example
    -------
    See `examples/detectionlimit.py` for complete examples
    demonstrating flux detection limit calculation from high-resolution eddy covariance data.

    References
    ----------
    (LAN15) Langford, B., et al. (2015). Eddy-covariance
        data with low signal-to-noise ratio: Time-lag determination,
        uncertainties and limit of detection. Atmospheric Measurement
        Techniques, 8(10), 4197–4213.
        https://doi.org/10.5194/amt-8-4197-2015

    (SAB18) Sabbatini, S., et al. (2018). Eddy covariance raw data
        processing for CO2 and energy fluxes calculation at ICOS
        ecosystem stations. International Agrophysics, 32(4), 495–515.
        https://doi.org/10.1515/intag-2017-0043

    (STR20) Striednig, M., et al. (2020). InnFLUX – an open-source
        code for conventional and disjunct eddy covariance analysis of
        trace gas measurements: An urban test case. Atmospheric
        Measurement Techniques, 13(3), 1447–1465.
        https://doi.org/10.5194/amt-13-1447-2020
    """

    def __init__(
            self,
            df: pd.DataFrame,
            u_col: str,
            v_col: str,
            w_col: str,
            c_col: str,
            ts_col: str,
            h2o_col: str,
            press_col: str,
            default_lag: float | None,
            noise_range: int,
            lag_range: list,
            lag_stepsize: int,
            sampling_rate: int,
            title_covariance_plot: str = None,
            verbose: bool = True,
    ):
        """Set up flux detection-limit estimation. See the class docstring."""
        # The noise windows are the ends of lag_range. A range that does not reach
        # past zero on both sides puts one of them on the flux itself, and a noise
        # window as wide as one side reaches into it: either way the detection
        # limit is read partly from the signal, with nothing to show for it.
        lag_lo, lag_hi = lag_range
        if not lag_lo < 0 < lag_hi:
            raise ValueError(
                f"lag_range must run from a negative to a positive lag, got {lag_range}: "
                f"the noise windows are its two ends, far from the flux on both sides.")
        if not 0 < noise_range < min(-lag_lo, lag_hi):
            raise ValueError(
                f"noise_range ({noise_range} s) must be positive and smaller than each "
                f"side of lag_range {lag_range}.")
        self.hires_df = df.copy()
        self.u_col = u_col
        self.v_col = v_col
        self.w_col = w_col
        self.c_col = c_col
        self.ts_col = ts_col
        self.h2o_col = h2o_col
        self.press_col = press_col
        self.noise_range = noise_range
        self.lag_range = lag_range
        self.default_lag = default_lag
        self.lag_stepsize = lag_stepsize
        self.sampling_rate = sampling_rate
        self.title_covariance_plot = title_covariance_plot
        self.verbose = verbose

        # Convert s to number or records,
        #   e.g. for 20Hz data (20 records = 1s): 180s * 20 = 3600 records
        self.lag_from = lag_range[0] * sampling_rate
        self.lag_to = lag_range[1] * sampling_rate

        # Calculate e
        # e = partial pressure of water vapor (Pa) = H2O mole fraction (mol mol-1) * air pressure (Pa)
        # CAM98, p39 eq.(3.5), solve for e
        self.e_col = 'e'
        self.hires_df[self.e_col] = self.hires_df[self.h2o_col] * self.hires_df[self.press_col]

        # Calculate pd
        # pd = dry air partial pressure (Pa)
        # pd (in Pa) = pa (in Pa) - e (in Pa)
        # SAB18, p513
        self.pd_col = 'pd'
        self.hires_df[self.pd_col] = self.hires_df[self.press_col] - self.hires_df['e']

        # Convert sonic temperature (K) to air temperature (K)
        self.hires_df[ts_col] = (
            air_temp_from_sonic_temp(sonic_temp=self.hires_df[ts_col], h2o=self.hires_df[self.h2o_col]))

        # New variables
        self.cov_df = pd.DataFrame()
        self.results = {}
        self.fig_cov = None
        self.maxcov_obj = None  # Stores covariances from lag search

    def run(self):
        """Run the flux detection-limit estimation."""

        # Calculate turbulent fluctuations
        self.hires_df = self._turbulent_fluctuations(df=self.hires_df)

        # Calculate covariances
        self.cov_df, cov_max_ix, cov_max_shift, self.maxcov_obj = self._crosscovariance()

        # Flux conversion factor
        # Convert covariance to flux units
        self.cov_df['cov_flux'] = self.cov_df['cov'].copy()
        self.cov_df = self._flux_conversion_factor(cov_df=self.cov_df)

        # Flux detection limit
        flux_detection_limit, flux_noise_rmse = self._flux_detection_limit(cov_df=self.cov_df.copy())

        # Calculate flux at default lag, this is the "signal" in the signal-to-noise ratio.
        # None leaves the signal out, for a caller that only knows the lag later
        # (see flux_at_lag).
        flux = None if self.default_lag is None else self.flux_at_lag(self.default_lag)

        # Get flux at maximum covariance lag
        if isinstance(cov_max_ix, numbers.Integral):
            maxcovflux = self.cov_df.iloc[cov_max_ix]['cov_flux']
        else:
            raise Exception("No default lag covariance found")

        self.results = {
            'flux_detection_limit': flux_detection_limit,
            'flux_noise_rmse': flux_noise_rmse,
            'cov_max_ix': cov_max_ix,
            'cov_max_shift': cov_max_shift,
            'flux_signal_at_default_lag': flux,
            'flux_signal_at_cov_max_lag': maxcovflux,
            'signal_to_noise': abs(flux) / flux_noise_rmse if flux is not None else None,
            'signal_to_detection_limit': abs(flux) / flux_detection_limit if flux is not None else None
        }

    def flux_at_lag(self, lag_s: float) -> float:
        """Flux read off the cross-covariance at *lag_s* seconds; call after `run()`.

        Positive means the scalar arrives after the wind, the same convention as
        `default_lag` and as the PWB lag, so a lag detected there can be passed
        straight in.
        """
        # MaxCovariance puts a scalar that arrives after the wind at a positive shift.
        # Rounding absorbs float error in s * Hz.
        shift = round(lag_s * self.sampling_rate)
        at_lag = self.cov_df.loc[self.cov_df['shift'] == shift, 'cov_flux']
        if at_lag.empty:
            raise ValueError(
                f"lag {lag_s} s is shift {shift} records, which is not on the lag grid "
                f"({self.lag_from} to {self.lag_to} in steps of {self.lag_stepsize} records). "
                f"Choose a lag on that grid, or a smaller lag_stepsize.")
        return float(at_lag.values[0])

    def get_detection_limit(self) -> dict:
        """Return the estimated flux detection limit."""
        return self.results

    def get_fig_cov(self) -> matplotlib.figure.Figure:
        """Return the covariance diagnostic figure."""
        fig_cov = self.maxcov_obj.plot_scatter_cov(
            title=f"Covariance vs time lag {self.title_covariance_plot}", showplot=False)
        return fig_cov

    def _turbulent_fluctuations(self, df: pd.DataFrame) -> pd.DataFrame:
        wr = WindDoubleRotation(u=df[self.u_col],
                                v=df[self.v_col],
                                w=df[self.w_col])
        primes_df = pd.DataFrame({
            f'{self.w_col}_TURB': reynolds_decomposition(wr.w2),
            f'{self.c_col}_TURB': reynolds_decomposition(df[self.c_col]),
        })
        df = pd.concat([df, primes_df], axis=1)
        return df

    def _crosscovariance(self) -> tuple[pd.DataFrame, int, int, MaxCovariance]:
        start_time = time.perf_counter()
        maxcov_obj = MaxCovariance(
            df=self.hires_df,
            var_reference=f"{self.w_col}_TURB",
            var_lagged=f"{self.c_col}_TURB",
            lgs_winsize_from=self.lag_from,
            lgs_winsize_to=self.lag_to,
            shift_stepsize=self.lag_stepsize,
            segment_name="cross-covariance",
        )

        maxcov_obj.run()
        end_time = time.perf_counter()
        elapsed_time_seconds = end_time - start_time
        if self.verbose:
            info(f"Time needed for covariance calculation: {elapsed_time_seconds:0.3f}s")
        cov_df, props_peak_auto = maxcov_obj.get()

        # Max covariance
        foundlag = cov_df.loc[cov_df['flag_peak_max_cov_abs']]

        # Location of max abs covariance
        cov_max_ix = foundlag.index[0]
        cov_max_shift = foundlag.iloc[0]['shift']

        return cov_df, cov_max_ix, cov_max_shift, maxcov_obj

    def _flux_conversion_factor(self, cov_df: DataFrame) -> pd.DataFrame:
        """Calculate flux conversion factor

        Conversion from covariances to flux units:
            see SAB18, eq.(16) (or similar: MAM16, eq.(2))
            flux_conversion_factor = 1 / ( (R Ta) / pd)   [same as: 1 / ( (Ta / pd) * R )]
                Ta  ... air temperature, K
                R   ... Universal gas constant, m3 Pa K-1 mol-1
                pd  ... dry air partial pressure, Pa

        From STR20 (innFLUX script):
            Calculate conversion factor for flux in nmol m-2 s-1
            flux_conversion_factor = 1/( (T_mean/273.15) * (1013.25/p_mean) * 22.414e-3 );
            They achieve the identical result (the dry air molar density),
            but the innFLUX script expresses the conversion as a ratio against STP reference
            conditions, while SAB18 uses the direct P/RT Ideal Gas Law formulation.

        """
        # Flux conversion factor, SAB18, eq.(16)
        R = 8.31446261815324  # Universal gas constant, m3 Pa K-1 mol-1
        ta_mean = self.hires_df[self.ts_col].mean()  # K
        pd_mean = self.hires_df[self.pd_col].mean()  # Pa
        flux_conversion_factor = 1 / ((R * ta_mean) / pd_mean)
        cov_df['cov_flux'] = cov_df['cov'] * flux_conversion_factor
        return cov_df

    def _flux_detection_limit(self, cov_df: DataFrame):
        """Calculate flux detection limit

        Flux noise criterion: RMSE of the covariance in the two noise windows,
        each `noise_range` seconds wide, at the ends of `lag_range` (e.g.
        +/- 160-180 s), see LAN15

        """

        # e.g. lag_range [-180, 180] at 20 Hz:
        # from -3600 to -3200 (-180 to -160s)
        # from +3200 to +3600 (+160 to +180s)

        # 20s @ 20Hz = 20 * 20 = 400 records
        # 20s @ 10Hz = 20 * 10 = 200 records
        winsize = self.noise_range * self.sampling_rate

        # LAN15, eq.(9)
        # Left lag window
        cov_df_left = cov_df.loc[(cov_df['shift'] >= self.lag_from) & (cov_df['shift'] <= self.lag_from + winsize)]
        # Right lag window
        cov_df_right = cov_df.loc[
            (cov_df['shift'] >= abs(self.lag_to) - winsize) & (cov_df['shift'] <= abs(self.lag_to))]

        # Implementation by STR20
        # their source code: https://git.uibk.ac.at/acinn/apc/innflux/-/blob/master/innFLUX_step1.m#L435
        #   flux_noise_rmse = sqrt(0.5 *
        #       (  nanstd(cov_wc(idx_left))^2 + nanmean(cov_wc(idx_left))^2 +
        #       nanstd(cov_wc(idx_right))^2 + nanmean(cov_wc(idx_right))^2)  )
        #       * flux_conversion_factor;
        flux_noise_rmse = math.sqrt(0.5 * (
                (cov_df_left['cov_flux'].std()) ** 2 + (cov_df_left['cov_flux'].mean()) ** 2 +
                (cov_df_right['cov_flux'].std()) ** 2 + (cov_df_right['cov_flux'].mean()) ** 2
        ))
        flux_detection_limit = flux_noise_rmse * 3
        if self.verbose:
            info(f"Flux noise RMSE: {flux_noise_rmse}")
            info(f"Flux detection limit: {flux_detection_limit}")
        return flux_detection_limit, flux_noise_rmse

