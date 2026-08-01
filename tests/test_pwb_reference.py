"""
TEST_PWB_REFERENCE: PWB AGAINST THE ORIGINAL R IMPLEMENTATION
==============================================================

Pins dyco's pre-whitening chain to the numbers RFlux v3.2.0's
`tlag_detection()` produces on the same input. Everything asserted tightly here
is deterministic given the fixture: the Breitung unit-root decision, the AIC AR
order, the AR coefficients, the pre-whitened CCF peak and the raw
cross-covariance. Only the block bootstrap differs between the two
implementations -- R and numpy do not share an RNG stream -- so the mode and the
HDI are checked loosely.

The two fixtures and the R script that produced these constants live in
`tests/data/` (`pwb_reference_generate.py`, `pwb_reference_rflux.R`). The
second fixture exists to exercise the differencing branch, which real turbulent
data almost never triggers and which no other test reaches.

Part of the dyco package: https://github.com/holukas/dyco
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dyco.pwb import PreWhiteningBootstrap

_DATA = Path(__file__).parent / 'data'
_HZ = 20
_LAG_MAX_S = 10.0
_TRUE_LAG_RECORDS = 169  # built into the fixtures by pwb_reference_generate.py

# Output of `Rscript pwb_reference_rflux.R`, RFlux v3.2.0 + R 4.5.3.
# Regenerating the fixtures invalidates these; rerun the script and update.
_R = {
    'stationary': dict(
        differenced=False,
        ar_orders=(5, 1, 3),                 # scalar, w, tsonic
        phi1=(0.831687873522, 0.908826103964, 0.778652566656),
        pww=169, cor_pww=0.300454145434,
        mcw=169, cov_mcw=1.82854726566,
        pwb=170, hdi=(167, 171), cov_pwb=1.67054592207,
    ),
    'differencing': dict(
        differenced=True,
        ar_orders=(67, 56, 56),
        phi1=(-0.157072443378, -0.070821662824, -0.229536829075),
        pww=169, cor_pww=0.295978189788,
        mcw=169, cov_mcw=1.78217007168,
        pwb=170, hdi=(167, 171), cov_pwb=1.62402236698,
    ),
}

# R printed 12 significant digits and dyco reproduced every one of them, so a
# 1e-9 relative band is loose enough to survive BLAS/FFT rounding differences
# and still tight enough that any change of substance fails.
_RTOL = 1e-9


def _load(case: str) -> pd.DataFrame:
    return pd.read_csv(_DATA / f'pwb_reference_{case}.csv.gz')


class TestPwbAgainstRFlux(unittest.TestCase):
    """Deterministic parity with RFlux v3.2.0 on both unit-root branches."""

    @classmethod
    def setUpClass(cls):
        # One run per fixture: the bootstrap dominates the runtime.
        cls.data = {}
        cls.pwb = {}
        for case in _R:
            df = _load(case)
            p = PreWhiteningBootstrap(
                df=df, var_w='w', var_scalar='scalar', var_tsonic='tsonic',
                hz=_HZ, lag_max_s=_LAG_MAX_S, n_bootstrap=99, wdt=5,
                random_state=42)
            p.run()
            cls.data[case] = df
            cls.pwb[case] = p

    def _close(self, got, ref, what):
        self.assertAlmostEqual(got, ref, delta=abs(ref) * _RTOL,
                               msg=f'{what}: dyco {got!r} vs R {ref!r}')

    def _series(self, case):
        """The three aligned series, differenced when the fixture calls for it."""
        df = self.data[case]
        s, w, t = (df[c].to_numpy(float) for c in ('scalar', 'w', 'tsonic'))
        if _R[case]['differenced']:
            s, w, t = np.diff(s), np.diff(w), np.diff(t)
        return s, w, t

    def test_the_unit_root_decision_matches_r(self):
        # dyco reads a hardcoded 1% critical value where R interpolates egcm's
        # quantile table; the two must reach the same verdict.
        for case, ref in _R.items():
            with self.subTest(case=case):
                df = self.data[case]
                stationary = [PreWhiteningBootstrap._is_stationary(
                    df[c].to_numpy(float)) for c in ('scalar', 'w', 'tsonic')]
                self.assertEqual(not all(stationary), ref['differenced'])

    def test_ar_orders_match_r(self):
        for case, ref in _R.items():
            with self.subTest(case=case):
                orders = self.pwb[case].results['ar_orders']
                self.assertEqual(
                    (orders['scalar'], orders['w'], orders['tsonic']),
                    ref['ar_orders'])

    def test_first_ar_coefficient_matches_r(self):
        # Guards the Levinson-Durbin recursion and the biased ACF against R's
        # Yule-Walker fit.
        for case, ref in _R.items():
            s, w, t = self._series(case)
            for name, x, expected in zip(('scalar', 'w', 'tsonic'), (s, w, t),
                                         ref['phi1']):
                with self.subTest(case=case, series=name):
                    phi, _ = self.pwb[case]._fit_ar_model(x)
                    self._close(float(phi[0]), expected, f'{case}/{name} phi1')

    def test_prewhitened_ccf_peak_matches_r(self):
        # R: tl_pww / cor_pww -- the whole pre-whitening chain in two numbers.
        for case, ref in _R.items():
            with self.subTest(case=case):
                res = self.pwb[case].results
                self.assertEqual(res['tlag_pw_records'], ref['pww'])
                self._close(res['corr_pw'], ref['cor_pww'], f'{case} cor_pww')

    def test_covariance_maximisation_peak_matches_r(self):
        # R: mcw / cov_mcw, the CM estimate on linearly detrended raw data.
        for case, ref in _R.items():
            with self.subTest(case=case):
                ccov = self.pwb[case]._raw_ccov
                lag_max = (len(ccov) - 1) // 2
                self.assertEqual(int(np.argmax(np.abs(ccov))) - lag_max,
                                 ref['mcw'])
                self._close(float(np.max(np.abs(ccov))), ref['cov_mcw'],
                            f'{case} cov_mcw')

    def test_raw_covariance_comes_from_the_undifferenced_series(self):
        # Regression: the raw cross-covariance used to be built from the
        # DIFFERENCED series whenever the unit-root test fired, making cov_pwb a
        # covariance of increments -- two orders of magnitude too small, and
        # free to flip sign. R reads it off set[,1]/set[,3], the originals.
        # Indexed at R's own selected lag so the bootstrap RNG cannot matter.
        for case, ref in _R.items():
            with self.subTest(case=case):
                ccov = self.pwb[case]._raw_ccov
                lag_max = (len(ccov) - 1) // 2
                self._close(float(ccov[ref['pwb'] + lag_max]), ref['cov_pwb'],
                            f'{case} cov at R lag {ref["pwb"]}')

    def test_the_bootstrap_lag_and_hdi_bracket_the_true_lag(self):
        # Stochastic: R and numpy resample differently, so only the neighbourhood
        # is asserted. R got 170 with a 95% HDI of [167, 171] on both fixtures.
        for case in _R:
            with self.subTest(case=case):
                res = self.pwb[case].results
                self.assertAlmostEqual(res['tlag_records'], _TRUE_LAG_RECORDS,
                                       delta=3)
                self.assertLess(res['hdi_range_s'], 0.5)  # S1-reliable
                self.assertLessEqual(res['hdi_lo_s'] * _HZ, _TRUE_LAG_RECORDS)
                self.assertGreaterEqual(res['hdi_hi_s'] * _HZ, _TRUE_LAG_RECORDS)


class TestRollingMeanMatchesZoo(unittest.TestCase):
    """Centred rolling mean parity with R's zoo::rollapply(align="center")."""

    # rollapply(1:10, width=w, FUN=mean, fill=NA) in R 4.5.3 / zoo.
    _EXPECTED = {
        4: [None, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, None, None],
        5: [None, None, 3, 4, 5, 6, 7, 8, None, None],
        6: [None, None, 3.5, 4.5, 5.5, 6.5, 7.5, None, None, None],
    }

    def test_odd_and_even_widths_match_r(self):
        # An even width used to raise ValueError from a shape mismatch, so the
        # paper's hz/2+1 smoothing width was unusable at 10 Hz (= 6).
        x = np.arange(1, 11, dtype=float)
        for width, expected in self._EXPECTED.items():
            with self.subTest(width=width):
                got = PreWhiteningBootstrap._smooth_series(x, width)
                for i, want in enumerate(expected):
                    if want is None:
                        self.assertTrue(np.isnan(got[i]), f'position {i}')
                    else:
                        self.assertAlmostEqual(got[i], want, places=12)

    def test_the_row_wise_form_agrees_with_the_single_series_form(self):
        x = np.arange(1, 11, dtype=float)
        for width in self._EXPECTED:
            with self.subTest(width=width):
                rows = PreWhiteningBootstrap._smooth_rows(np.tile(x, (3, 1)),
                                                          width)
                one = PreWhiteningBootstrap._smooth_series(x, width)
                for r in rows:
                    np.testing.assert_allclose(r, one)

    def test_a_window_wider_than_the_series_yields_no_values(self):
        got = PreWhiteningBootstrap._smooth_series(np.arange(5.0), 9)
        self.assertTrue(np.all(np.isnan(got)))

    def test_a_width_below_one_record_is_rejected(self):
        df = _load('stationary').head(400)
        with self.assertRaises(ValueError):
            PreWhiteningBootstrap(df=df, var_w='w', var_scalar='scalar',
                                  var_tsonic='tsonic', hz=_HZ, wdt=0)


if __name__ == '__main__':
    unittest.main()
