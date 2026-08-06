# PWB_REFERENCE_RFLUX: PRODUCE THE REFERENCE VALUES FROZEN IN test_pwb_reference.py
# ==================================================================================
#
# Runs the original RFlux v3.2.0 `tlag_detection()` on the two fixtures written by
# pwb_reference_generate.py and prints everything the Python test asserts against.
#
# Usage (adjust the two paths, then paste the output into test_pwb_reference.py):
#
#   Rscript pwb_reference_rflux.R
#
# Needs: zoo, boot, HDInterval, bayestestR, egcm, pracma, parallel, and a checkout
# of RFlux (https://github.com/icos-etc/RFlux) for R/tlag_detection.R.
#
# Only the deterministic outputs are frozen in the test. `pwb` and the HDI come
# out of the block bootstrap, whose RNG stream differs between R and numpy, so
# they are reported here for reference but asserted only loosely.

RFLUX_SRC <- "F:/Sync/luhk_work/dev-data/dyco-data/references/RFlux-master-v3.2.0/R/tlag_detection.R"
# Run this from the directory that holds the fixtures, or point FIXTURE_DIR at it.
FIXTURE_DIR <- "."

suppressMessages({
  library(zoo); library(boot); library(HDInterval); library(bayestestR)
  library(egcm); library(parallel); library(pracma)
})
source(RFLUX_SRC)

MFREQ <- 20

# The third case is the bundled real CH-LAE half hour. It is not a fixture file:
# the test derives it from examples/data/. To reproduce its constants, write the
# same input out first, then rerun this script --
#
#   uv run python -c "import sys; sys.path.insert(0, 'tests'); \
#     from test_pwb_reference import _load_real_chunk; \
#     _load_real_chunk().to_csv('tests/data/real_chunk.csv', index=False, \
#                               float_format='%.6f', lineterminator='\n')"
#
# real_chunk.csv is deliberately not committed -- it is derived data, and the
# test guards the derivation with a checksum instead.
CASES <- c("pwb_reference_stationary.csv.gz", "pwb_reference_differencing.csv.gz",
           "real_chunk.csv")

for (fx in CASES) {
  if (!file.exists(file.path(FIXTURE_DIR, fx))) {
    cat("\n########", fx, "absent, skipped (see the comment above) ########\n")
    next
  }
  d <- read.csv(file.path(FIXTURE_DIR, fx))
  set.seed(42)
  cat("\n########", fx, "########\n")

  res <- tlag_detection(scalar_var = d$scalar, tsonic_var = d$tsonic, w_var = d$w,
                        mfreq = MFREQ, LAG.MAX = MFREQ * 10, lws = 0, uws = 5,
                        Rboot = 99, plot.it = FALSE)
  for (nm in c("mcw", "pww", "cor_pww", "cov_mcw", "pwb", "pwb_lci", "pwb_uci", "cov_pwb")) {
    cat(sprintf("%-10s %s\n", nm, format(res[[nm]], digits = 12)))
  }

  # Deterministic intermediates: the unit-root decision and the AR fits.
  set <- na.omit(cbind(na.approx(d$scalar, na.rm = FALSE),
                       na.approx(d$tsonic, na.rm = FALSE),
                       na.approx(d$w,      na.rm = FALSE)))
  pv <- c(bvr.test(set[, 1])$p.val, bvr.test(set[, 2])$p.val, bvr.test(set[, 3])$p.val)
  differenced <- any(pv >= 0.01)
  cat(sprintf("bvr p (scalar,tsonic,w) %s -> differenced=%s\n",
              paste(format(pv, digits = 4), collapse = ", "), differenced))

  x <- set[, 1]; y <- set[, 2]; z <- set[, 3]
  if (differenced) { x <- diff(x); y <- diff(y); z <- diff(z) }
  om <- floor(10^2 * log10(length(x)))
  arx <- ar(x, aic = TRUE, order.max = om)
  ary <- ar(y, aic = TRUE, order.max = om)
  arz <- ar(z, aic = TRUE, order.max = om)
  cat(sprintf("order.max=%d  AR orders scalar=%d w=%d tsonic=%d\n",
              om, arx$order, arz$order, ary$order))
  cat("scalar phi[1] ", format(arx$ar[1], digits = 12), "\n")
  cat("w      phi[1] ", format(arz$ar[1], digits = 12), "\n")
  cat("tsonic phi[1] ", format(ary$ar[1], digits = 12), "\n")
}
