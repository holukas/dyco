# Migrating from v2

v3 carries **one** lag-detection method: pre-whitening block-bootstrap. The covariance-maximization
method that `dyco` shipped up to v2 was removed, and with it the v2 command-line interface.

`pip install dyco==2.0.3` still has the old method if you need it.

## Replacement for the v2 workflow

| v2 | v3 |
|---|---|
| `dyco` with short flags (`-lsw`, `-lsi`, `-lsf`, …) | [`dyco detect-remove`](cli/detect-remove.md), or [`dyco tui`](tui.md) |
| Covariance maximization | Pre-whitening block-bootstrap |
| Daily median lookup table, normalized toward a target lag | PWBOPT S1/S2/S3, per chunk, driven by each detection's own uncertainty interval |
| Lag counted in **records** | Lag expressed in **seconds** |
| Depends on [diive](https://github.com/holukas/diive) | Standalone |

There is **no flag-for-flag mapping**. The two methods take different parameters, so an old command
line cannot be mechanically translated. `dyco cm` and old v2 flags are still recognised by the
dispatcher, only so that an old command gets a pointer instead of a parse error.

## Reasons for removing covariance maximization

The v2 method pooled detections into a daily median lookup table and normalized them toward a target
lag. It gave no confidence measure for an individual detection, and low-SNR gases such as N₂O and
CH₄ need one. PWB gives each detection a 95% interval, and PWBOPT uses that interval to decide
whether the detection can be trusted at all.

The v3.0.0 entry in the
[changelog](https://github.com/holukas/dyco/blob/main/CHANGELOG.md) lists everything that was
removed.

## The published paper describes v1.1.2

> Hörtnagl, L., 2021. DYCO: A Python package to dynamically detect and compensate for time lags in
> ecosystem time series. *Journal of Open Source Software* 6(62), 2575.
> <https://doi.org/10.21105/joss.02575>

That paper documents the covariance-maximization method, in the version released for it on
16 Jun 2021. This is the method that v3 removed, so the algorithm in the paper is not the one a v3
run uses.

The pre-whitening block-bootstrap method has its own manuscript.

<!-- TODO: cite the PWB manuscript here once it is published
     (holukas/ms_fluxnet_ch4_n2o_timelag). -->
