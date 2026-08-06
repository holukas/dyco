# Migrating from v2

v3 carries **one** lag-detection method: pre-whitening block-bootstrap. The covariance-maximization
method that `dyco` shipped up to v2 was removed, and with it the v2 command-line interface.

`pip install dyco==2.0.3` still has the old method if you need it.

## What replaces what

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

## Why the old method went

It pooled detections into a daily median lookup table and normalized toward a target lag, with no
per-detection confidence — which is exactly what low-SNR gases such as N₂O and CH₄ need. PWB gives
each detection a 95% interval, and PWBOPT uses that interval to decide whether the detection can be
trusted at all.

<!-- TODO: link to the CHANGELOG entry for 3.0.0 once it is dated, rather than
     restating it here. -->

## The published paper describes v1.1.2

> Hörtnagl, L., 2021. DYCO: A Python package to dynamically detect and compensate for time lags in
> ecosystem time series. *Journal of Open Source Software* 6(62), 2575.
> <https://doi.org/10.21105/joss.02575>

That paper documents the covariance-maximization method, in the version released for it on
16 Jun 2021. If you arrived here from the paper, the algorithm described there is the one this page
is about migrating *away* from — the two describe different algorithms, and nothing in the paper
applies to a v3 run. `pip install dyco==2.0.3` still carries the old method if you need it.

The pre-whitening block-bootstrap method has its own manuscript.

<!-- TODO: cite the PWB manuscript here once it is published
     (holukas/ms_fluxnet_ch4_n2o_timelag). -->
