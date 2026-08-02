# dyco pwb-batch

Detect lags only, across files that are **already** split into averaging periods. Writes
`tlag_results.csv` and stops — nothing is shifted and no corrected raw data is produced. Pair it with
[`dyco apply-batch`](apply-batch.md).

Also available as the standalone `dyco-pwb-batch`.

:::{important}
Input files must already be **wind-rotation corrected** (double rotation or planar fit, e.g. EddyPro
"Advanced" rotated output). This command does not rotate; a non-zero mean `W` corrupts the
cross-correlation. [`dyco detect-remove`](detect-remove.md) handles rotation itself, and is the right
choice unless your files are already split.
:::

## All options

```{eval-rst}
.. argparse::
   :module: dyco.pwb
   :func: _build_parser
   :prog: dyco pwb-batch
```
