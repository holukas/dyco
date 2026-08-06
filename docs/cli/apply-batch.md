# dyco apply-batch

Remove lags listed in an existing `tlag_results.csv`. Detects nothing — it reads the numbers another
run produced and shifts columns by them. Step 2 of the two-step route, after
[`dyco pwb-batch`](pwb-batch.md).

Also available as the standalone `dyco-apply-batch`.

`--scalar LABEL:column` here reads the lag recorded under `LABEL` and applies it to `column`, so one
gas's detected lag can be removed from another gas's data.

## All options

```{eval-rst}
.. argparse::
   :module: dyco.apply_tlag
   :func: _build_parser
   :prog: dyco apply-batch
```
