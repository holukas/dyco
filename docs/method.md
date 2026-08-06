# The PWB method

How `dyco` finds a time lag, and why it does it this way.

<!-- This flowchart is also in README.md, deliberately -- GitHub renders mermaid
     natively and the README is the landing page. Edit both. -->

```{mermaid}
flowchart TD
    RAW["Raw EC file<br/>unrotated delimited text, plain or compressed<br/>--input-dir, --file-pattern"]
    RAW --> SPLIT["Cut into fixed-length chunks<br/>--chunk-seconds 1800<br/>boundaries snap to :00 / :30<br/>a remainder under --min-chunk-seconds writes nothing"]

    subgraph P1["Phase 1: detect (nothing is written yet)"]
        direction TB
        ROT["Double rotation<br/>in memory only, never reaches disk"]
        PW["Pre-whitening<br/>AR(p) filter, order chosen by AIC<br/>all three series differenced first if any fails the unit-root test"]
        BS["Block-bootstrap the CCF<br/>--n-bootstrap, --block-length<br/>4 combinations of W / T_SONIC"]
        EST["Lag for this chunk<br/>mode + 95% HDI<br/>a peak pinned to the window edge is a failure, not a lag"]
        ROT --> PW --> BS --> EST
    end

    SPLIT --> ROT

    EST --> PRE["Drop detections wider than --hdi-prefilter<br/>default 1 s, 0 disables<br/>stops S2 accepting a wide lag that merely sits close<br/>the unfiltered series is also computed and kept in the summary"]
    PRE --> OPT{"PWBOPT, per gas<br/>all chunks together, in time order"}
    OPT -->|"S1: HDI narrower than --hdi-thresh"| KEEP["trust the chunk's own lag"]
    OPT -->|"S2: close to the last trusted lag"| KEEP
    OPT -->|"S3: neither"| SUB["carry the last trusted lag forward<br/>--max-carry bounds how far"]

    KEEP --> GAP
    SUB --> GAP

    subgraph P3["Still no lag? fill the gap, best source first"]
        direction TB
        GAP{"any period still without a lag"}
        GAP -->|"its own trusted lag is within reach"| BFILL["fill backward from it<br/>--max-carry bounds this too"]
        GAP -->|"--scalar ...@lagfrom=CO2"| DONOR["take that period's lag from the other gas"]
        GAP -->|"nothing else left"| MED["median of this gas's raw detections<br/>(all of them rejected, a last resort)"]
    end

    BFILL --> APPLY
    DONOR --> APPLY
    MED --> APPLY

    subgraph P2["Phase 2: remove"]
        direction TB
        APPLY["Shift each scalar in the UNROTATED chunk<br/>by round(tlag * hz) records"]
        WRITE["Write one file per chunk<br/>header rows and column order intact<br/>--output-suffix picks .csv / .csv.gz / .zip"]
        APPLY --> WRITE
    end

    WRITE --> OUT["2_lag_removed/<br/>lag-compensated raw files<br/>-> flux software, lag maximization OFF"]
    EST -.-> CSV["1_lag_detection/<br/>summary CSV incl. {gas}_lag_source,<br/>checkpoints, plots if --save-plots"]
    WRITE -.-> LOG["log.txt<br/>what ran, and what it decided"]
```

Detection and removal are separate phases because PWBOPT cannot decide anything about one chunk until
it has seen the whole sequence.

## Why chunks

A multi-hour raw file is the wrong granularity for lag detection: both the rotation angles and the
tube delay drift over hours. Chunk boundaries snap to the wall-clock grid (:00 / :30) when the file
start time can be parsed, so downstream software bins them correctly. A file starting off-grid
produces a shorter leading chunk.

## The two phases

**Phase 1, detect.** Each chunk is read, double-rotated in memory, and passed to the PWB detector.
Nothing is written yet. Rotation happens in memory only; the rotated data never reach disk.

**PWBOPT.** With every chunk's raw detection in hand, the decision rule runs across the whole
sequence in temporal order. This is why the two phases are separate: the rule cannot decide anything
about one chunk until it has seen all of them.

**Phase 2, remove.** Each scalar in the **unrotated** chunk is shifted by `round(tlag * hz)` rows,
and the chunk is written out as its own file with the original header rows and column order intact.
A 6-hour input file yields up to twelve 30-minute output files. Compressed input is read
transparently, and `--output-suffix` decides how the chunks are written.

## Pre-whitening

Turbulent wind and trace-gas series are strongly autocorrelated, which broadens and distorts the peak
of the cross-correlation function and blurs the lag estimate. Pre-whitening fits an AR(p) filter
(order chosen by AIC) to each series so the residuals are approximately white noise, sharpening the
peak.

The AR order is allowed to go high: the ceiling is `floor(100 * log10(N))`, which for a 30-minute
20 Hz chunk is 455. Real chunks use a good deal of it. On the bundled CH-LAE half hour the three
fitted orders are 133, 87 and 312, which is what long-range turbulent autocorrelation costs to
capture.

### The differencing branch

An AR fit assumes the series is stationary, so each of the three series is first put through a
Breitung variance-ratio unit-root test. **If any one of them fails (p ≥ 0.01), all three are
first-differenced** before fitting. Stationarity of all three is required to use the originals; one
failure is enough to difference everything.

This is not the edge case it looks like. It fires on ordinary data, because sonic temperature drifts
with the diurnal cycle over half an hour: on the bundled CH-LAE chunk the test fails on `T_SONIC`
with p = 0.057, and all three series are differenced as a result. Expect to meet this branch.

Differencing feeds the **AR filters only**. The raw cross-covariance reported as `{gas}_cov_pwb` is
computed from the original, undifferenced series, which is what the reference implementation does.
Reading it off the differenced arrays instead turns it into a covariance of increments: on a drifting
record that is smaller by two orders of magnitude and can carry the opposite sign. dyco had that bug
and fixed it; the detected lag was never affected, only the reported covariance.

## Block-bootstrap

Block-bootstrap answers a question conventional lag detection cannot: *how sure are we?* Rather than
one cross-correlation over the whole averaging period, the method draws resampled series using
overlapping blocks that preserve local autocorrelation, finds the peak lag in each, and summarises
the resulting distribution as a mode (the estimate) plus a 95% highest-density interval (HDI). A
narrow HDI means repeated resampling keeps finding the same lag.

## The four combinations

Four cross-correlation combinations are evaluated per period, using either `W` or sonic temperature
`T_SONIC` as the reference and applying the AR filter to either the scalar or the reference. Because
`T_SONIC` and `W` are coupled through buoyant turbulence, the `T_SONIC` combinations often expose a
cleaner peak for gases whose direct scalar × `W` signal is weak. The combination with the highest
smoothed peak wins. **`T_SONIC` is required.**

## PWBOPT: S1, S2, S3

With every chunk's raw detection in hand, the S1/S2/S3 decision rule (Vitale et al. 2024, Section
2.3) runs across the whole sequence in temporal order. A chunk with a wide HDI has an untrustworthy
mode lag, and PWBOPT replaces it with the last lag that was trustworthy rather than accepting a
spurious value.

**Before the rule runs**, an optional pre-filter discards the worst detections outright: any lag
whose HDI range exceeds `--hdi-prefilter` (default `1.0` s) is set to `NaN`. This exists so that S2
cannot later accept a wide-uncertainty lag just because it happened to land near the previous one.
Set it to `0` to disable and get the paper's unfiltered behaviour.

Then, per chunk, in time order:

`S1_optimal`
: HDI range **below** `--hdi-thresh` (default `0.5` s). Repeated resampling keeps finding the same
  lag, so the chunk's own detection is accepted as-is.

`S2_optimal`
: HDI range at or above that threshold, **but** the lag is within `--dev-thresh` (default `0.5` s)
  of the preceding optimal lag. The detection is uncertain on its own terms, yet it agrees with a
  neighbour that was not — so it is accepted.

`S3_unreliable`
: Neither. The chunk's own lag is discarded and the last known optimal lag is carried forward.

`S3_expired`
: Neither, **and** the last optimal lag is further back than `--max-carry` periods, so it is no
  longer offered. Only occurs when that limit is set; see below.

### How far a lag may travel

The carry in S3 is unbounded in the paper: once an optimal lag exists, every later period that has
none inherits it, however distant. On a long run that means a single good half hour can supply days.

`--max-carry N` (**Max carry** in the TUI) caps the distance at N averaging periods, in **both**
directions — the forward carry above, and the backward fill below. Beyond it the period is left for
a donor gas or the median instead.

The limit has to cover both directions, or it does nothing. With detections either side of a long
unusable stretch, an unbounded backward fill would cover the whole of it from the later detection —
precisely the span the forward carry had just been forbidden to cross.

`{gas}_carry_periods` reports the distance for every period: `0` where the lag was detected there,
`n` where it travelled `n` periods, empty where it came from somewhere else. The default is
unlimited, which is the published behaviour.

:::{note}
**One deliberate difference from the paper.** In `dyco`, an S2 acceptance *updates* the
carry-forward reference, so a run of S2 chunks can drift away from the S1 lag that anchored them.
Section 2.3 of the paper is ambiguous here; its S3 wording implies this reading. The difference is
catalogued as low-severity in the `dyco/pwb.py` module docstring, alongside the rest of the
comparison against the reference R implementation.
:::

### Periods still without a lag

S3 carries a lag *forward*, so a period before the gas's first S1 or S2 detection has nothing to
inherit — and with `--max-carry` set, so does any period past the end of the carry's reach. Those are
filled afterwards, best source first:

1. **The gas's own lag, filled backward** from its next trusted detection, within `--max-carry`
   periods.
2. **A donor gas**, if you asked for one with `@lagfrom=`. Taken period by period, so a donor lag
   that drifts is followed rather than flattened into a constant.
3. **The median of this gas's raw detections** — every one of which PWBOPT has just rejected. This is
   a last resort, and on real data it is often a negative lag that no tube can produce. If a gas
   lands here, give it a donor.

So the full order of preference, per period, is: its own detection → its own lag carried forward →
its own lag filled backward → the donor's lag for that period → the median.

**The gas's own lag comes first in all three of its forms, and that is deliberate.** Two gases down
one tube still have different delays — a systematic 0.35 s between CH₄ and N₂O is ordinary — so
borrowing swaps a stale number for a biased one. Whether staleness has become the larger error is a
judgement about your site, and `--max-carry` is where you express it. Which also means the two
settings belong together: with no carry limit a gas reaches every period with its own lag, and a
donor is only consulted for a gas that never detects at all. dyco warns when a donor is named without
one.

`{gas}_lag_source` in the summary records which of these produced each period's lag, so a borrowed or
median-filled lag is never mistaken for a detected one.

### Which lag actually gets removed

`--lag-column-template` decides, and its default is `{prefix}_tlag_final_pf_s` — the **pre-filtered**
PWBOPT column. The summary CSV carries the unfiltered variant alongside it, so both are available for
comparison after the fact, but only one is applied to the data.

To read a finished run, though, you want neither: **`{gas}_lag_applied_s`** is the lag that reached
the data, measured back off the record shift, and `{gas}_lag_reason` says in words why that value was
chosen. The same reasoning, one block per output file, is written to
`detect_and_remove_tlag_decisions.txt`. See [What a run writes](output.md#the-summary-csv).

## Search windows are doing real work

`--lws` and `--uws` bound the search to physically possible lags. That sounds like a convenience. On
real data it is load-bearing.

On the bundled CH-LAE half hour, an *unwindowed* PWB search is unreliable — and unreliable in the
same way in both `dyco` and the reference R implementation. The cross-covariance peaks at **−7.45 s**,
which no tube delay can be, because a closed-path delay is always positive. The pipeline reports a
usable **8.45 s** for that same half hour only because `--lws 0 --uws 10` confines the search to lags
that could physically occur.

The consequence is worth internalising when you read a flag:

:::{important}
**An S1 flag from a windowed search is a weaker claim than an S1 flag from an unwindowed one.** A
narrow HDI says the bootstrap resamples agree with each other *within the window you specified*. It
does not say the window was the right one. If the true peak lies outside your window, a confident-
looking S1 is exactly what you will get.
:::

So the window is a real modelling assumption, not a speed optimisation. Keep the expected lag near
the middle of it; detections pinned to a boundary are unreliable and are discarded. And when a whole
site's lags come back suspiciously flat, widening the window is the first thing to try.

## Validation against the reference implementation

`tests/test_pwb_reference.py` pins the deterministic half of the implementation to RFlux v3.2.0's
`tlag_detection.R` at 12 significant digits: both branches of the unit-root test, and the bundled
real CH-LAE half hour, where the AR orders reach 133 / 87 / 312. The R script that produced the
frozen values ships alongside the fixtures in `tests/data/`, so the comparison can be rerun.

Everything deterministic matches. What is left over is the bootstrap, which is random by nature, and
a short list of deliberate choices. The authoritative catalogue with severities is the module
docstring in `dyco/pwb.py`; in summary:

**Differences that could move a number, all small.** dyco resamples with moving blocks starting
anywhere from 0 to `n - L`, while R's `tsboot` wraps around by default, joining the end of a
turbulence record to its start. The bootstrap mode comes from a Gaussian KDE rather than R's
`map_estimate`, a different bandwidth and grid but well inside bootstrap noise. AR-filter
initialisation NaN are zeroed where R's `acf` skips NA pairs, affecting about 1% of positions with
the same numerator. And an S2 acceptance updates the carry-forward reference, so a run of S2 periods
can drift; the paper's Section 2.3 is ambiguous and its S3 wording implies this reading.

**Differences that are choices, not approximations.** A lag pinned to the search-window boundary is
treated as a failed detection and discarded, where R returns it: a peak on the edge is undetermined
rather than measured, and this matches EddyPro. When `--lws`/`--uws` are set they constrain the
returned lag, where R computes windowed variants and then returns the unwindowed ones. `--max-carry`
and `@lagfrom=` are additions with no counterpart in R, both off unless you ask for them.

**One place where dyco follows R against the paper.** `--wdt` defaults to 5, R's value, not the
paper's `hz/2 + 1` (11 at 20 Hz, 6 at 10 Hz). The flag exists so you can have the paper's, and the
choice is not cosmetic: on the bundled CH-LAE hour, `--wdt 11` widens the 95% interval from
0.00/0.05 s to 0.30/0.20 s, against the 0.5 s threshold that decides S1.

## Further reading

Why this problem needs its own tool, and the literature it sits in:
[Motivation and background](background.md), which also carries the full reference list.

The method itself is Vitale, D., Fratini, G., Helfter, C., Hörtnagl, L., et al., 2024. A
pre-whitening with block-bootstrap cross-correlation procedure for temporal alignment of data sampled
by eddy covariance systems. *Environmental and Ecological Statistics* 31, 219–244.
<https://doi.org/10.1007/s10651-024-00615-9>
