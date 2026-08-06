"""
TEST_STOP_WRITES_OUTPUT: STOPPING A RUN STILL PRODUCES USABLE DATA
===================================================================

Stopping used to skip phase 2 outright, so a run that had detected for hours
before the user pressed Stop left a summary, a decisions report and plots, but
not one lag-corrected file. Detection is the expensive half; throwing away its
result because the user wanted the rest of the queue abandoned is the wrong
trade.

Now the chunks that did detect are aligned and written. PWBOPT has already
chosen a lag for each of them by that point, so nothing is guessed. The cancel
event is cleared at the phase boundary, which is what lets a second Stop abort
the alignment as well -- the run keeps whatever it wrote up to that point.

Part of the dyco package: https://github.com/holukas/dyco
"""

import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from dyco.pipeline import PerFilePipeline

_HZ = 20
_CHUNK_S = 60
_N = 2 * _CHUNK_S * _HZ    # two chunks per file
_N_FILES = 3               # -> six chunks in total


def _raw_frame(seed: int, n: int = _N, records: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    lagged = np.r_[np.zeros(records), w[:-records]]
    return pd.DataFrame({
        'u': rng.standard_normal(n), 'v': rng.standard_normal(n), 'w': w,
        'ts': 0.8 * w + 0.2 * rng.standard_normal(n),
        'ch4': lagged + 0.2 * rng.standard_normal(n),
    })


def _write(path: Path, df: pd.DataFrame) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(','.join(df.columns) + '\n')
        df.to_csv(fh, index=False, header=False, lineterminator='\n')


class _Run:
    """One pipeline run in a temp tree, stoppable from a progress callback."""

    def __init__(self, td: str, stop_after):
        self.root = Path(td)
        self.in_dir = self.root / 'in'
        self.out_dir = self.root / 'out'
        self.in_dir.mkdir()
        for i in range(_N_FILES):
            _write(self.in_dir / f'raw_{i}.csv', _raw_frame(seed=i))
        self.cancel = threading.Event()
        self.events: list = []
        self._stop_after = stop_after   # (phase, done) -> bool

    def _on_progress(self, done, total, row, phase):
        self.events.append((phase, done, row.get('period', '')))
        if self._stop_after(phase, done):
            self.cancel.set()

    def go(self) -> pd.DataFrame:
        self.pipeline = PerFilePipeline(
            input_dir=self.in_dir, output_dir=self.out_dir,
            col_u='u', col_v='v', col_w='w', col_tsonic='ts',
            scalars={'CH4': 'ch4'},
            hz=_HZ, chunk_seconds=_CHUNK_S, min_chunk_seconds=30,
            lag_max_s=2.0, block_length_s=4.0, n_bootstrap=9,
            file_pattern='*.csv', skiprows=0, extra_rows=0, sep=',',
            chunk_name_template='{stem}_{index:02d}{suffix}',
            n_workers=1, random_state=42, save_plots=False,
        )
        return self.pipeline.run(on_progress=self._on_progress,
                                 cancel_event=self.cancel)

    @property
    def aligned(self) -> list:
        d = self.out_dir / '2_lag_removed'
        return sorted(p.name for p in d.glob('*')) if d.is_dir() else []

    @property
    def detected(self) -> list:
        return [p for ph, _, p in self.events if ph == 'detect']

    @property
    def removed(self) -> list:
        return [p for ph, _, p in self.events if ph == 'remove']


class TestStopDuringDetect(unittest.TestCase):

    def test_detected_chunks_are_still_aligned(self):
        with TemporaryDirectory() as td:
            run = _Run(td, lambda phase, done: phase == 'detect' and done == 2)
            summary = run.go()

            # Detection stopped early, so not every chunk is in the summary.
            self.assertLess(len(run.detected), _N_FILES * 2)
            self.assertTrue(run.pipeline._cancelled)
            # Every chunk that detected was then written.
            self.assertEqual(len(run.aligned), len(run.detected))
            self.assertEqual(len(run.removed), len(run.detected))
            for name in run.aligned:
                self.assertGreater(
                    (run.out_dir / '2_lag_removed' / name).stat().st_size, 0)
            # And the run's own record of what happened is on disk too.
            det = run.out_dir / '1_lag_detection'
            self.assertTrue((det / 'detect_and_remove_tlag_summary.csv').is_file())
            self.assertTrue((det / 'detect_and_remove_tlag_decisions.txt').is_file())
            self.assertEqual(len(summary), len(run.detected))
            # A lag was applied, not left pending, for what was written.
            self.assertTrue(summary['ch4_lag_applied_s'].notna().all())

    def test_stopping_again_skips_the_alignment(self):
        with TemporaryDirectory() as td:
            # Stop detection after 3 chunks, then stop the align phase after
            # its first file.
            run = _Run(td, lambda phase, done: (
                (phase == 'detect' and done == 3)
                or (phase == 'remove' and done == 1)))
            run.go()

            self.assertEqual(len(run.detected), 3)
            self.assertTrue(run.pipeline._cancelled)
            # Alignment started and was cut short: fewer files than chunks.
            self.assertGreaterEqual(len(run.aligned), 1)
            self.assertLess(len(run.aligned), len(run.detected))

    def test_stopping_before_anything_detects_is_not_an_error(self):
        with TemporaryDirectory() as td:
            run = _Run(td, lambda phase, done: False)
            run.cancel.set()          # already stopped when run() is entered
            summary = run.go()

            self.assertEqual(len(run.detected), 0)
            self.assertEqual(run.aligned, [])
            self.assertTrue(summary.empty)


if __name__ == '__main__':
    unittest.main()
