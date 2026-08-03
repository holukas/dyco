"""
TEST_PROGRESS_TOTAL: THE PROGRESS BAR MUST NOT RUN PAST ITS OWN TOTAL
======================================================================

Phase 1 dispatches more chunks than a file has, so a sampling error in the
row-count estimate can never drop a trailing chunk. The extras land past EOF
and are discarded. But the completion counter fed to `on_progress` counted
those phantoms while the total it was compared against did not, so `done`
overtook `total`: the bar sat at 100% and the ETA at 0:00 while a long run
still had many files to go. With one phantom per file, a run over thousands of
files reached 100% early and stayed there -- the phantoms finish almost
instantly, so they front-run the real work.

Two things are asserted here. The reported count only ever covers real chunks,
and the reported total is revised as the run proves what each file actually
holds: a file's first past-EOF chunk index is its real chunk count.

Part of the dyco package: https://github.com/holukas/dyco
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from dyco.pipeline import PerFilePipeline

_HZ = 20
_CHUNK_S = 60
_CHUNK_ROWS = _CHUNK_S * _HZ      # 1200
_N = 2 * _CHUNK_ROWS              # exactly two chunks per file, no tail
_N_FILES = 3


def _raw_frame(n: int = _N, records: int = 20, seed: int = 5) -> pd.DataFrame:
    """EC-shaped data whose scalar lags the wind by `records` rows."""
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


class TestReportedCountNeverExceedsTotal(unittest.TestCase):
    """A real run over several files: `done` must stay inside `total`."""

    def _run(self, n_workers: int) -> list:
        events = []
        with TemporaryDirectory() as td:
            in_dir = Path(td) / 'in'
            in_dir.mkdir()
            for i in range(_N_FILES):
                _write(in_dir / f'raw_{i}.csv', _raw_frame(seed=i))
            PerFilePipeline(
                input_dir=in_dir, output_dir=Path(td) / 'out',
                col_u='u', col_v='v', col_w='w', col_tsonic='ts',
                scalars={'CH4': 'ch4'},
                hz=_HZ, chunk_seconds=_CHUNK_S, min_chunk_seconds=30,
                lag_max_s=2.0, block_length_s=4.0, n_bootstrap=9,
                file_pattern='*.csv', skiprows=0, extra_rows=0, sep=',',
                chunk_name_template='{stem}_{index:02d}{suffix}',
                n_workers=n_workers, random_state=42, save_plots=False,
            ).run(on_progress=lambda done, total, row, phase:
                  events.append((phase, done, total)))
        return events

    def _assert_sane(self, events):
        detect = [(d, t) for phase, d, t in events if phase == 'detect']
        self.assertTrue(detect, 'no detect progress was reported')
        for done, total in detect:
            self.assertLessEqual(
                done, total,
                f'reported {done}/{total}: the bar would sit past 100%')
        # One report per real chunk, counted 1..N with nothing skipped.
        self.assertEqual([d for d, _ in detect],
                         list(range(1, len(detect) + 1)))
        # And the phase ends exactly full, not short of it.
        self.assertEqual(detect[-1][0], detect[-1][1])
        self.assertEqual(detect[-1][0], _N_FILES * 2)

    def test_sequential(self):
        self._assert_sane(self._run(n_workers=1))

    def test_parallel(self):
        # The bug was worst in parallel, where the instant phantom chunks
        # complete long before the real ones they were dispatched alongside.
        self._assert_sane(self._run(n_workers=2))


class TestTotalIsRevisedWhileRunning(unittest.TestCase):
    """A past-EOF chunk measures its file, so the total stops being a guess."""

    @staticmethod
    def _fake_worker(kwargs, queue):
        """Emit one start/done pair per dispatched chunk, no file touched."""
        parent = kwargs['parent']
        ci = kwargs['chunk_index']
        queue.put({'event': 'start', 'pid': 1, 'parent': parent,
                   'chunk_index': ci, 'chunk_period': f'{parent}_{ci}'})
        queue.put({'event': 'done', 'pid': 1, 'row': {
            'parent': parent, 'chunk_index': ci, 'period': f'{parent}_{ci}',
            'status': 'empty:eof' if ci >= kwargs['real_chunks'] else 'ok',
        }})

    def _drive(self, estimates: dict, real: dict, dispatch: dict) -> list:
        pipeline = PerFilePipeline(
            input_dir=Path('.'), output_dir=Path('.'),
            col_u='u', col_v='v', col_w='w', col_tsonic='ts',
            scalars={'CH4': 'ch4'}, hz=_HZ, n_workers=1,
        )
        kwargs_list = [
            {'parent': name, 'chunk_index': ci, 'real_chunks': real[name]}
            for name in estimates for ci in range(dispatch[name])
        ]
        seen = []
        pipeline._run_pool(
            kwargs_list, self._fake_worker,
            total=sum(estimates.values()), phase='detect',
            checkpoint_path=None, parent_to_idx={}, total_files=len(estimates),
            on_progress=lambda done, total, row, phase: seen.append((done, total)),
            on_active=None, chunk_estimates=estimates,
        )
        return seen

    def test_over_estimate_is_corrected_down(self):
        # Both files were estimated at 3 chunks and really hold 2. File 'a'
        # proves itself first, which takes 1 off the total for everyone
        # watching; 'b' can only prove itself after its last real chunk, so
        # that final correction lands with no report left to show it.
        seen = self._drive(estimates={'a': 3, 'b': 3},
                           real={'a': 2, 'b': 2},
                           dispatch={'a': 4, 'b': 4})
        self.assertEqual(seen, [(1, 6), (2, 6), (3, 5), (4, 5)])

    def test_under_estimate_is_corrected_up(self):
        # Estimated at 2, really holds 3. Dispatch padding covers the extra.
        seen = self._drive(estimates={'a': 2, 'b': 2},
                           real={'a': 3, 'b': 3},
                           dispatch={'a': 4, 'b': 4})
        self.assertEqual([d for d, _ in seen], [1, 2, 3, 4, 5, 6])
        self.assertEqual(seen[-1], (6, 6))
        for done, total in seen:
            self.assertLessEqual(done, total)

    def test_a_file_that_never_reveals_itself_keeps_its_estimate(self):
        # Dispatch exactly the real count: no past-EOF chunk is ever produced,
        # so nothing measures the file and the estimate has to stand.
        seen = self._drive(estimates={'a': 2}, real={'a': 2}, dispatch={'a': 2})
        self.assertEqual(seen, [(1, 2), (2, 2)])


if __name__ == '__main__':
    unittest.main()
