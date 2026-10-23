"""
TEST_WORKER_CRASH: A LOST WORKER MUST NOT LOOK LIKE A WRITTEN CHUNK
===================================================================

When one alignment worker died (out of memory, say), the process pool broke
and the executor terminated every other worker as well, mid-write. The run
still printed "Done - 24/26 chunks written": the chunks that never reported
back stayed at status 'ok'. The output folder held 21 files, every one of them
a truncated gzip.

Two things now hold whatever the timing of the crash, and these tests check
both. A chunk is 'ok' only if its file is complete, and a chunk that was sent
to alignment but never reported is an 'error'. A file is only ever complete
under its real name, because `rawio.open_text_write` writes to a `.partial`
name and renames it on a clean close.

Part of the dyco package: https://github.com/holukas/dyco
"""

import gzip
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from dyco import pipeline, rawio

_HZ = 20
_CHUNK_S = 60
_N = 4 * _CHUNK_S * _HZ    # four chunks per file
_N_FILES = 2
_CRASH_CHUNK = 1


def _crash_remove_worker(kwargs, progress_queue=None):
    """Phase-2 worker that dies hard on one chunk, as an OOM kill would.

    Module-level so the spawned workers can unpickle it by reference.
    """
    if kwargs['chunk_index'] == _CRASH_CHUNK and kwargs['input_path'].name == 'raw_0.csv':
        time.sleep(0.5)     # let the siblings get into their writes
        os._exit(1)
    return pipeline.remove_one_chunk(progress_queue=progress_queue, **kwargs)


def _raw_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(_N)
    lagged = np.r_[np.zeros(20), w[:-20]]
    return pd.DataFrame({
        'u': rng.standard_normal(_N), 'v': rng.standard_normal(_N), 'w': w,
        'ts': 0.8 * w + 0.2 * rng.standard_normal(_N),
        'ch4': lagged + 0.2 * rng.standard_normal(_N),
    })


class TestLostWorker(unittest.TestCase):

    def test_lost_chunks_are_errors_and_written_files_are_complete(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            in_dir = root / 'in'
            in_dir.mkdir()
            for i in range(_N_FILES):
                df = _raw_frame(seed=i)
                with open(in_dir / f'raw_{i}.csv', 'w', newline='') as fh:
                    fh.write(','.join(df.columns) + '\n')
                    df.to_csv(fh, index=False, header=False, lineterminator='\n')

            original = pipeline._remove_worker
            pipeline._remove_worker = _crash_remove_worker
            try:
                p = pipeline.PerFilePipeline(
                    input_dir=in_dir, output_dir=root / 'out',
                    col_u='u', col_v='v', col_w='w', col_tsonic='ts',
                    scalars={'CH4': 'ch4'},
                    hz=_HZ, chunk_seconds=_CHUNK_S, min_chunk_seconds=30,
                    lag_max_s=2.0, block_length_s=4.0, n_bootstrap=9,
                    file_pattern='*.csv', skiprows=0, extra_rows=0, sep=',',
                    chunk_name_template='{stem}_{index:02d}{suffix}',
                    output_suffix='.csv.gz',
                    n_workers=4, random_state=42, save_plots=False,
                )
                summary = p.run()
            finally:
                pipeline._remove_worker = original

            data_dir = root / 'out' / '2_lag_removed'
            status = summary.set_index('period')['status']

            # The crashed chunk is an error, never 'ok'.
            self.assertEqual(status['raw_0_01.csv.gz'], 'error')
            self.assertIn('worker process was lost',
                          summary.set_index('period').at['raw_0_01.csv.gz',
                                                         'error'])
            # Every chunk is accounted for: siblings killed with the pool are
            # errors too, the rest were written.
            self.assertEqual(set(status), {'ok', 'error'})

            # 'ok' means a complete file under the real name, and nothing else
            # is there: no truncated file, no leftover .partial.
            ok = sorted(status[status == 'ok'].index)
            self.assertEqual(sorted(f.name for f in data_dir.iterdir()), ok)
            for name in ok:
                with gzip.open(data_dir / name, 'rt') as fh:
                    self.assertEqual(sum(1 for _ in fh), 1 + _CHUNK_S * _HZ)


class TestOpenTextWrite(unittest.TestCase):

    def test_a_failed_write_leaves_nothing_behind(self):
        with TemporaryDirectory() as td:
            for sfx in ('.csv', '.csv.gz', '.csv.zip'):
                path = Path(td) / f'chunk{sfx}'
                with self.assertRaises(RuntimeError):
                    with rawio.open_text_write(path) as fh:
                        fh.write('a,b\n')
                        raise RuntimeError('killed mid-write')
                self.assertFalse(path.exists(), sfx)
                self.assertFalse(rawio.partial_path(path).exists(), sfx)

    def test_gzip_header_names_the_final_file(self):
        # gzip records a name in its header; the .partial name the data was
        # written under must not end up there.
        with TemporaryDirectory() as td:
            path = Path(td) / 'chunk.csv.gz'
            with rawio.open_text_write(path) as fh:
                fh.write('a,b\n')
            raw = path.read_bytes()
            self.assertTrue(raw[3] & 0x08)      # FNAME flag set
            self.assertEqual(raw[10:raw.index(b'\0', 10)], b'chunk.csv')


if __name__ == '__main__':
    unittest.main()
