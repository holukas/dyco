"""
TEST_SPLIT: FILESPLITTERMULTI ON REAL FILES
===========================================

`FileSplitterMulti` named every CSV split with a double dot (`..csv.gz`), and it
passed `split_trim=True` to each `FileSplitter` whatever the caller set, so
`split_trim=False` had no effect and leaving out `split_trim_var` raised
`KeyError: None`. Both were found by running the library docs' own example.

Files are written and read back for real.

Part of the dyco package: https://github.com/holukas/dyco
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from dyco.split import FileSplitterMulti

HZ = 10
N = 30 * 60 * HZ          # one 30-min file, split into two 15-min parts
N_LEADING_GAP = 200       # missing N2O records at the start of the file


def _write_raw(folder: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'u': rng.normal(2, 1, N), 'v': rng.normal(0, 1, N),
                       'w': rng.normal(0, 0.3, N), 'N2O': rng.normal(0.33, 0.001, N)})
    df.loc[:N_LEADING_GAP - 1, 'N2O'] = np.nan
    df.to_csv(folder / 'SITE_202301010000.csv', index=False)


def _split(tmp: Path, **kw) -> Path:
    raw = tmp / 'raw'
    raw.mkdir(exist_ok=True)
    _write_raw(raw)
    out = tmp / 'out'
    FileSplitterMulti(
        outdir=out, searchdirs=str(raw), filename_pattern='*.csv',
        filename_date_format='SITE_%Y%m%d%H%M.csv', file_generation_freq='30min',
        data_split_duration='15min', data_nominal_res=1 / HZ, **kw).run()
    return out / 'splits'


class TestFileSplitterMulti(unittest.TestCase):

    def test_gzipped_csv_names_have_one_dot(self):
        with TemporaryDirectory() as tmp:
            splits = _split(Path(tmp), compress_splits=True, split_trim=True,
                            split_trim_var='N2O')
            names = sorted(p.name for p in splits.iterdir())
            self.assertEqual(names, ['20230101000000_TRIM.csv.gz', '20230101001500_TRIM.csv.gz'])
            # Written gzipped, and readable as such.
            df = pd.read_csv(splits / names[1], compression='gzip')
            self.assertEqual(len(df), N // 2)

    def test_plain_csv_and_parquet_names(self):
        for fmt, compress, suffix in (('csv', False, '.csv'), ('parquet', True, '.parquet')):
            with self.subTest(fmt=fmt), TemporaryDirectory() as tmp:
                splits = _split(Path(tmp), splits_output_format=fmt, compress_splits=compress)
                names = sorted(p.name for p in splits.iterdir())
                # Parquet ignores compress_splits: there is no .parquet.gz.
                self.assertEqual(names, [f'20230101000000{suffix}', f'20230101001500{suffix}'])

    def test_without_trim_every_record_is_kept(self):
        """split_trim=False is honoured, and needs no split_trim_var."""
        with TemporaryDirectory() as tmp:
            splits = _split(Path(tmp), split_trim=False)
            first = pd.read_csv(splits / '20230101000000.csv')
            self.assertEqual(len(first), N // 2)
            self.assertEqual(int(first['N2O'].isna().sum()), N_LEADING_GAP)

    def test_trim_cuts_the_leading_gap(self):
        with TemporaryDirectory() as tmp:
            splits = _split(Path(tmp), split_trim=True, split_trim_var='N2O')
            first = pd.read_csv(splits / '20230101000000_TRIM.csv')
            self.assertEqual(len(first), N // 2 - N_LEADING_GAP)
            self.assertFalse(first['N2O'].isna().any())


if __name__ == '__main__':
    unittest.main()
