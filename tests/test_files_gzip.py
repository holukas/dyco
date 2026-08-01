"""
TEST_FILES_GZIP: `files.read_raw_data` ACCEPTS COMPRESSED INPUT
===============================================================

`read_raw_data` dispatched on `Path(filepath).suffix`, which for `raw.csv.gz` is
`.gz` - so every compressed file was rejected outright with "File extension must
be '.csv' or '.parquet'". This is the reader the v2 path and `FileSplitter` use,
and `FileSplitterMulti` writes `.csv.gz` when `compress_splits=True`, so the
splitter's own output could not be read back in.

Distinct from the faults fixed in `dyco.pipeline` (see `test_rawio_gzip.py`).
That reader mishandled compressed files; this one refused them. The two readers
share no code, which is why the same class of gap had to be found twice.

Part of the dyco package: https://github.com/holukas/dyco
"""

import gzip
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dyco.files import data_suffix, read_raw_data

# files.read_raw_data_csv assumes a SINGLE header row - it is not the reader for
# multi-row logger headers. Use the format it actually supports.
HEADER = "u,v,w,co2\n"
N_ROWS = 300


def _content(n: int = N_ROWS) -> str:
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(n, 4))
    return HEADER + ''.join(','.join(f'{v:.4f}' for v in row) + '\n' for row in rows)


class TestDataSuffix(unittest.TestCase):
    """The format suffix is what dispatches; compression suffixes are ignored."""

    def test_compression_suffix_is_stripped(self):
        for name, expected in [('raw.csv', '.csv'),
                               ('raw.csv.gz', '.csv'),
                               ('raw.csv.bz2', '.csv'),
                               ('raw.parquet', '.parquet'),
                               ('raw.parquet.gz', '.parquet')]:
            with self.subTest(name=name):
                self.assertEqual(data_suffix(name), expected)

    def test_case_is_ignored(self):
        self.assertEqual(data_suffix('RAW.CSV.GZ'), '.csv')

    def test_no_suffix_gives_empty(self):
        self.assertEqual(data_suffix('raw'), '')

    def test_a_bare_compression_name_does_not_masquerade_as_a_format(self):
        # 'raw.gz' carries no format suffix, so it must not be read as CSV.
        self.assertEqual(data_suffix('raw.gz'), '')


class TestReadRawDataCompressed(unittest.TestCase):

    def setUp(self):
        self._tmp = TemporaryDirectory()
        d = Path(self._tmp.name)
        content = _content()
        self.plain = d / 'raw.csv'
        self.gz = d / 'raw.csv.gz'
        self.plain.write_text(content, encoding='utf-8', newline='')
        with gzip.open(self.gz, 'wt', encoding='utf-8', newline='') as fh:
            fh.write(content)

    def tearDown(self):
        self._tmp.cleanup()

    def test_gzipped_csv_reads_identically_to_plain(self):
        plain = read_raw_data(self.plain, data_timestamp_format=None)
        gz = read_raw_data(self.gz, data_timestamp_format=None)
        self.assertEqual(list(gz.columns), ['u', 'v', 'w', 'co2'])
        self.assertEqual(len(gz), N_ROWS)
        self.assertTrue(plain.equals(gz))

    def test_unsupported_format_still_rejected_with_a_useful_message(self):
        d = Path(self._tmp.name)
        bad = d / 'raw.txt.gz'
        bad.write_bytes(b'')
        with self.assertRaises(ValueError) as ctx:
            read_raw_data(bad, data_timestamp_format=None)
        msg = str(ctx.exception)
        self.assertIn('.csv', msg)
        self.assertIn('.parquet', msg)
        self.assertIn('.txt', msg)  # names what it actually saw


if __name__ == '__main__':
    unittest.main()
