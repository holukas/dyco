"""
TEST_RAWIO_GZIP: COMPRESSED RAW FILES READ THE SAME AS UNCOMPRESSED
===================================================================

`pandas.read_csv` infers gzip compression from the file suffix on its own, but
the header rows and the row counters in `dyco.pipeline` are read with the
builtin `open()`, which does not. Plain `open()` on a `.gz` returns compressed
bytes decoded as text, so the header parse produced garbage and the row count
was the compressed byte-stream's newline count rather than the real one.

That mattered because dyco's own file splitter writes `.csv.gz` when
`compress_splits=True`, so the toolchain could produce files its own pipeline
could not read.

These tests pin the fix by writing the same content twice - once plain, once
gzipped - and requiring identical results.

Part of the dyco package: https://github.com/holukas/dyco
"""

import gzip
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dyco.pipeline import (
    _DEFAULT_NA_VALUES,
    _count_data_rows,
    _detect_lineterm,
    _estimate_data_rows,
    _read_raw_file,
    _write_raw_file,
)

# Three header rows (names / units / instrument), then data - the layout of a
# real logger file.
HEADER = "u,v,w,ts,co2\n[m_s-1],[m_s-1],[m_s-1],[K],[umol_mol-1]\n[SONIC],[SONIC],[SONIC],[SONIC],[IRGA]\n"
N_ROWS = 500


def _body(n: int = N_ROWS) -> str:
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(n, 5))
    return ''.join(','.join(f'{v:.4f}' for v in row) + '\n' for row in rows)


class TestGzippedRawFiles(unittest.TestCase):
    """A .gz raw file must read exactly like its uncompressed twin."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        d = Path(self._tmp.name)
        content = HEADER + _body()
        self.plain = d / 'chunk.csv'
        self.gz = d / 'chunk.csv.gz'
        self.plain.write_text(content, encoding='utf-8', newline='')
        with gzip.open(self.gz, 'wt', encoding='utf-8', newline='') as fh:
            fh.write(content)

    def tearDown(self):
        self._tmp.cleanup()

    def test_row_count_matches(self):
        # The bug: this counted newlines in the *compressed* bytes.
        plain_n = _count_data_rows(self.plain, header_lines=3)
        gz_n = _count_data_rows(self.gz, header_lines=3)
        self.assertEqual(plain_n, N_ROWS)
        self.assertEqual(gz_n, N_ROWS)

    def test_header_and_data_match(self):
        # The bug: the header row was parsed from compressed bytes, so the
        # column count disagreed with the data and _read_raw_file raised.
        kept_p, df_p = _read_raw_file(self.plain, skiprows=0, extra_rows=2,
                                      sep=',', na_values=_DEFAULT_NA_VALUES)
        kept_g, df_g = _read_raw_file(self.gz, skiprows=0, extra_rows=2,
                                      sep=',', na_values=_DEFAULT_NA_VALUES)
        self.assertEqual(kept_p, kept_g)
        self.assertEqual(list(df_p.columns), ['u', 'v', 'w', 'ts', 'co2'])
        self.assertEqual(list(df_g.columns), list(df_p.columns))
        self.assertEqual(len(df_g), len(df_p))
        self.assertTrue(df_g.equals(df_p))

    def test_line_terminator_detected(self):
        # _detect_lineterm peeks at raw bytes; on a .gz those are the gzip
        # stream, whose first \n is meaningless.
        self.assertEqual(_detect_lineterm(self.plain), '\n')
        self.assertEqual(_detect_lineterm(self.gz), '\n')

    def test_row_estimate_matches_on_a_file_larger_than_the_sample(self):
        # The estimator extrapolates a bytes-per-line from a 256 KiB sample and
        # scales it by stat().st_size. On a .gz the sample is decompressed while
        # stat() is the compressed size - mixing the two underestimates the row
        # count by the compression ratio and silently drops most of the chunks
        # the pipeline would otherwise plan. Needs a file bigger than the sample
        # window, otherwise the estimator short-circuits to an exact count.
        d = Path(self._tmp.name)
        n_big = 40_000
        content = HEADER + _body(n_big)
        plain = d / 'big.csv'
        gz = d / 'big.csv.gz'
        plain.write_text(content, encoding='utf-8', newline='')
        with gzip.open(gz, 'wt', encoding='utf-8', newline='') as fh:
            fh.write(content)
        self.assertGreater(plain.stat().st_size, 1 << 18,
                           "fixture must exceed the estimator's sample window")

        exact = _count_data_rows(gz, header_lines=3)
        self.assertEqual(exact, n_big)

        est_plain = _estimate_data_rows(plain, header_lines=3)
        est_gz = _estimate_data_rows(gz, header_lines=3)
        # Plain: sampled estimate, within a percent. Gzip: must not be a wild
        # under-count - a 5x miss is the failure this pins.
        self.assertAlmostEqual(est_plain / n_big, 1.0, delta=0.02)
        self.assertAlmostEqual(est_gz / n_big, 1.0, delta=0.02)

    def test_write_round_trips_through_gzip(self):
        # The chunk filename template carries the input's suffix to the output,
        # so a .gz input produces a .gz output *name*. The writer must actually
        # compress, or the extension lies and downstream software cannot open it.
        d = Path(self._tmp.name)
        kept, df = _read_raw_file(self.gz, skiprows=0, extra_rows=2,
                                  sep=',', na_values=_DEFAULT_NA_VALUES)
        out = d / 'written.csv.gz'
        _write_raw_file(out, preserved_lines=kept, df=df, sep=',',
                        lineterm='\n', na_rep='NaN')

        # Really gzipped, not plain text wearing a .gz name.
        self.assertEqual(out.read_bytes()[:2], b'\x1f\x8b')

        kept2, df2 = _read_raw_file(out, skiprows=0, extra_rows=2,
                                    sep=',', na_values=_DEFAULT_NA_VALUES)
        self.assertEqual([l.rstrip('\r\n') for l in kept],
                         [l.rstrip('\r\n') for l in kept2])
        self.assertEqual(list(df2.columns), list(df.columns))
        self.assertEqual(len(df2), len(df))

    def test_write_stays_plain_without_a_gz_name(self):
        d = Path(self._tmp.name)
        kept, df = _read_raw_file(self.plain, skiprows=0, extra_rows=2,
                                  sep=',', na_values=_DEFAULT_NA_VALUES)
        out = d / 'written.csv'
        _write_raw_file(out, preserved_lines=kept, df=df, sep=',',
                        lineterm='\n', na_rep='NaN')
        self.assertNotEqual(out.read_bytes()[:2], b'\x1f\x8b')

    def test_crlf_survives_compression(self):
        d = Path(self._tmp.name)
        crlf = (HEADER + _body(50)).replace('\n', '\r\n')
        p = d / 'crlf.csv.gz'
        with gzip.open(p, 'wt', encoding='utf-8', newline='') as fh:
            fh.write(crlf)
        self.assertEqual(_detect_lineterm(p), '\r\n')
        self.assertEqual(_count_data_rows(p, header_lines=3), 50)


if __name__ == '__main__':
    unittest.main()
