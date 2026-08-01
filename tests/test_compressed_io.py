"""
TEST_COMPRESSED_IO: COMPRESSION MUST NOT CHANGE WHAT DYCO SEES
===============================================================

Raw EC data is delimited text that is routinely shipped compressed, and dyco's
own file splitter writes `.csv.gz`. Whether a file is compressed says nothing
about what it contains, so every reader and writer has to treat `.csv`,
`.csv.gz` and `.csv.zip` alike.

That was not true three times over. `pipeline.py` read compressed headers as
mojibake and mis-planned chunks; `apply_tlag.py` died on a `.csv.gz` with a bare
`StopIteration`; `tui.py` decoded gzip bytes into five replacement-character
"column names" and reported every configured column as missing, without
raising. Three modules, three separate accidents, none caught by a test --
which is the actual defect this file exists to fix.

So: the I/O layer is checked across every compression, and every entry point is
checked against a compressed input.

One warning about what guards what. The first zip writer here silently dropped
63 of 36000 rows because its TextIOWrapper was never flushed before the stream
beneath it closed -- a valid archive holding a short file, nothing raised, and
the detected lag was correct. Reverting that fix does *not* fail the round-trip
test below: with the wrapper held only by a local, CPython's refcounting
happens to flush it in time. It fails only where a real writer keeps the
wrapper alive a moment longer, i.e. the two tests that count rows after
`apply-batch` and after the pipeline. Those row counts are the guard; the
round-trip test checks content, not flush ordering.

Part of the dyco package: https://github.com/holukas/dyco
"""

import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from dyco import rawio

# Every compression dyco claims to handle, plus no compression at all. The
# format suffix underneath is always .csv -- that is the point.
_SUFFIXES = ['.csv', '.csv.gz', '.csv.bz2', '.csv.xz', '.csv.zip']


def _write_via_rawio(path: Path, text: str) -> None:
    with rawio.open_text_write(path) as fh:
        fh.write(text)


class TestRawioRoundTrip(unittest.TestCase):
    """Write then read must return the same text, whatever the suffix."""

    # Large enough to span several internal buffers: the zip write bug only
    # showed up past the first one.
    TEXT = ''.join(f'row{i},{i * 1.5},{i % 7}\n' for i in range(20000))

    def test_text_survives_the_round_trip(self):
        with TemporaryDirectory() as tmp:
            for suffix in _SUFFIXES:
                with self.subTest(suffix=suffix):
                    p = Path(tmp) / f'raw{suffix}'
                    _write_via_rawio(p, self.TEXT)
                    with rawio.open_text(p) as fh:
                        got = fh.read()
                    self.assertEqual(len(got), len(self.TEXT),
                                     f'{suffix}: {len(self.TEXT) - len(got)} '
                                     f'characters lost')
                    self.assertEqual(got, self.TEXT)

    def test_the_compressed_forms_are_actually_compressed(self):
        # A writer that quietly emits plain text under a .gz name produces a
        # file whose extension lies, which is how one of the earlier faults
        # escaped notice.
        with TemporaryDirectory() as tmp:
            for suffix in _SUFFIXES:
                with self.subTest(suffix=suffix):
                    p = Path(tmp) / f'raw{suffix}'
                    _write_via_rawio(p, self.TEXT)
                    head = p.read_bytes()[:4]
                    if suffix == '.csv':
                        self.assertTrue(head.startswith(b'row'))
                        self.assertEqual(p.stat().st_size, len(self.TEXT))
                    else:
                        self.assertFalse(head.startswith(b'row'),
                                         f'{suffix} was written uncompressed')
                        self.assertLess(p.stat().st_size, len(self.TEXT))

    def test_binary_reads_agree_with_text_reads(self):
        with TemporaryDirectory() as tmp:
            for suffix in _SUFFIXES:
                with self.subTest(suffix=suffix):
                    p = Path(tmp) / f'raw{suffix}'
                    _write_via_rawio(p, self.TEXT)
                    with rawio.open_binary(p) as fh:
                        self.assertEqual(fh.read().decode(), self.TEXT)

    def test_preserved_lines_come_back_in_order(self):
        with TemporaryDirectory() as tmp:
            for suffix in _SUFFIXES:
                with self.subTest(suffix=suffix):
                    p = Path(tmp) / f'raw{suffix}'
                    _write_via_rawio(p, self.TEXT)
                    head = rawio.read_preserved_lines(p, 3)
                    self.assertEqual([ln.rstrip('\n') for ln in head],
                                     ['row0,0.0,0', 'row1,1.5,1', 'row2,3.0,2'])

    def test_a_short_file_names_the_flags_to_check(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'short.csv.gz'
            _write_via_rawio(p, 'only one line\n')
            with self.assertRaises(ValueError) as ctx:
                rawio.read_preserved_lines(p, 5)
            self.assertIn('--skiprows', str(ctx.exception))

    def test_the_format_suffix_ignores_compression(self):
        for name, expected in [('raw.csv', '.csv'), ('raw.csv.gz', '.csv'),
                               ('raw.csv.zip', '.csv'), ('raw.dat.bz2', '.dat'),
                               ('raw.txt.xz', '.txt'), ('RAW.CSV.ZIP', '.csv'),
                               ('raw.gz', ''), ('raw', '')]:
            with self.subTest(name=name):
                self.assertEqual(rawio.data_suffix(name), expected)


class TestOutputSuffixIsIndependentOfInput(unittest.TestCase):
    """The output extension is one setting, format and compression together."""

    TMPL = '{stem}_chunk{index:02d}{suffix}'

    def _name(self, source: str, spec: str) -> str:
        from dyco.pipeline import _chunk_filename
        name, _ = _chunk_filename(Path(source), 0, 1800, self.TMPL, None,
                                  '%Y%m%d%H%M', spec)
        return name

    def test_the_output_extension_is_whatever_was_asked_for(self):
        for source, spec, expected in [
                # auto reuses the input's own extension, whatever it is
                ('file1.csv.gz', 'auto', 'file1_chunk00.csv.gz'),
                ('file1.gz', 'auto', 'file1_chunk00.gz'),
                ('file1.csv', 'auto', 'file1_chunk00.csv'),
                # compressed in, plain out
                ('file1.csv.gz', '.csv', 'file1_chunk00.csv'),
                # an input with no format suffix can still be labelled on the way out
                ('file1.gz', '.csv', 'file1_chunk00.csv'),
                ('file1.gz', '.csv.zip', 'file1_chunk00.csv.zip'),
                # plain in, compressed out; leading dot optional
                ('file1.csv', 'csv.gz', 'file1_chunk00.csv.gz'),
                ('file1.csv', '.dat', 'file1_chunk00.dat')]:
            with self.subTest(source=source, output_suffix=spec):
                self.assertEqual(self._name(source, spec), expected)

    def test_a_compression_dyco_cannot_write_is_refused(self):
        # Accepting .zst would write plain text under a name promising zstd.
        with self.assertRaises(ValueError) as ctx:
            self._name('file1.csv.gz', '.csv.zst')
        self.assertIn('cannot write', str(ctx.exception))

    def test_a_template_without_suffix_says_so_instead_of_ignoring_the_setting(self):
        from dyco.pipeline import _chunk_filename
        with self.assertRaises(ValueError) as ctx:
            _chunk_filename(Path('file1.csv.gz'), 0, 1800, '{stem}_chunk{index:02d}',
                            None, '%Y%m%d%H%M', '.csv')
        self.assertIn('{suffix}', str(ctx.exception))

    def test_the_pipeline_writes_the_format_that_was_asked_for(self):
        from dyco.pipeline import PerFilePipeline
        df = _raw_frame()
        for src_suffix, want, out_suffix in [('.csv.gz', '.csv', '.csv'),
                                             ('.csv', '.csv.gz', '.gz'),
                                             ('.csv', '.csv.zip', '.zip')]:
            with self.subTest(source=src_suffix, output_suffix=want):
                with TemporaryDirectory() as ind, TemporaryDirectory() as out:
                    _write_raw(Path(ind) / f'site_202401010000{src_suffix}', df,
                               extra_rows=0)
                    PerFilePipeline(
                        Path(ind), Path(out), 'u', 'v', 'w', 'ts',
                        {'CH4': 'ch4'}, hz=20, n_bootstrap=9, chunk_seconds=60,
                        min_chunk_seconds=30, sep=',', extra_rows=0,
                        n_workers=1, file_pattern=f'*{src_suffix}',
                        output_suffix=want, random_state=42).run()
                    written = sorted((Path(out) / '2_lag_removed').iterdir())
                    self.assertTrue(written, 'nothing was written')
                    for p in written:
                        self.assertEqual(p.suffix, out_suffix)
                        # and it really is that format, not just named so
                        with rawio.open_text(p) as fh:
                            first = fh.readline()
                        self.assertTrue(first.startswith('u,v,w'), first[:40])


class TestZipArchivesThatAreNotOneFile(unittest.TestCase):
    """A zip of many raw files is a different thing from a zipped raw file."""

    def test_a_multi_member_archive_says_so(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'two.csv.zip'
            with zipfile.ZipFile(p, 'w') as zf:
                zf.writestr('a.csv', 'x\n1\n')
                zf.writestr('b.csv', 'x\n2\n')
            with self.assertRaises(ValueError) as ctx:
                with rawio.open_text(p) as fh:
                    fh.read()
            msg = str(ctx.exception)
            self.assertIn('2 files', msg)
            self.assertIn('exactly one', msg)

    def test_an_empty_archive_says_so(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'empty.csv.zip'
            with zipfile.ZipFile(p, 'w'):
                pass
            with self.assertRaises(ValueError) as ctx:
                with rawio.open_text(p) as fh:
                    fh.read()
            self.assertIn('empty zip', str(ctx.exception))


def _raw_frame(n=2400, records=20, seed=3):
    """A small EC-shaped frame whose scalar lags the wind by `records` rows."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    return pd.DataFrame({
        'u': rng.standard_normal(n), 'v': rng.standard_normal(n), 'w': w,
        'ts': 0.8 * w + 0.2 * rng.standard_normal(n),
        'ch4': np.r_[np.zeros(records), w[:-records]] + 0.2 * rng.standard_normal(n),
    })


def _write_raw(path: Path, df: pd.DataFrame, extra_rows: int = 1) -> Path:
    """Write `df` as a raw file: header row, optional units row, then data."""
    body = ','.join(df.columns) + '\n'
    if extra_rows:
        body += ','.join(['-'] * len(df.columns)) + '\n'
    body += df.to_csv(index=False, header=False, lineterminator='\n')
    _write_via_rawio(path, body)
    return path


class TestEveryReaderAcceptsCompressedInput(unittest.TestCase):
    """One test per module that opens a raw file, across .gz and .zip.

    Each of these modules shipped broken on compressed input at some point;
    none of the breakages was caught by an existing test.
    """

    COMPRESSED = ['.csv.gz', '.csv.zip']

    def setUp(self):
        self.df = _raw_frame()

    def test_the_pipeline_reader_sees_the_real_header(self):
        from dyco.pipeline import _read_raw_file
        with TemporaryDirectory() as tmp:
            for suffix in self.COMPRESSED:
                with self.subTest(suffix=suffix):
                    p = _write_raw(Path(tmp) / f'in{suffix}', self.df)
                    preserved, data = _read_raw_file(p, 0, 1, ',', ['-9999'])
                    self.assertEqual(preserved[0].strip().split(','),
                                     list(self.df.columns))
                    self.assertEqual(len(data), len(self.df))

    def test_the_tui_column_scan_sees_the_real_header(self):
        # This one used to return five replacement characters and no error.
        from dyco.tui import _scan_columns
        with TemporaryDirectory() as tmp:
            for suffix in self.COMPRESSED:
                with self.subTest(suffix=suffix):
                    _write_raw(Path(tmp) / f'scan{suffix}', self.df)
                    _f0, _files, cols = _scan_columns(
                        tmp, f'*{suffix}', 0, 1, ',')
                    self.assertEqual(cols, list(self.df.columns))

    def test_the_splitter_reader_accepts_compression(self):
        from dyco.files import read_raw_data
        with TemporaryDirectory() as tmp:
            for suffix in self.COMPRESSED:
                with self.subTest(suffix=suffix):
                    p = _write_raw(Path(tmp) / f'split{suffix}', self.df,
                                   extra_rows=0)
                    got = read_raw_data(p, data_timestamp_format=None)
                    self.assertEqual(len(got), len(self.df))

    def test_apply_batch_reads_and_writes_compressed(self):
        from dyco.apply_tlag import _apply_tlag_file_worker
        with TemporaryDirectory() as tmp:
            for suffix in self.COMPRESSED:
                with self.subTest(suffix=suffix):
                    src = _write_raw(Path(tmp) / f'apply{suffix}', self.df)
                    out = Path(tmp) / f'applied{suffix}'
                    row = _apply_tlag_file_worker((
                        str(src), str(out), {'CH4': 'ch4'}, {'CH4': 1.0}, 20,
                        0, 1, ',', '\n', ['-9999'], '-9999', True))
                    self.assertEqual(row['status'], 'ok')
                    self.assertEqual(row['ch4_applied_records'], 20)
                    back = pd.read_csv(out, skiprows=[1], na_values=['-9999'])
                    self.assertEqual(len(back), len(self.df))
                    # The shift moved 20 rows off the end of the scalar column.
                    self.assertEqual(int(back['ch4'].isna().sum()), 20)


class TestPipelineEndToEndCompressed(unittest.TestCase):
    """detect-remove, in and out, for each compression.

    Row counts are asserted because the zip writer's dropped buffer produced a
    perfectly valid archive containing a short file -- nothing raised, and the
    lag was correct. Only counting rows catches that.
    """

    def _run(self, suffix: str):
        from dyco.pipeline import PerFilePipeline
        df = _raw_frame()
        with TemporaryDirectory() as ind, TemporaryDirectory() as out:
            _write_raw(Path(ind) / f'site_202401010000{suffix}', df,
                       extra_rows=0)
            summary = PerFilePipeline(
                Path(ind), Path(out), 'u', 'v', 'w', 'ts', {'CH4': 'ch4'},
                hz=20, n_bootstrap=19, chunk_seconds=60, min_chunk_seconds=30,
                sep=',', extra_rows=0, n_workers=1,
                file_pattern=f'*{suffix}', random_state=42).run()
            written = sorted((Path(out) / '2_lag_removed').iterdir())
            rows = [sum(1 for _ in fh) - 1  # minus the header row
                    for p in written
                    for fh in [self._open(p)]]
        return summary, written, rows, df

    @staticmethod
    def _open(path: Path):
        with rawio.open_text(path) as fh:
            return fh.read().splitlines()

    def test_each_compression_runs_and_writes_full_chunks(self):
        for suffix in ['.csv', '.csv.gz', '.csv.zip']:
            with self.subTest(suffix=suffix):
                summary, written, rows, df = self._run(suffix)
                errors = [e for e in summary['error'].fillna('') if e]
                self.assertEqual(errors, [])
                self.assertEqual(len(written), 2)   # 2400 rows = two 60 s chunks
                self.assertEqual(rows, [1200, 1200])
                self.assertEqual([p.suffix for p in written],
                                 [Path(f'x{suffix}').suffix] * 2)
                self.assertAlmostEqual(float(summary['ch4_tlag_s'].iloc[0]),
                                       1.0, delta=0.3)


if __name__ == '__main__':
    unittest.main()
