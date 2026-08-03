"""
TEST_QUOTED_HEADER: QUOTED COLUMN NAMES PARSE THE SAME AS BARE ONES
====================================================================

Loggers routinely quote the column-name row (`"TIMESTAMP","u","CH4"`) and leave
the data rows bare. `pandas.read_csv` strips those quotes when it parses the
data, but dyco split the header on the raw separator in six places across four
modules and none of them did. The columns came out named `"u"` and `"CH4"`,
quote characters included, so every `--col-u` / `--scalar` name a user could
reasonably pass failed to match and the run died reporting every column
"missing" from a file in which all of them were present.

Found on real CZ-Lnz QCL files (10 Hz, `"TIMESTAMP","RECORD","u",...`).

These tests pin the fix by writing the same content twice - once quoted, once
bare - and requiring identical column names and identical data.

Part of the dyco package: https://github.com/holukas/dyco
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dyco.apply_tlag import _WHITESPACE_SEP
from dyco.pipeline import _read_raw_file
from dyco.rawio import split_header_line

COLS = ['TIMESTAMP', 'u', 'v', 'w', 'ts', 'co2']
N_ROWS = 200


def _body(sep: str = ',', n: int = N_ROWS) -> str:
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(n, len(COLS) - 1))
    return ''.join(
        sep.join([f'2022-11-20 01:00:{i % 60:02d}']
                 + [f'{v:.4f}' for v in row]) + '\n'
        for i, row in enumerate(rows))


class TestSplitHeaderLine(unittest.TestCase):
    """The header splitter itself, across quoting and separator styles."""

    def test_quoted_comma(self):
        self.assertEqual(
            split_header_line('"TIMESTAMP","u","CH4"', ','),
            ['TIMESTAMP', 'u', 'CH4'])

    def test_bare_comma_unchanged(self):
        self.assertEqual(
            split_header_line('TIMESTAMP,u,CH4', ','),
            ['TIMESTAMP', 'u', 'CH4'])

    def test_mixed_quoting(self):
        self.assertEqual(
            split_header_line('TIMESTAMP,"u",CH4', ','),
            ['TIMESTAMP', 'u', 'CH4'])

    def test_whitespace_padding_stripped(self):
        self.assertEqual(
            split_header_line('"TIMESTAMP", "u" , "CH4"', ','),
            ['TIMESTAMP', 'u', 'CH4'])

    def test_separator_inside_quoted_field(self):
        # A quoted comma must not split the field -- this is the reason the
        # comma path goes through `csv` rather than str.split().
        self.assertEqual(
            split_header_line('"a,b",c', ','),
            ['a,b', 'c'])

    def test_tab_separator(self):
        self.assertEqual(
            split_header_line('"u"\t"v"\t"w"', '\t'),
            ['u', 'v', 'w'])

    def test_whitespace_regex_quoted(self):
        self.assertEqual(
            split_header_line('"u"  "v" "w"', _WHITESPACE_SEP),
            ['u', 'v', 'w'])

    def test_whitespace_regex_bare_matches_str_split(self):
        line = '  u   v w  '
        self.assertEqual(
            split_header_line(line, _WHITESPACE_SEP), line.split())

    def test_bracketed_names_survive(self):
        # dyco's own docs use names like CH4_DRY_[LGR-A]; brackets are not
        # quotes and must be left alone.
        self.assertEqual(
            split_header_line('"CH4_DRY_[LGR-A]","H2O_[LI-7200]"', ','),
            ['CH4_DRY_[LGR-A]', 'H2O_[LI-7200]'])

    def test_empty_header_line(self):
        self.assertEqual(split_header_line('', ','), [])


class TestQuotedHeaderReadsLikeBare(unittest.TestCase):
    """A quoted-header raw file must read exactly like its bare twin."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        d = Path(self._tmp.name)
        body = _body()
        self.bare = d / 'bare.csv'
        self.quoted = d / 'quoted.csv'
        self.bare.write_text(
            ','.join(COLS) + '\n' + body, encoding='utf-8', newline='')
        self.quoted.write_text(
            ','.join(f'"{c}"' for c in COLS) + '\n' + body,
            encoding='utf-8', newline='')

    def tearDown(self):
        self._tmp.cleanup()

    def _read(self, path):
        return _read_raw_file(path, skiprows=0, extra_rows=0, sep=',',
                              na_values=['-9999'])

    def test_column_names_have_no_quotes(self):
        _, df = self._read(self.quoted)
        self.assertEqual(list(df.columns), COLS)

    def test_data_identical_to_bare_twin(self):
        _, df_bare = self._read(self.bare)
        _, df_quoted = self._read(self.quoted)
        self.assertEqual(list(df_bare.columns), list(df_quoted.columns))
        for col in COLS[1:]:
            np.testing.assert_array_equal(
                df_bare[col].to_numpy(), df_quoted[col].to_numpy())

    def test_header_line_preserved_verbatim(self):
        # The written file must keep the logger's own quoting, so the output
        # stays a drop-in replacement for the input.
        preserved, _ = self._read(self.quoted)
        self.assertEqual(preserved[0].rstrip('\r\n'),
                         ','.join(f'"{c}"' for c in COLS))


if __name__ == '__main__':
    unittest.main()
