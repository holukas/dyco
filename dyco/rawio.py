"""
RAWIO: OPENING RAW DATA FILES, COMPRESSED OR NOT
=================================================

Raw eddy covariance data is delimited text -- ``.csv``, ``.dat``, ``.txt`` --
and it is routinely shipped compressed. dyco's own file splitter writes
``.csv.gz``. Whether a file is compressed is a property of how it was stored,
not of what it contains, so nothing above this module should have to care.

Every module that reads or writes raw files goes through here. That was not
always so: ``pipeline.py``, ``apply_tlag.py`` and ``tui.py`` each grew their own
``open()`` calls, and each was found, separately and by accident, to break on
compressed input -- the last one silently, returning five columns of mojibake
decoded from gzip bytes. One copy of this knowledge, not four.

Supported: ``.gz``/``.gzip``, ``.bz2``, ``.xz``/``.lzma`` and ``.zip``, all from
the standard library. A ``.zip`` must hold exactly one file; an archive of many
raw files is a different thing from a compressed raw file, and guessing which
member was meant would be worse than saying so.

Part of the dyco package: https://github.com/holukas/dyco
"""

import bz2
import gzip
import io
import lzma
import zipfile
from contextlib import contextmanager
from pathlib import Path

# Compression suffixes this module can open. `data_suffix` strips them, so the
# format suffix underneath is what dispatches. Everything listed here must
# actually be openable below -- a suffix we strip but cannot read would turn a
# clear "unsupported format" into a confusing failure further down.
COMPRESSION_SUFFIXES = {'.gz', '.gzip', '.bz2', '.xz', '.lzma', '.zip'}

_STREAM_OPENERS = {
    '.gz': gzip.open, '.gzip': gzip.open,
    '.bz2': bz2.open,
    '.xz': lzma.open, '.lzma': lzma.open,
}


def is_compressed(path) -> bool:
    """True when *path* carries a compression suffix this module handles."""
    return Path(path).suffix.lower() in COMPRESSION_SUFFIXES


def data_suffix(path) -> str:
    """Return the format-bearing suffix of *path*, ignoring compression.

    ``'raw.csv'``, ``'raw.csv.gz'`` and ``'raw.csv.zip'`` all give ``'.csv'``.
    Returns ``''`` for a name with no format suffix, so ``'raw.gz'`` is not
    mistaken for a CSV.
    """
    suffixes = [s.lower() for s in Path(path).suffixes]
    while suffixes and suffixes[-1] in COMPRESSION_SUFFIXES:
        suffixes.pop()
    return suffixes[-1] if suffixes else ''


# Compression suffixes that exist but that this module cannot open or write.
# Naming one is a mistake worth catching: the file would be written as plain
# text under a name promising otherwise.
_UNSUPPORTED_COMPRESSION = {'.zst', '.zstd', '.7z', '.rar', '.tar', '.tgz',
                            '.lz4', '.br'}

# The sentinel meaning "whatever the input was".
AUTO_SUFFIX = 'auto'


def normalise_output_suffix(spec: str) -> str:
    """Validate a user-supplied output suffix and return it.

    Takes the whole extension the output files should carry, so one setting
    covers both the text format and the compression: ``'.csv'``,
    ``'.csv.gz'``, ``'.dat.zip'``. ``'auto'`` (the default) means keep
    whatever the input used.

    The leading dot is required. A one-part extension is then written exactly
    as a two-part one -- ``.csv`` beside ``.csv.gz`` -- rather than the
    setting accepting ``csv`` for the first and never ``csv.gz`` for the
    second without one.

    The format part is never interpreted -- dyco writes delimited text
    whatever it is called -- but the *compression* part has to be one that can
    actually be written, or the file would be plain text under a name that
    lies.
    """
    spec = (spec or '').strip()
    if not spec or spec.lower() == AUTO_SUFFIX:
        return AUTO_SUFFIX
    if not spec.startswith('.'):
        raise ValueError(
            f'output suffix {spec!r} must start with a dot: write '
            f'{"." + spec!r} the way you would write {"." + spec + ".gz"!r}.')
    # Both forms have to be checked: Path('.zst').suffix is '' -- a name that
    # is nothing but a dotted word reads as a hidden file, not as a suffix.
    last = Path(spec).suffix.lower() or spec.lower()
    if last in _UNSUPPORTED_COMPRESSION:
        raise ValueError(
            f'output suffix {spec!r} asks for {last} compression, which dyco '
            f'cannot write. Supported: '
            f'{", ".join(sorted(COMPRESSION_SUFFIXES))}, or none at all.')
    return spec


def compression_suffix(path) -> str:
    """The compression suffix of *path* (``'.gz'``), or ``''`` if plain."""
    suffix = Path(path).suffix
    return suffix if suffix.lower() in COMPRESSION_SUFFIXES else ''


def strip_compression(name: str) -> str:
    """*name* without its compression suffixes: ``'a.csv.gz'`` -> ``'a.csv'``."""
    while True:
        suffix = Path(name).suffix
        if suffix and suffix.lower() in COMPRESSION_SUFFIXES:
            name = name[:-len(suffix)]
        else:
            return name


def resolve_output_suffix(spec: str, input_path) -> str:
    """The extension output files should carry, resolving ``'auto'``.

    Three shapes, each answering a different question:

    - a full extension (``'.csv.gz'``, ``'.dat'``) is used as given;
    - a bare compression (``'.zip'``, ``'.gz'``) keeps the input's text format
      in front of it, so ``file1.csv`` written as ``.zip`` becomes
      ``file1.csv.zip`` rather than ``file1.zip``;
    - ``'auto'`` reproduces the input's own extension -- ``file1.csv.gz`` gives
      ``'.csv.gz'``, ``file1.gz`` gives ``'.gz'``, ``file1.csv`` gives ``'.csv'``.
    """
    normalised = normalise_output_suffix(spec)
    name = Path(input_path).name
    if normalised == AUTO_SUFFIX:
        return data_suffix(name) + compression_suffix(name)
    if normalised.lower() in COMPRESSION_SUFFIXES:
        return data_suffix(name) + normalised
    return normalised


def _zip_member(zf: zipfile.ZipFile, path) -> zipfile.ZipInfo:
    """The single data member of *zf*, or a message explaining why there isn't one."""
    members = [i for i in zf.infolist() if not i.is_dir()]
    if len(members) == 1:
        return members[0]
    if not members:
        raise ValueError(f'{Path(path).name} is an empty zip archive.')
    names = ', '.join(repr(i.filename) for i in members[:5])
    more = f' (and {len(members) - 5} more)' if len(members) > 5 else ''
    raise ValueError(
        f'{Path(path).name} holds {len(members)} files ({names}{more}). A zipped '
        f'raw data file must contain exactly one; unzip the archive and point '
        f'dyco at the folder instead.')


@contextmanager
def open_text(path, encoding: str = 'utf-8', errors: str = 'replace'):
    """Open *path* for text reading, transparently decompressing."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.zip':
        with zipfile.ZipFile(path) as zf:
            # The wrapper gets its own `with`: it must be closed before the
            # stream beneath it, or its buffer goes nowhere. See open_text_write.
            with zf.open(_zip_member(zf, path)) as raw, \
                    io.TextIOWrapper(raw, encoding=encoding, errors=errors) as fh:
                yield fh
    elif suffix in _STREAM_OPENERS:
        with _STREAM_OPENERS[suffix](path, 'rt', encoding=encoding,
                                     errors=errors) as fh:
            yield fh
    else:
        with open(path, 'r', encoding=encoding, errors=errors) as fh:
            yield fh


@contextmanager
def open_binary(path):
    """Open *path* for binary reading, transparently decompressing."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.zip':
        with zipfile.ZipFile(path) as zf:
            with zf.open(_zip_member(zf, path)) as raw:
                yield raw
    elif suffix in _STREAM_OPENERS:
        with _STREAM_OPENERS[suffix](path, 'rb') as fh:
            yield fh
    else:
        with open(path, 'rb') as fh:
            yield fh


@contextmanager
def open_text_write(path, encoding: str = 'utf-8'):
    """Open *path* for text writing, compressing when the name says so.

    The chunk filename template carries the input's suffix through to the
    output, so a compressed input yields a compressed output *name*. Writing
    plain text to it would produce a file whose extension lies, and downstream
    software that trusts the extension would fail to open it.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.zip':
        # One member, named after the archive without the .zip -- so
        # `chunk.csv.zip` holds `chunk.csv`.
        member = path.stem or path.name
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # TextIOWrapper buffers. If the stream beneath it closes first, the
            # tail of that buffer is dropped -- silently, with a valid archive
            # and a short file. Give the wrapper its own `with` so it flushes
            # first. (Measured: 63 of 36000 rows lost before this.)
            with zf.open(member, 'w') as raw, \
                    io.TextIOWrapper(raw, encoding=encoding, newline='') as fh:
                yield fh
    elif suffix in _STREAM_OPENERS:
        with _STREAM_OPENERS[suffix](path, 'wt', encoding=encoding,
                                     newline='') as fh:
            yield fh
    else:
        with open(path, 'w', encoding=encoding, newline='') as fh:
            yield fh


def read_preserved_lines(path, n: int) -> list:
    """Read the first *n* lines of *path*, or say why they are not there.

    ``[next(fh) for _ in range(n)]`` raises a bare ``StopIteration`` with no
    message when the file is shorter than the header block -- the commonest
    symptom of a wrong ``--skiprows``/``--extra-rows``, and unreadable as an
    error row.
    """
    lines = []
    with open_text(path) as fh:
        for line in fh:
            lines.append(line)
            if len(lines) == n:
                return lines
    raise ValueError(
        f'{Path(path).name} has only {len(lines)} line(s) but --skiprows / '
        f'--extra-rows ask for {n} header line(s) before the data.')
