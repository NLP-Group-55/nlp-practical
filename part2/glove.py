"""Stream local Stanford and Gensim GloVe 100d archives."""

import gzip
from itertools import chain
from pathlib import Path
from zipfile import ZipFile

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "part2"
ARCHIVE_NAME = "glove.2024.wikigiga.100d.zip"
DEFAULT_ARCHIVE = RESULTS_DIR / "embeddings" / ARCHIVE_NAME
EMBEDDING_SIZE = 100


def iter_glove_rows(archive_path):
    """Stream each vocabulary key and its vector without extracting the archive."""
    archive_path = Path(archive_path)
    if archive_path.suffix == ".gz":
        with gzip.open(archive_path, "rb") as vectors:
            count, width = vectors.readline().split()
            if int(count) < 1 or int(width) != EMBEDDING_SIZE:
                raise ValueError(f"Expected a Gensim header for {EMBEDDING_SIZE}d vectors.")
            for line in vectors:
                key, _, values = line.partition(b" ")
                yield key, values
        return

    with ZipFile(archive_path) as archive:
        member, = [name for name in archive.namelist() if name.endswith(".txt")]
        with archive.open(member) as vectors:
            first_line = vectors.readline()
            if len(first_line.split()) != EMBEDDING_SIZE + 1:
                raise ValueError(f"Expected a token and {EMBEDDING_SIZE} GloVe values per row.")
            for line in chain((first_line,), vectors):
                key, _, values = line.partition(b" ")
                yield key, values


def load_vectors(archive_path, vocabulary):
    """Load float32 vectors for requested words that have a GloVe entry."""
    candidates = {word.encode("utf-8"): word for word in vocabulary}
    return {
        candidates[key]: np.fromstring(values, dtype=np.float32, sep=" ")
        for key, values in iter_glove_rows(archive_path)
        if key in candidates
    }
