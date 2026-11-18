"""Probe cosine similarity in GloVe 2024 100d with words from SST-2."""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from data_loader import SST2_ID, load_sst2_split

from .glove import DEFAULT_ARCHIVE, RESULTS_DIR, load_vectors
from .tokenizer import tokenize

# Each row compares the task-related word with a contrast for the same anchor.
PROBES = (
    ("sentiment", "good", "great", "bad"),
    ("sentiment", "boring", "dull", "engaging"),
    ("sentiment", "funny", "hilarious", "unfunny"),
    ("sentiment", "predictable", "formulaic", "unpredictable"),
    ("sentiment", "gripping", "riveting", "tedious"),
    ("word sense", "moving", "touching", "shifting"),
    ("film vocabulary", "plot", "storyline", "soundtrack"),
    ("film vocabulary", "thriller", "suspense", "comedy"),
)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"Missing GloVe archive: {args.archive}. See the README for the download.")

    texts, _, _, _ = load_sst2_split("train")
    vocabulary = {
        word
        for text in texts
        for word in tokenize(text, split_hyphens=True, slash_as_or=True)
        if word.isalpha()
    }
    vectors = load_vectors(args.archive, vocabulary)
    probe_words = {
        word for _, anchor, related, contrast in PROBES for word in (anchor, related, contrast)
    }
    missing = probe_words.difference(vectors)
    if missing:
        raise ValueError(
            f"Missing probe words in the covered SST-2 vocabulary: {', '.join(sorted(missing))}"
        )

    words = sorted(vectors)
    indices = {word: index for index, word in enumerate(words)}
    matrix = np.stack([vectors[word] for word in words])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    probes = []
    for category, anchor, related, contrast in PROBES:
        anchor_index = indices[anchor]
        scores = matrix @ matrix[anchor_index]
        related_score = float(scores[indices[related]])
        contrast_score = float(scores[indices[contrast]])
        scores[anchor_index] = -np.inf
        neighbors = np.argsort(-scores, kind="stable")[:5]
        probes.append({
            "category": category,
            "anchor": anchor,
            "task_related": {"word": related, "cosine": related_score},
            "contrast": {"word": contrast, "cosine": contrast_score},
            "nearest_neighbors": [
                {"word": words[index], "cosine": float(scores[index])} for index in neighbors
            ],
        })
        print(f"{anchor:11} {related:11} {related_score:+.3f}  {contrast:13} {contrast_score:+.3f}")

    report = {
        "dataset": SST2_ID,
        "checkpoint_archive": args.archive.name,
        "metric": "cosine similarity",
        "neighbor_candidates": (
            "Alphabetic SST-2 train tokens with exact GloVe vectors, "
            "after hyphen and slash normalization."
        ),
        "candidate_count": len(words),
        "probes": probes,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "glove_similarity.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved similarities to {output}")


if __name__ == "__main__":
    main()
