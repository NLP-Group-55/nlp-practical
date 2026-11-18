"""Measure exact GloVe 2024 100d vocabulary coverage on SST-2."""

import json
from argparse import ArgumentParser
from collections import Counter
from itertools import chain
from pathlib import Path

from data_loader import SST2_ID, load_sst2_split

from .glove import DEFAULT_ARCHIVE, RESULTS_DIR, iter_glove_rows
from .tokenizer import tokenize


def find_matches(archive_path, tokens):
    """Read only the vocabulary keys from the zipped vectors."""
    candidates = {token.encode("utf-8"): token for token in tokens}
    return {candidates[key] for key, _ in iter_glove_rows(archive_path) if key in candidates}


def summarize_coverage(rows, counts, matches):
    missing = Counter({token: count for token, count in counts.items() if token not in matches})
    token_count = counts.total()
    type_count = len(counts)
    covered_tokens = token_count - missing.total()
    covered_types = type_count - len(missing)
    return {
        "examples": len(rows),
        "examples_with_missing_tokens": sum(any(token in missing for token in row) for row in rows),
        "tokens": token_count,
        "covered_tokens": covered_tokens,
        "token_coverage": covered_tokens / token_count,
        "types": type_count,
        "covered_types": covered_types,
        "type_coverage": covered_types / type_count,
        "most_frequent_oov": dict(missing.most_common(15)),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--split-hyphens", action="store_true", help="Split hyphens between word characters."
    )
    parser.add_argument(
        "--slash-as-or", action="store_true", help="Replace each ASCII slash with an 'or' token."
    )
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"Missing GloVe archive: {args.archive}. See the README for the download.")

    rows_by_split = {}
    counts_by_split = {}
    for split in ("train", "validation"):
        texts, _, _, _ = load_sst2_split(split)
        rows = [
            tokenize(text, split_hyphens=args.split_hyphens, slash_as_or=args.slash_as_or)
            for text in texts
        ]
        rows_by_split[split] = rows
        counts_by_split[split] = Counter(chain.from_iterable(rows))

    tokens = set().union(*counts_by_split.values())
    matches = find_matches(args.archive, tokens)
    tokenization = (
        "Split on whitespace and on ASCII hyphens between word characters."
        if args.split_hyphens
        else "Split on whitespace."
    )
    output_suffix = "_hyphens" if args.split_hyphens else ""
    if args.slash_as_or:
        tokenization += " Replace each ASCII slash with a standalone 'or' token."
        output_suffix += "_slash_or"
    report = {
        "dataset": SST2_ID,
        "checkpoint_archive": args.archive.name,
        "tokenization": tokenization,
        "splits": {},
    }
    for split, counts in counts_by_split.items():
        coverage = summarize_coverage(rows_by_split[split], counts, matches)
        report["splits"][split] = coverage
        print(f"{split}: {coverage['token_coverage']:.2%} token coverage, "
              f"{coverage['type_coverage']:.2%} type coverage")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"glove_coverage{output_suffix}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved coverage to {output}")


if __name__ == "__main__":
    main()
