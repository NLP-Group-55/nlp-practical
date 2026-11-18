"""Word tokenization for the Project 2 experiments."""

import re

INTERNAL_HYPHEN = re.compile(r"(?<=\w)-(?=\w)")


def tokenize(text, *, split_hyphens=False, slash_as_or=False):
    """Split on whitespace with optional normalization of hyphens and slashes."""
    if split_hyphens:
        text = INTERNAL_HYPHEN.sub(" ", text)
    if slash_as_or:
        text = text.replace("/", " or ")
    return text.split()
