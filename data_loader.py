"""Load single-label texts for the experiments."""

import re
from collections import Counter

from datasets import load_dataset

DATASET_ID = "cardiffnlp/tweet_eval"
GOEMOTIONS_ID = "google-research-datasets/go_emotions"
SST2_ID = "stanfordnlp/sst2"
DATASET_CHOICES = ("stance", "goemotions", "sst2")
LABEL_NAMES = ("none", "against", "favor")
_SEMST = re.compile(r"\s*#SemST\b", flags=re.IGNORECASE)


def load_stance_split(topic, split="train", *, strip_semst=True, cache_dir=None):
    """Load a Hugging Face split expression such as train+validation."""
    data = load_dataset(DATASET_ID, f"stance_{topic}", split=split, cache_dir=cache_dir)
    names = tuple(data.features["label"].names)
    if names != LABEL_NAMES:
        raise RuntimeError(f"Unexpected label order for {topic}: {names}, expected {LABEL_NAMES}")

    texts = data["text"]
    if strip_semst:
        texts = [_SEMST.sub("", text) for text in texts]
    return [text.strip() for text in texts], list(data["label"])


def load_goemotions_split(split="train+validation"):
    """Keep simplified GoEmotions comments that have exactly one gold label."""
    data = load_dataset(GOEMOTIONS_ID, "simplified", split=split)
    pairs = [(row["text"], row["labels"][0]) for row in data if len(row["labels"]) == 1]
    texts, labels = zip(*pairs)
    label_names = tuple(data.features["labels"].feature.names)
    return list(texts), list(labels), label_names, len(data)


def load_sst2_split(split="train"):
    """Load labeled SST-2 movie-review sentences and phrases."""
    data = load_dataset(SST2_ID, split=split)
    labels = list(data["label"])
    label_names = tuple(data.features["label"].names)
    if any(label < 0 or label >= len(label_names) for label in labels):
        raise ValueError("SST-2 evaluation requires a labeled split. Use train or validation.")
    return list(data["sentence"]), labels, label_names, len(data)


if __name__ == "__main__":
    _, labels = load_stance_split("feminist")
    counts = Counter(labels)
    print(f"TweetEval feminist stance train split ({len(labels)} tweets)")
    for label_id, name in enumerate(LABEL_NAMES):
        print(f"{name}: {counts[label_id]}")
