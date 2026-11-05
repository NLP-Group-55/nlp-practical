"""Load texts and labels from TweetEval stance."""

import re
from collections import Counter

from datasets import load_dataset

DATASET_ID = "cardiffnlp/tweet_eval"
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


if __name__ == "__main__":
    _, labels = load_stance_split("feminist")
    counts = Counter(labels)
    print(f"TweetEval feminist stance train split ({len(labels)} tweets)")
    for label_id, name in enumerate(LABEL_NAMES):
        print(f"{name}: {counts[label_id]}")
