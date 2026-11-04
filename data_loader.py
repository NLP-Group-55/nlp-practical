import re

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value, load_dataset

DATASET_ID = "cardiffnlp/tweet_eval"
TOPICS = ("abortion", "atheism", "climate", "feminist", "hillary")
LABEL_NAMES = ("none", "against", "favor")
SPLITS = ("train", "validation", "test")

_SEMST = re.compile(r"\s*#SemST\b", flags=re.IGNORECASE)


def clean_text(text, strip_semst=True):
    if strip_semst:
        text = _SEMST.sub("", text)
    return text.strip()


def _as_tuple(value, allowed, what):
    items = (value,) if isinstance(value, str) else tuple(value)
    allowed = tuple(allowed)
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown {what}(s) {unknown}; choose from {allowed}")
    return items


def load_stance(topics=TOPICS, strip_semst=True, cache_dir=None):
    topics = _as_tuple(topics, TOPICS, "topic")
    columns = {split: {"text": [], "label": [], "topic": []} for split in SPLITS}

    for topic in topics:
        raw = load_dataset(DATASET_ID, f"stance_{topic}", cache_dir=cache_dir)
        names = tuple(raw["train"].features["label"].names)
        if names != LABEL_NAMES:
            raise RuntimeError(f"Unexpected label names for {topic}: {names}, expected {LABEL_NAMES}")
        for split in SPLITS:
            columns[split]["text"].extend(clean_text(t, strip_semst) for t in raw[split]["text"])
            columns[split]["label"].extend(raw[split]["label"])
            columns[split]["topic"].extend([topic] * len(raw[split]))

    features = Features({
        "text": Value("string"),
        "label": ClassLabel(names=list(LABEL_NAMES)),
        "topic": Value("string"),
    })
    return DatasetDict({
        split: Dataset.from_dict(columns[split], features=features) for split in SPLITS
    })


def to_pairs(dataset, splits="train", prepend_topic=False):
    pairs = []
    for split in _as_tuple(splits, dataset.keys(), "split"):
        data = dataset[split]
        for text, label, topic in zip(data["text"], data["label"], data["topic"]):
            if prepend_topic:
                text = f"[{topic}] {text}"
            pairs.append((text, label))
    return pairs


def describe(dataset):
    rows = []
    for split in dataset:
        data = dataset[split]
        present = sorted(set(data["topic"]), key=TOPICS.index)
        for topic in present + ["all"]:
            labels = [label for label, t in zip(data["label"], data["topic"]) if topic in ("all", t)]
            counts = {name: labels.count(i) for i, name in enumerate(LABEL_NAMES)}
            rows.append({"split": split, "topic": topic, **counts, "total": len(labels)})
    return pd.DataFrame(rows).set_index(["split", "topic"])


def tokenizer_corpus(dataset, extra_configs=("sentiment",), strip_semst=True, cache_dir=None):
    texts = [text for split in ("train", "validation") if split in dataset for text in dataset[split]["text"]]
    for config in extra_configs:
        extra = load_dataset(DATASET_ID, config, cache_dir=cache_dir)["train"]["text"]
        texts.extend(clean_text(t, strip_semst) for t in extra)
    return texts


if __name__ == "__main__":
    dataset = load_stance()
    print(dataset, end="\n\n")
    print(describe(dataset).to_string(), end="\n\n")