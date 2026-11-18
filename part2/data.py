"""Prepare the shared SST-2 word indices and batches for an embedding archive."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from data_loader import load_sst2_split

from .glove import EMBEDDING_SIZE, load_vectors
from .models import PAD_INDEX
from .tokenizer import tokenize

UNKNOWN_INDEX = 1
TOKENIZER_NAME = "Whitespace and internal ASCII hyphens"
Example = tuple[torch.Tensor, int]


@dataclass(frozen=True)
class PreparedData:
    train_rows: list[list[str]]
    train: list[Example]
    validation: list[Example]
    label_names: tuple[str, ...]
    embedding_weights: torch.Tensor
    validation_token_coverage: float


def make_dataset(rows, labels, word_indices):
    examples = []
    for words, label in zip(rows, labels, strict=True):
        ids = [word_indices.get(word, UNKNOWN_INDEX) for word in words]
        examples.append((torch.tensor(ids or [UNKNOWN_INDEX], dtype=torch.long), label))
    return examples


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([sequence.numel() for sequence in sequences])
    tokens = pad_sequence(sequences, batch_first=True, padding_value=PAD_INDEX)
    return tokens, lengths, torch.tensor(labels)


def prepare_data(archive_path):
    train_texts, train_labels, label_names, _ = load_sst2_split("train")
    val_texts, val_labels, val_label_names, _ = load_sst2_split("validation")
    if label_names != val_label_names or len(label_names) != 2:
        raise ValueError("Expected two matching sentiment labels in both SST-2 splits.")

    train_rows = [tokenize(text, split_hyphens=True) for text in train_texts]
    val_rows = [tokenize(text, split_hyphens=True) for text in val_texts]
    vocabulary = {word for rows in (train_rows, val_rows) for row in rows for word in row}
    vectors = load_vectors(archive_path, vocabulary)
    word_indices = {word: index for index, word in enumerate(sorted(vectors), start=2)}
    weights = np.zeros((len(word_indices) + 2, EMBEDDING_SIZE), dtype=np.float32)
    for word, index in word_indices.items():
        weights[index] = vectors[word]
    matched_tokens = sum(word in word_indices for row in val_rows for word in row)
    return PreparedData(
        train_rows=train_rows,
        train=make_dataset(train_rows, train_labels, word_indices),
        validation=make_dataset(val_rows, val_labels, word_indices),
        label_names=label_names,
        embedding_weights=torch.from_numpy(weights),
        validation_token_coverage=matched_tokens / sum(map(len, val_rows)),
    )
