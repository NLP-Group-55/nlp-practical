"""One-hidden-layer MLP experiments over available text encoders.

Run directly:
    uv run python MLP_One_Layer.py

Or import run_cross_validation() from a group experiment script.
Importing this module does not load data or start training.

Encoder interface:
    encoder = EncoderClass(tokenizer)
    features = encoder(list_of_texts)

Features must be a dense matrix with one row per text. If an encoder
needs fitting (e.g. IDF estimation), expose fit(training_texts). The CV
function calls it only on the fold-training texts.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from tokenizers import ByteLevelBPETokenizer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_loader import load_stance, to_pairs
from encoder import BagOfWords


SEED = 42
N_SPLITS = 5
VOCAB_SIZE = 2_000
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_IDS = [0, 1, 2]  # none, against, favor

# Add the other encoder classes after implementing and importing them.
# They must support the encoder interface described above.
ENCODER_CLASSES = [BagOfWords]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Seeding does not guarantee identical results across hardware,
    # PyTorch versions, or nondeterministic GPU operations.


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        return self.linear2(x)  # Raw logits for CrossEntropyLoss.


def make_loader(features, labels, shuffle, batch_size=BATCH_SIZE):
    features = torch.as_tensor(features, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, loader, optimizer, criterion, device=DEVICE):
    """Train for ONE epoch; return mean loss per example."""
    model.train()
    total_loss = 0.0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


def fit_mlp(
    train_loader,
    input_dim,
    *,
    hidden_dim=HIDDEN_DIM,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    device=DEVICE,
):
    """Fit the MLP for a fixed number of epochs on the training fold."""
    model = MLP(input_dim, hidden_dim, len(LABEL_IDS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    loss_history = [
        train_model(model, train_loader, optimizer, criterion, device)
        for _ in range(epochs)
    ]
    return model, loss_history


@torch.no_grad()
def evaluate_model(model, loader, device=DEVICE):
    model.eval()
    all_predictions = []
    all_labels = []

    for features, labels in loader:
        logits = model(features.to(device))
        predicted_labels = logits.argmax(dim=1)
        all_predictions.extend(predicted_labels.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return {
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "macro_f1": float(f1_score(
            all_labels,
            all_predictions,
            labels=LABEL_IDS,
            average="macro",
            zero_division=0,
        )),
        "classification_report": classification_report(
            all_labels,
            all_predictions,
            labels=LABEL_IDS,
            target_names=["none", "against", "favor"],
            output_dict=True,
            zero_division=0,
        ),
        "predictions": all_predictions,
        "true_labels": all_labels,
    }


def run_cross_validation(
    texts,
    labels,
    encoder_class=BagOfWords,
    *,
    seed=SEED,
    n_splits=N_SPLITS,
    vocab_size=VOCAB_SIZE,
    hidden_dim=HIDDEN_DIM,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    device=DEVICE,
):
    """Evaluate THIS MLP with ONE encoder; return serializable results.

    Use identical texts, labels, seed, and n_splits across systems to
    obtain matching held-out folds. main() calls this once per encoder.
    """
    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels, dtype=np.int64)
    device = torch.device(device)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_indices, heldout_indices) in enumerate(
        splitter.split(texts, labels), start=1
    ):
        set_seed(seed + fold)
        train_texts = texts[train_indices].tolist()
        heldout_texts = texts[heldout_indices].tolist()

        # Fit a fresh BPE tokenizer on this fold's training texts only.
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(
            train_texts,
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=False,
        )

        encoder = encoder_class(tokenizer)
        if hasattr(encoder, "fit"):
            encoder.fit(train_texts)

        train_features = encoder(train_texts)
        heldout_features = encoder(heldout_texts)
        train_loader = make_loader(
            train_features, labels[train_indices], True, batch_size
        )
        heldout_loader = make_loader(
            heldout_features, labels[heldout_indices], False, batch_size
        )

        model, loss_history = fit_mlp(
            train_loader,
            input_dim=train_features.shape[1],
            hidden_dim=hidden_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
        )

        scores = evaluate_model(model, heldout_loader, device)
        fold_results.append({
            "fold": fold,
            "training_seed": seed + fold,
            "n_train": len(train_indices),
            "n_heldout": len(heldout_indices),
            "input_dim": int(train_features.shape[1]),
            "heldout_indices": heldout_indices.tolist(),
            "training_loss": loss_history,
            **scores,
        })
        print(
            f"MLP + {encoder_class.__name__} | fold {fold}/{n_splits} | "
            f"macro-F1: {scores['macro_f1']:.4f} | "
            f"accuracy: {scores['accuracy']:.4f}"
        )

    f1_scores = np.array([result["macro_f1"] for result in fold_results])
    return {
        "architecture": "single_hidden_layer_mlp",
        "representation": encoder_class.__name__,
        "config": {
            "split_seed": seed,
            "n_splits": n_splits,
            "requested_vocab_size": vocab_size,
            "tokenizer_min_frequency": 2,
            "hidden_dim": hidden_dim,
            "activation": "ReLU",
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "loss": "unweighted_cross_entropy",
            "device": str(device),
        },
        "folds": fold_results,
        "mean_macro_f1": float(f1_scores.mean()),
        "std_macro_f1": float(f1_scores.std(ddof=1)),
        # Fold standard deviation is descriptive, not a confidence interval
        # or a substitute for the project's Bayesian/ROPE analysis.
    }


def main():
    dataset = load_stance(topics="feminist")
    pairs = to_pairs(dataset, splits="train")
    texts = [text for text, label in pairs]
    labels = [label for text, label in pairs]

    # Agree with your team whether to use train or train + validation.
    # This example preserves the current train-only protocol.
    all_results = []
    for encoder_class in ENCODER_CLASSES:
        result = run_cross_validation(
            texts, labels, encoder_class=encoder_class
        )
        all_results.append(result)

    output = {
        "data": {
            "dataset": "cardiffnlp/tweet_eval",
            "configuration": "stance_feminist",
            "splits_used": ["train"],
            "label_names": ["none", "against", "favor"],
            "n_examples": len(texts),
            "strip_semst": True,
        },
        "systems": all_results,
    }
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mlp_one_layer_results.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n--- Representation summary ---")
    for result in all_results:
        print(
            f"{result['representation']}: "
            f"mean macro-F1 = {result['mean_macro_f1']:.4f}"
        )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()