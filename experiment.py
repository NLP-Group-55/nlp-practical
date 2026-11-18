"""Evaluate the text classifiers with `uv run python experiment.py`."""

import json
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np
from baycomp import two_on_single
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tokenizers import ByteLevelBPETokenizer

from data_loader import DATASET_CHOICES, LABEL_NAMES, load_goemotions_split, load_sst2_split, load_stance_split
from encoder import BagOfWords, TfidfEncoder, TokenIndicators
from mlp_one_layer import (
    LABEL_IDS,
    N_SPLITS,
    SEED,
    VOCAB_SIZE,
    fit_mlp,
    make_loader,
    predict_mlp,
    set_seed,
)
from svm import train_svm


ENCODERS = {
    "Token indicators": TokenIndicators,
    "Bag of words": BagOfWords,
    "TF-IDF": TfidfEncoder,
}
ROPE = 0.01  # Treat one macro-F1 point or less as practically equivalent.
TOPIC = "feminist"
DATA_SPLIT = "train+validation"
RESULTS_PATH = Path(__file__).with_name("results") / "validation" / "results.json"


def split_folds(texts, labels):
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(texts, labels)


def fit_fold(train_texts, train_labels, heldout_texts, *, seed, n_classes=len(LABEL_IDS)):
    """Fit each representation and its classifiers on the training examples."""
    # All systems share this vocabulary, learned only from the training examples.
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        train_texts, vocab_size=VOCAB_SIZE, min_frequency=2, show_progress=False
    )

    for representation, encoder_class in ENCODERS.items():
        encoder = encoder_class(tokenizer)
        if isinstance(encoder, TfidfEncoder):
            encoder.fit(train_texts)
        train_features = encoder(train_texts)
        heldout_features = encoder(heldout_texts)

        # Text vectors are mostly zero. Keep the linear solvers' input sparse.
        linear_features = csr_matrix(train_features)
        logistic = LogisticRegression(max_iter=1_000).fit(linear_features, train_labels)
        svm = train_svm(linear_features, train_labels, seed=seed)
        set_seed(seed)
        train_loader = make_loader(train_features, train_labels, shuffle=True)
        mlp, _ = fit_mlp(train_loader, input_dim=train_features.shape[1], n_classes=n_classes)
        predictors = {
            "Logistic regression": logistic.predict,
            "Linear SVM": svm.predict,
            "One-layer MLP": partial(predict_mlp, mlp),
        }
        yield representation, train_features, heldout_features, predictors


def score_split(train_texts, train_labels, heldout_texts, heldout_labels, *, seed, label_ids):
    scores = {}
    for representation, _, heldout_features, predictors in fit_fold(
        train_texts, train_labels, heldout_texts, seed=seed, n_classes=len(label_ids)
    ):
        for name, predict in predictors.items():
            predictions = predict(heldout_features)
            scores[representation, name] = {
                "accuracy": accuracy_score(heldout_labels, predictions),
                "macro_f1": f1_score(
                    heldout_labels, predictions, labels=label_ids,
                    average="macro", zero_division=0,
                ),
            }
        print(f"  {representation} done", flush=True)
    return scores


def run_cross_validation(texts, labels, *, label_ids=LABEL_IDS):
    """Return paired fold scores for every encoder and classifier combination."""
    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=len(label_ids))
    if np.any(counts < N_SPLITS):
        raise ValueError(f"Every class needs at least {N_SPLITS} examples for stratified CV.")
    scores = defaultdict(list)

    for fold, (train_idx, heldout_idx) in enumerate(split_folds(texts, labels), 1):
        print(f"Fold {fold}/{N_SPLITS} ({len(train_idx):,} training examples)", flush=True)
        train_texts, heldout_texts = texts[train_idx].tolist(), texts[heldout_idx].tolist()
        train_labels, heldout_labels = labels[train_idx], labels[heldout_idx]
        fold_scores = score_split(
            train_texts, train_labels, heldout_texts, heldout_labels, seed=SEED + fold, label_ids=label_ids
        )
        for system, metrics in fold_scores.items():
            scores[system].append(metrics)

    return scores


def run_sst2_validation():
    train_texts, train_labels, label_names, n_train = load_sst2_split("train")
    val_texts, val_labels, val_label_names, n_val = load_sst2_split("validation")
    if label_names != val_label_names:
        raise ValueError("SST-2 train and validation label names must match.")

    print(f"SST-2 official split | {n_train:,} train | {n_val:,} validation", flush=True)
    scores = score_split(
        train_texts, train_labels, val_texts, val_labels,
        seed=SEED + 1, label_ids=list(range(len(label_names))),
    )
    systems = []
    for system, metrics in sorted(scores.items(), key=lambda item: item[1]["macro_f1"], reverse=True):
        print(f"{' + '.join(system)} | macro-F1 {metrics['macro_f1']:.3f} | accuracy {metrics['accuracy']:.3f}")
        systems.append({"name": system, "metrics": metrics})

    results = {
        "dataset": "SST-2 sentiment",
        "evaluation": "official_validation",
        "train_split": "train",
        "evaluation_split": "validation",
        "n_train": n_train,
        "n_examples": n_val,
        "systems": systems,
    }
    results_path = RESULTS_PATH.parent / "sst2" / "official" / RESULTS_PATH.name
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved results to {results_path}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="stance")
    args = parser.parse_args()
    if args.dataset == "sst2":
        run_sst2_validation()
        return

    # Pool the labeled development splits for cross-validation.
    if args.dataset == "goemotions":
        dataset_name = "GoEmotions single-label (train + validation)"
        texts, labels, label_names, original_size = load_goemotions_split(DATA_SPLIT)
    else:
        dataset_name = "TweetEval feminist stance (train + validation)"
        texts, labels = load_stance_split(TOPIC, split=DATA_SPLIT)
        label_names, original_size = LABEL_NAMES, len(texts)

    results_path = (
        RESULTS_PATH if args.dataset == "stance"
        else RESULTS_PATH.parent / args.dataset / RESULTS_PATH.name
    )

    label_ids = list(range(len(label_names)))
    class_counts = dict(zip(label_names, np.bincount(labels, minlength=len(label_ids)).tolist()))
    print(f"{dataset_name} | {len(texts):,} of {original_size:,} examples | {N_SPLITS} folds", flush=True)
    print(f"Class counts: {class_counts}", flush=True)
    scores = run_cross_validation(texts, labels, label_ids=label_ids)
    macro_f1 = {
        system: np.array([fold["macro_f1"] for fold in folds])
        for system, folds in scores.items()
    }

    print(f"\n{dataset_name} | {len(texts):,} examples | {N_SPLITS} folds")
    for system in sorted(scores, key=lambda system: macro_f1[system].mean(), reverse=True):
        accuracy = np.mean([fold["accuracy"] for fold in scores[system]])
        print(f"{' + '.join(system)} | macro-F1 {macro_f1[system].mean():.3f} | accuracy {accuracy:.3f}")

    print(f"\nBayesian comparisons on paired macro-F1 scores (ROPE ±{ROPE:.2f})")
    comparisons = []
    for left, right in combinations(scores, 2):
        # Change one component at a time to isolate its effect.
        if left[0] == right[0] or left[1] == right[1]:
            p_left, p_rope, p_right = two_on_single(macro_f1[left], macro_f1[right], rope=ROPE)
            comparisons.append({
                "left": left,
                "right": right,
                "probabilities": [p_left, p_rope, p_right],
            })
            print(
                f"{' + '.join(left)} vs {' + '.join(right)} | "
                f"P(left) {p_left:.1%} | P(ROPE) {p_rope:.1%} | P(right) {p_right:.1%}"
            )

    results = {
        "dataset": dataset_name,
        "n_examples": len(texts),
        "n_examples_before_filter": original_size,
        "class_counts": class_counts,
        "n_splits": N_SPLITS,
        "rope": ROPE,
        "systems": [{"name": system, "folds": folds} for system, folds in scores.items()],
        "comparisons": comparisons,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
