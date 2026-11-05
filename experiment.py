"""Cross-validate the stance classifiers with `uv run python experiment.py`."""

import json
from collections import defaultdict
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np
from baycomp import two_on_single
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tokenizers import ByteLevelBPETokenizer

from data_loader import load_stance_split
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


def fit_fold(train_texts, train_labels, heldout_texts, *, seed):
    """Fit each representation and its classifiers on the training fold."""
    # All systems share this vocabulary, learned only from training tweets.
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

        logistic = LogisticRegression(max_iter=1_000).fit(train_features, train_labels)
        svm = train_svm(train_features, train_labels, seed=seed)
        set_seed(seed)
        train_loader = make_loader(train_features, train_labels, shuffle=True)
        mlp, _ = fit_mlp(train_loader, input_dim=train_features.shape[1])
        predictors = {
            "Logistic regression": logistic.predict,
            "Linear SVM": svm.predict,
            "One-layer MLP": partial(predict_mlp, mlp),
        }
        yield representation, train_features, heldout_features, predictors


def run_cross_validation(texts, labels):
    """Return paired fold scores for every encoder and classifier combination."""
    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels, dtype=np.int64)
    scores = defaultdict(list)

    for fold, (train_idx, heldout_idx) in enumerate(split_folds(texts, labels), 1):
        train_texts, heldout_texts = texts[train_idx].tolist(), texts[heldout_idx].tolist()
        train_labels, heldout_labels = labels[train_idx], labels[heldout_idx]
        for representation, _, heldout_features, predictors in fit_fold(
            train_texts, train_labels, heldout_texts, seed=SEED + fold
        ):
            for name, predict in predictors.items():
                predictions = predict(heldout_features)
                scores[representation, name].append({
                    "accuracy": accuracy_score(heldout_labels, predictions),
                    "macro_f1": f1_score(
                        heldout_labels, predictions, labels=LABEL_IDS,
                        average="macro", zero_division=0,
                    ),
                })

    return scores


def main():
    # Pool the development splits and reserve the official test split for later.
    dataset_name = "TweetEval feminist stance (train + validation)"
    texts, labels = load_stance_split(TOPIC, split=DATA_SPLIT)
    scores = run_cross_validation(texts, labels)
    macro_f1 = {
        system: np.array([fold["macro_f1"] for fold in folds])
        for system, folds in scores.items()
    }

    print(f"{dataset_name} | {len(texts)} tweets | {N_SPLITS} folds")
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
        "n_splits": N_SPLITS,
        "rope": ROPE,
        "systems": [{"name": system, "folds": folds} for system, folds in scores.items()],
        "comparisons": comparisons,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
