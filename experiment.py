"""Cross-validate the stance classifiers with `uv run python experiment.py`."""

from collections import defaultdict
from itertools import combinations

import numpy as np
from baycomp import two_on_single
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tokenizers import ByteLevelBPETokenizer

from data_loader import load_stance, to_pairs
from encoder import BagOfWords, TfidfEncoder, TokenIndicators
from mlp_one_layer import (
    LABEL_IDS,
    N_SPLITS,
    SEED,
    VOCAB_SIZE,
    evaluate_model,
    fit_mlp,
    make_loader,
    set_seed,
)
from svm import train_svm


ENCODERS = {
    "Token indicators": TokenIndicators,
    "Bag of words": BagOfWords,
    "TF-IDF": TfidfEncoder,
}
ROPE = 0.01  # Treat one macro-F1 point or less as practically equivalent.


def run_cross_validation(texts, labels):
    """Return paired fold scores for every encoder and classifier combination."""
    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scores = defaultdict(list)

    for fold, (train_idx, heldout_idx) in enumerate(splitter.split(texts, labels), 1):
        train_texts, heldout_texts = texts[train_idx].tolist(), texts[heldout_idx].tolist()
        train_labels, heldout_labels = labels[train_idx], labels[heldout_idx]
        seed = SEED + fold

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
            for name, model in (("Logistic regression", logistic), ("Linear SVM", svm)):
                predictions = model.predict(heldout_features)
                scores[representation, name].append({
                    "accuracy": accuracy_score(heldout_labels, predictions),
                    "macro_f1": f1_score(
                        heldout_labels, predictions, labels=LABEL_IDS,
                        average="macro", zero_division=0,
                    ),
                })

            set_seed(seed)
            train_loader = make_loader(train_features, train_labels, shuffle=True)
            heldout_loader = make_loader(heldout_features, heldout_labels, shuffle=False)
            mlp, _ = fit_mlp(train_loader, input_dim=train_features.shape[1])
            metrics = evaluate_model(mlp, heldout_loader)
            scores[representation, "One-layer MLP"].append({
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            })

    return scores


def main():
    # Use the same TweetEval topic and train split as the existing MLP run.
    dataset = load_stance(topics="feminist")
    texts, labels = zip(*to_pairs(dataset, splits="train"))
    scores = run_cross_validation(texts, labels)
    macro_f1 = {
        system: np.array([fold["macro_f1"] for fold in folds])
        for system, folds in scores.items()
    }

    print(f"TweetEval feminist stance | {len(texts)} tweets | {N_SPLITS} folds")
    for system in sorted(scores, key=lambda system: macro_f1[system].mean(), reverse=True):
        accuracy = np.mean([fold["accuracy"] for fold in scores[system]])
        print(f"{' + '.join(system)} | macro-F1 {macro_f1[system].mean():.3f} | accuracy {accuracy:.3f}")

    print(f"\nBayesian comparisons on paired macro-F1 scores (ROPE ±{ROPE:.2f})")
    for left, right in combinations(scores, 2):
        # Change one component at a time to isolate its effect.
        if left[0] == right[0] or left[1] == right[1]:
            p_left, p_rope, p_right = two_on_single(macro_f1[left], macro_f1[right], rope=ROPE)
            print(
                f"{' + '.join(left)} vs {' + '.join(right)} | "
                f"P(left) {p_left:.1%} | P(ROPE) {p_rope:.1%} | P(right) {p_right:.1%}"
            )


if __name__ == "__main__":
    main()
