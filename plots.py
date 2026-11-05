"""Plot the saved validation results with `uv run python plots.py`."""

import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter


RESULTS_DIR = Path(__file__).with_name("results") / "validation"


def plot_scores(results):
    systems = sorted(
        results["systems"],
        key=lambda system: np.mean([fold["macro_f1"] for fold in system["folds"]]),
        reverse=True,
    )
    names = [" + ".join(system["name"]) for system in systems]
    rows = np.arange(len(systems))

    fig, axes = plt.subplots(
        1, 2, sharey=True, figsize=(10, 1.8 + 0.4 * len(systems)), layout="constrained"
    )
    fig.suptitle("Stance classification performance", fontsize=17, fontweight="bold")
    for ax, metric, title, color in zip(
        axes, ("macro_f1", "accuracy"), ("Macro-F1", "Accuracy"), ("#117C83", "#6850A1")
    ):
        values = np.array([[fold[metric] for fold in system["folds"]] for system in systems])
        means = values.mean(axis=1)
        # Show the folds so a high average can be read alongside its spread.
        ax.scatter(values.ravel(), np.repeat(rows, values.shape[1]), s=24, color=color, alpha=0.38)
        ax.scatter(means, rows, s=56, marker="D", color=color, edgecolor="white", linewidth=0.8)
        ax.set_title(title, loc="left", color=color, fontweight="bold")
        ax.set_xlim(max(0, values.min() - 0.03), min(1, values.max() + 0.03))
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="x", color="#E1E7ED")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
    axes[0].set_yticks(rows, names)
    axes[0].invert_yaxis()
    fig.supxlabel(
        f"{results['dataset']} · {results['n_examples']} tweets · "
        f"{results['n_splits']} folds per system · Diamonds show means",
        fontsize=9,
    )
    fig.savefig(RESULTS_DIR / "scores.png", dpi=180)
    plt.close(fig)


def plot_rope(results):
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(14, 6.5), layout="constrained")
    fig.suptitle(
        f"Bayesian ROPE comparisons\nPractical equivalence is within ±{results['rope']:.2f} macro-F1",
        fontsize=16, fontweight="bold",
    )

    for ax, fixed, title in zip(axes, (0, 1), ("Compare classifiers", "Compare representations")):
        groups = {}
        for pair in results["comparisons"]:
            if pair["left"][fixed] == pair["right"][fixed]:
                groups.setdefault(pair["left"][fixed], []).append(pair)

        names, headers, rows, probabilities = [], [], [], []
        for group, pairs in groups.items():
            headers.append(len(names))
            names.append(group)
            for pair in pairs:
                rows.append(len(names))
                names.append(f"{pair['left'][1 - fixed]} vs {pair['right'][1 - fixed]}")
                probabilities.append(pair["probabilities"])

        offsets = np.zeros(len(rows))
        for widths, label, color, text_color in zip(
            np.array(probabilities).T,
            ("First system better", "Practically equivalent", "Second system better"),
            ("#117C83", "#E5BE6D", "#6850A1"),
            ("white", "#263340", "white"),
        ):
            bars = ax.barh(rows, widths, left=offsets, height=0.66, color=color, label=label)
            ax.bar_label(
                bars, labels=[f"{width:.1%}" if width >= 0.10 else "" for width in widths],
                label_type="center", color=text_color, fontsize=9,
            )
            offsets += widths

        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_yticks(range(len(names)), names, fontsize=9)
        for index in headers:
            ax.get_yticklabels()[index].set_fontweight("bold")
        ax.set_ylim(len(names) - 0.45, -0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Posterior probability")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(axis="both", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E1E7ED")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(RESULTS_DIR / "rope.png", dpi=180)
    plt.close(fig)


def plot_boundaries():
    """Show the fitted classifiers on a PCA slice through the first training fold."""
    from sklearn.decomposition import PCA

    from data_loader import LABEL_NAMES, load_stance_split
    from experiment import DATA_SPLIT, ENCODERS, SEED, TOPIC, fit_fold, split_folds

    texts, labels = load_stance_split(TOPIC, split=DATA_SPLIT)
    train, heldout = next(split_folds(texts, labels))
    labels = np.asarray(labels)
    fitted = fit_fold([texts[i] for i in train], labels[train],
                      [texts[i] for i in heldout], seed=SEED + 1)

    colors = ("#0072B2", "#D55E00", "#009E73")
    class_colors = ListedColormap(colors)
    fig, axes = plt.subplots(
        len(ENCODERS), 3, sharex="row", sharey="row", figsize=(11.5, 9), layout="constrained"
    )
    fig.suptitle(
        "Stance decision regions · Cross-validation fold 1\n"
        "Background shows the model prediction on the PCA slice · Dots show true held-out labels",
        fontsize=14, fontweight="bold",
    )
    for row, (representation, train_features, heldout_features, predictors) in enumerate(fitted):
        pca = PCA(n_components=2, svd_solver="randomized", random_state=SEED).fit(train_features)
        points = pca.transform(heldout_features)
        projected = np.vstack((pca.transform(train_features), points))
        low, high = projected.min(axis=0), projected.max(axis=0)
        margin = 0.06 * (high - low)
        x, y = np.meshgrid(*[np.linspace(a, b, 110) for a, b in zip(low - margin, high + margin)])

        # Map the grid back to the full feature space, holding other PCs at zero.
        grid = pca.inverse_transform(np.column_stack((x.ravel(), y.ravel())))
        for ax, (name, predict) in zip(axes[row], predictors.items()):
            regions = predict(grid).reshape(x.shape)
            ax.pcolormesh(x, y, regions, cmap=class_colors, vmin=-0.5, vmax=len(LABEL_NAMES) - 0.5,
                          shading="nearest", alpha=0.24)
            for label, (name_label, color, marker) in enumerate(zip(LABEL_NAMES, colors, ("o", "s", "^"))):
                selected = points[labels[heldout] == label]
                ax.scatter(*selected.T, s=20, c=color, marker=marker,
                           linewidths=0.4, edgecolors="white", label=name_label)
            if row == 0:
                ax.set_title(name, fontweight="bold")
            ax.tick_params(labelbottom=True, labelsize=8)

        pc1, pc2 = pca.explained_variance_ratio_
        axes[row, 0].set_ylabel(f"{representation}\nPC 2 ({pc2:.1%})", fontweight="bold")
        axes[row, 1].set_xlabel(f"PC 1 ({pc1:.1%})")

    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="outside lower center", ncol=3, frameon=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "decision_boundaries.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved decision boundaries to {output}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", action="store_true", help="Fit the first fold and plot its decision regions.")
    if parser.parse_args().boundaries:
        plot_boundaries()
        return

    path = RESULTS_DIR / "results.json"
    if not path.is_file():
        raise SystemExit("Run `uv run python experiment.py` to generate the results first.")
    results = json.loads(path.read_text(encoding="utf-8"))
    with plt.style.context("seaborn-v0_8-white"):
        plot_scores(results)
        plot_rope(results)
    print(f"Saved scores.png and rope.png to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
