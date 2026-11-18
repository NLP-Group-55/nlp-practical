"""Plot training and held-out loss curves for the four encoders."""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def save_loss_curves(models, output, *, heldout_label="Validation"):
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 6), layout="constrained")
    for axis, model in zip(axes.flat, models.values(), strict=True):
        history = model["epochs"]
        epochs = [entry["epoch"] for entry in history]
        for key, label, color in (
            ("train_loss", "Training", "#26736F"),
            ("validation_loss", heldout_label, "#B05234"),
        ):
            axis.plot(epochs, [entry[key] for entry in history], "o-", label=label, color=color)
        if "selected_epoch" in model:
            axis.axvline(
                model["selected_epoch"], color="#555555", linestyle="--", label="Selected epoch"
            )
        axis.set_title(model["name"], loc="left", fontweight="bold")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncol=len(labels), frameon=False)
    fig.suptitle("SST-2 encoder comparison", x=0.06, ha="left", fontweight="bold")
    fig.supxlabel("Epoch")
    fig.supylabel("Cross-entropy loss")
    fig.savefig(output, dpi=180)
    plt.close(fig)
