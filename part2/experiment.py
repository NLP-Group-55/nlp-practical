"""Compare four sentence encoders on the official SST-2 validation split."""

import json
import os
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Subset

from data_loader import SST2_ID

from .data import TOKENIZER_NAME, collate_batch, prepare_data
from .glove import DEFAULT_ARCHIVE, RESULTS_DIR
from .models import ENCODER_NAMES, SentimentClassifier
from .plots import save_loss_curves

OPTIMIZER_SETTINGS = {"lr": 0.001, "weight_decay": 0.01}
DEVELOPMENT_FOLDS = 10


def save_report(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(f".{os.getpid()}.tmp")
    partial.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    partial.replace(output)


def train_epoch(model, batches, optimizer, loss_fn):
    model.train()
    loss_total = 0.0
    for tokens, lengths, labels in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(tokens, lengths), labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item() * labels.numel()
    return loss_total / len(batches.dataset)


@torch.inference_mode()
def evaluate(model, batches, loss_fn):
    model.eval()
    loss_total = 0.0
    targets, predictions = [], []
    for tokens, lengths, labels in batches:
        logits = model(tokens, lengths)
        loss_total += loss_fn(logits, labels).item() * labels.numel()
        targets.extend(labels.tolist())
        predictions.extend(logits.argmax(dim=1).tolist())
    metrics = {
        "validation_loss": loss_total / len(targets),
        "macro_f1": f1_score(targets, predictions, labels=[0, 1], average="macro", zero_division=0),
        "accuracy": accuracy_score(targets, predictions),
    }
    return metrics, predictions


def start_training(encoder, embedding_weights, train_data, args, **model_options):
    torch.manual_seed(args.seed)
    train_batches = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = SentimentClassifier(encoder, embedding_weights, **model_options)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        **OPTIMIZER_SETTINGS,
    )
    return model, train_batches, optimizer


def train_encoder(
    encoder, embedding_weights, train_data, val_batches, args, *,
    patience=None, heldout_name="val", **model_options,
):
    model, train_batches, optimizer = start_training(
        encoder, embedding_weights, train_data, args, **model_options
    )
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_f1, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        train_loss = train_epoch(model, train_batches, optimizer, loss_fn)
        metrics, predictions = evaluate(model, val_batches, loss_fn)
        elapsed = perf_counter() - start
        history.append({
            "epoch": epoch, "train_loss": train_loss, "elapsed_seconds": elapsed, **metrics,
        })
        print(
            f"  Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} | "
            f"{heldout_name} loss {metrics['validation_loss']:.4f} | "
            f"macro-F1 {metrics['macro_f1']:.4f} | accuracy {metrics['accuracy']:.4f} "
            f"| {elapsed:.1f}s",
            flush=True,
        )
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
        if patience is not None and epoch - best_epoch >= patience:
            print(f"  Stopped after {patience} epochs without higher {heldout_name} macro-F1.")
            break
    return {
        "name": ENCODER_NAMES[encoder],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs": history,
        "final_validation_predictions": predictions,
    }


def select_and_refit(
    encoder, embedding_weights, train_data, selection_data, dev_batches, val_batches, args,
):
    selection = train_encoder(
        encoder, embedding_weights, selection_data, dev_batches, args,
        patience=args.patience, heldout_name="dev",
    )
    best = max(selection["epochs"], key=lambda entry: entry["macro_f1"])
    selected_epoch = best["epoch"]
    print(f"  Selected epoch {selected_epoch} (dev macro-F1 {best['macro_f1']:.4f}).")
    print(f"  Refitting on all {len(train_data):,} official training examples.", flush=True)
    model, batches, optimizer = start_training(encoder, embedding_weights, train_data, args)
    loss_fn = nn.CrossEntropyLoss()
    refit_losses = [
        train_epoch(model, batches, optimizer, loss_fn) for _ in range(selected_epoch)
    ]
    metrics, predictions = evaluate(model, val_batches, loss_fn)
    print(
        f"  Official validation | loss {metrics['validation_loss']:.4f} | "
        f"macro-F1 {metrics['macro_f1']:.4f} | accuracy {metrics['accuracy']:.4f}",
        flush=True,
    )
    return {
        **selection,
        "selected_epoch": selected_epoch,
        "refit_train_losses": refit_losses,
        "official_validation": metrics,
        "final_validation_predictions": predictions,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--patience", type=int,
        help="Stop after this many epochs without higher development macro-F1.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("Epochs, batch size, and thread count must be positive.")
    if args.patience is not None and args.patience < 1:
        parser.error("Patience must be positive.")
    if not args.archive.is_file():
        parser.error(f"Missing GloVe archive: {args.archive}. See the README for the download.")
    torch.set_num_threads(args.threads)

    data = prepare_data(args.archive)
    train_data, val_data, embedding_weights = data.train, data.validation, data.embedding_weights
    n_train, n_val = len(train_data), len(val_data)
    val_labels = [label for _, label in val_data]
    val_batches = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_batch)

    report = {
        "dataset": SST2_ID,
        "train_split": "train",
        "validation_split": "validation",
        "train_examples": n_train,
        "validation_examples": n_val,
        "label_names": data.label_names,
        "tokenizer": TOKENIZER_NAME,
        "checkpoint_archive": args.archive.name,
        "embeddings_frozen": True,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": "cpu",
        "threads": args.threads,
        "optimizer": {"name": "AdamW", **OPTIMIZER_SETTINGS},
        "validation_labels": val_labels,
        "models": {},
    }
    suffix = ""
    if args.patience is not None:
        # Keep identical tokenized examples together during epoch selection.
        groups = [" ".join(row) for row in data.train_rows]
        train_labels = [label for _, label in train_data]
        splitter = StratifiedGroupKFold(
            n_splits=DEVELOPMENT_FOLDS, shuffle=True, random_state=args.seed
        )
        fit_indices, dev_indices = next(splitter.split(data.train_rows, train_labels, groups))
        selection_data = Subset(train_data, fit_indices)
        dev_batches = DataLoader(
            Subset(train_data, dev_indices), batch_size=args.batch_size, collate_fn=collate_batch
        )
        report["epoch_selection"] = {
            "monitor": "development macro-F1",
            "patience": args.patience,
            "folds": DEVELOPMENT_FOLDS,
            "fit_examples": len(fit_indices),
            "development_examples": len(dev_indices),
            "grouping": "identical tokenized training examples",
            "refit_on_full_train": True,
        }
        suffix = "_patience"
        print(f"Epoch selection | {len(fit_indices):,} fit | {len(dev_indices):,} development")

    output = RESULTS_DIR / f"encoder_comparison{suffix}.json"
    vector_count = embedding_weights.size(0) - 2
    print(f"SST-2 | {n_train:,} train | {n_val:,} validation | {vector_count:,} GloVe vectors")
    for encoder in ENCODER_NAMES:
        print(f"\n{ENCODER_NAMES[encoder]}", flush=True)
        if args.patience is None:
            result = train_encoder(encoder, embedding_weights, train_data, val_batches, args)
        else:
            result = select_and_refit(
                encoder, embedding_weights, train_data, selection_data, dev_batches, val_batches, args
            )
        report["models"][encoder] = result
        save_report(output, report)
    save_loss_curves(
        report["models"], RESULTS_DIR / f"encoder_loss_curves{suffix}.png",
        heldout_label="Development" if args.patience is not None else "Validation",
    )
    print(f"\nSaved metrics and final predictions to {output}")


if __name__ == "__main__":
    main()
