"""Screen seven SST-2 hyperparameter variants and repeat the finalists with paired seeds."""

import json
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, replace
from statistics import mean, stdev
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from data_loader import SST2_ID

from .data import TOKENIZER_NAME, collate_batch, prepare_data
from .experiment import OPTIMIZER_SETTINGS, save_report, train_encoder
from .glove import ARCHIVE_NAME, RESULTS_DIR
from .models import ENCODER_NAMES

EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
BASELINE_REPORT = RESULTS_DIR / "encoder_comparison.json"
OUTPUT = RESULTS_DIR / "hyperparameter_sweep.json"
SEEDS = (42, 7, 123)
EPOCHS = 8
BATCH_SIZE = 256
THREADS = 4


@dataclass(frozen=True)
class Configuration:
    name: str
    encoder: str
    archive: str = ARCHIVE_NAME
    pooling: str = "mean"
    head_dropout: float = 0.0
    freeze_embeddings: bool = True
    compared_with: str | None = None


CONTROLS = {encoder: Configuration(encoder, encoder) for encoder in ENCODER_NAMES}
REFERENCE = CONTROLS["lstm"]
POOLING_VARIANTS = (
    replace(CONTROLS["mean_pool"], name="max_pool", pooling="max", compared_with="mean_pool"),
    replace(CONTROLS["mean_pool"], name="sum_pool", pooling="sum", compared_with="mean_pool"),
)
EMBEDDING_VARIANTS = (
    replace(
        REFERENCE, name="lstm_twitter", archive="glove-twitter-100.gz", compared_with="lstm",
    ),
    replace(
        REFERENCE, name="lstm_wikigiga_2014", archive="glove-wiki-gigaword-100.gz",
        compared_with="lstm",
    ),
)
DROPOUT_VARIANTS = (
    replace(CONTROLS["gru"], name="gru_dropout", head_dropout=0.2, compared_with="gru"),
    replace(REFERENCE, name="lstm_dropout", head_dropout=0.2, compared_with="lstm"),
)


def read_baseline():
    if not BASELINE_REPORT.is_file():
        raise SystemExit("Generate the fixed eight-epoch controls with uv run python -m part2.experiment.")
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    expected = {
        "dataset": SST2_ID,
        "train_split": "train",
        "validation_split": "validation",
        "train_examples": 67349,
        "validation_examples": 872,
        "tokenizer": TOKENIZER_NAME,
        "checkpoint_archive": ARCHIVE_NAME,
        "embeddings_frozen": True,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEEDS[0],
        "device": "cpu",
        "threads": THREADS,
        "optimizer": {"name": "AdamW", **OPTIMIZER_SETTINGS},
    }
    for key, value in expected.items():
        if baseline.get(key) != value:
            raise ValueError(f"Baseline setting {key!r} does not match the sweep.")
    if set(baseline["models"]) != set(CONTROLS):
        raise ValueError("Expected all four models in the baseline report.")
    if any(len(entry["epochs"]) != EPOCHS for entry in baseline["models"].values()):
        raise ValueError("Expected eight completed epochs per baseline model.")
    return baseline


def seed_summary(by_seed):
    final = [by_seed[str(seed)]["epochs"][-1] for seed in SEEDS]
    scores = [metrics["macro_f1"] for metrics in final]
    accuracies = [metrics["accuracy"] for metrics in final]
    return {
        "mean_macro_f1": mean(scores),
        "sample_sd_macro_f1": stdev(scores),
        "mean_accuracy": mean(accuracies),
        "sample_sd_accuracy": stdev(accuracies),
    }


def main():
    ArgumentParser(description=__doc__).parse_args()
    baseline = read_baseline()
    for config in (REFERENCE, *EMBEDDING_VARIANTS):
        path = EMBEDDINGS_DIR / config.archive
        if not path.is_file():
            raise SystemExit(f"Missing {path}. See the Project 2 README for the download.")
    torch.set_num_threads(THREADS)

    screening_seed = str(SEEDS[0])
    report = {
        key: value for key, value in baseline.items()
        if key not in {"checkpoint_archive", "seed", "models", "embeddings_frozen"}
    }
    report["seeds"] = SEEDS
    runs = report["runs"] = {
        name: {"configuration": asdict(config), "by_seed": {screening_seed: baseline["models"][name]}}
        for name, config in CONTROLS.items()
    }
    data_by_archive = {}

    def record_run(config, seed):
        print(f"\n{config.name} | seed {seed}", flush=True)
        if config.archive not in data_by_archive:
            data_by_archive[config.archive] = prepare_data(EMBEDDINGS_DIR / config.archive)
        data = data_by_archive[config.archive]
        if [label for _, label in data.validation] != report["validation_labels"]:
            raise ValueError("Validation labels differ from the reference report.")
        val_batches = DataLoader(data.validation, batch_size=BATCH_SIZE, collate_fn=collate_batch)
        args = SimpleNamespace(seed=seed, epochs=EPOCHS, batch_size=BATCH_SIZE)
        result = train_encoder(
            config.encoder, data.embedding_weights, data.train, val_batches, args,
            pooling=config.pooling, head_dropout=config.head_dropout,
            freeze_embeddings=config.freeze_embeddings,
        )
        entry = runs.setdefault(config.name, {"configuration": asdict(config), "by_seed": {}})
        entry["validation_token_coverage"] = data.validation_token_coverage
        entry["by_seed"][str(seed)] = result
        save_report(OUTPUT, report)

    def screening_f1(config):
        return runs[config.name]["by_seed"][screening_seed]["epochs"][-1]["macro_f1"]

    for config in (*POOLING_VARIANTS, *EMBEDDING_VARIANTS, *DROPOUT_VARIANTS):
        record_run(config, SEEDS[0])
    best_embedding = max((REFERENCE, *EMBEDDING_VARIANTS), key=screening_f1)
    fine_tuned = replace(
        best_embedding, name="lstm_fine_tuned", freeze_embeddings=False,
        compared_with=best_embedding.name,
    )
    record_run(fine_tuned, SEEDS[0])

    for name in CONTROLS:
        runs[name]["validation_token_coverage"] = data_by_archive[ARCHIVE_NAME].validation_token_coverage
    candidates = (*CONTROLS.values(), *POOLING_VARIANTS, *EMBEDDING_VARIANTS, *DROPOUT_VARIANTS, fine_tuned)
    ranked = sorted(candidates, key=screening_f1, reverse=True)
    winner = ranked[0]
    report["screening_rank"] = [config.name for config in ranked]
    report["best_frozen_embedding"] = best_embedding.name
    report["screening_winner"] = winner.name
    report["reference"] = REFERENCE.name
    print(f"\nScreening winner: {winner.name} ({screening_f1(winner):.4f} macro-F1).", flush=True)
    save_report(OUTPUT, report)

    finalists = dict.fromkeys((winner, REFERENCE))
    for seed in SEEDS[1:]:
        for config in finalists:
            record_run(config, seed)
    report["finalist_summary"] = {
        config.name: seed_summary(runs[config.name]["by_seed"]) for config in finalists
    }
    if winner != REFERENCE:
        report["paired_macro_f1_differences"] = {
            str(seed): (
                runs[winner.name]["by_seed"][str(seed)]["epochs"][-1]["macro_f1"]
                - runs[REFERENCE.name]["by_seed"][str(seed)]["epochs"][-1]["macro_f1"]
            )
            for seed in SEEDS
        }
    save_report(OUTPUT, report)
    print(f"Saved the screening runs and seed repeats to {OUTPUT}")


if __name__ == "__main__":
    main()
