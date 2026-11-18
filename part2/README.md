# Project 2

Use this guide to reproduce the SST-2 experiments.
[Results and interpretation](RESULTS.md)

## GloVe vocabulary coverage on SST-2

We check exact matches against the uncased **GloVe 2024 Wikipedia + Gigaword
100d** checkpoint. Stanford used a lowercasing Penn Treebank tokenizer to train
the embeddings. SST-2 already stores lowercase text with punctuation and
contraction splits such as `do n't`. Whitespace splitting preserves those
boundaries. [GloVe training notes](https://github.com/stanfordnlp/GloVe/blob/master/Training_README.md)

Token coverage counts every occurrence. Type coverage counts each distinct
token once. Both metrics are computed for the official train and validation
splits, including punctuation.

Run these commands from the `nlp/` directory. The download uses a
[mirror of Stanford's archive](https://huggingface.co/jasonmusespresso/glove-2024-wikigiga-100d).

```sh
mkdir -p results/part2/embeddings
curl --fail --location \
  'https://huggingface.co/jasonmusespresso/glove-2024-wikigiga-100d/resolve/main/glove.2024.wikigiga.100d.zip?download=true' \
  --output results/part2/embeddings/glove.2024.wikigiga.100d.zip
uv run python -m part2.glove_coverage
uv run python -m part2.glove_coverage --split-hyphens
uv run python -m part2.glove_coverage --split-hyphens --slash-as-or
```

The script reads the archive in place and writes counts and the most frequent
missing tokens under `results/part2/` in `glove_coverage.json`,
`glove_coverage_hyphens.json`, and `glove_coverage_hyphens_slash_or.json`.

## Word similarity probes

Run `uv run python -m part2.glove_similarity` from `nlp/` to write the full
scores and neighbors to `results/part2/glove_similarity.json`.

## Sentence encoder comparison

Run `uv run python -m part2.experiment` from `nlp/` with the same GloVe archive
as the coverage check. The four encoders share a linear sentiment head and
frozen GloVe 2024 WikiGigaword 100d embeddings. Tokenization splits on whitespace
and internal hyphens. Unknown words use a zero vector and padding is excluded
from pooling and recurrence.

| Encoder | Sentence representation |
| --- | --- |
| Embed-and-pool | Mean of the 100-dimensional word vectors |
| GRU | Final state of a one-layer, 128-unit GRU |
| Unidirectional LSTM | Final hidden state of a one-layer, 128-unit LSTM |
| Bidirectional LSTM | Final hidden states from both directions of a one-layer LSTM with 64 units per direction |

Each encoder trains for eight epochs on all 67,349 official training examples
with batch size 256, seed 42, and AdamW (learning rate 0.001, weight decay
0.01). We report the final epoch on the same 872 official validation examples
used in Project 1. The validation loss at each epoch helps show where continued
training starts to hurt generalization.

The run writes epoch metrics and final validation predictions in dataset order
to `results/part2/encoder_comparison.json` and saves the loss curves to
`results/part2/encoder_loss_curves.png`.

### Early stopping with patience

Run `uv run python -m part2.experiment --epochs 16 --patience 3` to select
training duration separately for each encoder. All four share one development
fold from a stratified ten-fold split of the official training data.
Identical tokenized examples stay together. Stop epoch selection after three
consecutive epochs without a strict improvement in development macro-F1, or
at 16 epochs. Select the earliest epoch with the best development macro-F1.

For each encoder, reset to the same seed and refit on all 67,349 official
training examples for the selected number of epochs. Score the refitted model
on official validation. The comparison uses the same training data, settings,
and evaluation split as the fixed eight-epoch run.

The run saves the selection history, refit losses, final metrics, and
predictions to `results/part2/encoder_comparison_patience.json`. The early
stopping curves in `results/part2/encoder_loss_curves_patience.png` show the
development loss and the epoch selected by development macro-F1.

### Hyperparameter screening

The screening uses seed 42 and the same eight-epoch training settings as
the encoder comparison. Each variant changes one setting against a recorded
control. It reuses the four controls in `results/part2/encoder_comparison.json`.

| Setting | Screening runs |
| --- | --- |
| Pooling | Max and sum, each compared with mean pooling |
| Pretrained vectors | Frozen GloVe Twitter 100d and WikiGigaword 2014 100d in the unidirectional LSTM, compared with its frozen WikiGigaword 2024 result |
| Head dropout | 0.2 dropout on the sentence vector before the linear head, once for GRU and once for the unidirectional LSTM |
| Embedding fine-tuning | Unfreeze the strongest of the three frozen embeddings in the unidirectional LSTM, using the same optimizer settings |

The highest final official validation macro-F1 determines the screening
winner. The winner and original unidirectional LSTM are evaluated with
seeds 7 and 123, reusing their seed-42 scores. These repeats measure training
variation on the same validation split used for screening.

Download the two additional checkpoints from
[Gensim's model catalog](https://github.com/piskvorky/gensim-data#models).
Run from `nlp/` after completing the fixed-epoch comparison above.

```sh
mkdir -p results/part2/embeddings
curl --fail --location \
  https://github.com/piskvorky/gensim-data/releases/download/glove-twitter-100/glove-twitter-100.gz \
  --output results/part2/embeddings/glove-twitter-100.gz
curl --fail --location \
  https://github.com/piskvorky/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz \
  --output results/part2/embeddings/glove-wiki-gigaword-100.gz
uv run python -m part2.sweep
```

The sweep writes settings, per-epoch scores, and paired seed comparisons to
`results/part2/hyperparameter_sweep.json`.
