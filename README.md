# nlp-practical

NLP1 practicals: Stance Detection on the stance subsets of [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval).

The [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
experiment uses the `simplified` configuration and keeps comments with exactly
one label. It runs five-fold stratified cross-validation on the pooled train
and validation splits. Each fold fits its tokenizer and TF-IDF weights on its
training comments. The official test split is reserved for final evaluation.

```sh
uv run python experiment.py --dataset goemotions
uv run python plots.py --dataset goemotions
```

Results and figures are written to `results/validation/goemotions/`.

[SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) is the group's selected
dataset. The experiment fits the same nine setups on the official training split
and reports macro-F1 and accuracy on the official validation split. Its single
holdout provides one score per setup.

```sh
uv run python experiment.py --dataset sst2
uv run python plots.py --dataset sst2
```

Results and the score chart are written to `results/validation/sst2/official/`.
