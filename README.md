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
