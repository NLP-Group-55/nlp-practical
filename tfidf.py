"""Turn BPE token counts into TF-IDF features."""

from sklearn.feature_extraction.text import TfidfTransformer
from encoder import BagOfWords

class TfidfEncoder:
    def __init__(self, tokenizer):
        self.token_counts = BagOfWords(tokenizer)
        self.transformer = TfidfTransformer()

    def fit(self, texts):
        """Learn how common each token is across the training texts."""
        self.transformer.fit(self.token_counts(texts))
        return self

    def __call__(self, texts):
        # Apply the learned weights to the token counts for each tweet.
        return self.transformer.transform(self.token_counts(texts)).toarray()
