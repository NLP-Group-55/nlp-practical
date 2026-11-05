import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer


class BagOfWords:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Number of features: one per vocabulary token."""
        return self.tokenizer.get_vocab_size()

    def __call__(self, texts: list[str]) -> np.ndarray:
        """Return float32 counts with shape (batch_size, vocab_size)."""
        encodings = self.tokenizer.encode_batch(texts)
        vectors = np.zeros((len(texts), len(self)), dtype=np.float32)

        for i, encoding in enumerate(encodings):
            np.add.at(vectors[i], encoding.ids, 1)

        return vectors


class TokenIndicators(BagOfWords):
    """Record whether each BPE token occurs in a tweet."""

    def __call__(self, texts: list[str]) -> np.ndarray:
        return np.minimum(super().__call__(texts), 1)


class TfidfEncoder:
    """Turn BPE token counts into TF-IDF features."""

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
