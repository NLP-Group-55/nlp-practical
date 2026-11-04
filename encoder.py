import numpy as np


class BagOfWords:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Number of features: one per vocabulary token."""
        return self.tokenizer.get_vocab_size()

    def __call__(self, text: str) -> np.ndarray:
        """Return a float32 vector containing BPE token counts."""
        token_ids = self.tokenizer.encode(text).ids
        vector = np.zeros(len(self), dtype=np.float32)
        np.add.at(vector, token_ids, 1)
        return vector