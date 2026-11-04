import numpy as np


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
    