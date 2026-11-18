"""Sentence encoders and their shared linear sentiment head."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

PAD_INDEX = 0
ENCODER_NAMES = {
    "mean_pool": "Embed-and-pool",
    "gru": "GRU",
    "lstm": "Unidirectional LSTM",
    "bilstm": "Bidirectional LSTM",
}


class SentimentClassifier(nn.Module):
    """Encode a batch of sentences using pretrained word vectors."""

    def __init__(
        self, encoder, embedding_weights, *, pooling="mean", head_dropout=0.0, freeze_embeddings=True,
    ):
        super().__init__()
        if pooling not in ("mean", "max", "sum"):
            raise ValueError(f"Unknown pooling method: {pooling}")
        self.pooling = pooling
        if not freeze_embeddings:
            embedding_weights = embedding_weights.clone()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights, freeze=freeze_embeddings, padding_idx=PAD_INDEX
        )
        embedding_size = embedding_weights.size(1)
        match encoder:
            case "mean_pool":
                self.recurrent = None
                output_size = embedding_size
            case "gru":
                output_size = 128
                self.recurrent = nn.GRU(embedding_size, output_size, batch_first=True)
            case "lstm" | "bilstm":
                output_size = 128
                bidirectional = encoder == "bilstm"
                self.recurrent = nn.LSTM(
                    embedding_size, 64 if bidirectional else 128,
                    batch_first=True, bidirectional=bidirectional,
                )
            case _:
                raise ValueError(f"Unknown encoder: {encoder}")
        self.head_dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(output_size, 2)

    def forward(self, tokens, lengths):
        embeddings = self.embedding(tokens)
        if self.recurrent is None:
            lengths = lengths.to(tokens.device).unsqueeze(1)
            positions = torch.arange(tokens.size(1), device=tokens.device)
            mask = (positions < lengths).unsqueeze(2)
            if self.pooling == "max":
                encoded = embeddings.masked_fill(~mask, -torch.inf).amax(dim=1)
            else:
                encoded = (embeddings * mask).sum(1)
                if self.pooling == "mean":
                    encoded = encoded / lengths
        else:
            packed = pack_padded_sequence(
                embeddings, lengths, batch_first=True, enforce_sorted=False
            )
            _, state = self.recurrent(packed)
            hidden = state[0] if isinstance(state, tuple) else state
            encoded = hidden.transpose(0, 1).flatten(1)
        return self.classifier(self.head_dropout(encoded))
