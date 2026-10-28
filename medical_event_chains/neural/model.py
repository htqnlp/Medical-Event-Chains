from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class TinyEncoder(nn.Module):
    """Random character embeddings for offline execution tests, NOT pretrained BERT."""
    def __init__(self, vocab_size, hidden_size=32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TriggerModel(nn.Module):
    """Sentence BiLSTM features + context BiGRU features, fused by addition.

    All three slots receive predictions. Padding is removed before both recurrent
    layers; different-length sentences never introduce padding into context.
    """
    def __init__(self, encoder, num_labels, hidden_size=128, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.sentence = nn.LSTM(encoder.config.hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.context = nn.GRU(2 * hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        batch, slots, length = input_ids.shape
        if slots != 3:
            raise ValueError("Exactly three sentence slots are required")
        flat_mask = attention_mask.reshape(batch * slots, length)
        lengths = flat_mask.sum(1)
        if (lengths <= 0).any():
            raise ValueError("Empty sentence slot")
        encoded = self.encoder(input_ids=input_ids.reshape(batch * slots, length), attention_mask=flat_mask).last_hidden_state
        packed = pack_padded_sequence(encoded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        sentence, _ = self.sentence(packed)
        sentence, _ = pad_packed_sequence(sentence, batch_first=True, total_length=length)
        sentence = sentence.reshape(batch, slots, length, -1)
        lengths = lengths.reshape(batch, slots)
        sequences = [torch.cat([sentence[b, s, :int(lengths[b, s])] for s in range(slots)], 0) for b in range(batch)]
        context_lengths = lengths.sum(1)
        packed_context = pack_padded_sequence(pad_sequence(sequences, batch_first=True), context_lengths.cpu(), batch_first=True, enforce_sorted=False)
        context, _ = self.context(packed_context)
        context, _ = pad_packed_sequence(context, batch_first=True)
        scattered = torch.zeros_like(sentence)
        for b in range(batch):
            cursor = 0
            for s in range(slots):
                count = int(lengths[b, s])
                scattered[b, s, :count] = context[b, cursor:cursor + count]
                cursor += count
        logits = self.classifier(self.dropout(sentence + scattered))
        loss = None
        if labels is not None:
            if not (labels != -100).any():
                raise ValueError("Batch has no supervised tokens")
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}
