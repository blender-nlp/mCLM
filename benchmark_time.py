import torch
import time
import random
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

def benchmark(vocab_size=10000, seq_len=128, batch_size=16, device='cuda'):
    total_tokens = batch_size * seq_len
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    indices = list(range(vocab_size))  # Simulated "custom" indices
    random.shuffle(indices)  # To simulate lookup cost

    # First version (with re-indexing)
    start = time.time()
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1).tolist()  # required for .index()

    # Re-indexing
    reindexed_labels = torch.LongTensor([
        indices.index(x) for x in shift_labels
    ]).to(device)

    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits, reindexed_labels)
    torch.cuda.synchronize()
    t1 = time.time() - start

    # Second version (no re-indexing)
    start = time.time()
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    loss = loss_fct(shift_logits, shift_labels)
    torch.cuda.synchronize()
    t2 = time.time() - start

    return t1, t2


for vocab_size in [1000, 5000, 10000, 20000, 50000, 100000, 500000, 1000000, 25000000]:
    t1, t2 = benchmark(vocab_size)
    print(f"Vocab size: {vocab_size:6d} | With indexing: {t1:.4f}s | No indexing: {t2:.4f}s | Ratio: {t1/t2:.2f}x")
