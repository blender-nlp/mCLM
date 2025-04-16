import torch
import time
import random
from torch.nn import CrossEntropyLoss

def test_reindexing_equivalence(vocab_size=10000, num_indices=800, batch_size=8, seq_len=128, device='cuda'):

    # Setup
    total_tokens = batch_size * seq_len
    logits = torch.randn(batch_size, seq_len, num_indices, device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Simulate a sub-vocabulary with shuffled entries
    indices = random.sample(range(vocab_size), num_indices)  # assume subset of vocab
    loss_fct = CrossEntropyLoss()

    # === 1. Original slow version ===
    shift_logits = logits[..., :-1, :].contiguous().view(-1, num_indices)
    shift_labels = labels[..., 1:].contiguous().view(-1)

    start = time.time()
    slow_labels = torch.LongTensor([
        indices.index(x) if x in indices else -1
        for x in shift_labels.tolist()
    ]).to(device)
    mask = slow_labels != -1
    loss_slow = loss_fct(shift_logits[mask], slow_labels[mask])
    torch.cuda.synchronize()
    t_slow = time.time() - start

    # === 2. Fast reindexing ===
    shift_labels = labels[..., 1:].contiguous().view(-1)

    start = time.time()
    mapping_tensor = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
    for new_index, original_index in enumerate(indices):
        mapping_tensor[original_index] = new_index
    fast_labels = mapping_tensor[shift_labels]
    mask = fast_labels != -1
    loss_fast = loss_fct(shift_logits[mask], fast_labels[mask])
    torch.cuda.synchronize()
    t_fast = time.time() - start

    # === 3. Baseline (no reindexing) ===
    logits_base = torch.randn(batch_size, seq_len, vocab_size, device=device)
    shift_logits_base = logits_base[..., :-1, :].contiguous().view(-1, vocab_size)
    shift_labels_base = labels[..., 1:].contiguous().view(-1)

    start = time.time()
    loss_base = loss_fct(shift_logits_base, shift_labels_base)
    torch.cuda.synchronize()
    t_base = time.time() - start

    # === Check correctness ===
    assert torch.allclose(loss_slow, loss_fast, rtol=1e-5), "Mismatch between slow and fast reindexing!" + f" {loss_slow.item():.6f} (slow) != {loss_fast.item():.6f} (fast)"
    print(f"✅ Loss match: {loss_slow.item():.6f} (slow) == {loss_fast.item():.6f} (fast)")

    # === Print timings ===
    print(f"⏱️ Timings:")
    print(f"  Slow reindexing: {t_slow:.4f}s")
    print(f"  Fast reindexing: {t_fast:.4f}s")
    print(f"  No reindexing  : {t_base:.4f}s")

test_reindexing_equivalence()

for vocab_size in [1000, 5000, 10000, 20000, 50000, 100000, 500000, 1000000, 25000000]:
    print('Vocab Size:', vocab_size)
    test_reindexing_equivalence(vocab_size)
