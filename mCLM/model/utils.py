import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch_geometric.data import Batch


# mCLM embedding function
def embed_chemical_language(
        input_ids, text_vocab_size, embed_text, embed_molecules):
    is_text = input_ids < text_vocab_size

    text_input_ids = input_ids * is_text
    inputs_embeds = embed_text(text_input_ids)

    mol_input_ids = input_ids.clone()
    mol_input_ids[is_text] = -1
    mol_inputs_embeds = embed_molecules(mol_input_ids)

    #print(inputs_embeds.device, mol_inputs_embeds.device, is_text.device)
    inputs_embeds = inputs_embeds * is_text[..., None] + \
        mol_inputs_embeds * (~is_text[..., None])
    return inputs_embeds


# mCLM molecule adaptation layer for logits
def molecule_adaptation_layer(hidden_size, activation):
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        activation,
        nn.Linear(hidden_size, hidden_size),
    )


# mCLM training finishes, finalize the molecule embeddings for inference
# batch mode
def finalized_molecule_embeddings(
        vocab_size, mol_vocab_size, embed_molecules, hidden_size,
        batch_size, device):
    all_molecule_ids = torch.arange(
        vocab_size,
        mol_vocab_size + vocab_size,
        dtype=torch.long,
        device=device)
    finalized_molecule_embeddings = nn.Embedding(
        mol_vocab_size,
        hidden_size,
    )
    with torch.no_grad():
        for i in range(0, mol_vocab_size, batch_size):
            all_molecule_embeddings = embed_molecules(
                all_molecule_ids[i:i + batch_size])
            finalized_molecule_embeddings.weight.data[i:i + batch_size] = \
                all_molecule_embeddings
    # all_molecule_embeddings = embed_molecules(all_molecule_ids)
    # finalized_molecule_embeddings.weight = nn.Parameter(
    #     all_molecule_embeddings, requires_grad=False)
    return finalized_molecule_embeddings


class mCLMSparseLogits:
    def __init__(self, indices, logits, vocab_size=None):
        self.indices = indices
        #if self.indices is not None:
        #    self.indices = self.indices.cpu().tolist()
        self.logits = logits
        self.vocab_size = vocab_size


def compute_loss(logits, labels, mapping_tensor=None):
    if not isinstance(logits, mCLMSparseLogits):
        self = mCLMSparseLogits(
            indices=None,
            logits=logits,
        )
    else:
        self = logits
    # Shift so that tokens < n predict n
    shift_logits = self.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    # re-indexing
    if self.indices is not None:
        if self.vocab_size == None: #Too slow:
            shift_labels = torch.LongTensor([
                self.indices.index(x)
                for x in shift_labels.tolist()
            ]).to(shift_labels.device)
        else: #faster version
            if mapping_tensor is None:
                mapping_tensor = torch.full((self.vocab_size,), -1, dtype=torch.long, device=shift_labels.device)
            mapping_tensor[self.indices] = torch.arange(len(self.indices), dtype=torch.long, device=shift_labels.device)
        # Apply mapping (vectorized)
        shift_labels = mapping_tensor[shift_labels]
    loss = loss_fct(shift_logits, shift_labels.to(torch.long))
    return loss


# mCLM logit function
def mclm_logit_head(
    lm_head, embed_molecules, finalized_molecule_embeddings,
    vocab_size, mol_vocab_size, total_vocab_size,
    negative_sampling_size,
    hidden_states, is_training, labels=None
):
    #print(f"[mclm_logit_head] got: {negative_sampling_size}")
    text_logits = lm_head(hidden_states)
    device = text_logits.device
    if labels is not None and negative_sampling_size is not None:
        assert labels is not None
        negative_set = torch.randperm(
            mol_vocab_size
        )[:negative_sampling_size] + vocab_size
        molecule_ids_trained = torch.LongTensor(
            sorted(list(
                set(negative_set) |
                set(filter(lambda x: x >= vocab_size,
                           labels.flatten().tolist()))
            ))
        ).to(device)
        #print('negative_set size:', negative_set.shape)
        #print('molecule_ids_trained size:', molecule_ids_trained.shape)
        trained_mol_logits = hidden_states.matmul(
            embed_molecules(molecule_ids_trained).transpose(0, 1)
        )
        logits = torch.cat([text_logits, trained_mol_logits], dim=-1)
        #print('logits size:', logits.shape)
        all_ids_trained = torch.cat(
            [torch.arange(vocab_size, dtype=torch.long, device=device),
             molecule_ids_trained], dim=0
        )
        logits = mCLMSparseLogits(
            indices=all_ids_trained,
            logits=logits,
            vocab_size=total_vocab_size,
        )
        # mol_logits = torch.zeros(
        #     hidden_states.shape[:-1] + (mol_vocab_size,),
        #     device=hidden_states.device,
        #     dtype=hidden_states.dtype,
        # ) - 1e6
        # mol_logits[..., molecule_ids_trained - vocab_size] = \
        #     trained_mol_logits
    else:
        # mol_logits = hidden_states.matmul(
        #     finalized_molecule_embeddings.weight.transpose(0, 1))
        molecule_ids_trained = "all"
        mol_logits = hidden_states.matmul(
            embed_molecules(molecule_ids_trained).transpose(0, 1)
        )
        logits = torch.cat([text_logits, mol_logits], dim=-1)

    return logits


def embed_molecules_fn(
    mol_input_ids, out_channels, mol_vocab, mol_gnn, mol_adaptor,
    _finalized_molecule_embeddings, _use_mol_embeddings, text_vocab_size,
    dtype, device
):
    if mol_input_ids == "all":
        assert _use_mol_embeddings
        return mol_adaptor(
            _finalized_molecule_embeddings.weight
        )

    if _use_mol_embeddings:
        output_features = _finalized_molecule_embeddings(
            (mol_input_ids - text_vocab_size).clamp(0, None).to('cpu')
        ).to(device)
        output_features = mol_adaptor(output_features)
        output_features[mol_input_ids < 0] = 0
    else:
        output_features = torch.zeros(
            mol_input_ids.size() + (out_channels,),
            dtype=dtype,
        )
        # get greater than 0 mol_input_ids
        graph_ids = mol_input_ids[mol_input_ids >= 0]
        graphs = [mol_vocab[graph_id.item()] for graph_id in graph_ids]
        if len(graphs) == 0:
            return output_features
        graphs = Batch.from_data_list(graphs).to(device)
        # embed the molecules using the GNN
        mol_embeddings = mol_gnn(graphs)
        mol_embeddings = mol_adaptor(mol_embeddings)
        # assign the embeddings to the output features
        output_features[mol_input_ids >= 0] = mol_embeddings

    return output_features






# mCLM logit function
def mclm_logit_head_optimized(
    lm_head, embed_molecules, finalized_molecule_embeddings,
    vocab_size, mol_vocab_size, total_vocab_size,
    negative_sampling_size,
    hidden_states, is_training, labels=None
):
    text_logits = lm_head(hidden_states)
    device = text_logits.device
    if labels is not None and negative_sampling_size is not None:
        assert labels is not None
        neg = torch.multinomial(
            torch.ones(mol_vocab_size, device=device),
            negative_sampling_size,
            replacement=False
        ) + vocab_size

        # 2) pull out the “positive” molecule IDs that actually appear in labels:
        labels_flat = labels.view(-1)
        mol_labels = labels_flat[labels_flat >= vocab_size]

        # 3) union them via a single tensor concat + unique (all GPU):
        all_mol_ids = torch.cat([neg, mol_labels], dim=0)
        molecule_ids_trained = torch.unique(all_mol_ids)

        # 4) lookup embeddings by direct weight‐slice (faster than embed()):
        #mol_embeds = embed_molecules.weight[molecule_ids_trained - vocab_size]  # (M, H)
        mol_embeds = embed_molecules(molecule_ids_trained)
        trained_mol_logits = hidden_states @ mol_embeds.t()                    # (B, L, M)

        # 5) stitch back into a sparse‐logits object:
        logits = torch.cat([text_logits, trained_mol_logits], dim=-1)
        all_ids_trained = torch.cat([
            torch.arange(vocab_size, device=device),
            molecule_ids_trained
        ], dim=0)

        logits = mCLMSparseLogitsOptimized(
            indices=all_ids_trained,
            logits=logits,
            vocab_size=total_vocab_size,
        )
    else:
        molecule_ids_trained = "all"
        mol_logits = hidden_states.matmul(
            embed_molecules(molecule_ids_trained).transpose(0, 1)
        )
        logits = torch.cat([text_logits, mol_logits], dim=-1)

    return logits



# Optimized Version
class mCLMSparseLogitsOptimized:
    def __init__(self, indices, logits, vocab_size=None):
        self.indices = indices
        #if self.indices is not None:
        #    self.indices = torch.tensor(self.indices, dtype=torch.long)
        self.logits = logits
        self.vocab_size = vocab_size


def compute_loss_optimized(logits, labels, mapping_tensor=None):
    if not isinstance(logits, mCLMSparseLogitsOptimized):
        self = mCLMSparseLogitsOptimized(
            indices=None,
            logits=logits,
        )
    else:
        self = logits

    shift_logits = self.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = CrossEntropyLoss()

    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    shift_labels = shift_labels.to(shift_logits.device)

    if self.indices is not None:
        if mapping_tensor is None:
            mapping_tensor = torch.full((self.vocab_size,), -1, dtype=torch.long, device=shift_labels.device)
            mapping_tensor[self.indices] = torch.arange(len(self.indices), dtype=torch.long, device=shift_labels.device)

        shift_labels = mapping_tensor[shift_labels]

    loss = loss_fct(shift_logits, shift_labels)
    return loss




class MLPAdaptor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout=0.1):
        super(MLPAdaptor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)
