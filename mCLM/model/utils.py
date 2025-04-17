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
def finalized_molecule_embeddings(
        vocab_size, mol_vocab_size, embed_molecules, hidden_size, device):
    all_molecule_ids = torch.arange(
        vocab_size,
        mol_vocab_size + vocab_size,
        dtype=torch.long,
        device=device)
    all_molecule_embeddings = embed_molecules(all_molecule_ids)
    finalized_molecule_embeddings = nn.Embedding(
        mol_vocab_size,
        hidden_size,
    )
    finalized_molecule_embeddings.weight = nn.Parameter(
        all_molecule_embeddings, requires_grad=False)
    return finalized_molecule_embeddings


class mCLMSparseLogits:
    def __init__(self, indices, logits, vocab_size):
        self.indices = indices.cpu().tolist()
        self.logits = logits
        self.vocab_size = vocab_size

    #pass a mapping_tensor object to avoid memory leaks
    def compute_loss(self, labels, mapping_tensor=None):
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
            #Too slow:
            #shift_labels = torch.LongTensor([
            #    self.indices.index(x)
            #    for x in shift_labels.tolist()
            #]).to(shift_labels.device)
            #ChatGPT-4o wrote this faster version for me:
            if mapping_tensor is not None:
                device = shift_labels.device  # or torch.device("cuda")
                mapping_tensor = mapping_tensor.to(device)
                for new_index, original_index in enumerate(self.indices):
                    mapping_tensor[original_index] = new_index
            else:
                mapping_tensor = torch.full((self.vocab_size,), -1, dtype=torch.long, device=shift_labels.device)
                for new_index, original_index in enumerate(self.indices):
                    mapping_tensor[original_index] = new_index
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
    text_logits = lm_head(hidden_states)
    device = text_logits.device
    if labels is not None:
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
        trained_mol_logits = hidden_states.matmul(
            embed_molecules(molecule_ids_trained).transpose(0, 1)
        )
        logits = torch.cat([text_logits, trained_mol_logits], dim=-1)
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
        mol_logits = hidden_states.matmul(
            finalized_molecule_embeddings.weight.transpose(0, 1))
        logits = torch.cat([text_logits, mol_logits], dim=-1)

    return logits


def embed_molecules_fn(mol_input_ids, out_channels, mol_vocab, mol_gnn,
                       dtype, device):
    output_features = torch.zeros(
        mol_input_ids.size() + (out_channels,),
        dtype=dtype,
        device=device,
    )
    # get greater than 0 mol_input_ids
    graph_ids = mol_input_ids[mol_input_ids >= 0]
    graphs = [mol_vocab[graph_id.item()] for graph_id in graph_ids]
    if len(graphs) == 0:
        return output_features
    graphs = Batch.from_data_list(graphs).to(device)
    # embed the molecules using the GNN
    mol_embeddings = mol_gnn(graphs)
    # assign the embeddings to the output features
    output_features[mol_input_ids >= 0] = mol_embeddings

    return output_features
