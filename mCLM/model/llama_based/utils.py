import torch
import torch.nn as nn


# mCLM embedding function
def embed_chemical_language(
        input_ids, text_vocab_size, embed_text, embed_molecules):
    is_text = input_ids < text_vocab_size
    is_molecule = ~is_text

    text_input_ids = input_ids * is_text
    inputs_embeds = embed_text(text_input_ids)

    mol_input_ids = input_ids[is_molecule]
    mol_inputs_embeds = embed_molecules(mol_input_ids)

    inputs_embeds.masked_scatter_(
        is_molecule.unsqueeze(-1), mol_inputs_embeds
    )
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
        mol_vocab_size, embed_molecules, hidden_size, device):
    all_molecule_ids = torch.arange(
        mol_vocab_size,
        dtype=torch.long,
        device=device)
    all_molecule_embeddings = embed_molecules(all_molecule_ids)
    finalized_molecule_embeddings = nn.Linear(
        mol_vocab_size,
        hidden_size,
    )
    finalized_molecule_embeddings.weight = nn.Parameter(
        all_molecule_embeddings, requires_grad=False)
    return finalized_molecule_embeddings


# mCLM logit function
def mclm_logit_head(
    lm_head, molecule_mlp, embed_molecules, finalized_molecule_embeddings,
    vocab_size, mol_vocab_size, total_vocab_size,
    negative_sampling_size,
    hidden_states, is_training, molecule_ids_to_keep=None
):
    text_logits = lm_head(hidden_states)
    mol_feature = molecule_mlp(hidden_states)
    if is_training:
        assert molecule_ids_to_keep is not None
        negative_set = torch.randperm(
            mol_vocab_size
        )[:negative_sampling_size] + vocab_size
        molecule_ids_trained = torch.LongTensor(
            sorted(list(
                set(negative_set) | set(molecule_ids_to_keep.tolist())
            ))
        )
        trained_mol_logits = mol_feature.matmul(
            embed_molecules(molecule_ids_trained).transpose(0, 1)
        )
        mol_logits = torch.zeros(
            mol_feature.shape[:-1] + (total_vocab_size,),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        mol_logits[:, molecule_ids_trained] = trained_mol_logits
    else:
        mol_logits = finalized_molecule_embeddings(mol_feature)

    logits = torch.cat([text_logits, mol_logits], dim=-1)
    return logits
