import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCELoss
from torch_geometric.data import Batch

from cut_cross_entropy import linear_cross_entropy
#from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

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
    dtype, device, on_device=False
):
    #python immutable tensor issue
    if mol_input_ids == "all":
        #assert _use_mol_embeddings
        if _use_mol_embeddings:
            return mol_adaptor(
                _finalized_molecule_embeddings.weight
            )
        else:
            #print('number of graphs:', len(graph_ids))
            graphs = [mol_vocab.get(graph_id) for graph_id in range(mol_vocab.start_idx, mol_vocab.start_idx+len(mol_vocab))]
            #if len(graphs) == 0:
            #    return output_features
            graphs = Batch.from_data_list(graphs).to(device)
            # embed the molecules using the GNN
            mol_embeddings = mol_gnn(graphs)
            mol_embeddings = mol_adaptor(mol_embeddings)
            # assign the embeddings to the output features
            return mol_embeddings


    if _use_mol_embeddings:
        #print('on_device2:',on_device)
        if on_device:
            if _finalized_molecule_embeddings.weight.device != device:
                _finalized_molecule_embeddings = _finalized_molecule_embeddings.to(device)

            output_features = _finalized_molecule_embeddings(
                (mol_input_ids - text_vocab_size).clamp(0, None)
            )
        else:

            output_features = _finalized_molecule_embeddings(
                (mol_input_ids - text_vocab_size).clamp(0, None).to('cpu')
            ).to(device)

        output_features = mol_adaptor(output_features)
        output_features[mol_input_ids < 0] = 0
    else:
        output_features = torch.zeros(
            mol_input_ids.size() + (out_channels,),
            dtype=dtype,
        ).to(device)
        # get greater than 0 mol_input_ids
        graph_ids = mol_input_ids[mol_input_ids >= 0].cpu()
        #print('number of graphs:', len(graph_ids))
        graphs = [mol_vocab.get(graph_id) for graph_id in graph_ids.tolist()]
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
    #print(f"[mclm_logit_head_optimized] got: {negative_sampling_size}")
    text_logits = lm_head(hidden_states)
    device = hidden_states.device
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
        #print('mol_embeds:', mol_embeds.shape)
        trained_mol_logits = hidden_states @ mol_embeds.t()                    # (B, L, M)

        # 5) stitch back into a sparse‐logits object:
        logits = torch.cat([text_logits, trained_mol_logits], dim=-1)
        all_ids_trained = torch.cat([
            torch.arange(vocab_size, device=device),
            molecule_ids_trained
        ], dim=0)

        logits = mCLMSparseLogits(
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

def compute_loss_optimized(logits, labels, mapping_tensor=None):
    if not isinstance(logits, mCLMSparseLogits):
        self = mCLMSparseLogits(
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

        labels = mapping_tensor[labels]

    #C = shift_logits.size(-1)
    #lo, hi = int(shift_labels.min()), int(shift_labels.max())
    #print(f"[DEBUG] labels ∈ [{lo}, {hi}], allowed range is [0, {C-1}]")
    #assert 0 <= lo and hi < C, "Some target labels are out of bounds!"

    loss = loss_fct(shift_logits, shift_labels)
    return loss



# mCLM logit function
def mclm_logit_head_optimized2(
    lm_head, embed_molecules, finalized_molecule_embeddings,
    vocab_size, mol_vocab_size, total_vocab_size,
    negative_sampling_size,
    hidden_states, is_training, labels=None, mCLMSparseLogits=None
):
    text_logits = lm_head(hidden_states)
    text_class = lm_head.weight
    device = text_logits.device
    if labels is not None and negative_sampling_size is not None:
        assert labels is not None
        negative_set = torch.multinomial(
            torch.ones(mol_vocab_size, device=device),
            min(mol_vocab_size, negative_sampling_size),
            replacement=False
        ) + vocab_size

        # 2) pull out the “positive” molecule IDs that actually appear in labels:
        labels_flat = labels.view(-1)
        mol_labels = labels_flat[labels_flat >= vocab_size]

        # 3) union them via a single tensor concat + unique (all GPU):
        all_mol_ids = torch.cat([negative_set, mol_labels], dim=0)
        molecule_ids_trained = torch.unique(all_mol_ids)

        # 4) lookup embeddings by direct weight‐slice (faster than embed()):
        #mol_embeds = embed_molecules.weight[molecule_ids_trained - vocab_size]  # (M, H)
        mol_embeds = embed_molecules(molecule_ids_trained)

        all_ids_trained = torch.cat([
            torch.arange(vocab_size, device=device),
            molecule_ids_trained
        ], dim=0)

        logits = mCLMSparseLogitsOptimized(
            indices=all_ids_trained,
            logits=None,
            vocab_size=total_vocab_size,
            embeddings = hidden_states,
            classifier = torch.cat([text_class, mol_embeds], dim=0),
        )
    else:
        molecule_ids_trained = "all"
        mol_embeds = embed_molecules(molecule_ids_trained)
        logits = mCLMSparseLogitsOptimized(
            indices=None,
            logits=None,
            vocab_size=total_vocab_size,
            embeddings = hidden_states,
            classifier = torch.cat([text_class, mol_embeds], dim=0),
        )
    if not is_training:
        logits = logits.embeddings @ logits.classifier.t()


    return logits


# Optimized Version
class mCLMSparseLogitsOptimized:
    def __init__(self, indices, logits, vocab_size=None,
        embeddings = None, classifier=None):

        self.indices = indices
        #if self.indices is not None:
        #    self.indices = torch.tensor(self.indices, dtype=torch.long)
        self.logits = logits
        self.vocab_size = vocab_size
        self.embeddings = embeddings
        self.classifier = classifier


def compute_loss_optimized2(logits, labels, mapping_tensor=None):
    if not isinstance(logits, mCLMSparseLogitsOptimized):
        self = mCLMSparseLogitsOptimized(
            indices=None,
            logits=logits,
        )
    else:
        self = logits

    loss_fct_opt = linear_cross_entropy
    loss_fct_opt2 = CrossEntropyLoss()

    #print("Labels Pre:", labels.min(), labels.max())

    if self.indices is not None:
        if mapping_tensor is None:
            mapping_tensor = torch.full((self.vocab_size,), -1, dtype=torch.long, device=labels.device)
        mapping_tensor[self.indices] = torch.arange(len(self.indices), dtype=torch.long, device=labels.device)

        labels = mapping_tensor[labels]

    #print(self.embeddings.shape, self.classifier.shape, labels.shape)
    #print()

    loss = loss_fct_opt(self.embeddings, \
        self.classifier, labels, shift=1)#, impl='torch_compile')#, impl='cce_kahan_full_c_full_e')

    if False:
        logits = self.embeddings @ self.classifier.t()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        loss = loss_fct_opt2(shift_logits, shift_labels)

    #print('Loss:', loss)

    if False:
        logits = self.embeddings @ self.classifier.t()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        print(shift_logits.shape, shift_labels.shape)
        print("Labels:", shift_labels.min(), shift_labels.max())
        loss_trad = loss_fct_opt2(shift_logits, shift_labels)

        print('CE loss:', loss_trad.item())
        print('CCE loss:', loss.item())

    return loss





# mCLM logit function
def mclm_logit_head_optimized2_sep(
    lm_head, embed_molecules, finalized_molecule_embeddings,
    vocab_size, mol_vocab_size, total_vocab_size,
    negative_sampling_size,
    hidden_states, is_training, is_generating_mol,
    labels=None, mCLMSparseLogits=None,
):
    text_logits = lm_head(hidden_states)
    text_class = lm_head.weight
    device = text_logits.device
    if labels is not None:# and negative_sampling_size is not None:
        if negative_sampling_size is not None:
            negative_set = torch.multinomial(
                torch.ones(mol_vocab_size, device=device),
                min(mol_vocab_size, negative_sampling_size),
                replacement=False
            ) + vocab_size

            # 2) pull out the “positive” molecule IDs that actually appear in labels:
            labels_flat = labels.view(-1)
            mol_labels = labels_flat[labels_flat >= vocab_size]

            # 3) union them via a single tensor concat + unique (all GPU):
            all_mol_ids = torch.cat([negative_set, mol_labels], dim=0)
            molecule_ids_trained = torch.unique(all_mol_ids)

            # 4) lookup embeddings by direct weight‐slice (faster than embed()):
            #mol_embeds = embed_molecules.weight[molecule_ids_trained - vocab_size]  # (M, H)
            mol_embeds = embed_molecules(molecule_ids_trained)

            all_ids_trained = torch.cat([
                torch.arange(vocab_size, device=device),
                molecule_ids_trained
            ], dim=0)

            logits = mCLMSparseLogitsOptimized_sep(
                indices=all_ids_trained,
                logits=None,
                total_vocab_size=total_vocab_size,
                vocab_size=vocab_size,
                mol_vocab_size=mol_vocab_size,
                embeddings = hidden_states,
                text_classifier = text_class,
                mol_classifier = mol_embeds,
                #mol_labels_mask = labels >= vocab_size,
            )
        else:
            mol_embeds = embed_molecules("all")

            logits = mCLMSparseLogitsOptimized_sep(
                indices=None,
                logits=None,
                total_vocab_size=total_vocab_size,
                mol_vocab_size=mol_vocab_size,
                text_vocab_size=vocab_size,
                embeddings = hidden_states,
                text_classifier = text_class,
                mol_classifier = mol_embeds,
            )
        return logits
    else:
        #print(is_generating_mol.shape)
        #print(text_logits.shape)
        # Chi: must be generating, assert
        assert is_generating_mol.ndim == 2
        #assert is_generating_mol.shape[0] == 1
        bs = is_generating_mol.shape[0]
        length = is_generating_mol.shape[1]
        mol_logits = torch.full((bs, length, mol_vocab_size), float('-inf')) #torch.ones(1, length, mol_vocab_size) * -1e5
        #full_logits = torch.cat(
        #    (text_logits, mol_logits),
        #    dim=-1
        #)
        #print(mol_logits.shape)
        #print(is_generating_mol[:,-1].sum())
        if is_generating_mol[:,-1].any():
            molecule_ids_trained = "all"
            mol_embeds = embed_molecules(molecule_ids_trained)
        ml = []
        for i in range(bs):
            #print(i)
            if is_generating_mol[i, -1].item():
                text_logits[i, :, :-1] = torch.full_like(text_logits[i, :, :-1], float('-inf'))
                mol_logits = hidden_states[i, -1].matmul(
                    mol_embeds.transpose(-1, -2)
                )
                mol_logits = mol_logits.unsqueeze(0).unsqueeze(0)  # Shape: [1,1,99607]
                mol_logits = mol_logits.repeat(1, text_logits.shape[1], 1)           # Shape: [1,X,99607]
                #print(mol_logits.shape, text_logits.shape)
            else:
                mol_logits = torch.full((1, length, mol_vocab_size), float('-inf'))
            ml.append(mol_logits)

        mol_logits = torch.cat(ml, dim=0)
        #print(mol_logits.shape)

        full_logits = torch.cat(
            (text_logits, mol_logits),
            dim=-1
        )
        #print(full_logits.shape)


        return full_logits


# Optimized Version
class mCLMSparseLogitsOptimized_sep:
    def __init__(self, indices, logits, text_vocab_size=None, total_vocab_size=None, mol_vocab_size=None,
        embeddings = None, text_classifier=None, mol_classifier=None):

        self.indices = indices
        #if self.indices is not None:
        #    self.indices = torch.tensor(self.indices, dtype=torch.long)
        self.logits = logits
        self.text_vocab_size = text_vocab_size
        self.total_vocab_size = total_vocab_size
        self.mol_vocab_size = mol_vocab_size
        self.embeddings = embeddings
        self.text_classifier = text_classifier
        self.mol_classifier = mol_classifier
        #self.mol_labels_mask = mol_labels_mask
        #self.num_mols = mol_labels_mask.sum()


def compute_loss_optimized2_sep(logits, labels, mapping_tensor=None, MOL_start=None):
    if not isinstance(logits, mCLMSparseLogitsOptimized_sep):
        self = mCLMSparseLogitsOptimized_sep(
            indices=None,
            logits=logits,
        )
    else:
        self = logits

    loss_fct_opt = linear_cross_entropy
    loss_fct_opt2 = CrossEntropyLoss()
    #print("Labels Pre:", labels.min(), labels.max())

    if self.indices is not None:
        if mapping_tensor is None:
            mapping_tensor = torch.full((self.total_vocab_size,), -1, dtype=torch.long, device=labels.device)
        mapping_tensor[self.indices] = torch.arange(len(self.indices), dtype=torch.long, device=labels.device)

        labels = mapping_tensor[labels]

    if False:
        print(self.embeddings.shape, self.text_classifier.shape, self.mol_classifier.shape, labels.shape)
        print(self.mol_labels_mask.shape)
        print(self.embeddings[~self.mol_labels_mask].shape, self.embeddings[self.mol_labels_mask].shape)
        print(labels.min(), labels.max())
        print(labels[~self.mol_labels_mask].min(), labels[~self.mol_labels_mask].max())
        print(labels[self.mol_labels_mask].min(), labels[self.mol_labels_mask].max())



    #mol_labels_mask = (labels == self.text_vocab_size - 2).logical_or(
    #            labels >= self.text_vocab_size) 
    #print('mask test:')
    #print(mol_labels_mask.shape, mol_labels_mask.sum())
    #mol_labels_mask = labels >= (self.text_vocab_size - 1) 
    #print(mol_labels_mask.shape, mol_labels_mask.sum())
    #print('self.text_vocab_size:', self.text_vocab_size)
    #print('self.mol_vocab_size:', self.mol_vocab_size)


    #can't use built-in shift for the split loss :(
    shift_embeddings = self.embeddings[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_labels = shift_labels.long()
    #mol_labels_mask = mol_labels_mask[..., 1:].contiguous()

    mol_labels_mask = (shift_labels == self.text_vocab_size - 1) | (shift_labels >= self.text_vocab_size) # - 1 because [/MOL] needs to be produced by the chem model

    #print('mol labels:', shift_labels[mol_labels_mask])

    num_mols = mol_labels_mask.sum()
    num_text = (~mol_labels_mask).sum()
    #print('num_mols:', num_mols)
    #print('num_text:', num_text)

    #print("shift_labels:", shift_labels.shape)
    #print("mol_labels_mask:", mol_labels_mask.shape)
    #print("shift_labels[mol_labels_mask]:", shift_labels[mol_labels_mask].shape)
    #print("shift_labels[~mol_labels_mask]:", shift_labels[~mol_labels_mask].shape)


    text_classifier = self.text_classifier
    mol_classifier = self.mol_classifier
    #print("text_classifier:", text_classifier.shape)
    #print("mol_classifier:", mol_classifier.shape)
    mol_classifier = torch.cat([mol_classifier, text_classifier[self.text_vocab_size-1].clone().unsqueeze(0)], dim=0) #move [/MOL] over to mol_classifier
    mol_labels = shift_labels[mol_labels_mask] - self.text_vocab_size
    mol_labels[mol_labels == -1] = self.mol_vocab_size #set the [/MOL] token label
    #mol_labels = mol_labels.long()

    #print('mol labels2:', mol_labels)

    #print('Labels:')
    #print(labels.min(), labels.max())
    #print('shift_labels:', shift_labels.min(), shift_labels.max())
    #print('text_labels:', shift_labels[~mol_labels_mask].min(), shift_labels[~mol_labels_mask].max())
    #print('mol_labels:', mol_labels.min(), mol_labels.max())
    #print('Labels Shape:')
    #print(labels.shape)
    #print('shift_labels:', shift_labels.shape)
    #print('text_labels:', shift_labels[~mol_labels_mask].shape)
    #print('mol_labels:', mol_labels.shape)

    if labels.device.type == 'cpu':
        impl = "torch_compile"
    else:
        impl = 'cce'
    
    #print("shift_embeddings[mol_labels_mask]:", shift_embeddings[mol_labels_mask].shape)
    #print("shift_embeddings[~mol_labels_mask]:", shift_embeddings[~mol_labels_mask].shape)

    #print('Final Values for Loss:')


    if MOL_start != None:
        loss_fct_opt2 = CrossEntropyLoss(reduction='none')
        if num_text != 0:
            text_logits = shift_embeddings[~mol_labels_mask] @ text_classifier.t()
            text_labels = shift_labels[~mol_labels_mask]
            text_loss = loss_fct_opt2(text_logits, text_labels)
            #text_loss = loss_fct_opt(shift_embeddings[~mol_labels_mask], \
            #    text_classifier, text_labels, impl=impl, reduction='none')#, shift=1)#, impl='cce_kahan_full_c_full_e')
            text_loss = text_loss.clone()
            text_loss[text_labels == MOL_start] = text_loss[text_labels == MOL_start]* 10
            #print('MOL_start loss:', text_loss[text_labels == MOL_start], 'for', MOL_start)
            text_loss = text_loss.mean()
        else:
            text_loss = None

        if num_mols != 0:
            mol_logits = shift_embeddings[mol_labels_mask] @ mol_classifier.t()
            mol_loss = loss_fct_opt2(mol_logits, mol_labels)
            #mol_loss = loss_fct_opt(shift_embeddings[mol_labels_mask], \
            #    mol_classifier, mol_labels, impl=impl, reduction='none')#, shift=1)#, impl='cce_kahan_full_c_full_e')

            mol_loss = mol_loss.mean()
        else:
            mol_loss = None
    else:
        if num_text != 0:
            text_logits = shift_embeddings[~mol_labels_mask] @ text_classifier.t()
            text_labels = shift_labels[~mol_labels_mask]
            text_loss = loss_fct_opt2(text_logits, text_labels)
        else:
            text_loss = None

        if num_mols != 0:
            mol_logits = shift_embeddings[mol_labels_mask] @ mol_classifier.t()
            mol_loss = loss_fct_opt2(mol_logits, mol_labels)

        else:
            mol_loss = None

    if False: #there's a bug in this loss implementation
        if num_text != 0:
            text_loss = loss_fct_opt(shift_embeddings[~mol_labels_mask], \
                text_classifier, shift_labels[~mol_labels_mask], impl=impl)#, shift=1)#, impl='cce_kahan_full_c_full_e')
        else:
            text_loss = None

        if num_mols != 0:
            mol_loss = loss_fct_opt(shift_embeddings[mol_labels_mask], \
                mol_classifier, mol_labels, impl=impl)#, shift=1)#, impl='cce_kahan_full_c_full_e')
        else:
            mol_loss = None

    if False:
        text_logits = shift_embeddings[~mol_labels_mask] @ text_classifier.t()
        mol_logits = shift_embeddings[mol_labels_mask] @ mol_classifier.t()
        print("Labels:", shift_labels.min(), shift_labels.max())
        print("Text Labels:", shift_labels[~mol_labels_mask].min(), shift_labels[~mol_labels_mask].max())
        print("Molecule Labels:", mol_labels.min(), mol_labels.max())
        print(mol_logits.shape, text_logits.shape, shift_labels.shape, shift_labels[~mol_labels_mask].shape, mol_labels.shape)
        mol_loss_trad = loss_fct_opt2(mol_logits, mol_labels)
        text_loss_trad = loss_fct_opt2(text_logits, shift_labels[~mol_labels_mask])

        print('Mol Cut loss:', mol_loss.item())
        print('Mol Reg loss:', mol_loss_trad.item())
        print('Text Cut loss:', text_loss.item())
        print('Text Reg loss:', text_loss_trad.item())

    return text_loss, mol_loss





def compute_loss_optimized2_sep_batch(logits, labels, mapping_tensor=None):
    if not isinstance(logits, mCLMSparseLogitsOptimized_sep):
        self = mCLMSparseLogitsOptimized_sep(
            indices=None,
            logits=logits,
        )
    else:
        self = logits

    if self.indices is not None:
        if mapping_tensor is None:
            mapping_tensor = torch.full((self.total_vocab_size,), -1, dtype=torch.long, device=labels.device)
        mapping_tensor[self.indices] = torch.arange(len(self.indices), dtype=torch.long, device=labels.device)

        labels = mapping_tensor[labels]

    #can't use built-in shift for the split loss :(
    shift_embeddings = self.embeddings[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_labels = shift_labels.long()

    mol_labels_mask = (shift_labels == self.text_vocab_size - 1) | (shift_labels >= self.text_vocab_size) # - 1 because [/MOL] needs to be produced by the chem model


    num_mols = mol_labels_mask.sum()
    num_text = (~mol_labels_mask).sum()

    text_classifier = self.text_classifier
    mol_classifier = self.mol_classifier
    mol_classifier = torch.cat([mol_classifier, text_classifier[self.text_vocab_size-1].clone().unsqueeze(0)], dim=0) #move [/MOL] over to mol_classifier
    mol_labels = shift_labels[mol_labels_mask] - self.text_vocab_size
    mol_labels[mol_labels == -1] = self.mol_vocab_size #set the [/MOL] token label


    loss_fct_opt2 = CrossEntropyLoss(reduction='none')
    if num_text != 0:
        text_logits = shift_embeddings[~mol_labels_mask] @ text_classifier.t()
        text_labels = shift_labels[~mol_labels_mask]
        text_loss = loss_fct_opt2(text_logits, text_labels)

        #text_loss = text_loss.clone()
        #text_loss[text_labels == MOL_start] = text_loss[text_labels == MOL_start]* 10
        print('text_loss', text_loss.shape)
        text_loss = text_loss.mean(dim=tuple(range(1, text_loss.ndim)))
        print('text_loss', text_loss.shape)
    else:
        text_loss = None

    if num_mols != 0:
        mol_logits = shift_embeddings[mol_labels_mask] @ mol_classifier.t()
        mol_loss = loss_fct_opt2(mol_logits, mol_labels)
        #mol_loss = loss_fct_opt(shift_embeddings[mol_labels_mask], \
        #    mol_classifier, mol_labels, impl=impl, reduction='none')#, shift=1)#, impl='cce_kahan_full_c_full_e')

        print('mol_loss', mol_loss.shape)
        mol_loss = mol_loss.mean(dim=tuple(range(1, mol_loss.ndim)))
        print('mol_loss', mol_loss.shape)
    else:
        mol_loss = None


    return text_loss, mol_loss




def compute_loss_BCE(logits, labels, Yes_token, No_token, mapping_tensor=None):
    if not isinstance(logits, mCLMSparseLogitsOptimized_sep):
        self = mCLMSparseLogitsOptimized_sep(
            indices=None,
            logits=logits,
        )
    else:
        self = logits

    loss_fct_opt = linear_cross_entropy
    loss_fct_opt2 = CrossEntropyLoss()
    #print("Labels Pre:", labels.min(), labels.max())

    if self.indices is not None:
        if mapping_tensor is None:
            mapping_tensor = torch.full((self.total_vocab_size,), -1, dtype=torch.long, device=labels.device)
        mapping_tensor[self.indices] = torch.arange(len(self.indices), dtype=torch.long, device=labels.device)

        labels = mapping_tensor[labels]


    #can't use built-in shift for the split loss :(
    shift_embeddings = self.embeddings[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_labels = shift_labels.long()

    print(shift_labels.shape, shift_embeddings.shape)

    ans_labels_mask = (shift_labels == Yes_token) | (shift_labels == No_token) 

    print(ans_labels_mask.shape)
    
    num_yesno = ans_labels_mask.sum()

    text_classifier = self.text_classifier[[No_token, Yes_token]]
    mol_classifier = self.mol_classifier

    loss_fct_opt2 = CrossEntropyLoss(reduction='none')
    if num_yesno != 0:
        text_logits = shift_embeddings[ans_labels_mask] @ text_classifier.t()
        text_labels = (shift_labels[ans_labels_mask] == Yes_token).long()
        text_loss = loss_fct_opt2(text_logits, text_labels)

        #text_loss = text_loss.clone()
        #text_loss[text_labels == MOL_start] = text_loss[text_labels == MOL_start]* 10

        text_loss = text_loss.mean(dim=tuple(range(1, text_loss.ndim)))
        print('text_loss', text_loss.shape)
    
    _, mol_loss = compute_loss_optimized2_sep_batch(logits, labels)

    batch_labels_mask = (ans_labels_mask.sum(dim=tuple(range(1, ans_labels_mask.ndim))) > 0).long()

    print('batch_labels_mask', batch_labels_mask, batch_labels_mask.shape)

    if batch_labels_mask.sum() != 0:
        text_loss = text_loss[batch_labels_mask].mean()
    else:
        text_loss = None


    if (~batch_labels_mask).sum() != 0:
        mol_loss = mol_loss[~batch_labels_mask].mean()
    else:
        mol_loss = None

    return text_loss, mol_loss







class MLPAdaptor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(MLPAdaptor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)




class MLPAdaptorStable(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout=0.1):
        super(MLPAdaptor, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.PReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        # Initialize weights carefully
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight, gain=0.01)  # or gain=1.0
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x

