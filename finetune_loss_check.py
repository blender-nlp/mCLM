import unittest
import torch
from torch.nn import CrossEntropyLoss
import types


if False:
        
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

# Mocking mCLMSparseLogitsOptimized_sep
class mCLMSparseLogitsOptimized_sep:
    def __init__(self, indices, logits):
        self.indices = indices
        self.logits = logits
        self.embeddings = logits  # for test simplicity
        self.text_vocab_size = 100
        self.mol_vocab_size = 1000
        self.text_classifier = torch.randn(self.text_vocab_size, logits.shape[-1])  # 2-class classifier
        self.mol_classifier = torch.randn(1000, logits.shape[-1])  # large mol vocab
        self.total_vocab_size = self.text_vocab_size + self.mol_vocab_size

# Mocking linear_cross_entropy
def linear_cross_entropy(*args, **kwargs):
    return torch.tensor(0.0, requires_grad=True)

# Function under test (you should import it in practice)
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

    assert torch.is_tensor(shift_labels), "shift_labels is not a tensor"
    assert isinstance(Yes_token, int) and isinstance(No_token, int), "Tokens must be ints"

    print(shift_labels.shape, shift_embeddings.shape)

    ans_labels_mask = (shift_labels == Yes_token) | (shift_labels == No_token) 

    print(ans_labels_mask.shape)
    
    num_yesno = ans_labels_mask.sum()

    text_classifier = self.text_classifier[[No_token, Yes_token]]
    mol_classifier = self.mol_classifier

    loss_fct_opt2 = CrossEntropyLoss()#reduction='none')
    if num_yesno != 0:
        text_logits = shift_embeddings[ans_labels_mask] @ text_classifier.t()
        text_labels = (shift_labels[ans_labels_mask] == Yes_token).long()
        text_loss = loss_fct_opt2(text_logits, text_labels)

        #text_loss = text_loss.clone()
        #text_loss[text_labels == MOL_start] = text_loss[text_labels == MOL_start]* 10

        #text_loss = text_loss.mean(dim=tuple(range(1, text_loss.ndim)))
        print('text_loss', text_loss.shape)
    
    batch_labels_mask = ans_labels_mask.sum(dim=tuple(range(1, ans_labels_mask.ndim))) > 0

    print('batch_labels_mask', batch_labels_mask, batch_labels_mask.shape)
    
    mol_labels_mask = (shift_labels == self.text_vocab_size - 1) | (shift_labels >= self.text_vocab_size) # - 1 because [/MOL] needs to be produced by the chem model
    print('mol_labels_mask', mol_labels_mask.shape, mol_labels_mask.sum())
    mol_labels_mask[batch_labels_mask,:] = False
    print('mol_labels_mask', mol_labels_mask.shape, mol_labels_mask.sum())
    mol_labels = shift_labels[mol_labels_mask] - self.text_vocab_size

    num_mols = mol_labels_mask.sum()
    print('num_mols', num_mols)

    if num_mols != 0:
        mol_logits = shift_embeddings[mol_labels_mask] @ mol_classifier.t()
        mol_loss = loss_fct_opt2(mol_logits, mol_labels)
        #mol_loss = loss_fct_opt(shift_embeddings[mol_labels_mask], \
        #    mol_classifier, mol_labels, impl=impl, reduction='none')#, shift=1)#, impl='cce_kahan_full_c_full_e')

        print('mol_loss', mol_loss.shape)
        #mol_loss = mol_loss.mean(dim=tuple(range(1, mol_loss.ndim)))
        #print('mol_loss', mol_loss.shape)
    else:
        mol_loss = None


    if batch_labels_mask.sum() == 0:
        text_loss = None


    if (~batch_labels_mask).sum() == 0:
        mol_loss = None

    return text_loss, mol_loss

class TestComputeLossBCE(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 6
        self.hidden_dim = 8

        # Random logits
        self.logits = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)

        # Random label values, including Yes and No tokens
        self.Yes_token = 42
        self.No_token = 43
        self.labels = torch.randint(0, 1100, (self.batch_size, self.seq_len))

        # Insert some Yes and No tokens
        self.labels[0, 2] = self.Yes_token
        self.labels[1, 3] = self.No_token
        self.labels[2, 1] = self.Yes_token

        # Patch global references
        global mCLMSparseLogitsOptimized_sep, linear_cross_entropy
        mCLMSparseLogitsOptimized_sep = mCLMSparseLogitsOptimized_sep
        linear_cross_entropy = linear_cross_entropy

    def test_loss_computation(self):
        text_loss, mol_loss = compute_loss_BCE(
            self.logits, self.labels, self.Yes_token, self.No_token
        )

        # Ensure types and that losses are scalars or None
        self.assertTrue(
            text_loss is None or (isinstance(text_loss, torch.Tensor) and text_loss.ndim == 0),
            f"Unexpected text_loss shape/type: {text_loss}"
        )
        self.assertTrue(
            mol_loss is None or (isinstance(mol_loss, torch.Tensor) and mol_loss.ndim == 0),
            f"Unexpected mol_loss shape/type: {mol_loss}"
        )

if __name__ == "__main__":
    unittest.main()