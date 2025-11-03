
from mCLM.model.models import mCLM
from mCLM.tokenizer.utils import convert_instruction_to_input, message_ids_to_string, get_processor, extract_mol_content, \
    ConditionalMolEndProcessor, insert_sublists
import torch

from transformers import LogitsProcessorList, LogitsProcessor


class RestrictTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        allowed_ids = list(self.allowed_token_ids)
        mask[:, allowed_ids] = scores[:, allowed_ids]
        return mask

def get_molecule(instruct, tokenizer, molecule_tokenizer, num_beams=5):

    MOL_start = tokenizer.convert_tokens_to_ids('[MOL]')
    MOL_end = tokenizer.convert_tokens_to_ids('[/MOL]')

    allowed_token_ids = list(molecule_tokenizer.idx_to_block.keys())

    library = set()
    for block in molecule_tokenizer.block_to_idx:
        library.add(block)

    di_blocks = set([b for b in library if '1*' in b and '2*' in b])
    mono_blocks = set([b for b in library if ('3*' in b)])

    response = ''

    mol_list1, cleaned_instruct = extract_mol_content(instruct)
    mol_list2, cleaned_response = extract_mol_content(response)

    mol_list = mol_list1 + mol_list2


    messages = [
        {"role": "user", "content": cleaned_instruct},
        {"role": "assistant", "content": cleaned_response + '[MOL]'},
    ]
    message_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    flat_x = message_tokens.flatten()
    indices = (flat_x == MOL_start).nonzero(as_tuple=True)[0]
    last_index = indices[-1].item()
    message_tokens = message_tokens[0,:last_index]

    frags = [[molecule_tokenizer.get_Idx(m) for m in mol.split('^')] for mol in mol_list]
    
    message_tokens = torch.Tensor(insert_sublists(message_tokens.squeeze(), frags, MOL_start, MOL_end)+[MOL_start]).to(torch.int).to(device)
    
    processor = LogitsProcessorList([ RestrictTokensLogitsProcessor(allowed_token_ids + [MOL_end, tokenizer.pad_token_id]), \
        ConditionalMolEndProcessor(MOL_start, MOL_end, tokenizer.convert_tokens_to_ids(tokenizer.eos_token), \
            set([molecule_tokenizer.get_Idx(b) for b in mono_blocks]), \
            set([molecule_tokenizer.get_Idx(b) for b in di_blocks])), \
        ])
        
    bad_words_ids = None

    input_ids = message_tokens.long().unsqueeze(0)

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids), #This is to turn off the attention mask warning
        pad_token_id=tokenizer.eos_token_id, #This is to turn off the pad token warning
        max_new_tokens=10,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        do_sample=False,
        bad_words_ids=bad_words_ids,
        logits_processor=processor,
        num_beam_groups=num_beams, diversity_penalty=1.0,
        #no_repeat_ngram_size=2,
    )
    
    all_mol_list = []
    all_smiles_list = []

    for j in range(num_beams):
        message_ids = generated[j, len(message_tokens):] 

        tokens = tokenizer.convert_ids_to_tokens(message_ids, skip_special_tokens=True)
        tokens = [molecule_tokenizer.get_block(int(message_ids[i]))+'^' if t == None else t for i, t in enumerate(tokens)]

        mol_msg, smiles_msg, mol_list, smiles_list = message_ids_to_string(message_ids, molecule_tokenizer, tokenizer, add_mol=True)

        all_mol_list.append(mol_list)
        all_smiles_list.append(smiles_list)

    mols = [mol_list[0] for mol_list in all_mol_list]
    smiles = [smiles_list[0] for smiles_list in all_smiles_list]

    return smiles, mols


# ===========================
# Settings
# ===========================
DTYPE = torch.bfloat16         # use float32 for debugging (less rounding noise)
DEVICE = torch.device("cpu")  # set to same device for both



if __name__ == "__main__":

    device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
    model = mCLM.from_pretrained("language-plus-molecules/mCLM_1k-3b")

    tokenizer = model.tokenizer
    molecule_tokenizer = model.molecule_tokenizer

    model.to(DEVICE).to(DTYPE) #This is important for the HF model
        
    while True:
        user_input = input("Enter a task (type 'quit' to exit): ")
        if user_input == 'quit': break
        user_input = user_input.strip()

            
        smiles, mols = get_molecule(user_input, tokenizer, molecule_tokenizer, num_beams=10)

        if len(mols) != 0:
            
            headers = ["#", "SMILES", "Blocks"]

            # 3. Print the table (Using standard print and string formatting)

            print()
            print(f"| {headers[0]:<7} | {headers[1]:<40} | {headers[2]:<60} |")
            print(f"|:-------:|:{'-'*41}|:{'-'*61}|") # Separator line

            for i, (smi, blocks) in enumerate(zip(smiles, mols), 1):
                print(f"| {i:<7} | {smi:<40} | {blocks:<60} |")
            print(f"|:-------:|:{'-'*41}|:{'-'*61}|") # Separator line

            print("\n")            
        else:
            print(f'"{user_input}"', 'failed.')
            

        