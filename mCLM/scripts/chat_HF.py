
from mCLM.model.models import mCLM
from mCLM.tokenizer.utils import convert_instruction_to_input, message_ids_to_string, get_processor
import torch
import argparse

# ===========================
# Settings
# ===========================
DTYPE = torch.bfloat16         # use float32 for debugging (less rounding noise)
DEVICE = torch.device("cpu")  # set to same device for both



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script with a synth_only flag.")

    parser.add_argument(
        '--synth_only',
        action='store_true',
        help='Run in synthesis-only mode (default: False)'
    )
    args = parser.parse_args()


    model = mCLM.from_pretrained("language-plus-molecules/mCLM_1k-3b")

    tokenizer = model.tokenizer
    molecule_tokenizer = model.molecule_tokenizer

    if args.synth_only:
        bad_words_ids = set()

        synth_only_blocks = set([b.strip() for b in open('resources/synth_blocks_all.txt').readlines()])

        for id, smi in molecule_tokenizer.idx_to_block.items():
            if smi not in synth_only_blocks:
                bad_words_ids.add(id)
        bad_words_ids = [[bwi] for bwi in bad_words_ids]

    else:
        bad_words_ids = None

    model.to(DEVICE).to(DTYPE) #This is important for the HF model
        
    while True:
        user_input = input("Enter an instruction (type 'quit' to exit): ")
        if user_input == 'quit': break
        user_input = user_input.strip()

        message_tokens = convert_instruction_to_input(user_input, model, molecule_tokenizer, tokenizer)
            
        ################## Generate results ###################################

        beam_size = 5            

        input_ids = message_tokens.to(DEVICE)

        processor = get_processor(molecule_tokenizer, tokenizer) #we do this every time in case vocab was expanded

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids), #This is to turn off the attention mask warning
            pad_token_id=tokenizer.eos_token_id, #This is to turn off the pad token warning
            max_new_tokens=32,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            logits_processor=processor,
            do_sample=False,
            bad_words_ids=bad_words_ids,
            diversity_penalty=1.0,
            num_beam_groups=beam_size,
        )

        for i in [0]: #range(beam_size):
            message_ids = generated[i, message_tokens.shape[1]:]
            
            mol_msg, smiles_msg, mol_list, smiles_list = message_ids_to_string(message_ids, molecule_tokenizer, tokenizer)

            if smiles_msg != None:
                print(mol_msg)
                if len(smiles_list) > 0:
                    print("SMILES list:", smiles_list)
            else:
                print(mol_msg)

            print()
            

        