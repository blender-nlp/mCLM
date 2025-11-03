"""
Script to upload mCLM model to Hugging Face Hub.

This script handles:
1. Loading the trained mCLM model with all components
2. Saving model weights, configurations, and tokenizers
3. Uploading to Hugging Face Hub with proper documentation

Usage:
    python mCLM/scripts/upload_to_huggingface.py \
        --ckpt_path /path/to/checkpoint.ckpt \
        --tokenizer_path /path/to/tokenizer/ \
        --base_model /path/to/Qwen2.5-3B/ \
        --repo_id your-username/mclm-model \
        --load_GNN_ckpt /path/to/gnn_checkpoint.ckpt \
        --pretrained_embeddings /path/to/embeddings/ \
        [--private] \
        [--push_to_hub]
"""

import argparse
import os
import os.path as osp
import torch
import json
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import HfApi, create_repo, upload_folder

# Import mCLM components
from mCLM.tokenizer.molecule_tokenizer import MoleculeTokenizer
from mCLM.model.qwen_based.configuration import Qwen2Config
from mCLM.model.qwen_based.model import Qwen2ForCausalLM
from mCLM.model.models import mCLM
from mCLM.data.processing import load_with_tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Upload mCLM model to Hugging Face")
    
    # Model paths
    parser.add_argument("--ckpt_path", type=str, required=True,
                       help="Path to the trained model checkpoint (.ckpt file)")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                       help="Path to the directory containing tokenizer files")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Path to the base Qwen2 model")
    parser.add_argument("--load_GNN_ckpt", type=str, default=None,
                       help="Path to GNN checkpoint directory")
    parser.add_argument("--pretrained_embeddings", type=str, default=None,
                       help="Path to pretrained molecule embeddings")
    
    # Model configuration
    parser.add_argument("--node_dim", default=138, type=int)
    parser.add_argument("--edge_dim", default=12, type=int)
    parser.add_argument("--hidden_dim_graph", default=512, type=int)
    parser.add_argument("--num_mp_layers", default=5, type=int)
    parser.add_argument("--num_readout_layers", default=1, type=int)
    parser.add_argument("--dropout", default=0.13, type=float)
    parser.add_argument("--aggr", default="mean", type=str)
    parser.add_argument("--jk", default="cat", type=str)
    parser.add_argument("--latent_size", default=256, type=int)
    
    # Hub configuration
    parser.add_argument("--repo_id", type=str, required=True,
                       help="Hugging Face repository ID (e.g., 'username/model-name')")
    parser.add_argument("--private", action="store_true",
                       help="Make the repository private")
    parser.add_argument("--output_dir", type=str, default="./hf_upload_temp",
                       help="Temporary directory to save model files before upload")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Actually push to hub (otherwise just prepare files locally)")
    
    # Optional metadata
    parser.add_argument("--model_description", type=str,
                       default="mCLM: A Molecular Chemistry Language Model",
                       help="Description for the model card")
    parser.add_argument("--license", type=str, default="apache-2.0",
                       help="License for the model")
    
    parser.add_argument("--PEFT", type=bool, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    return args


def create_model_card(args, output_dir):
    """Create a comprehensive model card for the mCLM model."""
    
    model_card = f"""---
license: {args.license}
language:
- en
tags:
- chemistry
- molecules
- drug-discovery
- molecular-generation
- causal-lm
base_model: Qwen/Qwen2.5-3B
library_name: transformers
---

# {args.repo_id.split('/')[-1]}

{args.model_description}

## Model Description

mCLM (Molecular Chemistry Language Model) is a specialized language model that combines:
- **Text Understanding**: Based on Qwen2.5-3B architecture
- **Molecular Understanding**: Graph Neural Network (GNN) encoder for molecular structures
- **Unified Representation**: MLP adaptor to align molecule and text embeddings

The model can understand and generate both natural language and molecular structures (SMILES), making it suitable for:
- Molecular property prediction
- Drug discovery and design
- Chemistry question answering
- Molecule-text retrieval

## Architecture

- **Base Model**: Qwen2.5-3B
- **Molecular Encoder**: GNN with {args.num_mp_layers} message passing layers
- **Hidden Dimension**: {args.hidden_dim_graph}
- **Molecular Tokenizer**: Custom block-based tokenizer for SMILES representations

## Usage

```python
import torch
from transformers import AutoTokenizer
from mCLM.model.qwen_based.model import Qwen2ForCausalLM
from mCLM.tokenizer.molecule_tokenizer import MoleculeTokenizer
from mCLM_tokenizer.tokenizer import convert_SMILES_strings
from mCLM.data.processing import extract_mol_content, load_with_tqdm

# Load tokenizers
tokenizer = AutoTokenizer.from_pretrained("{args.repo_id}")
tokenizer.pad_token = tokenizer.eos_token

# Load molecule tokenizer
torch.serialization.add_safe_globals([MoleculeTokenizer])
molecule_tokenizer = torch.load("molecule_tokenizer.pth", weights_only=False)
molecule_tokenizer.change_start_idx(len(tokenizer))

# Load model
model = Qwen2ForCausalLM.from_pretrained(
    "{args.repo_id}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Setup model
model.model.set_mol_vocab(molecule_tokenizer.GNN_input_map)
model.model.extend_text_vocab_size(len(tokenizer.vocab))

# Load pretrained embeddings if available
if hasattr(model, 'load_pretrained_embeddings'):
    model.load_pretrained_embeddings()

# Example: Generate molecule description
user_input = "What is the molecular weight of aspirin <SMILES> CC(=O)Oc1ccccc1C(=O)O </SMILES>?"

# Convert SMILES to molecule tokens
mols_list, MOL_input = convert_SMILES_strings(user_input)
mol_list, cleaned_text = extract_mol_content(MOL_input)

messages = [{{"role": "user", "content": cleaned_text}}]
inputs = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 
    return_tensors="pt"
)

# Generate
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Data

The model was trained on:
- [Molecular instruction-following data from activity cliffs](https://huggingface.co/datasets/language-plus-molecules/mCLM_Pretrain_1k)
- [General text instruction data](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)

## Limitations

- The model is trained on specific molecular representations and may not generalize to all chemistry domains
- Generations should not be trusted without validation
- Not a substitute for expert chemical analysis

## Citation

If you use this model, please cite:

```bibtex
@misc{{edwards2025mclmmodularchemicallanguage,
      title={{mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules}}, 
      author={{Carl Edwards and Chi Han and Gawon Lee and Thao Nguyen and Sara Szymkuć and Chetan Kumar Prasad and Bowen Jin and Jiawei Han and Ying Diao and Ge Liu and Hao Peng and Bartosz A. Grzybowski and Martin D. Burke and Heng Ji}},
      year={{2025}},
      eprint={{2505.12565}},
      archivePrefix={{arXiv}},
      primaryClass={{cs.AI}},
      url={{https://arxiv.org/abs/2505.12565}}, 
}}
```

## Model Card Contact

For questions or issues, please open an issue in the repository.
"""
    
    with open(osp.join(output_dir, "README.md"), "w") as f:
        f.write(model_card)
    
    print(f"✓ Created model card at {output_dir}/README.md")


def prepare_model_files(args, output_dir):
    """Load and save all model components."""
    
    print("=" * 50)
    print("Loading Model Components")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load and save text tokenizer
    print("\n[1/6] Loading text tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(['[MOL]', '[/MOL]'])
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Text tokenizer saved to {output_dir}")
    
    # 2. Load and save molecule tokenizer
    print("\n[2/6] Loading molecule tokenizer...")
    torch.serialization.add_safe_globals([MoleculeTokenizer])
    molecule_tokenizer_path = osp.join(args.tokenizer_path, "molecule_tokenizer.pth")
    
    if not osp.exists(molecule_tokenizer_path):
        raise FileNotFoundError(f"Molecule tokenizer not found at {molecule_tokenizer_path}")
    
    molecule_tokenizer = load_with_tqdm(
        molecule_tokenizer_path,
        map_location=torch.device('cpu'),
        weights_only=False
    )
    
    start_idx = len(tokenizer)
    molecule_tokenizer.change_start_idx(start_idx)
    
    # Save molecule tokenizer
    torch.save(molecule_tokenizer, osp.join(output_dir, "molecule_tokenizer.pth"))
    print(f"✓ Molecule tokenizer saved to {output_dir}/molecule_tokenizer.pth")
    
    # Load GNN input cache if exists
    GNN_cache_path = osp.join(args.tokenizer_path, 'molecule_tokenizer.graphs.pth')
    if osp.exists(GNN_cache_path):
        print("  Loading GNN input cache...")
        molecule_tokenizer.GNN_input_map = load_with_tqdm(
            GNN_cache_path,
            map_location=torch.device('cpu'),
            weights_only=False
        )
        torch.save(
            molecule_tokenizer.GNN_input_map,
            osp.join(output_dir, "molecule_tokenizer.graphs.pth")
        )
        print(f"✓ GNN cache saved to {output_dir}/molecule_tokenizer.graphs.pth")
    else:
        print("  Creating GNN input cache...")
        molecule_tokenizer.create_input()
        torch.save(
            molecule_tokenizer.GNN_input_map,
            osp.join(output_dir, "molecule_tokenizer.graphs.pth")
        )
    
    # 3. Create model configuration
    print("\n[3/6] Creating model configuration...")
    config = vars(args)
    
    # Initialize model
    model = mCLM(config)
    
    # 4. Load pretrained embeddings if provided
    if args.pretrained_embeddings is not None:
        print("\n[4/6] Loading pretrained molecule embeddings...")
        embedding_path = osp.join(args.pretrained_embeddings, 'precomputed_tokens.pt')
        if osp.exists(embedding_path):
            loaded_pretrain_mol_embeddings = torch.load(embedding_path).to(torch.bfloat16)
            from torch import nn
            pretrain_mol_embeddings = nn.Embedding.from_pretrained(
                loaded_pretrain_mol_embeddings,
                freeze=True
            )
            pretrain_mol_embeddings.weight.data = pretrain_mol_embeddings.weight.data.to(torch.bfloat16)
            
            model.model.finalize_molecule_embeddings(embeddings=pretrain_mol_embeddings)
            model.model.use_mol_embeddings(True)
            
            # Save embeddings
            torch.save(
                loaded_pretrain_mol_embeddings,
                osp.join(output_dir, "precomputed_tokens.pt")
            )
            print(f"✓ Pretrained embeddings saved to {output_dir}/precomputed_tokens.pt")
        else:
            print(f"  Warning: Pretrained embeddings not found at {embedding_path}")
    else:
        print("\n[4/6] Skipping pretrained embeddings (not provided)")
    
    # 5. Load GNN checkpoint if provided
    if args.load_GNN_ckpt is not None:
        print("\n[5/6] Loading GNN checkpoint...")
        gnn_ckpt_path = osp.join(args.load_GNN_ckpt, 'best_val_checkpoint.ckpt')
        if osp.exists(gnn_ckpt_path):
            gnn_ckpt_sd = torch.load(gnn_ckpt_path, map_location='cpu', weights_only=False)['state_dict']
            for key in list(gnn_ckpt_sd.keys()):
                gnn_ckpt_sd[key.replace('mol_encoder.', '')] = gnn_ckpt_sd.pop(key)
            
            model.model.model.model.mol_gnn.load_state_dict(gnn_ckpt_sd)
            print(f"✓ GNN weights loaded from {gnn_ckpt_path}")
        else:
            print(f"  Warning: GNN checkpoint not found at {gnn_ckpt_path}")
    else:
        print("\n[5/6] Skipping GNN checkpoint (not provided)")
    
    # Set up model vocab
    model.model.set_mol_vocab(molecule_tokenizer.GNN_input_map)
    model.model.extend_text_vocab_size(len(tokenizer.vocab))
    model = model.to(torch.bfloat16)
    
    # 6. Load main checkpoint
    print("\n[6/6] Loading main model checkpoint...")
    if not osp.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.ckpt_path}")
    
    sd = load_with_tqdm(args.ckpt_path, map_location='cpu')['state_dict']
    
    # Remove GNN keys (already loaded separately)
    for key in list(sd.keys()):
        if 'mol_gnn' in key:
            sd.pop(key)
    
    model.load_state_dict(sd, strict=False)
    print("✓ Main checkpoint loaded")
    
    # 7. Save the model
    print("\nSaving model to disk...")
    model.model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    print(f"✓ Model saved to {output_dir}")
    
    # 8. Save configuration with mCLM-specific settings
    print("\nSaving configuration...")
    config_dict = model.model.config.to_dict()
    with open(osp.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"✓ Configuration saved to {output_dir}/config.json")
    
    # 9. Create requirements file
    print("\nCreating requirements file...")
    requirements = """transformers>=4.40.0
torch>=2.0.0
rdkit>=2023.3.1
lightning>=2.0.0
huggingface_hub>=0.20.0
torch-geometric>=2.3.0
"""
    with open(osp.join(output_dir, "requirements.txt"), "w") as f:
        f.write(requirements)
    print(f"✓ Requirements saved to {output_dir}/requirements.txt")
    
    # 10. Create usage example
    print("\nCreating usage example...")
    usage_example = """# mCLM Usage Example

import torch
from transformers import AutoTokenizer
from mCLM.model.qwen_based.model import Qwen2ForCausalLM
from mCLM.tokenizer.molecule_tokenizer import MoleculeTokenizer

# Load model and tokenizers
model = Qwen2ForCausalLM.from_pretrained(
    "YOUR_REPO_ID",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("YOUR_REPO_ID")
tokenizer.pad_token = tokenizer.eos_token

# Load molecule tokenizer
torch.serialization.add_safe_globals([MoleculeTokenizer])
molecule_tokenizer = torch.load("molecule_tokenizer.pth", weights_only=False)

# Run inference
user_input = "What is aspirin used for?"
messages = [{"role": "user", "content": user_input}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

outputs = model.generate(input_ids=inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
"""
    with open(osp.join(output_dir, "usage_example.py"), "w") as f:
        f.write(usage_example)
    print(f"✓ Usage example saved to {output_dir}/usage_example.py")
    
    return output_dir


def upload_to_hub(repo_id, output_dir, private=False):
    """Upload the model to Hugging Face Hub."""
    
    print("\n" + "=" * 50)
    print("Uploading to Hugging Face Hub")
    print("=" * 50)
    
    api = HfApi()
    
    # Create repository
    print(f"\nCreating repository: {repo_id}")
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"✓ Repository created/verified: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"✗ Error creating repository: {e}")
        return False
    
    # Upload files
    print(f"\nUploading files from {output_dir}...")
    try:
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload mCLM model"
        )
        print(f"✓ Files uploaded successfully!")
        print(f"\n🎉 Model available at: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"✗ Error uploading files: {e}")
        return False


def main():
    args = parse_args()
    
    print("\n" + "=" * 50)
    print("mCLM Hugging Face Upload Script")
    print("=" * 50)
    print(f"\nCheckpoint: {args.ckpt_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Repository: {args.repo_id}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Private: {args.private}")
    print(f"Push to Hub: {args.push_to_hub}")
    
    try:
        # Prepare model files
        output_dir = prepare_model_files(args, args.output_dir)
        
        # Create model card
        create_model_card(args, output_dir)
        
        # Upload to hub if requested
        if args.push_to_hub:
            success = upload_to_hub(args.repo_id, output_dir, args.private)
            if success:
                print("\n✓ Upload completed successfully!")
            else:
                print("\n✗ Upload failed. Files are saved locally at:", output_dir)
        else:
            print("\n" + "=" * 50)
            print("Model prepared locally (not uploaded)")
            print("=" * 50)
            print(f"\nFiles saved to: {output_dir}")
            print("\nTo upload later, run:")
            print(f"  python -c \"from huggingface_hub import upload_folder; upload_folder('{output_dir}', '{args.repo_id}')\"")
            print("\nOr re-run this script with --push_to_hub flag")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

