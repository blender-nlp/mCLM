# mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules


[:globe_with_meridians: Website](https://thaonguyen217.github.io/mclm.github.io/) | [:octocat: Code](https://github.com/blender-nlp/mCLM) | [:hugs: Data and Model](https://huggingface.co/collections/language-plus-molecules/mclm) | [:desktop_computer: Demo](https://blender02.cs.illinois.edu/mCLM) | [:page_with_curl: Paper](https://arxiv.org/abs/2505.12565)

## Installation

```bash

git clone git@github.com:blender-nlp/mCLM.git

mamba env create -f environment.yml

mamba activate mCLM

pip install -e ./

```


## Chat

```bash
python mCLM/scripts/chat_HF.py
```
`What is the BBBP of <SMILES> Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C </SMILES>?`
> The molecule is predicted to have a Blood Brain Barrier permeability (BBB) of 0.000000000000

`Does [MOL] [3*]C(=O)CCCC^[2*]OC/C=C(/[1*])C^[3*]CCC=C(C)C [/MOL] exhibit strawberry smell?`
> Yes, this molecule has a strawberry smell. [...]

#### To only generate molecules:
```bash
python mCLM/scripts/chat_HF_molgen.py
```

`Generate a molecule that has higher blood brain barrier permeability than [MOL] [3*]C(=O)CCCC^[2*]OC/C=C(/[1*])C^[3*]CCC=C(C)C [/MOL].`
> | #       | SMILES                                   | Blocks                                                       |
> |:-------:|:-----------------------------------------|:-------------------------------------------------------------|
> | 1       | CCCCCC(=O)OCCCOC(=O)c1ccccc1             | [3*]C(=O)CCCCC^[2*]OCCCO[1*]^[3*]C(=O)c1ccccc1               |
> | 2       | CCCCCOOCC=C(C)C(=O)CCC                   | [3*]C(=O)CCC^[2*]OC/C=C(/[1*])C^[3*]OCCCCC                   |
> | 3       | CCCCC(=O)C(C)=CCOCCC=C(C)C               | [3*]C(=O)CCCC^[2*]OC/C=C(/[1*])C^[3*]CCC=C(C)C               |
> | 4       | CCC(=O)C(C)=CCONc1ccc(S(N)(=O)=O)cc1     | [3*]C(=O)CC^[2*]OC/C=C(/[1*])C^[3*]Nc1ccc(S(N)(=O)=O)cc1     |
> | 5       | CCC(=O)N(CC)CC                           | [3*]N(CC)CC^[3*]C(=O)CC                                      |
> | 6       | CN(C)c1ccccc1                            | [3*]N(C)C^[3*]c1ccccc1                                       |
> | 7       | CCN(CC)C(C)=O                            | [3*]N(CC)CC^[3*]C(C)=O                                       |
> | 8       | C=CC(=O)OCCCCOC(=O)CCCC                  | [3*]C(=O)CCCC^[2*]OCCCCO[1*]^[3*]C(=O)C=C                    |
> | 9       | CCCCCC(=O)OCCCOC(=O)c1ccco1              | [3*]C(=O)CCCCC^[2*]OCCCO[1*]^[3*]C(=O)c1ccco1                |
> | 10      | CCCCCCCC(=O)OCCCOC(=O)CCCCCCC            | [3*]C(=O)CCCCCCC^[2*]OCCCO[1*]^[3*]C(=O)CCCCCCC              |

`Generate a molecule related to <SMILES> Cc1cc(C)nc(NS(=O)(=O)c2ccc(N)cc2)n1 </SMILES>.`
> | #       | SMILES                                   | Blocks                                                       |
> |:-------:|:-----------------------------------------|:-------------------------------------------------------------|
> | 1       | CCN(CC)C(C)=O                            | [3*]N(CC)CC^[3*]C(C)=O                                       |
> | 2       | CN(C)C(=O)Nc1ccc(Cl)cc1                  | [3*]N(C)C^[3*]C(=O)Nc1ccc(Cl)cc1                             |
> | 3       | O=CC=Cc1ccccc1                           | [3*]c1ccccc1^[3*]C=CC=O                                      |
> | 4       | O=C(C=Cc1ccccc1)NCCO                     | [3*]NCCO^[3*]C(=O)C=Cc1ccccc1                                |
> | 5       | O=C(Nc1ccccc1)Nc1ccccc1                  | [3*]Nc1ccccc1^[3*]C(=O)Nc1ccccc1                             |
> | 6       | Cc1cc(C)nc(NS(=O)(=O)c2ccc(N)cc2)n1      | [3*]NS(=O)(=O)c1ccc(N)cc1^[3*]c1nc(C)cc(C)n1                 |
> | 7       | CCCCCC(=O)OCCc1ccccc1                    | [3*]C(=O)CCCCC^[3*]OCCc1ccccc1                               |
> | 8       | CCCCCCCCc1ccc(O)cc1                      | [3*]CCCCCCCC^[3*]c1ccc(O)cc1                                 |
> | 9       | CCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCC | [3*]C(=O)CCCCCCCCCCCCCCCC^[1*]OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)O[2*]^[3*]C(=O)CCCCCCCCCCCCCC |
> | 10      | CCCCCCCCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C | [3*]C(=O)CCCCCCCCCCCCCCC^[1*]O[C@H](CO[2*])COP(=O)([O-])OCC[N+](C)(C)C^[3*]C(=O)CCCCCCCCCCCCC |


* **Note:** Converting the HF model to the right dtype is very important for getting the right outputs: `model.to(torch.bfloat16)`. 

## Data

The pretraining datasets and model are [on Huggingface](https://huggingface.co/collections/language-plus-molecules/mclm).


## Training the model

Training code is available, but not configured to be run with publicly available data. 

### Training the GNN

To be released. 

### Pretraining

Updated training code based on the Huggingface dataset to be released. Files for this current version have not been made available. 

#### Pretraining Adapter

```bash
PYTHONPATH=. srun python mCLM/scripts/main.py \
    --pretrained_text_model LLMs/Qwen2.5-3B/ \
    --pretrained_tokenizer LLMs/Qwen2.5-3B/ \
    --check_val_every_n_steps 10000 \
    --batch_size=4 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/Qwen2.5-3B_TotalTop1k_Adaptor_splitLoss/ \
    --version Qwen2.5-3B_Adaptor_splitLoss \
    --max_epochs 1 \
    --train_adapter \
    --accumulate_grad_batches 4 \
    --data_module TotalTopK \
    --task TotalTop1k \
    --num_warmup_steps 2000 \
    --save_checkpoint_every_n_steps 1000 \
    --instruction_data_path mCLM/data/instruction_onlyblocks_top_1k/ \
    --synthetic_data_path mCLM/data/synthetic_onlyblocks_top_1k/ \
    --downsample_tulu 0.1 \
    --tokenizer_cache GNN_input_cache/Total.molecule_tokenizer.1k.pth \
    --pretrained_embeddings final_embeddings/OnlyBlocks/Top1k/128_dim/ \
```

#### Main Pretraining
```bash
PYTHONPATH=. python mCLM/scripts/main.py 
    --pretrained_text_model LLMs/Qwen2.5-3B/ \
    --pretrained_tokenizer LLMs/Qwen2.5-3B/ \
    --check_val_every_n_steps 10000 \
    --batch_size=4 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/Pretraining/ 
    --version Qwen2.5-3B_splitLoss \
    --max_epochs 5 \
    --accumulate_grad_batches 4 \
    --data_module TotalTopK 
    --task TotalTop1k \
    --num_warmup_steps 2000 \
    --save_checkpoint_every_n_steps 1000 \
    --instruction_data_path data/instruction_onlyblocks_top_1k/ \
    --synthetic_data_path data/synthetic_onlyblocks_top_1k/ \
    --downsample_tulu 0.1 \
    --load_GNN_ckpt ckpts_GNN/OnlyBlocks/128_dim/best_val_checkpoint.ckpt \
    --tokenizer_cache GNN_input_cache/Total.molecule_tokenizer.1k.pth \
    --pretrained_embeddings final_embeddings/OnlyBlocks/Top1k/128_dim/ \
```

### Finetuning
```bash
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=. srun python mCLM/scripts/main.py 
    --pretrained_text_model LLMs/Qwen2.5-3B/ \
    --pretrained_tokenizer LLMs/Qwen2.5-3B/ \
    --check_val_every_n_steps 10000 \
    --batch_size=4 --lr 2e-5 --mol_lr 2e-6 \
    --ckpt_path ckpts/Output_Checkpoint/ \
    --version YOUR_NAME_HERE \
    --max_epochs 5 \
    --finetune \
    --accumulate_grad_batches 4 \
    --data_module FinetuneTopK 
    --task FinetuneTop1k \
    --num_warmup_steps 5000 \
    --save_checkpoint_every_n_steps 1000 \
    --synthetic_data_path data/finetune_top_1k/ \
    --tokenizer_cache GNN_input_cache/Total.molecule_tokenizer.1k.pth \
    --load_ckpt ckpts/Pretraining/latest_checkpoint-epoch=04-step=129000.ckpt \
    --pretrained_embeddings final_embeddings/OnlyBlocks/Top1k/128_dim/ \
```

* `pretrained_text_model` - A local download of `Qwen/Qwen2.5-3B`
* `pretrained_tokenizer` - A local download of `Qwen/Qwen2.5-3B`
* `check_val_every_n_steps` - Set how often to run validation metrics
* `batch_size` - batch size per GPU
* `lr` - LLM learning rate
* `mol_lr` - Molecule encoder learning rate
* `ckpt_path` - The output path for checkpoints
* `version` - A name for WandB
* `max_epochs` 
* `accumulate_grad_batches` - Number of gradient batches to accumulate before backprop. 
* `data_module` - The data module to use. 
* `task` - The name of your task for WandB
* `num_warmup_steps` - Number of warmup steps for the learning rate. 
* `save_checkpoint_every_n_steps` - How often to save backup checkpoints. 
* `instruction_data_path` - Processed data from other projects. 
* `synthetic_data_path` - Either the mCLM activity cliff data (pretraining) or finetuning data. 
* `tokenizer_cache` - The cached inputs for the molecule tokenizer.
* `load_ckpt` - The checkpoint that is being finetuned. 
* `pretrained_embeddings` - The location of the pretrained GNN embeddings. 
* `load_GNN_ckpt` - The pretrained GNN checkpoint. 
* `downsample_tulu` - A float to downsample Tulu3 in the pretraining data mixture. 

## Why so many tokenizers?

There are three tokenizers used for the model:
- **mCLM_tokenizer**: This tokenizer is packaged separately. It is used to run the molecule tokenizer (without synthesis guarantees) to convert a SMILES string into our block notation and vice versa. Relevant functions are `join_fragments` and `get_blocks`. 
- **mCLM/tokenizer**: This contains `MoleculeTokenizer`, which is a class for converting molecule token IDs into [torch-geometric data objects](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.data.Data.html#torch_geometric.data.Data). It contains and manages a cache for these objects, allows data type conversion, and allows switching between base language models.
- **tokenizer**: This is an `AutoTokenizer` from HuggingFace.

You can use mCLM_tokenizer to get the blocks from a molecule SMILES. Then, use the MoleculeTokenizer and AutoTokenizer to interact with the model. mCLM.from_pretrained will already create those two for you! After you're done generating, you can use mCLM_tokenizer to convert back to SMILES. Please see [chat_HF.py](mCLM/scripts/chat_HF.py) for an example.

## Citation
If you found our work useful, please cite:
```bibtex
@misc{edwards2025mclmmodularchemicallanguage,
      title={mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules}, 
      author={Carl Edwards and Chi Han and Gawon Lee and Thao Nguyen and Sara Szymkuć and Chetan Kumar Prasad and Bowen Jin and Jiawei Han and Ying Diao and Ge Liu and Hao Peng and Bartosz A. Grzybowski and Martin D. Burke and Heng Ji},
      year={2025},
      eprint={2505.12565},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.12565}, 
}
```
