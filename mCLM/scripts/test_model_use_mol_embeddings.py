import argparse
import os
# import os.path as osp
# import lightning as L
# import sys
import torch
# from lightning import Trainer, seed_everything
# from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
# from lightning.pytorch.loggers import WandbLogger

# import pandas as pd

from mCLM.data.dataloaders import KinaseDataModule
from mCLM_tokenizer.tokenizer import get_blocks

import subprocess


if __name__ == "__main__":
    torch.cuda.empty_cache()

    try:
        subprocess.run(["nvidia-smi"])
    except Exception as e:
        print("Error running nvidia-smi:", e)

    config = {
        "pretrained_text_model": "michiyasunaga/BioLinkBERT-base",
        "trunc_length": 512,
        "num_warmup_steps": 1000,
        "max_epochs": 2,
        "batch_size": 128,
        "val_batch_size": None,
        "node_dim": 133,
        "edge_dim": 12,
        "hidden_dim_graph": 512,
        "num_mp_layers": 5,
        "num_readout_layers": 1,
        "dropout": 0.13,
        "aggr": "mean",
        "jk": "cat",
        "latent_size": 256,
        "validate_every_n": 1000,
        "lr": 2e-5,
        "data_module": "S1B",
        "ckpt_path": "ckpts/",
        "loss": "CLIP",
        "model": "GNN",
        "load_ckpt": None,
        "seed": 42,
    }

    parser = argparse.ArgumentParser(description="Biencoder")
    parser.add_argument(
        "--pretrained_text_model",
        default="michiyasunaga/BioLinkBERT-base",
        type=str,
        help="Which text encoder to use from HuggingFace",
    )
    parser.add_argument("--trunc_length", default=512, type=int)

    parser.add_argument("--num_warmup_steps", default=1000, type=int)
    parser.add_argument("--max_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--val_batch_size", default=None, type=int)

    parser.add_argument("--node_dim", default=133, type=int)
    parser.add_argument("--edge_dim", default=12, type=int)
    parser.add_argument("--hidden_dim_graph", default=512, type=int)
    parser.add_argument("--num_mp_layers", default=5, type=int)
    parser.add_argument("--num_readout_layers", default=1, type=int)
    parser.add_argument("--dropout", default=0.13, type=float)
    parser.add_argument("--aggr", default="mean", type=str)
    parser.add_argument("--jk", default="cat", type=str)

    parser.add_argument("--latent_size", default=256, type=int)
    parser.add_argument("--validate_every_n", default=1000, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--ckpt_path", default="ckpts/MOA/", type=str)
    parser.add_argument("--loss", default="CLIP", type=str)
    parser.add_argument("--load_ckpt", default=None, type=str)
    parser.add_argument("--load_GNN_ckpt", default=None, type=str)

    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--model", default="mCLM", type=str)
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct", type=str)
    #parser.add_argument(
    #    "--freeze_text_encoder", type=bool, action=argparse.BooleanOptionalAction
    #)

    parser.add_argument(
        "--freeze_GNN", type=bool, action=argparse.BooleanOptionalAction
    )

    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    parser.add_argument("--resume_wandb_run", default=None, type=str)

    parser.add_argument("--task", default=None, type=str)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    parser.add_argument("--data_module", type=str, default='Kinase')
    parser.add_argument("--caption_source", type=str)
    parser.add_argument("--fold_idx", type=int)
    parser.add_argument("--data_path", type=str, default='kinase_data_processing/')

    args = parser.parse_args()

    if args.val_batch_size == None:
        args.val_batch_size = args.batch_size

    config = vars(args)
    print(config)

    os.makedirs(config["ckpt_path"], exist_ok=True)

    if config["val_batch_size"] == None:
        config["val_batch_size"] = config["batch_size"]

    # seed_everything(config["seed"])

    experiment_id = (
        config["model"]
        + ":"
        + config["data_module"]
        + ":"
        + str(config["fold_idx"])
        + ":"
        + "mCLM"
    )

    config["task"] = args.task

    task_type = 'NTP'  # next token prediction

    molecule_tokenizer = get_blocks

    if config["data_module"] == "Kinase":
        output_dim = 1
        dm = KinaseDataModule(
            config,
            molecule_tokenizer = molecule_tokenizer,
            data_path = config["data_path"],
            base_model=config["base_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
        )

    dm.setup('test')
    tokenizer = dm.tokenizer
    test_loader = dm.test_dataloader()

    # test GNN input dict
    block_ID_to_data = dm.GNN_input_map
    print('GNN Input Dict')
    print(dict(list(block_ID_to_data.items())[:10]))

    print("Loading model...")
    if "Llama" in config["base_model"]:
        from mCLM.model.llama_based.model import LlamaForCausalLM
        mCLM_Model = LlamaForCausalLM
    elif "Qwen" in config["base_model"]:
        from mCLM.model.qwen_based.model import Qwen2ForCausalLM
        mCLM_Model = Qwen2ForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = config["pretrained_text_model"]
    model = mCLM_Model.from_pretrained(ckpt_path)
    model.to(device)

    print("extending text vocab size and setting molecule vocab...")
    model.extend_text_vocab_size(len(dm.tokenizer.vocab))
    model.set_mol_vocab(block_ID_to_data)
    print(model.config)

    test_iter = iter(test_loader)
    item = next(test_iter)

    print("Testing GNN forwarding...")
    graph = list(block_ID_to_data.values())[0]
    graph_feature = model.model.mol_gnn(graph)
    print("graph_feature", graph_feature.shape)
    print(graph_feature)

    print("Testing model training time forwarding...")
    model.train(True)
    # currently I am still use gnn to calculate the mol_embeddings
    # to load a custom Tensor feature, use keyword `embeddings=`:
    # model.finalize_molecule_embeddings(embeddings=your_mol_embeddings)
    model.finalize_molecule_embeddings(batch_size=1024)
    model.use_mol_embeddings(True)
    training_output = model(
        input_ids=item["input"]["input_ids"][:, 0],
        attention_mask=item["input"]["attention_mask"][:, 0],
        labels=item["input"]["input_ids"][:, 0]
    )
    print("training_output")
    print(training_output)

    print("Testing model inference time forwarding...")
    model.train(False)
    inference_output = model(
        input_ids=item["input"]["input_ids"][:, 0],
        attention_mask=item["input"]["attention_mask"][:, 0],
        labels=item["input"]["input_ids"][:, 0]
    )
    print("inference_output")
    print(inference_output)

    print("Testing model generation...")
    model.train(False)
    prompt = item["input"]["input_ids"][0, 0].tolist()
    prompt = prompt[:prompt.index(tokenizer.eos_token_id)]
    generated = model.generate(
        input_ids=torch.tensor([prompt]).to(device),
        max_new_tokens=50,
        num_beams=1,
        do_sample=False,
    )
    print("prompt:", tokenizer.decode(prompt))
    print("generated:", tokenizer.decode(
        generated[0].tolist()[len(prompt):]))
