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

from mCLM.model.models import mCLM

import subprocess


if __name__ == "__main__":
    torch.cuda.empty_cache()

    #subprocess.run(["nvidia-smi"])

    config = {
        "pretrained_text_model": "michiyasunaga/BioLinkBERT-base",
        "trunc_length": 512,
        "num_warmup_steps": 1000,
        "max_epochs": 2,
        "batch_size": 128,
        "val_batch_size": None,
        "node_dim": 142,
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
    parser.add_argument("--trunc_length", default=512, type=int)

    parser.add_argument("--num_warmup_steps", default=1000, type=int)
    parser.add_argument("--max_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--val_batch_size", default=None, type=int)

    parser.add_argument("--node_dim", default=142, type=int)
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
    parser.add_argument("--base_model", default="/home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/", type=str)
    parser.add_argument("--pretrained_text_model", default="/home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/", type=str)

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
            data_path = config["data_path"],
            base_model=config["base_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
        )

    dm.setup('test')
    llama_tokenizer = dm.tokenizer
    test_loader = dm.test_dataloader()

    # test GNN input dict
    block_ID_to_data = dm.molecule_tokenizer.GNN_input_map
    #print('GNN Input Dict')
    #print(block_ID_to_data)

    print("Loading model...")
    from mCLM.model.llama_based.model import LlamaForCausalLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = config["pretrained_text_model"]
    #model = LlamaForCausalLM.from_pretrained(ckpt_path)
    model = mCLM(config)

    model.model.extend_text_vocab_size(len(dm.tokenizer.vocab))
    model.model.set_mol_vocab(block_ID_to_data)
    #print(model.config)

    # test graph forwarding
    if False:
        graph = block_ID_to_data[128258]
        graph_feature = model.model.embed_molecules(graph)


    # model forwarding, testing mode
    test_iter = iter(test_loader)
    item = next(test_iter)
    #print(item["input"]["input_ids"].shape, item["input"]["attention_mask"].shape,
    #    item["input"]["input_ids"].shape)
    model.train(False)
    model.model.post_training()
    inference_output = model.compute_step(
        item,
        'test',
    )
        #input_ids=item["input"]["input_ids"],
        #attention_mask=item["input"]["attention_mask"],
        #labels=item["input"]["input_ids"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    # model forwarding, training mode
    model.train(True)
    training_output = model.compute_step(
        item,
        'train',
    )
        #input_ids=item["input"]["input_ids"],
        #attention_mask=item["input"]["attention_mask"],
        #labels=item["input"]["labels"]
    training_output['loss'].backward()

    for p in model.named_parameters():
        print(p[0], p[1].requires_grad)

    grad_dict = {k:v.grad for k, v in zip(model.state_dict(), model.parameters())}
    print(grad_dict['model.base_model.model.model.mol_gnn.convs.1.nn.0.bias'])
    print(grad_dict['model.base_model.model.lm_head.weight'])

    optimizer.step()

    orig_model = mCLM(config)
    orig_model.model.extend_text_vocab_size(len(dm.tokenizer.vocab))
    orig_model.model.set_mol_vocab(dm.molecule_tokenizer.GNN_input_map)

    key = 'model.base_model.model.model.mol_gnn.convs.1.nn.0.bias'
    print(key, (model.state_dict()[key] == orig_model.state_dict()[key]).all())
    key = 'model.base_model.model.lm_head.weight'
    print(key, (model.state_dict()[key] == orig_model.state_dict()[key]).all())
    key = 'model.base_model.model.model.embed_tokens.weight'
    print(key, (model.state_dict()[key] == orig_model.state_dict()[key]).all())
    key = 'model.base_model.model.model.layers.15.mlp.up_proj.lora_A.default.weight'
    print(key, (model.state_dict()[key] == orig_model.state_dict()[key]).all())
