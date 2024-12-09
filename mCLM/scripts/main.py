import argparse
import os
import os.path as osp
import lightning as L
import sys
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import pandas as pd

from MolCapArena.data.dataloaders import GraphDataModule
from MolCapArena.model.models import (
    mCLM,
)

import subprocess


if __name__ == "__main__":
    torch.cuda.empty_cache()

    subprocess.run(["nvidia-smi"])

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
    parser.add_argument("--batch_size", default=128, type=int)
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

    parser.add_argument("--data_module", type=str)
    parser.add_argument("--caption_source", type=str)
    parser.add_argument("--fold_idx", type=int)

    args = parser.parse_args()

    if args.val_batch_size == None:
        args.val_batch_size = args.batch_size

    config = vars(args)
    print(config)

    os.makedirs(config["ckpt_path"], exist_ok=True)

    if config["val_batch_size"] == None:
        config["val_batch_size"] = config["batch_size"]

    seed_everything(config["seed"])

    if config["caption_source"] != None:
        experiment_id = (
            config["model"]
            + ":"
            + config["data_module"]
            + ":"
            + str(config["fold_idx"])
            + ":"
            + config["caption_source"]
        )
    else:
        experiment_id = (
            config["model"]
            + ":"
            + config["data_module"]
            + ":"
            + str(config["fold_idx"])
            + ":"
            + "GNN"
        )


    config["task"] = task

    model_type = mCLM

    task_type = 'NTP' #next token prediction

    model = model_type(
        config,
        task=task_type,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    molecule_tokenizer = mCLMGNNModel()


    if config["load_GNN_ckpt"] != None:
        gnn_ckpt_sd = MoleculeTextModel.load_from_checkpoint(
            config["load_GNN_ckpt"],
            config=config,
            encoder=GNNOnlyModel(config, output_dim=output_dim),
        ).encoder.mol_encoder.state_dict()
        #gnn_ckpt_sd.pop("classifier.1.weight")
        #gnn_ckpt_sd.pop("classifier.1.bias")

        molecule_tokenizer.encoder.load_state_dict(gnn_ckpt_sd, strict=True)


    processor = mCLMProcessor(molecule_tokenizer=molecule_tokenizer)

    


    if config["data_module"] == "BBBP":
        task = "BBBP"
        output_dim = 1
        dm = GraphDataModule(
            config,
            pretrained_text_model=config["pretrained_text_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            task=task,
            fold_idx=config["fold_idx"],
        )

    if config["caption_source"] != None:
        name = (
            task
            + "_"
            + config["model"]
            + "_"
            + config["caption_source"]
            + "_"
            + str(config["fold_idx"])
        )
    else:
        name = task + "_" + config["model"] + "_" + str(config["fold_idx"])

    if config["resume_wandb_run"] != None:
        wandb_logger = WandbLogger(
            entity="",
            project="mCLM",
            config=config,
            id=config["resume_wandb_run"],
            resume="must",
            name=name,
        )
    else:
        wandb_logger = WandbLogger(
            entity="",
            project="mCLM",
            config=config,
            name=name,
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [lr_monitor]

    dirpath = config["ckpt_path"]

    val_checkpoint = ModelCheckpoint(
        dirpath=dirpath,
        monitor="val/loss",
        mode="min",
        filename="best_val_checkpoint",
        save_top_k=1,
    )
    callbacks.append(val_checkpoint)

    trainer = Trainer(
        default_root_dir=dirpath,
        max_epochs=config["max_epochs"],
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        callbacks=callbacks,
    )

    if config["resume_from_checkpoint"] != None:
        trainer.fit(model, datamodule=dm, ckpt_path=config["resume_from_checkpoint"])
    else:
        trainer.validate(model, datamodule=dm)  # bug in regression it seems
        zz
        trainer.fit(model, datamodule=dm)

    tokenizer.save_model(dirpath)
    trainer.save_model(dirpath)



