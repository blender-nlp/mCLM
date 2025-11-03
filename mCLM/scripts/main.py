import argparse
import os
import os.path as osp
import lightning as L
import sys
import torch
import torch.nn as nn
from lightning import Trainer, seed_everything, Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch.strategies import DeepSpeedStrategy

import pandas as pd

import pickle

from mCLM.data.dataloaders import KinaseDataModule, SMolInstructDataModule, TotalDataModule, MolGenSMolInstructDataModule, MolGenTotalDataModule, FinetuneDataModule
from mCLM.model.models import (
    mCLM,
)

import subprocess


def main(args):

    os.makedirs(config["ckpt_path"], exist_ok=True)

    if config["val_batch_size"] == None:
        config["val_batch_size"] = config["batch_size"]

    seed_everything(config["seed"])

    callbacks = []

    if config["data_module"] == "SMolInstruct":
        dm = SMolInstructDataModule(
            config,
            instruction_data_path = config['instruction_data_path'],
            synthetic_data_path = config['synthetic_data_path'],
            base_model=config["base_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            GNN_cache = '../GNN_input_cache/Total.molecule_tokenizer.v4.pth',
        )

    elif config["data_module"] == "Total":
        dm = TotalDataModule(
            config,
            instruction_data_path = config['instruction_data_path'],
            synthetic_data_path = config['synthetic_data_path'],
            base_model=config["base_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            shrink_data=25000,
        )
    elif config["data_module"] == "TotalTop50k":
        dm = TotalDataModule(
            config,
            instruction_data_path = config['instruction_data_path'],
            synthetic_data_path = config['synthetic_data_path'],
            base_model=config["base_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            GNN_cache = '../GNN_input_cache/Total.molecule_tokenizer.50kV2.pth',
        )

    elif config["data_module"] == "SMolInstructTop50k":
        dm = SMolInstructDataModule(
            config,
            instruction_data_path = config['instruction_data_path'],
            synthetic_data_path = config['synthetic_data_path'],
            base_model=config["base_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            GNN_cache = '../GNN_input_cache/Total.molecule_tokenizer.50kV2.pth',
        )

    elif config["data_module"] == "TotalTopK":
        dm = TotalDataModule(
            config,
            instruction_data_path = config['instruction_data_path'],
            synthetic_data_path = config['synthetic_data_path'],
            base_model=config["base_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            GNN_cache = config['tokenizer_cache'],
        )

    elif config["data_module"] == "FinetuneTopK":
        dm = FinetuneDataModule(
            config,
            synthetic_data_path = config['synthetic_data_path'],
            base_model=config["base_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            GNN_cache = config['tokenizer_cache'],
        )


        

        class CallDataModuleFunctionCallback(Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                epoch = trainer.current_epoch
                datamodule = trainer.datamodule

                # Check if the datamodule has the required function
                if hasattr(datamodule, "set_new_epoch") and callable(getattr(datamodule, "set_new_epoch")):
                    datamodule.set_new_epoch(epoch + 1)
                else:
                    print("Warning: datamodule does not have a callable 'set_new_epoch' method")

        callbacks.append(CallDataModuleFunctionCallback())



    name = config['task'] + "_" + config["model"] + "_" + config['version']

    if config["resume_wandb_run"] != None:
        wandb_logger = WandbLogger(
            entity="mCLM",
            project="mCLM",
            config=config,
            id=config["resume_wandb_run"],
            resume="must",
            name=name,
        )
    else:
        wandb_logger = WandbLogger(
            entity="mCLM",
            project="mCLM",
            config=config,
            name=name,
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks.append(lr_monitor)

    dirpath = config["ckpt_path"]

    val_checkpoint = ModelCheckpoint(
        dirpath=dirpath,
        monitor="val/loss",
        mode="min",
        filename="best_val_checkpoint",
        save_top_k=1,
    )
    callbacks.append(val_checkpoint)

    
    model = mCLM(config)

    if config['only_molecule_loss']:
        model.model.use_only_molecule_loss(True)

    if config['only_text_loss']:
        model.model.use_only_text_loss(True)


    if config["load_GNN_ckpt"] != None:
        gnn_ckpt_sd = torch.load(config["load_GNN_ckpt"], map_location='cpu', weights_only=False)['state_dict']
        for key in list(gnn_ckpt_sd.keys()):
            gnn_ckpt_sd[key.replace('mol_encoder.', '')] = gnn_ckpt_sd.pop(key)

        model.model.model.model.mol_gnn.load_state_dict(gnn_ckpt_sd)
        print('Loaded GNN Weights')

    if config['freeze_GNN']:
        for param in model.model.model.model.mol_gnn.parameters():
            param.requires_grad = False


    if config['train_adapter']:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'adaptor' in name:
                param.requires_grad = True


    if config['pretrained_embeddings'] != None:
        print('Initializing pretrained_embeddings')
        pretrain_mol_embeddings = torch.load(config['pretrained_embeddings'] + 'precomputed_tokens.pt').to(torch.bfloat16)
        pretrain_mol_embeddings = nn.Embedding.from_pretrained(pretrain_mol_embeddings, freeze=True)
        pretrain_mol_embeddings.weight.data = pretrain_mol_embeddings.weight.data.to(torch.bfloat16)

        
        model.model.finalize_molecule_embeddings(embeddings=pretrain_mol_embeddings)
        model.model.use_mol_embeddings(True)

    model = model.to(torch.bfloat16)


    class ShuffleTrainingData(Callback):
        def on_train_start(self, trainer, pl_module):
            trainer.datamodule.train_ds.seed(pl_module.current_epoch)

    callbacks.append(ShuffleTrainingData())

    if config['save_checkpoint_every_n_steps']:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="step",
            mode="max",
            dirpath=dirpath,
            every_n_train_steps=config['save_checkpoint_every_n_steps'],
            filename="latest_checkpoint-{epoch:02d}-{step}",
        )
    else:
        # saves last-K checkpoints based on "step" metric
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="step",
            mode="max",
            dirpath=dirpath,
            every_n_train_steps=config['check_val_every_n_steps'],
            filename="latest_checkpoint-{epoch:02d}-{step}",
        )

    callbacks.append(checkpoint_callback)


    if config['pretrained_embeddings'] == None:
        strategy = 'ddp_find_unused_parameters_true'
    else:
        strategy = 'ddp'

    if config['check_val_every_n_steps']:
        val_check_interval = config['check_val_every_n_steps']
    else:
        val_check_interval = None

    trainer = Trainer(
        default_root_dir=dirpath,
        max_epochs=config["max_epochs"],
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        callbacks=callbacks,
        val_check_interval=val_check_interval,
        strategy=strategy,
        precision='bf16',
        accumulate_grad_batches=config['accumulate_grad_batches'],
    )


    if config["resume_from_checkpoint"] != None:
        trainer.fit(model, datamodule=dm, ckpt_path=config["resume_from_checkpoint"])
    else:
        trainer.fit(model, datamodule=dm)


if __name__ == "__main__":

    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser(description="mCLM")
    parser.add_argument("--trunc_length", default=512, type=int)

    parser.add_argument("--num_warmup_steps", default=1000, type=int)
    parser.add_argument("--max_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=4, type=int) 
    parser.add_argument("--val_batch_size", default=None, type=int)

    #GNN args
    parser.add_argument("--node_dim", default=137, type=int)
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
    parser.add_argument("--mol_lr", default=None, type=float)
    parser.add_argument("--ckpt_path", default="ckpts/", type=str)
    parser.add_argument("--load_ckpt", default=None, type=str)
    parser.add_argument("--load_GNN_ckpt", default=None, type=str)

    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--model", default="mCLM", type=str)
    parser.add_argument("--pretrained_text_model", default="LLMs/Qwen2.5-0.5B/", type=str)
    parser.add_argument("--pretrained_tokenizer", default="LLMs/Qwen2.5-0.5B/", type=str)
    parser.add_argument("--pretrained_embeddings", default=None, type=str)
    parser.add_argument("--GNN_cache", default=None, type=str)

    parser.add_argument("--freeze_GNN", type=bool, action=argparse.BooleanOptionalAction)

    parser.add_argument("--PEFT", type=bool, action=argparse.BooleanOptionalAction)

    parser.add_argument("--only_molecule_loss", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--only_text_loss", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--train_adapter", type=bool, action=argparse.BooleanOptionalAction)

    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    parser.add_argument("--resume_wandb_run", default=None, type=str)

    parser.add_argument("--task", default='Task', type=str)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    parser.add_argument("--data_module", type=str, default='Total')
    parser.add_argument("--version", type=str, default='')

    parser.add_argument("--check_val_every_n_steps", default=None, type=int)
    parser.add_argument("--save_checkpoint_every_n_steps", default=None, type=int)

    parser.add_argument("--instruction_data_path", type=str, default='mCLM/data/instruction_onlyblocks_top_50000/')
    parser.add_argument("--synthetic_data_path", type=str, default='mCLM/data/synthetic_onlyblocks_top_50000/')
    parser.add_argument("--tokenizer_cache", type=str, default=None)
    parser.add_argument("--downsample_tulu", default=None, type=float)

    parser.add_argument("--accumulate_grad_batches", default=1, type=int)

    args = parser.parse_args()

    if args.val_batch_size == None:
        args.val_batch_size = args.batch_size

    if args.mol_lr == None:
        args.mol_lr = args.lr

    config = vars(args)

    config["task"] = args.task
    
    print(config)
    # TRAIN
    main(config)

    