import argparse
import os
import os.path as osp
import lightning as L
import sys
import torch
from lightning import Trainer, seed_everything, Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import pandas as pd

import pickle

from mCLM.data.dataloaders import KinaseDataModule, SMolInstructDataModule, TotalDataModule
from mCLM.model.models import (
    mCLM,
)

import subprocess


def main(args):

    os.makedirs(config["ckpt_path"], exist_ok=True)

    if config["val_batch_size"] == None:
        config["val_batch_size"] = config["batch_size"]

    seed_everything(config["seed"])

    experiment_id = (
        config["model"]
        + ":"
        + config["data_module"]
        + ":"
        + str(config["fold_idx"])
        + ":"
        + "mCLM"
    )



    task_type = 'NTP' #next token prediction

    if False: #debug
        model = model_type(
            config,
            task=task_type,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )



    if False: #config["load_GNN_ckpt"] != None:
        gnn_ckpt_sd = MoleculeTextModel.load_from_checkpoint(
            config["load_GNN_ckpt"],
            config=config,
            encoder=GNNOnlyModel(config, output_dim=output_dim),
        ).encoder.mol_encoder.state_dict()
        #gnn_ckpt_sd.pop("classifier.1.weight")
        #gnn_ckpt_sd.pop("classifier.1.bias")

        molecule_tokenizer.encoder.load_state_dict(gnn_ckpt_sd, strict=True)
    


    if config["data_module"] == "Kinase":
        output_dim = 1
        dm = KinaseDataModule(
            config,
            data_path = 'kinase_data_processing/',
            base_model=config["base_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
        )
    elif config["data_module"] == "SMolInstruct":
        output_dim = 1
        dm = SMolInstructDataModule(
            config,
            instruction_data_path = config['instruction_data_path'],
            synthetic_data_path = config['synthetic_data_path'],
            base_model=config["base_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
        )
    elif config["data_module"] == "Total":
        output_dim = 1
        dm = TotalDataModule(
            config,
            instruction_data_path = config['instruction_data_path'],
            synthetic_data_path = config['synthetic_data_path'],
            base_model=config["base_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            shrink_data=25000,
        )
    

    name = config['task'] + "_" + config["model"] + "_" + config['version']

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

    
    #ckpt_path = config["pretrained_text_model"]
    #model = LlamaForCausalLM.from_pretrained(ckpt_path)
    model = mCLM(config)

    #model.extend_text_vocab_size(len(dm.tokenizer.vocab))
    #model.set_mol_vocab(block_ID_to_data)

    #class SetupCallback(Callback):
    #    def on_fit_start(self, trainer, pl_module):
    #        #print(dir(trainer))
    #        pl_module.model.extend_text_vocab_size(len(trainer.datamodule.tokenizer.vocab))
    #        pl_module.model.set_mol_vocab(trainer.datamodule.molecule_tokenizer.GNN_input_map)

    class MoveMoleculeDevice(Callback):
        def setup(self, trainer, pl_module, stage):
            GNN_input_map = trainer.datamodule.molecule_tokenizer.GNN_input_map
            #print(pl_module.device)
            
            #this is no longer needed because of moving to device in parallelized mol_embed code
            for key in GNN_input_map:
                trainer.datamodule.molecule_tokenizer.GNN_input_map[key] = GNN_input_map[key].to(pl_module.device)

            with open(config["ckpt_path"] + "molecule_tokenizer.pth", "wb") as f:
                torch.save(trainer.datamodule.molecule_tokenizer, f)


    class ClearMoleculeTokenizerCache(Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            trainer.datamodule.molecule_tokenizer.clear_data()

    #callbacks.append(ClearMoleculeTokenizerCache())


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


    class LossThresholdCallback(Callback):
        def __init__(self, every_n_steps=100, loss_threshold=0.1):
            self.every_n_steps = every_n_steps
            self.loss_threshold = loss_threshold
            self.total_loss = 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            global_step = trainer.global_step

            if 'loss' in outputs:
                self.total_loss += outputs['loss'].item()

            # Check every n steps
            if global_step % self.every_n_steps == 0 and 'loss' in outputs:
                loss = self.total_loss / self.every_n_steps
                self.total_loss = 0
                if loss < self.loss_threshold:
                    self.make_config_change(pl_module)

        def make_config_change(self, pl_module):
            if pl_module.model.negative_sampling_size < config['max_negative_sampling_schedule']:
                #print(f"Current negative sampling is {pl_module.model.negative_sampling_size}")
                pl_module.model.negative_sampling_size *= 2
                print(f"Loss below threshold — set negative sampling size to {pl_module.model.negative_sampling_size}.")

    if config['max_negative_sampling_schedule'] != None:
        callbacks.append(LossThresholdCallback(every_n_steps=100, loss_threshold=config['negative_sampling_schedule_loss']))

    if config['check_val_every_n_steps']:
        trainer = Trainer(
            default_root_dir=dirpath,
            max_epochs=config["max_epochs"],
            accelerator="auto",
            devices="auto",#torch.cuda.device_count() if torch.cuda.is_available() else None,  # limiting got iPython runs
            logger=wandb_logger,
            callbacks=callbacks,
            val_check_interval=config['check_val_every_n_steps'],
            strategy='ddp',
        )
    else:
        trainer = Trainer(
            default_root_dir=dirpath,
            max_epochs=config["max_epochs"],
            accelerator="auto",
            devices="auto",#torch.cuda.device_count() if torch.cuda.is_available() else None,  # limiting got iPython runs
            logger=wandb_logger,
            callbacks=callbacks,
            strategy='ddp',
        )


    if config["resume_from_checkpoint"] != None:
        trainer.fit(model, datamodule=dm, ckpt_path=config["resume_from_checkpoint"])
    else:
        #trainer.validate(model, datamodule=dm) 
        trainer.fit(model, datamodule=dm)


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
    parser.add_argument("--batch_size", default=4, type=int) #2 takes up 29733MiB
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
    parser.add_argument("--ckpt_path", default="ckpts/", type=str)
    parser.add_argument("--loss", default="CLIP", type=str)
    parser.add_argument("--load_ckpt", default=None, type=str)
    parser.add_argument("--load_GNN_ckpt", default=None, type=str)

    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--model", default="mCLM", type=str)
    parser.add_argument("--base_model", default="/home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B/", type=str)
    parser.add_argument("--pretrained_text_model", default="/home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B/", type=str)
    parser.add_argument("--pretrained_tokenizer", default="/home/a-m/cne2/MMLI_projects/LLMs/Llama-3.2-1B-Instruct/", type=str)

    parser.add_argument(
        "--freeze_GNN", type=bool, action=argparse.BooleanOptionalAction
    )

    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    parser.add_argument("--resume_wandb_run", default=None, type=str)

    parser.add_argument("--task", default='Kinase', type=str)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    parser.add_argument("--data_module", type=str, default='Kinase')
    parser.add_argument("--version", type=str, default='')
    parser.add_argument("--caption_source", type=str)
    parser.add_argument("--fold_idx", type=int)

    parser.add_argument("--check_val_every_n_steps", default=None, type=int)
    parser.add_argument("--save_checkpoint_every_n_steps", default=None, type=int)

    parser.add_argument("--instruction_data_path", type=str, default='/home/a-m/cne2/MMLI_projects/mCLM/data/instruction/')
    parser.add_argument("--synthetic_data_path", type=str, default='/home/a-m/cne2/MMLI_projects/mCLM/data/synthetic/')

    parser.add_argument("--max_negative_sampling_schedule", default=None, type=int)
    #parser.add_argument("--negative_sampling_schedule", default=100, type=int)
    parser.add_argument("--negative_sampling_schedule_loss", default=0.1, type=float)

    args = parser.parse_args()

    if args.val_batch_size == None:
        args.val_batch_size = args.batch_size

    config = vars(args)

    config["task"] = args.task
    
    print(config)
    # TRAIN
    main(config)

    