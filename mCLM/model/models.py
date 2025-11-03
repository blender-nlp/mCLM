
from typing import List, Dict, Callable, Union, Tuple

import torch
from torch import nn

import lightning as L
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR

from mCLM.data.processing import load_with_tqdm

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

import torch
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import hf_hub_download
from mCLM.model.qwen_based.model import Qwen2ForCausalLM
from mCLM.tokenizer.molecule_tokenizer import MoleculeTokenizer
from peft import get_peft_model, LoraConfig


class mCLM(L.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.save_hyperparameters(ignore=[])

        ckpt_path = config["base_model"]

        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "gate_proj",
                "v_proj",
                "o_proj",
                "k_proj",
                "up_proj",
                "down_proj"
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )

        if isinstance(config["base_model"], Qwen2ForCausalLM):
            self.model = config["base_model"]

        elif "Qwen" in config["base_model"]:
            mCLM_Model = Qwen2ForCausalLM
            self.model = mCLM_Model.from_pretrained(ckpt_path)

        if self.config['PEFT']:
            self.model = get_peft_model(self.model, peft_config)

            #PEFT turns these off
            for param in self.model.base_model.model.model.mol_gnn.parameters():
                param.requires_grad = True
            for param in self.model.base_model.model.model.mol_adaptor.parameters():
                param.requires_grad = True
            
        else:
            class ModelWrapper(nn.Module):
                def __init__(self, model: nn.Module):
                    super(ModelWrapper, self).__init__()
                    self.model = model
                def forward(self, *args, **kwargs):
                    return self.model(*args, **kwargs)
                def __getattr__(self, name):
                    # If attribute not found on self, delegate to model
                    try:
                        return super().__getattr__(name)
                    except AttributeError:
                        return getattr(self.model, name)

            self.model = ModelWrapper(self.model)

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def on_fit_start(self):
        # In case PL tries to move it before first batch
        self.model._finalized_molecule_embeddings.cpu()

    def setup(self, stage=None):
        self.model.extend_text_vocab_size(len(self.trainer.datamodule.tokenizer.vocab))
        self.model.set_mol_vocab(self.trainer.datamodule.molecule_tokenizer)

        if self.config['finetune']:
            print('Setting Finetune to On')
            tokenizer = self.trainer.datamodule.tokenizer
            self.model.use_BCE_loss(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Yes"))[0], tokenizer.convert_tokens_to_ids(tokenizer.tokenize("No"))[0])

        if self.config['load_ckpt'] != None:
        
            sd = load_with_tqdm(self.config["load_ckpt"], map_location='cpu')['state_dict']

            self.load_state_dict(sd, strict=False)

    def get_metrics(
        self,
    ) -> Dict[str, Callable[[torch.tensor, torch.tensor], torch.tensor]]:
        metrics = dict()

        return metrics


    def _log_metric(self, step_name, preds, y, task_name=None):
        for metric_name, metric_func in self.metrics.items():
            if task_name is not None:
                metric_str = f"{step_name}/{task_name}-{metric_name}"
            else:
                metric_str = f"{step_name}/{metric_name}"
            self.log(
                metric_str,
                metric_func(preds, y).mean().item(),
                prog_bar=True,
                sync_dist=True,
            )


    def compute_step(
        self,
        batch,
        prefix: str,
        task_id = None,
    ) -> torch.Tensor:
        # 1. compute loss

        output, text_loss, mol_loss = self.model(
            input_ids=batch["input"]["input_ids"],
            attention_mask=batch["input"]["attention_mask"],
            labels=batch["input"]["labels"],
            stage=prefix,
        )

        loss = output.loss


        if task_id != None:
            
            self.log(
                f"{prefix}/{task_id[0]}/loss",
                loss.item(),
                prog_bar=False,
                batch_size=self.config['batch_size'],
                sync_dist=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"{prefix}/{task_id[0]}/text_loss",
                text_loss,
                prog_bar=False,
                batch_size=self.config['batch_size'],
                sync_dist=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"{prefix}/{task_id[0]}/mol_loss",
                mol_loss,
                prog_bar=False,
                batch_size=self.config['batch_size'],
                sync_dist=True,
                add_dataloader_idx=False,
            )
        else:
            self.log(
                f"{prefix}/loss",
                loss.item(),
                prog_bar=True,
                batch_size=self.config['batch_size'],
                sync_dist=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"{prefix}/text_loss",
                text_loss,
                prog_bar=True,
                batch_size=self.config['batch_size'],
                sync_dist=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"{prefix}/mol_loss",
                mol_loss,
                prog_bar=True,
                batch_size=self.config['batch_size'],
                sync_dist=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"sampling_size",
                self.model.negative_sampling_size,
                prog_bar=True,
                batch_size=self.config['batch_size'],
                sync_dist=True,
                add_dataloader_idx=False,
            )

        return {"loss": loss, "text_loss":text_loss, "mol_loss":mol_loss}

    def on_validation_epoch_end(self):
        all_validation_loss = torch.cat(
            [i["loss"].unsqueeze(0) for i in self.validation_step_outputs]
        )
        self.log(
            "val/loss",
            all_validation_loss.nanmean().item(),
            prog_bar=True,
            sync_dist=True,
        )
        all_validation_loss = torch.Tensor(
            [i["text_loss"] for i in self.validation_step_outputs]
        )
        self.log(
            "val/text_loss",
            all_validation_loss.nanmean().item(),
            prog_bar=True,
            sync_dist=True,
        )
        all_validation_loss = torch.Tensor(
            [i["mol_loss"] for i in self.validation_step_outputs]
        )
        self.log(
            "val/mol_loss",
            all_validation_loss.nanmean().item(),
            prog_bar=True,
            sync_dist=True,
        )
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        all_test_loss = torch.cat(
            [i["loss"].unsqueeze(0) for i in self.test_step_outputs]
        )
        self.log(
            "test/loss",
            all_test_loss.nanmean().item(),
            prog_bar=True,
            sync_dist=True,
        )
        all_test_loss = torch.Tensor(
            [i["text_loss"] for i in self.test_step_outputs]
        )
        self.log(
            "test/text_loss",
            all_test_loss.nanmean().item(),
            prog_bar=True,
            sync_dist=True,
        )
        all_test_loss = torch.Tensor(
            [i["mol_loss"] for i in self.test_step_outputs]
        )
        self.log(
            "test/mol_loss",
            all_test_loss.nanmean().item(),
            prog_bar=True,
            sync_dist=True,
        )
        self.test_step_outputs.clear()

    def training_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        step = self.compute_step(batch, prefix="train")
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"Parameter {name} has no gradient.")

        return step

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        step_outputs = self.compute_step(batch, prefix="val", task_id = batch['task_id'])
        self.validation_step_outputs.append(step_outputs)
        return step_outputs

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        step_outputs = self.compute_step(batch, prefix="test", task_id = batch['task_id'])
        self.test_step_outputs.append(step_outputs)
        return step_outputs

    def configure_optimizers(self):
        # 1. Split parameters
        group1_params = []
        group2_params = []

        for name, param in self.named_parameters():
            if "mol_adaptor" in name or "lm_head" in name:
                group1_params.append(param)
            else:
                group2_params.append(param)

        # 2. Create the optimizer with parameter groups

        optimizer = torch.optim.AdamW(
            [
                {'params': group1_params, 'lr': self.config['mol_lr']},
                {'params': group2_params, 'lr': self.config['lr']},
            ],
            weight_decay=self.config['weight_decay'],
            eps=1e-7, fused=True
        )


        num_training_steps = self.trainer.estimated_stepping_batches

        scheduler_warmup_cosine = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.config["num_warmup_steps"],
            max_epochs=num_training_steps,
            eta_min=self.config['lr'] / 10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler_warmup_cosine, "interval": "step"},
        }

    def on_fit_start(self):
        # fix metrics devices
        pass

    def on_validation_start(self):
        # fix metrics devices
        pass

    def on_test_start(self):
        # fix metrics devices
        pass

    def predict_step(self, batch) -> torch.Tensor:
        if not self.pred_embs:
            out = self(batch)
            return out, batch
        else:
            return self.encoder(batch, get_embedding=True), batch

    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids=input_ids, **kwargs)


    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs):
        """
        Load an mCLM model and molecule tokenizer from a Hugging Face repo.

        Args:
            repo_id (str): Hugging Face repo id, e.g. "language-plus-molecules/mCLM_1k-3b".
            **kwargs: Overrides for config values, e.g. PEFT=True, finetune=False, etc.

        Returns:
            mCLM: A fully initialized LightningModule ready for inference or finetuning.
        """
        # -------------------------
        # 1. Load base model config
        # -------------------------
        config = AutoConfig.from_pretrained(repo_id)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        tokenizer.pad_token = tokenizer.eos_token

        # Make sure config vocab size matches checkpoint, not tokenizer
        if config.vocab_size != len(tokenizer):
            print(f"[mCLM.from_pretrained] Adjusting vocab size from {config.vocab_size} → {len(tokenizer)}")
            config.vocab_size = len(tokenizer)

        # -------------------------
        # 2. Load molecule tokenizer (.pth from repo)
        # -------------------------
        torch.serialization.add_safe_globals([MoleculeTokenizer])
        molecule_tokenizer_path = hf_hub_download(repo_id, "molecule_tokenizer.pth")
        molecule_tokenizer = torch.load(molecule_tokenizer_path, weights_only=False)
        molecule_tokenizer.change_start_idx(len(tokenizer))

        GNN_input_map_path = hf_hub_download(repo_id, "molecule_tokenizer.graphs.pth")
        molecule_tokenizer.GNN_input_map = torch.load(GNN_input_map_path, weights_only=False)

        molecule_tokenizer.set_bfloat16()

        # -------------------------
        # 3. Load Qwen2 model safely
        # -------------------------
        base_model = Qwen2ForCausalLM.from_pretrained(
            repo_id,
            config=config,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )

        # Extend with molecule vocab
        base_model.set_mol_vocab(molecule_tokenizer)
        base_model.extend_text_vocab_size(len(tokenizer.vocab))


        # -------------------------
        # 5. Wrap into LightningModule
        # -------------------------
        full_config = {
            "base_model": base_model,
            "lr": None,
            "mol_lr": None,
            "weight_decay": None,
            "num_warmup_steps": None,
            "PEFT": kwargs.get("PEFT", None),
            "finetune": kwargs.get("finetune", None),
            "load_ckpt": kwargs.get("load_ckpt", None),
            "batch_size": kwargs.get("batch_size", None),
        }



        # Load Pretrained Embeddings
        loaded_mol_embeddings_path = hf_hub_download(repo_id, "precomputed_tokens.pt")
        pretrained_mol_embeddings = torch.load(loaded_mol_embeddings_path, weights_only=False)

        #Create the model using the loaded values
        model = cls(full_config)
        model.tokenizer = tokenizer
        model.molecule_tokenizer = molecule_tokenizer

        model.pretrained_mol_embeddings = pretrained_mol_embeddings

        # Initialize the model to use the loaded embeddings
        pretrain_mol_embeddings = nn.Embedding.from_pretrained(model.pretrained_mol_embeddings, freeze=True)
        pretrain_mol_embeddings.weight.data = pretrain_mol_embeddings.weight.data.to(torch.bfloat16)

        
        model.model.finalize_molecule_embeddings(embeddings=pretrain_mol_embeddings)
        model.model.use_mol_embeddings(True)

        return model
