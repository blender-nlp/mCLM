
from typing import List, Dict, Callable, Union, Tuple

import torch

import lightning as L
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR


from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


class mCLM(L.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.save_hyperparameters(ignore=[])

        ckpt_path = config["pretrained_text_model"]

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

        if "Llama" in config["base_model"]:
            from mCLM.model.llama_based.model import LlamaForCausalLM
            mCLM_Model = LlamaForCausalLM
        elif "Qwen" in config["base_model"]:
            from mCLM.model.qwen_based.model import Qwen2ForCausalLM
            mCLM_Model = Qwen2ForCausalLM
        self.model = mCLM_Model.from_pretrained(ckpt_path)
        self.model = get_peft_model(self.model, peft_config)

        #PEFT turns these off
        for param in self.model.base_model.model.model.mol_gnn.parameters():
            param.requires_grad = True

        #self.model.print_trainable_parameters()

    #def train(self, mode=True):
    #    super().train(mode=mode)
    #
    #    self.model.base_model.model.model.mol_gnn.train(mode=mode)

    def on_fit_start(self):
        # In case PL tries to move it before first batch
        self.model._finalized_molecule_embeddings.cpu()

    def setup(self, stage=None):
        self.model.extend_text_vocab_size(len(self.trainer.datamodule.tokenizer.vocab))
        self.model.set_mol_vocab(self.trainer.datamodule.molecule_tokenizer)

        #self.model.mapping_tensor = torch.full((self.model.total_vocab_size,), -1, dtype=torch.long)
        #print('mapping:', self.model.mapping_tensor.shape)
        #self.model.mapping_tensor.requires_grad = False

    def get_metrics(
        self,
    ) -> Dict[str, Callable[[torch.tensor, torch.tensor], torch.tensor]]:
        metrics = dict()

        #metrics["auc"] = torchmetrics.classification.BinaryAUROC()
        #metrics["auprc"] = torchmetrics.classification.BinaryAveragePrecision()
        #metrics["accuracy"] = torchmetrics.classification.BinaryAccuracy()
        #metrics["precision"] = torchmetrics.classification.BinaryPrecision()
        #metrics["recall"] = torchmetrics.classification.BinaryRecall()

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

        #print('the input:', batch["input"]["input_ids"].shape, batch["input"]["attention_mask"].shape, batch["input"]["labels"].shape)

        #print('in mCLM', self.model.negative_sampling_size)

        output = self.model(
            input_ids=batch["input"]["input_ids"],
            attention_mask=batch["input"]["attention_mask"],
            labels=batch["input"]["labels"],
            stage=prefix,
        )
        #print('out mCLM', self.model.negative_sampling_size)

        loss = output.loss


        self.log(
            f"{prefix}/loss",
            loss.item(),
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
        if task_id != None:
            self.log(
                f"{prefix}/{task_id[0]}/loss",
                loss.item(),
                prog_bar=True,
                batch_size=self.config['batch_size'],
                sync_dist=True,
                add_dataloader_idx=False,
            )

        #if prefix == "train":
        #    self._log_metric(step_name=prefix, preds=outputs, y=y)

        return {"loss": loss}

    def on_validation_epoch_end(self):
        #all_validation_probs = torch.cat(
        #    [i["probs"] for i in self.validation_step_outputs]
        #)
        #all_validation_labels = torch.cat(
        #    [i["labels"] for i in self.validation_step_outputs]
        #)
        #self._log_metric("val", all_validation_probs, all_validation_labels)
        #self.validation_step_outputs.clear()
        pass

    def on_test_epoch_end(self):
        pass
        #all_test_probs = torch.cat([i["probs"] for i in self.test_step_outputs])
        #all_test_labels = torch.cat([i["labels"] for i in self.test_step_outputs])
        #self._log_metric("test", all_test_probs, all_test_labels)

    def training_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        return self.compute_step(batch, prefix="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        step_outputs = self.compute_step(batch, prefix="val", task_id = batch['task_id'])
        #self.validation_step_outputs.append(step_outputs)
        return step_outputs

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        step_outputs = self.compute_step(batch, prefix="test", task_id = batch['task_id'])
        #self.test_step_outputs.append(step_outputs)
        return step_outputs

    def configure_optimizers(self):
        #have to include certain parameters separately for some reason (lightning optimizer issue)
        #this is maybe because embed_tokens and lm_head are the same parameters under the hood
        #all_params = list(self.parameters())
        #embed_tokens_params = list(self.model.base_model.model.model.embed_tokens.parameters())
        #lm_head_params = list(self.model.base_model.model.lm_head.parameters())
        #print(embed_tokens_params)
        #print(lm_head_params)

        # Remove duplicates
        #excluded_ids = {id(p) for p in embed_tokens_params}
        #filtered_params = [p for p in all_params if id(p) not in excluded_ids]

        optimizer = torch.optim.Adam(
            self.parameters(),
            #[
            #    {'params': filtered_params}, 
            #    {'params': embed_tokens_params}, 
            #],
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'],
            eps = 1e-7, # default is 1e-8 which causes issues with bf16
            #https://discuss.pytorch.org/t/nan-loss-issues-with-precision-16-in-pytorch-lightning-gan-training/204369/4
        )


        # don't use self.trainer.num_training_batches as it is inf here
        # num_train_batches = self.hparams.num_train_batches
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
        #for k, v in self.metrics.items():
        #    self.metrics[k] = v.to(self.device)

    def on_validation_start(self):
        # fix metrics devices
        pass
        #for k, v in self.metrics.items():
        #    self.metrics[k] = v.to(self.device)

    def on_test_start(self):
        # fix metrics devices
        pass
        #for k, v in self.metrics.items():
        #    self.metrics[k] = v.to(self.device)

    def predict_step(self, batch) -> torch.Tensor:
        if not self.pred_embs:
            out = self(batch)
            return out, batch
        else:
            return self.encoder(batch, get_embedding=True), batch

    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids=input_ids, **kwargs)

