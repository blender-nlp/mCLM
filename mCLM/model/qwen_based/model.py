from functools import partial
from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    LossKwargs,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    # can_return_tuple,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg

from transformers.models.qwen2.modeling_qwen2 import \
    Qwen2Model as OriginalQwen2Model, \
    Qwen2PreTrainedModel, Qwen2DecoderLayer, \
    Qwen2RMSNorm, Qwen2RotaryEmbedding, \
    QWEN2_START_DOCSTRING, QWEN2_INPUTS_DOCSTRING

from ..components import GNNMolEncoder
from .configuration import Qwen2Config as Qwen2Config
from ..utils import embed_chemical_language, \
    finalized_molecule_embeddings, mclm_logit_head, embed_molecules_fn, \
    MLPAdaptor, compute_loss, \
    mclm_logit_head_optimized, compute_loss_optimized, \
    mclm_logit_head_optimized2, compute_loss_optimized2, \
    mclm_logit_head_optimized2_sep, compute_loss_optimized2_sep


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-qwen2/Qwen2-2-7b-hf"
_CONFIG_FOR_DOC = "Qwen2Config"


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2Model(OriginalQwen2Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        # self.vocab_size = config.vocab_size
        self.text_vocab_size = config.vocab_size
        # mCLM mol-related init
        self.mol_gnn = GNNMolEncoder(
            **config.molecule_config)
        self.mol_adaptor = MLPAdaptor(
            config.molecule_config["input_dim_adapter"],
            config.molecule_config["hidden_dim_adapter"],
            config.molecule_config["out_channels_adapter"],
            dropout=config.molecule_config["dropout"]
        )
        self.mol_vocab = None
        self._use_mol_embeddings = False
        self._finalized_molecule_embeddings = [None]

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.on_device = True #self.config.molecule_config["out_channels_adapter"] == 896 #check if 0.5B model or not
        #print(self.config.molecule_config["out_channels"])
        #print('self.on_device', self.on_device)
        #zz

        # Initialize weights and apply final processing
        self.post_init()

    def embed_molecules(self, mol_input_ids):
        #if isinstance(self._finalized_molecule_embeddings, torch.Tensor): #I need to do this after lightning moves things to gpu
        #    self._finalized_molecule_embeddings = nn.Embedding.from_pretrained(self._finalized_molecule_embeddings, freeze=True)

        #print('on_device:', self.config.molecule_config["out_channels"] == 768)

        return embed_molecules_fn(
            mol_input_ids, self.config.molecule_config["out_channels_adapter"],
            self.mol_vocab, self.mol_gnn, self.mol_adaptor,
            self._finalized_molecule_embeddings[0], self._use_mol_embeddings,
            self.text_vocab_size,
            self.dtype, self.device,
            on_device=self.on_device
        )

    # mCLM extend text embedding
    def extend_text_vocab_size(self, new_vocab_size):
        # assert new_vocab_size > self.vocab_size
                    
        self.embed_tokens.weight = nn.Parameter(
            F.pad(self.embed_tokens.weight,
                (0, 0, 0, new_vocab_size - self.text_vocab_size), "constant", 0)
        )
        #https://www.cs.columbia.edu/~johnhew/vocab-expansion.html
        old_weight = self.embed_tokens.weight[:new_vocab_size-2, :] #2 is this is hardcoded for mCLM
        mean_emb = old_weight.mean(dim=0, keepdim=True)
        new_rows = mean_emb.repeat(2, 1) 
        self.embed_tokens.weight = nn.Parameter(torch.cat([old_weight, new_rows], dim=0))

        self.text_vocab_size = new_vocab_size
        self.config.vocab_size = new_vocab_size + self.config.mol_vocab_size

    # @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_tokens(input_ids)
        # mCLM embedding function
        if inputs_embeds is None:
            inputs_embeds = embed_chemical_language(
                input_ids, self.text_vocab_size, self.embed_tokens, self.embed_molecules
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)

        # mCLM config
        config = Qwen2Config.from_dict(config.to_dict())
        self._finalized_molecule_embeddings = [None]

        self.model = Qwen2Model(config)
        self.config = config
        # self.vocab_size = config.vocab_size
        self.config.text_vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.mapping_tensor = torch.full((self.mol_vocab_size,), -1, dtype=torch.long)
        self.mapping_tensor.requires_grad = False

        #self.negative_sampling_size = int(self.config.negative_sampling_size)

        self.register_buffer("negative_sampling_size", torch.tensor(self.config.negative_sampling_size, dtype=torch.long), persistent=True)

        # Initialize weights and apply final processing
        self.post_init()
        self.use_mol_embeddings(False)

        self.only_molecule_loss= False



    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # mCLM total vocab size (to work with beam search)
    @property
    def vocab_size(self):
        return self.config.vocab_size

    # mCLM molecule vocab sizes
    @property
    def mol_vocab_size(self):
        return self.config.mol_vocab_size

    # mCLM total vocab sizes
    @property
    def text_vocab_size(self):
        return self.config.text_vocab_size

    # mCLM extend text embedding
    def extend_text_vocab_size(self, new_vocab_size):
        # In Qwen2, the embedding size is rounded up to multiple of 256,
        # so we do not check if new_vocab_size is larger
        # assert new_vocab_size > self.vocab_size

        #new_size = new_vocab_size
        #old_size = self.vocab_size
        #if new_size <= old_size:
        #    print(f"Skipping vocab extension: new_size={new_size}, old_size={old_size}")

        self.lm_head.weight = nn.Parameter(
            F.pad(self.lm_head.weight,
                (0, 0, 0, new_vocab_size - self.text_vocab_size), "constant", 0)
        )
        #https://www.cs.columbia.edu/~johnhew/vocab-expansion.html
        old_weight = self.lm_head.weight[:new_vocab_size-2, :] #2 is this is hardcoded for mCLM
        mean_emb = old_weight.mean(dim=0, keepdim=True)
        new_rows = mean_emb.repeat(2, 1) 
        self.lm_head.weight = nn.Parameter(torch.cat([old_weight, new_rows], dim=0))

        self.model.extend_text_vocab_size(new_vocab_size)
        self.config.text_vocab_size = new_vocab_size
        self.config.vocab_size = self.text_vocab_size + self.config.mol_vocab_size


    # mCLM force use molecule embeddings
    def use_mol_embeddings(self, use_mol_embeddings):
        self._use_mol_embeddings = use_mol_embeddings
        self.model._use_mol_embeddings = use_mol_embeddings

    # mCLM set molecule vocab
    def set_mol_vocab(self, mol_vocab):
        self.mol_vocab = mol_vocab
        self.model.mol_vocab = mol_vocab
        self.config.mol_vocab_size = len(mol_vocab)

        self.config.vocab_size = self.text_vocab_size + self.config.mol_vocab_size

    # @can_return_tuple
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        #torch.set_printoptions(threshold=float('inf'))
        #print('input ids:')
        #print(input_ids)
        #zz

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(hidden_states[:, slice_indices, :])

        #print(f"[Forward] negative_sampling_size: {self.negative_sampling_size}")
        is_generating_mol = (input_ids == self.text_vocab_size - 2).logical_or(
                input_ids >= self.text_vocab_size)
        #print('is_generating_mol:')
        #print(self.vocab_size, is_generating_mol, input_ids)
        #print(is_generating_mol.shape)
        # mCLM logit head
        logits = mclm_logit_head_optimized2_sep(
            self.lm_head, self.model.embed_molecules,
            self._finalized_molecule_embeddings[0],
            self.text_vocab_size, self.mol_vocab_size, self.vocab_size,
            #self.negative_sampling_size.item(),
            None,
            hidden_states,
            is_generating_mol=is_generating_mol,
            is_training=self.training,
            labels=labels,
        )


        # mCLM loss
        loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.total_vocab_size)
        #     shift_labels = shift_labels.view(-1)
        #     # Enable model parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels.to(torch.long))
        #print('mapping in forward:', self.mapping_tensor.shape)

        if labels is not None:
            if len(self.mapping_tensor) != self.total_vocab_size:
                self.mapping_tensor = torch.full((self.total_vocab_size,), -1, dtype=torch.long, device=hidden_states.device)
                self.mapping_tensor.requires_grad = False

            #loss = compute_loss_optimized2(logits, labels, mapping_tensor=self.mapping_tensor)
            #text_loss, mol_loss = None, None
            text_loss, mol_loss = compute_loss_optimized2_sep(logits, labels, mapping_tensor=self.mapping_tensor)
            if mol_loss == None:
                loss = text_loss
                text_loss, mol_loss = text_loss.item(), torch.nan
            elif text_loss == None:
                loss = mol_loss
                text_loss, mol_loss = torch.nan, mol_loss.item()
            else:
                loss = (text_loss + mol_loss)/2
                text_loss, mol_loss = text_loss.item(), mol_loss.item()

            if self.only_molecule_loss:
                loss = mol_loss

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ), text_loss, mol_loss #for logging

        else:
            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    def finalize_molecule_embeddings(self, batch_size=None, embeddings=None):
        if batch_size is None:
            assert embeddings is not None
            self._finalized_molecule_embeddings = [embeddings] #this is a trick to prevent lightning from seeing this
            self._finalized_molecule_embeddings[0].requires_grad = False
        else:
            self._finalized_molecule_embeddings = \
                [finalized_molecule_embeddings(
                    self.text_vocab_size, self.mol_vocab_size,
                    self.model.embed_molecules,
                    self.config.hidden_size, batch_size, self.device
                )]
        self.model._finalized_molecule_embeddings = \
            self._finalized_molecule_embeddings


__all__ = [
    "Qwen2ForCausalLM",
    "Qwen2Model",
]
