# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, Dict, List

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from collections import defaultdict

from transformers.activations import ACT2FN, get_activation
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    auto_docstring,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

# --- IMPORTS FROM YOUR ENVIRONMENT ---
from models.l0 import HardConcreteGate

logger = logging.get_logger(__name__)


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


def eager_attention_forward(module, query, key, value, attention_mask, head_mask=None, **kwargs):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    if not module.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with torch.amp.autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)

        return attn_output, attn_weights

    @deprecate_kwarg("layer_past", new_name="past_key_value", version="4.53.0", raise_if_both_names=True)
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        if past_key_value is not None:
            if isinstance(past_key_value, EncoderDecoderCache):
                if is_cross_attention:
                    past_key_value = past_key_value.cross_attention_cache
                else:
                    past_key_value = past_key_value.self_attention_cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs=cache_kwargs
            )

        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
                using_eager = True
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
                # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
                # not necessarily to eager (if mentioned options are provided).
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config=config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    @deprecate_kwarg("layer_past", new_name="past_key_value", version="4.53.0", raise_if_both_names=True)
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, self_attn_weights = self.attn(
            hidden_states,
            past_key_value=past_key_value,
            cache_position=cache_position,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_output, cross_attn_weights = self.crossattention(
                hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            # residual connection
            hidden_states = residual + cross_attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
            if encoder_hidden_states is not None:
                outputs += (cross_attn_weights,)

        return outputs


# Copied from transformers.models.xlm.modeling_xlm.XLMSequenceSummary with XLM->GPT2
class GPT2SequenceSummary(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.summary_type = getattr(config, "summary_type", "last")
        if self.summary_type == "attn":
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.summary = nn.Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        activation_string = getattr(config, "summary_activation", None)
        self.activation: Callable = get_activation(activation_string) if activation_string else nn.Identity()

        self.first_dropout = nn.Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = nn.Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(
        self, hidden_states: torch.FloatTensor, cls_index: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                cls_index = torch.full_like(
                    hidden_states[..., :1, :],
                    hidden_states.shape[-2] - 1,
                    dtype=torch.long,
                )
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


@auto_docstring
class GPT2PreTrainedModel(PreTrainedModel):
    config_class = GPT2Config
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))


@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    mc_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.
"""


PRUNING_FACTOR = 5  # Default pruning factor for the hard concrete gates
@dataclass
class PruningConfig:
    init_value: float = 1.0
    sparsity_warmup_steps: int = 1000

    # --- Fine-grained pruning (existing) ---
    # Attention Head Pruning
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 0.01 * PRUNING_FACTOR

    # MLP neuron pruning
    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 0.005 * PRUNING_FACTOR
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 0.005 * PRUNING_FACTOR
    
    prune_embedding: bool = True
    lambda_embedding: float = 1 * PRUNING_FACTOR
    
    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 0.002 * PRUNING_FACTOR
    
    # --- NEW: Block-level pruning ---
    # Prune entire attention blocks
    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 0.02 * PRUNING_FACTOR
    
    # Prune entire MLP blocks
    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 0.02 * PRUNING_FACTOR
    
    # Prune entire transformer layers
    prune_full_layers: bool = True
    lambda_full_layers: float = 0.05 * PRUNING_FACTOR
    
class PrunableAttention(nn.Module):
    def __init__(self, original_attention, gpt_config: GPT2Config, pruning_config: PruningConfig):
        super().__init__()
        self.original_attention = original_attention
        self.num_heads = gpt_config.n_head
        self.head_dim = gpt_config.hidden_size // self.num_heads
        
        # --- Head-level gates (Level 2) ---
        if pruning_config.prune_attention_heads:
            self.head_gates = HardConcreteGate(self.num_heads)
        else:
            self.head_gates = None
            
        ### NEW: Neuron-level gates (Level 3) ###
        if pruning_config.prune_attention_neurons:
            # One gate for each dimension within each head
            self.neuron_gates = HardConcreteGate(self.num_heads * self.head_dim)
        else:
            self.neuron_gates = None
        ### END NEW ###

        
    def forward(self, hidden_states, **kwargs):
        # Forward pass through the original attention mechanism
        attn_outputs = self.original_attention(hidden_states, **kwargs)
        output = attn_outputs[0]
        
        # Start with the clean output
        gated_output = output
        
        if self.head_gates or self.neuron_gates:
            b, s, d = output.shape
            
            # Reshape output to expose head and head_dim
            gated_output_reshaped = output.view(b, s, self.num_heads, self.head_dim)
            
            # --- Apply Level 2: Head Gates ---
            # Zero Ablation: mask * clean
            if self.head_gates:
                head_gate = self.head_gates().view(1, 1, self.num_heads, 1)
                gated_output_reshaped = head_gate * gated_output_reshaped
            
            ### NEW: Apply Level 3: Neuron Gates ###
            # Zero Ablation: mask * clean
            if self.neuron_gates:
                neuron_gate = self.neuron_gates().view(1, 1, self.num_heads, self.head_dim)
                gated_output_reshaped = gated_output_reshaped * neuron_gate
            ### END NEW ###
            
            # Reshape back to the original tensor shape
            gated_output = gated_output_reshaped.view(b, s, d)
        
        # Return gated output and original attention weights (if any)
        return (gated_output,) + attn_outputs[1:]

class PrunableMLP(nn.Module):
    def __init__(self, original_mlp, gpt_config: GPT2Config, pruning_config: PruningConfig):
        super().__init__()
        self.original_mlp = original_mlp
        self.pruning_config = pruning_config
        
        # --- Create gates based on the PruningConfig ---
        self.hidden_gates = None
        if self.pruning_config.prune_mlp_hidden:
            intermediate_size = gpt_config.n_inner if gpt_config.n_inner is not None else 4 * gpt_config.hidden_size
            self.hidden_gates = HardConcreteGate(intermediate_size)

        self.output_gates = None
        if self.pruning_config.prune_mlp_output:
            self.output_gates = HardConcreteGate(gpt_config.hidden_size)
            
    def forward(self, hidden_states):
        # --- Deconstructed Forward Pass for Zero Ablation ---
        
        # 1. Get hidden activations
        act = self.original_mlp.act(self.original_mlp.c_fc(hidden_states))

        # 2. Apply gates to the HIDDEN layer (Zero Ablation)
        gated_act = act
        if self.hidden_gates:
            gate = self.hidden_gates().view(1, 1, -1)
            gated_act = gate * act
        
        # 3. Get final outputs from the second linear layer
        output = self.original_mlp.dropout(self.original_mlp.c_proj(gated_act))

        # 4. Apply gates to the OUTPUT layer (Zero Ablation)
        gated_output = output
        if self.output_gates:
            gate = self.output_gates().view(1, 1, -1)
            gated_output = gate * output

        return gated_output

    def get_sparsity_loss(self) -> Dict[str, torch.Tensor]:
        """Calculates the sparsity loss for any gates present in this module."""
        losses = {}
        if self.hidden_gates:
            losses['mlp_hidden'] = self.hidden_gates.get_sparsity_loss()
        if self.output_gates:
            losses['mlp_output'] = self.output_gates.get_sparsity_loss()
        return losses

from collections import defaultdict


class PrunableGPT2Block(nn.Module):
    def __init__(self, original_block, gpt_config: GPT2Config, pruning_config: PruningConfig):
        super().__init__()
        self.ln_1 = original_block.ln_1
        self.ln_2 = original_block.ln_2
        
        self.attn = PrunableAttention(original_block.attn, gpt_config, pruning_config)
        self.mlp = PrunableMLP(original_block.mlp, gpt_config, pruning_config)
        
        self.attention_block_gate = None
        if pruning_config.prune_attention_blocks:
            self.attention_block_gate = HardConcreteGate(1)
            
        self.mlp_block_gate = None
        if pruning_config.prune_mlp_blocks:
            self.mlp_block_gate = HardConcreteGate(1)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        **kwargs
    ) -> Tuple[torch.FloatTensor]:
        # --- First Sub-block: Multi-Head Self-Attention ---
        
        # Apply LayerNorm
        ln_states = self.ln_1(hidden_states)
        
        # Pass through PrunableAttention
        attn_outputs = self.attn(ln_states, **kwargs)
        attn_output = attn_outputs[0]
        
        # NEW: Apply attention block gate (Zero Ablation)
        if self.attention_block_gate:
            gate = self.attention_block_gate()
            attn_output = gate * attn_output
        
        # Residual connection
        hidden_states = hidden_states + attn_output

        # --- Second Sub-block: MLP ---
        
        # Apply LayerNorm
        ln_after_attn = self.ln_2(hidden_states)
        
        # Pass through PrunableMLP
        mlp_output = self.mlp(ln_after_attn)
        
        # NEW: Apply MLP block gate (Zero Ablation)
        if self.mlp_block_gate:
            gate = self.mlp_block_gate()
            mlp_output = gate * mlp_output
        
        # Final residual connection
        final_states = hidden_states + mlp_output
        
        return final_states, attn_outputs

    def get_sparsity_loss(self) -> Dict[str, torch.Tensor]:
        """Get sparsity losses for block-level gates."""
        losses = {}
        if self.attention_block_gate:
            losses['attention_blocks'] = self.attention_block_gate.get_sparsity_loss()
        if self.mlp_block_gate:
            losses['mlp_blocks'] = self.mlp_block_gate.get_sparsity_loss()
        return losses


from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class PrunableGPT2LMHeadModel(GPT2LMHeadModel):
    @classmethod
    def from_pretrained_with_pruning(cls, model_name: str, pruning_config: PruningConfig, **kwargs):
        # Load the standard pre-trained model
        model = cls.from_pretrained(model_name, **kwargs)
        model.embedding_gate = HardConcreteGate(1)
        
        # Replace each block in the transformer with our prunable wrapper
        prunable_blocks = nn.ModuleList([
            PrunableGPT2Block(block, model.config, pruning_config)
            for block in model.transformer.h
        ])
        model.transformer.h = prunable_blocks
        
        # NEW: Create layer-level gates if enabled
        if pruning_config.prune_full_layers:
            model.layer_gates = nn.ModuleList([
                HardConcreteGate(1) for _ in range(len(model.transformer.h))
            ])
        else:
            model.layer_gates = None
        
        # Store the config for later use
        model.pruning_config = pruning_config
        print("Model successfully adapted for pruning (Zero Ablation) with block-level gates.")
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Tuple[Tuple[torch.Tensor]], Cache]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        # Note: corrupted_input_ids and corrupted_inputs_embeds are removed or ignored 
        # in this implementation as we are doing Zero Ablation (out = mask * out)
        # on a single stream.
        
        transformer = self.transformer

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        return_legacy_cache = False
        if use_cache:
            if past_key_values is None:
                return_legacy_cache = True
                past_key_values = DynamicCache()
            elif not isinstance(past_key_values, Cache):
                return_legacy_cache = True
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if self.is_gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = transformer.wte(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeds = transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        # --- ZERO ABLATION: Apply embedding gate ---
        gate = self.embedding_gate()
        hidden_states = gate * hidden_states

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
            token_type_embeds = transformer.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = transformer.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)
        
        causal_mask = transformer._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        
        encoder_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        for i, block in enumerate(transformer.h):
            if transformer.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if causal_mask is not None:
                    causal_mask = causal_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            if self.is_gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            hidden_states=inputs[0],
                            past_key_value=inputs[1], cache_position=inputs[2],
                            attention_mask=inputs[3], head_mask=inputs[4],
                            encoder_hidden_states=inputs[5], encoder_attention_mask=inputs[6],
                            use_cache=inputs[7], output_attentions=inputs[8],
                        )

                checkpointed_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    past_key_values, cache_position,
                    causal_mask, head_mask[i],
                    encoder_hidden_states, encoder_attention_mask,
                    use_cache, output_attentions,
                    use_reentrant=False,
                )
                hidden_states, outputs = checkpointed_outputs[0], checkpointed_outputs[1]
            else:
                block_outputs = block(
                    hidden_states,
                    past_key_value=past_key_values, cache_position=cache_position,
                    attention_mask=causal_mask, head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache, output_attentions=output_attentions,
                )
                hidden_states = block_outputs[0]
                outputs = block_outputs

            # --- ZERO ABLATION: Apply layer-level gate if enabled ---
            if self.layer_gates is not None:
                layer_gate = self.layer_gates[i]()
                hidden_states = layer_gate * hidden_states

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
                if self.config.add_cross_attention and len(outputs) > 2:
                    all_cross_attentions = all_cross_attentions + (outputs[2],)

            if transformer.model_parallel:
                for k, v in transformer.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != transformer.last_device:
                        next_device = "cuda:" + str(k + 1)
                        hidden_states = hidden_states.to(next_device)

        hidden_states = transformer.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        past_key_values = past_key_values if use_cache else None
        if return_legacy_cache and past_key_values is not None:
            past_key_values = past_key_values.to_legacy_cache()

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if kwargs.get("labels") is not None:
            labels = kwargs["labels"]
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + (past_key_values, all_hidden_states, all_self_attentions, all_cross_attentions)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        
    def set_pruning_config(self, pruning_config: PruningConfig):
        self.pruning_config = pruning_config
        print("Pruning configuration updated.")
        
    def set_final_circuit_mode(self, enabled: bool):
        """
        Recursively finds all HardConcreteGate modules and sets their final_mode.
        
        Args:
            enabled (bool): If True, gates will output hard 0/1 values. 
                            If False, they return to normal eval/train behavior.
        """
        print(f"\n--- Setting final circuit mode to: {enabled} ---")
        gate_count = 0
        
        # Recursively find all HardConcreteGate modules in the model
        for name, module in self.named_modules():
            if isinstance(module, HardConcreteGate):
                module.final_mode = enabled
                gate_count += 1
                
        print(f"    Updated {gate_count} HardConcreteGate modules.")
        
        # Optionally, you can also print which specific gates were found
        if enabled:
            print("    Gates are now in hard 0/1 mode for final inference.")
        else:
            print("    Gates are back to soft/stochastic mode.")
            
    def gate_group_sizes(self) -> Dict[str, int]:
        sizes = defaultdict(int)

        # Embedding
        if self.pruning_config.prune_embedding and hasattr(self, 'embedding_gate'):
            sizes['embedding'] += self.embedding_gate.num_gates()

        # Full layers (layer gates)
        if getattr(self, 'layer_gates', None) is not None:
            for layer_gate in self.layer_gates:
                # each layer gate is typically a single scalar gate; adjust if vector
                sizes['full_layers'] += layer_gate.num_gates()

        # Blocks + fine-grained
        for block in self.transformer.h:
            # Block-level gates
            if hasattr(block, 'get_sparsity_loss'):  # adapt if your API differs
                # If block.get_sparsity_loss() returns keys like 'attention_blocks', 'mlp_blocks',
                # estimate their sizes from the underlying gate modules:
                if hasattr(block, 'attention_block_gate') and block.attention_block_gate:
                    sizes['attention_blocks'] += block.attention_block_gate.num_gates()
                if hasattr(block, 'mlp_block_gate') and block.mlp_block_gate:
                    sizes['mlp_blocks'] += block.mlp_block_gate.num_gates()

            # Attention heads
            if hasattr(block.attn, 'head_gates') and block.attn.head_gates is not None:
                sizes['attention_heads'] += block.attn.head_gates.num_gates()

            # Attention neurons (your custom)
            if hasattr(block.attn, 'neuron_gates') and block.attn.neuron_gates is not None:
                sizes['attention_neurons'] += block.attn.neuron_gates.num_gates()

            # MLP (hidden/output)
            if hasattr(block.mlp, 'hidden_gates') and block.mlp.hidden_gates is not None:
                sizes['mlp_hidden'] += block.mlp.hidden_gates.num_gates()
            if hasattr(block.mlp, 'output_gates') and block.mlp.output_gates is not None:
                sizes['mlp_output'] += block.mlp.output_gates.num_gates()

        return dict(sizes)
    
    def get_sparsity_loss(self, step: int = 0) -> Dict[str, torch.Tensor]:
        losses, total_loss = {}, torch.tensor(0.0, device=self.device)
        warmup_mult = min(
            1.0,
            step / self.pruning_config.sparsity_warmup_steps
            if self.pruning_config.sparsity_warmup_steps > 0 else 1.0
        )

        # --- build raw expected L0 counts per group (your current code) ---
        if self.pruning_config.prune_embedding and hasattr(self, 'embedding_gate'):
            losses.setdefault('embedding', torch.tensor(0.0, device=self.device))
            losses['embedding'] += self.embedding_gate.get_sparsity_loss()

        if self.layer_gates is not None:
            losses.setdefault('full_layers', torch.tensor(0.0, device=self.device))
            for layer_gate in self.layer_gates:
                losses['full_layers'] += layer_gate.get_sparsity_loss()

        for block in self.transformer.h:
            if hasattr(block, 'get_sparsity_loss'):
                block_losses = block.get_sparsity_loss()
                for key, loss in block_losses.items():
                    losses.setdefault(key, torch.tensor(0.0, device=self.device))
                    losses[key] += loss

            if hasattr(block.attn, 'head_gates') and block.attn.head_gates is not None:
                losses.setdefault('attention_heads', torch.tensor(0.0, device=self.device))
                losses['attention_heads'] += block.attn.head_gates.get_sparsity_loss()

            if hasattr(block.attn, 'neuron_gates') and block.attn.neuron_gates is not None:
                losses.setdefault('attention_neurons', torch.tensor(0.0, device=self.device))
                losses['attention_neurons'] += block.attn.neuron_gates.get_sparsity_loss()

            if hasattr(block.mlp, 'get_sparsity_loss'):
                mlp_losses = block.mlp.get_sparsity_loss()
                for key, loss in mlp_losses.items():
                    losses.setdefault(key, torch.tensor(0.0, device=self.device))
                    losses[key] += loss

        # --- NEW: normalize by group sizes before weighting ---
        sizes = self.gate_group_sizes()  # dict of ints
        normalized = {}
        for k, v in losses.items():
            denom = float(max(1, sizes.get(k, 0)))  # safe scalar
            normalized[k] = v / denom

        # --- apply lambdas on normalized terms ---
        def add(term_key, lam):
            nonlocal total_loss
            if term_key in normalized:
                total_loss = total_loss + lam * warmup_mult * normalized[term_key]

        add('embedding',          self.pruning_config.lambda_embedding)
        add('full_layers',        self.pruning_config.lambda_full_layers)
        add('attention_blocks',   self.pruning_config.lambda_attention_blocks)
        add('mlp_blocks',         self.pruning_config.lambda_mlp_blocks)
        add('attention_heads',    self.pruning_config.lambda_attention_heads)
        add('attention_neurons',  self.pruning_config.lambda_attention_neurons)
        add('mlp_hidden',         self.pruning_config.lambda_mlp_hidden)
        add('mlp_output',         self.pruning_config.lambda_mlp_output)

        # Return both raw and normalized for logging/plots
        out = {}
        for k in losses:
            out[f'{k}_raw_E_L0'] = losses[k].detach()
            out[f'{k}_norm']     = normalized[k].detach()
        out['total_sparsity'] = total_loss
        return out