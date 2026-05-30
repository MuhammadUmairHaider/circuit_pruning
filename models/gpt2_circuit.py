"""
Prunable GPT-2 Model for Circuit Discovery.

Mirrors the dual-stream (clean/corrupted) architecture of llama_circuit.py,
adapted for GPT-2 models. Compatible with transformers >= 5.0.

Key differences from Llama:
- Learned positional embeddings (wpe) instead of RoPE
- Single c_attn projection for Q/K/V (instead of separate projections)
- c_proj + resid_dropout inside the attention module
- LayerNorm instead of RMSNorm
- Retains embedding gate (Llama version removed it)
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention, GPT2MLP, GPT2Block, GPT2Model,
)
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.cache_utils import Cache, DynamicCache

from models.l0 import HardConcreteGate


# ==============================================================================
# PRUNING CONFIGURATION
# ==============================================================================

PRUNING_FACTOR = 5


@dataclass
class PruningConfig:
    init_value: float = 1.0
    sparsity_warmup_steps: int = 1000
    depth_penalty_scaling: float = 0.1

    # Attention head pruning
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 0.01 * PRUNING_FACTOR

    # MLP neuron pruning
    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 0.005 * PRUNING_FACTOR
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 0.005 * PRUNING_FACTOR

    # Embedding gate (GPT-2 retains this, unlike Llama)
    prune_embedding: bool = True
    lambda_embedding: float = 1 * PRUNING_FACTOR

    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 0.002 * PRUNING_FACTOR

    # Block-level pruning
    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 0.02 * PRUNING_FACTOR

    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 0.02 * PRUNING_FACTOR

    prune_full_layers: bool = True
    lambda_full_layers: float = 0.05 * PRUNING_FACTOR


# ==============================================================================
# PRUNABLE GPT-2 ATTENTION
# ==============================================================================

class PrunableGPT2Attention(nn.Module):
    """
    Wraps a GPT2Attention with pruning gates.

    In single-stream mode: delegates entirely to original_attention.
    In dual-stream mode: intercepts pre-c_proj output, applies head/neuron gates,
    then applies c_proj + resid_dropout.
    """

    def __init__(self, original_attention: GPT2Attention, gpt_config: GPT2Config,
                 pruning_config: PruningConfig):
        super().__init__()
        self.original_attention = original_attention
        self.num_heads = gpt_config.num_attention_heads
        self.head_dim = gpt_config.hidden_size // self.num_heads

        self.head_gates = None
        if pruning_config.prune_attention_heads:
            self.head_gates = HardConcreteGate(self.num_heads)

        self.neuron_gates = None
        if pruning_config.prune_attention_neurons:
            self.neuron_gates = HardConcreteGate(self.num_heads * self.head_dim)

    def _forward_pre_proj(self, hidden_states: torch.Tensor, **kwargs):
        """
        Run GPT2Attention.forward but intercept output before c_proj + resid_dropout.
        Temporarily replaces both with Identity, restores after.

        Returns:
            pre_proj: [batch, seq_len, hidden_size] merged-head representation
            attn_weights: attention weights (or None if not computed)
        """
        attn = self.original_attention
        saved_c_proj = attn.c_proj
        saved_dropout = attn.resid_dropout
        attn.c_proj = nn.Identity()
        attn.resid_dropout = nn.Identity()
        try:
            pre_proj, attn_weights = attn(hidden_states, **kwargs)
        finally:
            attn.c_proj = saved_c_proj
            attn.resid_dropout = saved_dropout
        return pre_proj, attn_weights

    def forward(
        self,
        clean_states: torch.Tensor,
        corrupted_states: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Single-stream: returns (attn_output, attn_weights) like GPT2Attention.
        Dual-stream: returns ((gated_output, attn_weights), corrupted_output).
        """
        # Single-stream: delegate entirely
        if corrupted_states is None:
            return self.original_attention(clean_states, **kwargs)

        # Dual-stream
        clean_pre, clean_weights = self._forward_pre_proj(clean_states, **kwargs)

        # Disable cache for corrupted stream to avoid state contamination
        corrupted_kwargs = dict(kwargs)
        corrupted_kwargs['past_key_values'] = None
        corr_pre, _ = self._forward_pre_proj(corrupted_states, **corrupted_kwargs)

        # Reshape to [batch, seq_len, num_heads, head_dim] for per-head gating
        bsz, seq_len = clean_pre.shape[:2]
        clean_heads = clean_pre.view(bsz, seq_len, self.num_heads, self.head_dim)
        corr_heads = corr_pre.view(bsz, seq_len, self.num_heads, self.head_dim)

        gated = clean_heads
        if self.head_gates is not None:
            g = self.head_gates().to(clean_heads.dtype).view(1, 1, self.num_heads, 1)
            gated = g * gated + (1 - g) * corr_heads
        if self.neuron_gates is not None:
            g = self.neuron_gates().to(gated.dtype).view(1, 1, self.num_heads, self.head_dim)
            gated = g * gated + (1 - g) * corr_heads

        # Flatten heads back to [batch, seq_len, hidden_size]
        gated_flat = gated.reshape(bsz, seq_len, -1).contiguous()
        corr_flat = corr_heads.reshape(bsz, seq_len, -1).contiguous()

        # Apply output projection
        attn = self.original_attention
        gated_out = attn.resid_dropout(attn.c_proj(gated_flat))
        corr_out = attn.resid_dropout(attn.c_proj(corr_flat))

        return (gated_out, clean_weights), corr_out


# ==============================================================================
# PRUNABLE GPT-2 MLP
# ==============================================================================

class PrunableGPT2MLP(nn.Module):
    """
    Wraps a GPT2MLP with pruning gates.

    GPT-2 MLP: c_fc -> act -> c_proj -> dropout
    Gates at intermediate (hidden) and output (residual) dimensions.
    """

    def __init__(self, original_mlp: GPT2MLP, gpt_config: GPT2Config,
                 pruning_config: PruningConfig):
        super().__init__()
        self.original_mlp = original_mlp
        self.pruning_config = pruning_config

        self.hidden_gates = None
        if pruning_config.prune_mlp_hidden:
            intermediate_size = gpt_config.n_inner if gpt_config.n_inner else 4 * gpt_config.hidden_size
            self.hidden_gates = HardConcreteGate(intermediate_size)

        self.output_gates = None
        if pruning_config.prune_mlp_output:
            self.output_gates = HardConcreteGate(gpt_config.hidden_size)

    def forward(
        self,
        clean_states: torch.Tensor,
        corrupted_states: Optional[torch.Tensor] = None,
    ):
        mlp = self.original_mlp

        # Single-stream
        if corrupted_states is None:
            return mlp(clean_states)

        # Dual-stream: c_fc -> act
        clean_act = mlp.act(mlp.c_fc(clean_states))
        corrupted_act = mlp.act(mlp.c_fc(corrupted_states))

        gated_act = clean_act
        if self.hidden_gates is not None:
            g = self.hidden_gates().to(clean_act.dtype).view(1, 1, -1)
            gated_act = g * clean_act + (1 - g) * corrupted_act

        # c_proj -> dropout
        clean_out = mlp.dropout(mlp.c_proj(gated_act))
        corrupted_out = mlp.dropout(mlp.c_proj(corrupted_act))

        gated_out = clean_out
        if self.output_gates is not None:
            g = self.output_gates().to(clean_out.dtype).view(1, 1, -1)
            gated_out = g * clean_out + (1 - g) * corrupted_out

        return gated_out, corrupted_out

    def get_sparsity_loss(self) -> Dict[str, torch.Tensor]:
        losses = {}
        if self.hidden_gates:
            losses['mlp_hidden'] = self.hidden_gates.get_sparsity_loss()
        if self.output_gates:
            losses['mlp_output'] = self.output_gates.get_sparsity_loss()
        return losses


# ==============================================================================
# PRUNABLE GPT-2 BLOCK
# ==============================================================================

class PrunableGPT2Block(nn.Module):
    """
    Wraps a GPT2Block with prunable sub-modules and block-level gates.

    Architecture (Pre-LayerNorm):
        residual = x
        x = ln_1(x)
        x = attn(x)       <- PrunableGPT2Attention
        x = residual + x
        residual = x
        x = ln_2(x)
        x = mlp(x)        <- PrunableGPT2MLP
        x = residual + x

    Single-stream return: hidden_states tensor (compatible with GPT2Model loop).
    Dual-stream return: (final_clean, final_corrupted, attn_outputs).
    """

    def __init__(self, original_block: GPT2Block, gpt_config: GPT2Config,
                 pruning_config: PruningConfig):
        super().__init__()
        self.ln_1 = original_block.ln_1
        self.ln_2 = original_block.ln_2

        self.attn = PrunableGPT2Attention(original_block.attn, gpt_config, pruning_config)
        self.mlp = PrunableGPT2MLP(original_block.mlp, gpt_config, pruning_config)

        self.attention_block_gate = None
        if pruning_config.prune_attention_blocks:
            self.attention_block_gate = HardConcreteGate(1)

        self.mlp_block_gate = None
        if pruning_config.prune_mlp_blocks:
            self.mlp_block_gate = HardConcreteGate(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        corrupted_states: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        attn_kwargs = dict(
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
        )

        # --- Single-stream mode (HuggingFace compatibility) ---
        if corrupted_states is None:
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
            attn_out, _ = self.attn(hidden_states, **attn_kwargs)
            hidden_states = residual + attn_out

            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)
            hidden_states = residual + self.mlp(hidden_states)
            return hidden_states

        # --- Dual-stream pruning mode ---
        clean_states = hidden_states

        ln_clean = self.ln_1(clean_states)
        ln_corrupted = self.ln_1(corrupted_states)

        attn_outputs, corrupted_attn_out = self.attn(ln_clean, ln_corrupted, **attn_kwargs)
        attn_out = attn_outputs[0]

        if self.attention_block_gate is not None:
            g = self.attention_block_gate().to(attn_out.dtype)
            attn_out = g * attn_out + (1 - g) * corrupted_attn_out

        clean_states = clean_states + attn_out
        corrupted_states = corrupted_states + corrupted_attn_out

        ln_clean2 = self.ln_2(clean_states)
        ln_corrupted2 = self.ln_2(corrupted_states)

        mlp_out, corrupted_mlp_out = self.mlp(ln_clean2, ln_corrupted2)

        if self.mlp_block_gate is not None:
            g = self.mlp_block_gate().to(mlp_out.dtype)
            mlp_out = g * mlp_out + (1 - g) * corrupted_mlp_out

        final_clean = clean_states + mlp_out
        final_corrupted = corrupted_states + corrupted_mlp_out

        return final_clean, final_corrupted, attn_outputs

    def get_sparsity_loss(self) -> Dict[str, torch.Tensor]:
        losses = {}
        if self.attention_block_gate:
            losses['attention_blocks'] = self.attention_block_gate.get_sparsity_loss()
        if self.mlp_block_gate:
            losses['mlp_blocks'] = self.mlp_block_gate.get_sparsity_loss()
        return losses


# ==============================================================================
# PRUNABLE GPT-2 LM HEAD MODEL
# ==============================================================================

class PrunableGPT2LMHeadModel(GPT2LMHeadModel):
    """
    Extends GPT2LMHeadModel with circuit discovery pruning capabilities.

    Supports dual-stream forward pass (clean + corrupted inputs) for
    discovering minimal circuits via differentiable pruning.
    """

    @classmethod
    def from_pretrained_with_pruning(cls, model_name: str, pruning_config: PruningConfig, **kwargs):
        """Load a pretrained GPT-2 model and wrap it with pruning gates."""
        model = cls.from_pretrained(model_name, **kwargs)

        # Embedding gate (GPT-2 retains this unlike Llama)
        model.embedding_gate = HardConcreteGate(1)

        # Replace each transformer block with our prunable wrapper
        prunable_blocks = nn.ModuleList([
            PrunableGPT2Block(block, model.config, pruning_config)
            for block in model.transformer.h
        ])
        model.transformer.h = prunable_blocks

        # Layer-level gates
        if pruning_config.prune_full_layers:
            model.layer_gates = nn.ModuleList([
                HardConcreteGate(1) for _ in range(len(model.transformer.h))
            ])
        else:
            model.layer_gates = None

        model.pruning_config = pruning_config
        print("GPT-2 model successfully adapted for pruning with block-level gates.")
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        corrupted_input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        corrupted_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        When corrupted_input_ids is None, falls back to standard GPT2LMHeadModel forward.
        When provided, runs the dual-stream circuit discovery forward pass.
        """
        is_pruning_run = corrupted_input_ids is not None or corrupted_inputs_embeds is not None

        if not is_pruning_run:
            return super().forward(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        # =====================================================================
        # DUAL-STREAM PRUNING FORWARD PASS
        # =====================================================================
        transformer = self.transformer

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # KV cache not meaningful for dual-stream circuit discovery
        use_cache = False

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("Specify either input_ids or inputs_embeds for the clean stream.")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Embeddings
        if inputs_embeds is None:
            inputs_embeds = transformer.wte(input_ids)
        if corrupted_inputs_embeds is None:
            corrupted_inputs_embeds = transformer.wte(corrupted_input_ids)

        # Detach corrupted stream — only gates need gradients through it
        corrupted_inputs_embeds = corrupted_inputs_embeds.detach()

        # Positional embeddings
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=device).unsqueeze(0)

        position_embeds = transformer.wpe(position_ids).to(inputs_embeds.device)
        hidden_states_clean = inputs_embeds + position_embeds
        hidden_states_corrupted = corrupted_inputs_embeds + position_embeds

        # Embedding gate
        g = self.embedding_gate().to(hidden_states_clean.dtype)
        hidden_states_clean = g * hidden_states_clean + (1 - g) * hidden_states_corrupted

        if token_type_ids is not None:
            token_type_embeds = transformer.wte(token_type_ids.view(-1, input_shape[-1]))
            hidden_states_clean = hidden_states_clean + token_type_embeds
            hidden_states_corrupted = hidden_states_corrupted + token_type_embeds

        hidden_states_clean = transformer.drop(hidden_states_clean)
        hidden_states_corrupted = transformer.drop(hidden_states_corrupted)

        output_shape = (-1,) + input_shape[1:] + (hidden_states_clean.size(-1),)

        # Causal mask
        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        cache_position = torch.arange(inputs_embeds.shape[1], device=device)
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, block in enumerate(transformer.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states_clean,)

            hidden_states_clean, hidden_states_corrupted, attn_outputs = block(
                hidden_states_clean,
                cache_position=cache_position,
                attention_mask=causal_mask,
                use_cache=False,
                corrupted_states=hidden_states_corrupted,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )

            # Layer-level gate
            if self.layer_gates is not None:
                layer_gate = self.layer_gates[i]().to(hidden_states_clean.dtype)
                hidden_states_clean = layer_gate * hidden_states_clean + (1 - layer_gate) * hidden_states_corrupted

            if output_attentions and attn_outputs is not None:
                all_self_attentions = all_self_attentions + (attn_outputs[1],)

        hidden_states_clean = transformer.ln_f(hidden_states_clean)
        hidden_states_clean = hidden_states_clean.view(output_shape)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states_clean,)

        lm_logits = self.lm_head(hidden_states_clean)

        loss = None
        if kwargs.get("labels") is not None:
            labels = kwargs["labels"]
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (lm_logits, None, all_hidden_states, all_self_attentions, None)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    # ------------------------------------------------------------------
    # PRUNING UTILITIES
    # ------------------------------------------------------------------

    def set_pruning_config(self, pruning_config: PruningConfig):
        self.pruning_config = pruning_config

    def set_final_circuit_mode(self, enabled: bool):
        """Set all HardConcreteGate modules to final (hard 0/1) mode."""
        print(f"\n--- Setting final circuit mode to: {enabled} ---")
        gate_count = 0
        for name, module in self.named_modules():
            if isinstance(module, HardConcreteGate):
                module.final_mode = enabled
                gate_count += 1
        print(f"    Updated {gate_count} HardConcreteGate modules.")

    def gate_group_sizes(self) -> Dict[str, int]:
        sizes = defaultdict(int)

        if hasattr(self, 'embedding_gate') and self.embedding_gate is not None:
            sizes['embedding'] += self.embedding_gate.num_gates()

        if getattr(self, 'layer_gates', None) is not None:
            for gate in self.layer_gates:
                sizes['full_layers'] += gate.num_gates()

        for block in self.transformer.h:
            if hasattr(block, 'attention_block_gate') and block.attention_block_gate:
                sizes['attention_blocks'] += block.attention_block_gate.num_gates()
            if hasattr(block, 'mlp_block_gate') and block.mlp_block_gate:
                sizes['mlp_blocks'] += block.mlp_block_gate.num_gates()
            if hasattr(block.attn, 'head_gates') and block.attn.head_gates:
                sizes['attention_heads'] += block.attn.head_gates.num_gates()
            if hasattr(block.attn, 'neuron_gates') and block.attn.neuron_gates:
                sizes['attention_neurons'] += block.attn.neuron_gates.num_gates()
            if hasattr(block.mlp, 'hidden_gates') and block.mlp.hidden_gates:
                sizes['mlp_hidden'] += block.mlp.hidden_gates.num_gates()
            if hasattr(block.mlp, 'output_gates') and block.mlp.output_gates:
                sizes['mlp_output'] += block.mlp.output_gates.num_gates()

        return dict(sizes)

    def get_sparsity_loss(self, step: int = 0) -> Dict[str, torch.Tensor]:
        """Compute sparsity loss across all gate groups with depth weighting."""
        losses, total_loss = {}, torch.tensor(0.0, device=self.device)

        warmup_mult = min(
            1.0,
            step / self.pruning_config.sparsity_warmup_steps
            if self.pruning_config.sparsity_warmup_steps > 0 else 1.0
        )

        density_count = 0

        def add_weighted(term_key, raw_loss, lam, layer_idx=None):
            nonlocal total_loss, density_count
            depth_mult = 1.0
            if layer_idx is not None and self.pruning_config.depth_penalty_scaling > 0:
                n_layers = len(self.transformer.h)
                fraction = (n_layers - 1 - layer_idx) / max(1, n_layers - 1)
                depth_mult = 1.0 + self.pruning_config.depth_penalty_scaling * fraction
            term_loss = lam * warmup_mult * depth_mult * raw_loss
            total_loss = total_loss + term_loss
            density_count += 1
            if term_key not in losses:
                losses[term_key] = 0.0
            losses[term_key] += term_loss

        # Embedding
        if self.pruning_config.prune_embedding and hasattr(self, 'embedding_gate') and self.embedding_gate is not None:
            add_weighted('embedding', self.embedding_gate.get_sparsity_loss(),
                         self.pruning_config.lambda_embedding)

        # Full layers
        if getattr(self, 'layer_gates', None) is not None:
            for i, gate in enumerate(self.layer_gates):
                add_weighted('full_layers', gate.get_sparsity_loss(),
                             self.pruning_config.lambda_full_layers, layer_idx=i)

        # Per-block components
        for i, block in enumerate(self.transformer.h):
            if hasattr(block, 'attention_block_gate') and block.attention_block_gate:
                add_weighted('attention_blocks', block.attention_block_gate.get_sparsity_loss(),
                             self.pruning_config.lambda_attention_blocks, layer_idx=i)
            if hasattr(block, 'mlp_block_gate') and block.mlp_block_gate:
                add_weighted('mlp_blocks', block.mlp_block_gate.get_sparsity_loss(),
                             self.pruning_config.lambda_mlp_blocks, layer_idx=i)
            if hasattr(block.attn, 'head_gates') and block.attn.head_gates:
                add_weighted('attention_heads', block.attn.head_gates.get_sparsity_loss(),
                             self.pruning_config.lambda_attention_heads, layer_idx=i)
            if hasattr(block.attn, 'neuron_gates') and block.attn.neuron_gates:
                add_weighted('attention_neurons', block.attn.neuron_gates.get_sparsity_loss(),
                             self.pruning_config.lambda_attention_neurons, layer_idx=i)
            if hasattr(block.mlp, 'hidden_gates') and block.mlp.hidden_gates:
                add_weighted('mlp_hidden', block.mlp.hidden_gates.get_sparsity_loss(),
                             self.pruning_config.lambda_mlp_hidden, layer_idx=i)
            if hasattr(block.mlp, 'output_gates') and block.mlp.output_gates:
                add_weighted('mlp_output', block.mlp.output_gates.get_sparsity_loss(),
                             self.pruning_config.lambda_mlp_output, layer_idx=i)

        losses['total_sparsity'] = total_loss / density_count if density_count > 0 else total_loss
        return losses
