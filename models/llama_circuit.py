"""
Prunable Llama Model for Circuit Discovery.

Mirrors the dual-stream (clean/corrupted) architecture of gpt2_circuit.py,
adapted for Llama 3 family models (Llama 3, 3.1, 3.2).

Key architectural differences from GPT-2:
- RoPE (Rotary Positional Embeddings) instead of learned positional embeddings
- GQA (Grouped Query Attention): num_key_value_heads < num_attention_heads
- SwiGLU MLP: gate_proj * silu(up_proj) -> down_proj  (instead of fc -> gelu -> proj)
- RMSNorm instead of LayerNorm
- Separate q_proj, k_proj, v_proj (instead of single c_attn)
"""

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import (
    LlamaForCausalLM,
    LlamaConfig,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

from models.l0 import HardConcreteGate


# ==============================================================================
# PRUNING CONFIGURATION (same structure as GPT-2, works for any model)
# ==============================================================================

PRUNING_FACTOR = 5

@dataclass
class PruningConfig:
    init_value: float = 1.0
    sparsity_warmup_steps: int = 1000

    depth_penalty_scaling: float = 0.1

    # Attention Head Pruning
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 0.01 * PRUNING_FACTOR

    # MLP neuron pruning
    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 0.005 * PRUNING_FACTOR
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 0.005 * PRUNING_FACTOR

    # EMBEDDING GATE COMPLETELY REMOVED - always uses clean embeddings (gate = 1.0)
    # No prune_embedding or lambda_embedding parameters

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
# PRUNABLE LLAMA ATTENTION
# ==============================================================================

class PrunableLlamaAttention(nn.Module):
    """
    Wraps a LlamaAttention module with pruning gates.
    
    Gates:
        - head_gates: one gate per query head (num_attention_heads)
        - neuron_gates: one gate per neuron across all heads (num_attention_heads * head_dim)
    
    Handles GQA: Llama uses grouped query attention where multiple query heads
    share the same key/value head. Gates operate at the query-head level.
    """

    def __init__(self, original_attention: LlamaAttention, llama_config: LlamaConfig, 
                 pruning_config: PruningConfig):
        super().__init__()
        self.original_attention = original_attention
        self.config = llama_config

        self.num_heads = llama_config.num_attention_heads
        self.num_kv_heads = llama_config.num_key_value_heads
        self.head_dim = llama_config.hidden_size // self.num_heads
        self.hidden_size = llama_config.hidden_size
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # Head-level gates
        if pruning_config.prune_attention_heads:
            self.head_gates = HardConcreteGate(self.num_heads)
        else:
            self.head_gates = None

        # Neuron-level gates (within heads)
        if pruning_config.prune_attention_neurons:
            self.neuron_gates = HardConcreteGate(self.num_heads * self.head_dim)
        else:
            self.neuron_gates = None

    def _forward_attention_pre_oproj(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run the original attention module's forward but intercept output BEFORE o_proj.
        
        This guarantees numerical equivalence with the base model by using the exact
        same computation path (SDPA, flash attention, etc.), while still allowing us
        to gate at the per-head level before the output projection.
        
        Returns attention output in shape [batch, seq_len, num_heads * head_dim] and weights.
        """
        attn = self.original_attention
        
        # Temporarily replace o_proj with Identity to get pre-projection output
        saved_o_proj = attn.o_proj
        attn.o_proj = nn.Identity()
        
        try:
            attn_output, attn_weights = attn(hidden_states, **kwargs)
        finally:
            # Always restore o_proj, even if forward fails
            attn.o_proj = saved_o_proj
        
        return attn_output, attn_weights

    def forward(
        self,
        clean_states: torch.Tensor,
        corrupted_states: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass supporting both standard (single-stream) and dual-stream modes.
        
        When corrupted_states is None (standard HuggingFace forward):
            Returns: (output, attn_weights)
        When corrupted_states is provided (dual-stream pruning):
            Returns: (gated_output, attn_weights), corrupted_output
        """
        # --- Standard single-stream mode (called by HuggingFace's LlamaModel.forward) ---
        if corrupted_states is None:
            # Delegate entirely to the original attention (including o_proj)
            attn_out, weights = self.original_attention(clean_states, **kwargs)
            return attn_out, weights

        # --- Dual-stream pruning mode ---
        # Get pre-o_proj outputs using the original module's computation
        clean_attn_out, clean_weights = self._forward_attention_pre_oproj(clean_states, **kwargs)
        
        corrupted_kwargs = dict(kwargs)
        corrupted_kwargs['past_key_value'] = None
        corr_attn_out, _ = self._forward_attention_pre_oproj(corrupted_states, **corrupted_kwargs)

        # Reshape to [batch, seq_len, num_heads, head_dim] for per-head gating
        bsz, seq_len = clean_attn_out.shape[:2]
        clean_attn_out = clean_attn_out.view(bsz, seq_len, self.num_heads, self.head_dim)
        corr_attn_out = corr_attn_out.view(bsz, seq_len, self.num_heads, self.head_dim)

        # Apply gates
        gated_output = clean_attn_out

        if self.head_gates is not None or self.neuron_gates is not None:
            if self.head_gates is not None:
                head_gate = self.head_gates().to(clean_attn_out.dtype).view(1, 1, self.num_heads, 1)
                gated_output = head_gate * gated_output + (1 - head_gate) * corr_attn_out

            if self.neuron_gates is not None:
                neuron_gate = self.neuron_gates().to(gated_output.dtype).view(1, 1, self.num_heads, self.head_dim)
                # Mix with corrupted stream (consistent with all other gate types)
                gated_output = neuron_gate * gated_output + (1 - neuron_gate) * corr_attn_out

        # Flatten heads: [batch, seq_len, hidden_size]
        gated_output = gated_output.reshape(bsz, seq_len, -1).contiguous()
        corr_attn_out = corr_attn_out.reshape(bsz, seq_len, -1).contiguous()

        # Apply output projection
        gated_output = self.original_attention.o_proj(gated_output)
        corr_output = self.original_attention.o_proj(corr_attn_out)

        return (gated_output, clean_weights), corr_output


# ==============================================================================
# PRUNABLE LLAMA MLP
# ==============================================================================

class PrunableLlamaMLP(nn.Module):
    """
    Wraps a LlamaMLP module with pruning gates.
    
    Llama MLP uses SwiGLU activation:
        output = down_proj(silu(gate_proj(x)) * up_proj(x))
    
    Gates:
        - hidden_gates: one gate per intermediate neuron (intermediate_size)
        - output_gates: one gate per output neuron (hidden_size)
    """

    def __init__(self, original_mlp: LlamaMLP, llama_config: LlamaConfig, 
                 pruning_config: PruningConfig):
        super().__init__()
        self.original_mlp = original_mlp
        self.pruning_config = pruning_config

        # Hidden gates (intermediate neurons)
        self.hidden_gates = None
        if pruning_config.prune_mlp_hidden:
            self.hidden_gates = HardConcreteGate(llama_config.intermediate_size)

        # Output gates (residual stream neurons)
        self.output_gates = None
        if pruning_config.prune_mlp_output:
            self.output_gates = HardConcreteGate(llama_config.hidden_size)

    def forward(
        self,
        clean_states: torch.Tensor,
        corrupted_states: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass supporting both standard and dual-stream modes.
        
        SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        """
        mlp = self.original_mlp

        # --- Standard single-stream mode ---
        if corrupted_states is None:
            act = mlp.act_fn(mlp.gate_proj(clean_states)) * mlp.up_proj(clean_states)
            return mlp.down_proj(act)

        # --- Dual-stream pruning mode ---
        clean_act = mlp.act_fn(mlp.gate_proj(clean_states)) * mlp.up_proj(clean_states)
        corrupted_act = mlp.act_fn(mlp.gate_proj(corrupted_states)) * mlp.up_proj(corrupted_states)

        gated_act = clean_act
        if self.hidden_gates is not None:
            gate = self.hidden_gates().to(clean_act.dtype).view(1, 1, -1)
            gated_act = gate * clean_act + (1 - gate) * corrupted_act

        clean_output = mlp.down_proj(gated_act)
        corrupted_output = mlp.down_proj(corrupted_act)

        gated_output = clean_output
        if self.output_gates is not None:
            gate = self.output_gates().to(clean_output.dtype).view(1, 1, -1)
            gated_output = gate * clean_output + (1 - gate) * corrupted_output

        return gated_output, corrupted_output

    def get_sparsity_loss(self) -> Dict[str, torch.Tensor]:
        losses = {}
        if self.hidden_gates:
            losses['mlp_hidden'] = self.hidden_gates.get_sparsity_loss()
        if self.output_gates:
            losses['mlp_output'] = self.output_gates.get_sparsity_loss()
        return losses


# ==============================================================================
# PRUNABLE LLAMA DECODER LAYER
# ==============================================================================

class PrunableLlamaDecoderLayer(nn.Module):
    """
    Wraps a LlamaDecoderLayer with prunable sub-modules and block-level gates.
    
    Architecture (Pre-RMSNorm):
        residual = x
        x = input_layernorm(x)
        x = self_attn(x)      <-- PrunableLlamaAttention
        x = residual + x
        
        residual = x
        x = post_attention_layernorm(x)
        x = mlp(x)            <-- PrunableLlamaMLP
        x = residual + x
    """

    def __init__(self, original_layer: LlamaDecoderLayer, llama_config: LlamaConfig,
                 pruning_config: PruningConfig):
        super().__init__()
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm

        self.attn = PrunableLlamaAttention(original_layer.self_attn, llama_config, pruning_config)
        self.mlp = PrunableLlamaMLP(original_layer.mlp, llama_config, pruning_config)

        # Block-level gates
        self.attention_block_gate = None
        if pruning_config.prune_attention_blocks:
            self.attention_block_gate = HardConcreteGate(1)

        self.mlp_block_gate = None
        if pruning_config.prune_mlp_blocks:
            self.mlp_block_gate = HardConcreteGate(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        corrupted_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Forward pass supporting both standard (single-stream) and dual-stream modes.
        
        When corrupted_states is None (standard HuggingFace forward):
            Returns: tuple matching LlamaDecoderLayer output format
        When corrupted_states is provided (dual-stream pruning):
            Returns: (final_clean, final_corrupted, attn_outputs)
        """
        # Build kwargs dict for attention
        attn_kwargs = dict(
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            cache_position=cache_position,
            output_attentions=output_attentions,
        )

        # --- Standard single-stream mode (HuggingFace compatibility) ---
        if corrupted_states is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            attn_out, weights = self.attn(hidden_states, **attn_kwargs)  # single-stream
            hidden_states = residual + attn_out

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + self.mlp(hidden_states)  # single-stream

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (weights,)
            return outputs

        # --- Dual-stream pruning mode ---
        clean_states = hidden_states

        ln_clean = self.input_layernorm(clean_states)
        ln_corrupted = self.input_layernorm(corrupted_states)

        attn_outputs, corrupted_attn_output = self.attn(
            ln_clean, ln_corrupted, **attn_kwargs
        )
        attn_output = attn_outputs[0]

        if self.attention_block_gate is not None:
            gate = self.attention_block_gate().to(attn_output.dtype)
            attn_output = gate * attn_output + (1 - gate) * corrupted_attn_output

        clean_states = clean_states + attn_output
        corrupted_states = corrupted_states + corrupted_attn_output

        ln_clean_after_attn = self.post_attention_layernorm(clean_states)
        ln_corrupted_after_attn = self.post_attention_layernorm(corrupted_states)

        mlp_output, corrupted_mlp_output = self.mlp(
            ln_clean_after_attn, ln_corrupted_after_attn
        )

        if self.mlp_block_gate is not None:
            gate = self.mlp_block_gate().to(mlp_output.dtype)
            mlp_output = gate * mlp_output + (1 - gate) * corrupted_mlp_output

        final_clean = clean_states + mlp_output
        final_corrupted = corrupted_states + corrupted_mlp_output

        return final_clean, final_corrupted, attn_outputs

    def get_sparsity_loss(self) -> Dict[str, torch.Tensor]:
        losses = {}
        if self.attention_block_gate:
            losses['attention_blocks'] = self.attention_block_gate.get_sparsity_loss()
        if self.mlp_block_gate:
            losses['mlp_blocks'] = self.mlp_block_gate.get_sparsity_loss()
        return losses


# ==============================================================================
# PRUNABLE LLAMA FOR CAUSAL LM
# ==============================================================================

class PrunableLlamaForCausalLM(LlamaForCausalLM):
    """
    Extends LlamaForCausalLM with circuit discovery pruning capabilities.
    
    Supports dual-stream forward pass (clean + corrupted inputs) for
    discovering minimal circuits via differentiable pruning.
    """

    @classmethod
    def from_pretrained_with_pruning(cls, model_name: str, pruning_config: PruningConfig, **kwargs):
        """Load a pretrained Llama model and wrap it with pruning gates."""
        model = cls.from_pretrained(model_name, **kwargs)

        # NO EMBEDDING GATE - completely removed to avoid training interference
        # Always uses clean embeddings (equivalent to gate = 1.0)

        # Replace each decoder layer with our prunable wrapper
        prunable_layers = nn.ModuleList([
            PrunableLlamaDecoderLayer(layer, model.config, pruning_config)
            for layer in model.model.layers
        ])
        model.model.layers = prunable_layers

        # Layer-level gates
        if pruning_config.prune_full_layers:
            model.layer_gates = nn.ModuleList([
                HardConcreteGate(1) for _ in range(len(model.model.layers))
            ])
        else:
            model.layer_gates = None

        model.pruning_config = pruning_config
        print("Llama model successfully adapted for pruning with block-level gates.")
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        corrupted_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        corrupted_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Forward pass supporting both standard inference and dual-stream pruning.
        
        When corrupted_input_ids is None, falls back to standard LlamaForCausalLM forward.
        When provided, runs the dual-stream circuit discovery forward pass.
        """
        is_pruning_run = corrupted_input_ids is not None or corrupted_inputs_embeds is not None

        if not is_pruning_run:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )

        # =====================================================================
        # DUAL-STREAM PRUNING FORWARD PASS
        # =====================================================================
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        if corrupted_inputs_embeds is None:
            corrupted_inputs_embeds = self.model.embed_tokens(corrupted_input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]
        device = inputs_embeds.device

        # Detach corrupted stream — only gates need gradients, not the corrupted path
        corrupted_inputs_embeds = corrupted_inputs_embeds.detach()

        # NO EMBEDDING GATE - always use clean embeddings (gate = 1.0)
        hidden_states_clean = inputs_embeds
        hidden_states_corrupted = corrupted_inputs_embeds.clone()

        # Handle cache
        return_legacy_cache = False
        if use_cache:
            if past_key_values is None:
                return_legacy_cache = True
                past_key_values = DynamicCache()

        # Compute position IDs and cache position
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length, device=device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create causal mask
        # Note: Using manual causal mask creation for compatibility across transformers versions
        if hasattr(self.model, '_update_causal_mask'):
            causal_mask = self.model._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )
        else:
            # Manual causal mask creation for older transformers versions
            if attention_mask is None:
                causal_mask = None
            else:
                # Convert 2D attention mask to 4D causal mask
                # Shape: (batch_size, 1, seq_length, seq_length)
                if attention_mask.dim() == 2:
                    # Create causal mask
                    causal_mask = torch.triu(
                        torch.ones(seq_length, seq_length, dtype=torch.bool, device=device),
                        diagonal=1
                    )
                    # Invert it (1 where we attend, 0 where we mask)
                    causal_mask = ~causal_mask
                    # Expand to batch
                    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_length, seq_length)
                    # Apply the attention_mask (typically all 1s for padding)
                    if attention_mask is not None:
                        # Expand attention_mask to match causal_mask shape
                        expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, seq_length, seq_length)
                        causal_mask = causal_mask & expanded_mask.bool()
                    # Convert to float mask for scaled_dot_product_attention
                    # 0.0 for positions we attend to, -inf for masked positions
                    causal_mask = torch.where(causal_mask, 0.0, float('-inf'))
                    causal_mask = causal_mask.to(inputs_embeds.dtype)
                else:
                    causal_mask = attention_mask

        # Compute rotary embeddings once
        position_embeddings = self.model.rotary_emb(hidden_states_clean, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Forward through each layer
        for i, layer in enumerate(self.model.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states_clean,)

            hidden_states_clean, hidden_states_corrupted, attn_outputs = layer(
                hidden_states_clean,
                hidden_states_corrupted,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_key_values,
                cache_position=cache_position,
                output_attentions=output_attentions,
            )

            # Apply layer-level gate
            if self.layer_gates is not None:
                layer_gate = self.layer_gates[i]().to(hidden_states_clean.dtype)
                hidden_states_clean = layer_gate * hidden_states_clean + (1 - layer_gate) * hidden_states_corrupted

            if output_attentions:
                all_self_attns = all_self_attns + (attn_outputs[1],)

        # Final norm
        hidden_states_clean = self.model.norm(hidden_states_clean)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states_clean,)

        # LM head
        logits = self.lm_head(hidden_states_clean)

        # Compute loss if labels provided
        loss = None
        if kwargs.get("labels") is not None:
            labels = kwargs["labels"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + (past_key_values, all_hidden_states, all_self_attns)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # ------------------------------------------------------------------
    # PRUNING UTILITIES
    # ------------------------------------------------------------------

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

        # NO EMBEDDING GATE - completely removed

        if getattr(self, 'layer_gates', None) is not None:
            for layer_gate in self.layer_gates:
                sizes['full_layers'] += layer_gate.num_gates()

        for block in self.model.layers:
            if hasattr(block, 'attention_block_gate') and block.attention_block_gate is not None:
                sizes['attention_blocks'] += block.attention_block_gate.num_gates()
            if hasattr(block, 'mlp_block_gate') and block.mlp_block_gate is not None:
                sizes['mlp_blocks'] += block.mlp_block_gate.num_gates()

            if hasattr(block.attn, 'head_gates') and block.attn.head_gates is not None:
                sizes['attention_heads'] += block.attn.head_gates.num_gates()
            if hasattr(block.attn, 'neuron_gates') and block.attn.neuron_gates is not None:
                sizes['attention_neurons'] += block.attn.neuron_gates.num_gates()

            if hasattr(block.mlp, 'hidden_gates') and block.mlp.hidden_gates is not None:
                sizes['mlp_hidden'] += block.mlp.hidden_gates.num_gates()
            if hasattr(block.mlp, 'output_gates') and block.mlp.output_gates is not None:
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

        def add_weighted(term_key, raw_loss, lam, layer_idx=None):
            nonlocal total_loss
            depth_mult = 1.0
            if layer_idx is not None and self.pruning_config.depth_penalty_scaling > 0:
                n_layers = len(self.model.layers)
                fraction = (n_layers - 1 - layer_idx) / max(1, n_layers - 1)
                depth_mult = 1.0 + self.pruning_config.depth_penalty_scaling * fraction

            term_loss = lam * warmup_mult * depth_mult * raw_loss
            total_loss = total_loss + term_loss

            if term_key not in losses:
                losses[term_key] = 0.0
            losses[term_key] += term_loss

        # Global components
        # NO EMBEDDING GATE - completely removed

        if getattr(self, 'layer_gates', None) is not None:
            for i, gate in enumerate(self.layer_gates):
                add_weighted('full_layers', gate.get_sparsity_loss(),
                             self.pruning_config.lambda_full_layers, layer_idx=i)

        # Per-layer components
        for i, block in enumerate(self.model.layers):
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

        losses['total_sparsity'] = total_loss
        return losses
