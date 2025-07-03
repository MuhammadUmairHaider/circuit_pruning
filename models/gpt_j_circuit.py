# coding=utf-8
# Pruning-enabled GPT-J model implementation following the same pattern as GPT-2

import warnings
from typing import Optional, Union, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.gptj.modeling_gptj import (
    GPTJModel,
    GPTJForCausalLM,
    GPTJAttention,
    GPTJMLP,
    GPTJBlock,
    GPTJConfig,
    apply_rotary_pos_emb,
    GPTJ_ATTENTION_CLASSES,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Import the pruning components
from models.l0 import HardConcreteGate

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


class PrunableGPTJAttention(nn.Module):
    def __init__(self, original_attention, gptj_config: GPTJConfig, pruning_config: PruningConfig):
        super().__init__()
        self.original_attention = original_attention
        self.num_heads = gptj_config.num_attention_heads
        self.head_dim = gptj_config.hidden_size // self.num_heads
        
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
        
    def forward(self, clean_states, corrupted_states, **kwargs):
        # Forward through clean attention
        clean_attn_outputs = self.original_attention(clean_states, **kwargs)
        clean_outputs = clean_attn_outputs[0]
        clean_attn_weights_tuple = clean_attn_outputs[1:]

        # Forward through corrupted attention (without caching)
        corrupted_kwargs = kwargs.copy()
        corrupted_kwargs['use_cache'] = False
        if 'layer_past' in corrupted_kwargs:
            corrupted_kwargs['layer_past'] = None
        
        corrupted_attn_outputs = self.original_attention(corrupted_states, **corrupted_kwargs)
        corrupted_outputs = corrupted_attn_outputs[0]
        corrupted_attn_weights_tuple = corrupted_attn_outputs[1:]
        
        # Start with the clean output
        gated_output = clean_outputs
        
        if self.head_gates or self.neuron_gates:
            b, s, d = clean_outputs.shape
            
            # Reshape both clean and corrupted outputs to expose head and head_dim
            clean_reshaped = clean_outputs.view(b, s, self.num_heads, self.head_dim)
            corrupted_reshaped = corrupted_outputs.view(b, s, self.num_heads, self.head_dim)
            
            # Start with the clean reshaped output
            gated_output_reshaped = clean_reshaped

            # --- Apply Level 2: Head Gates ---
            if self.head_gates:
                head_gate = self.head_gates().view(1, 1, self.num_heads, 1)
                gated_output_reshaped = head_gate * gated_output_reshaped + (1 - head_gate) * corrupted_reshaped
            
            ### NEW: Apply Level 3: Neuron Gates ###
            if self.neuron_gates:
                # This gate is applied AFTER the head gate has potentially mixed in corrupted state
                neuron_gate = self.neuron_gates().view(1, 1, self.num_heads, self.head_dim)
                # The neuron gate acts as a final filter on each dimension's output.
                # If a neuron is pruned, its output becomes 0.
                gated_output_reshaped = gated_output_reshaped * neuron_gate
            ### END NEW ###
            
            # Reshape back to the original tensor shape
            gated_output = gated_output_reshaped.view(b, s, d)
        
        # Gate the attention weights tuple for analysis (optional, but good practice)
        gated_attn_weights_tuple = ()
        if self.head_gates and (clean_attn_weights_tuple and corrupted_attn_weights_tuple and
            len(clean_attn_weights_tuple) > 1 and clean_attn_weights_tuple[1] is not None and 
            len(corrupted_attn_weights_tuple) > 1 and corrupted_attn_weights_tuple[1] is not None):
            gate_val = self.head_gates.get_gate_values() # Assuming a method to get values without re-sampling
            clean_weights = clean_attn_weights_tuple[1]
            corrupted_weights = corrupted_attn_weights_tuple[1]
            attn_gate = gate_val.view(1, self.num_heads, 1, 1)
            gated_weights = attn_gate * clean_weights + (1 - attn_gate) * corrupted_weights
            gated_attn_weights_tuple = (clean_attn_weights_tuple[0], gated_weights) + clean_attn_weights_tuple[2:]
        else:
            gated_attn_weights_tuple = clean_attn_weights_tuple

        return (gated_output,) + gated_attn_weights_tuple, corrupted_outputs


class PrunableGPTJMLP(nn.Module):
    def __init__(self, original_mlp, gptj_config: GPTJConfig, pruning_config: PruningConfig):
        super().__init__()
        self.original_mlp = original_mlp
        self.pruning_config = pruning_config
        
        # --- Create gates based on the PruningConfig ---
        self.hidden_gates = None
        if self.pruning_config.prune_mlp_hidden:
            intermediate_size = gptj_config.n_inner if gptj_config.n_inner is not None else 4 * gptj_config.n_embd
            self.hidden_gates = HardConcreteGate(intermediate_size)

        self.output_gates = None
        if self.pruning_config.prune_mlp_output:
            self.output_gates = HardConcreteGate(gptj_config.n_embd)
            
    def forward(self, clean_states, corrupted_states):
        # --- Deconstructed Forward Pass to allow gating at two points ---
        
        # 1. Get hidden activations for both streams
        clean_act = self.original_mlp.act(self.original_mlp.fc_in(clean_states))
        corrupted_act = self.original_mlp.act(self.original_mlp.fc_in(corrupted_states))

        # 2. Apply gates to the HIDDEN layer, if enabled
        gated_act = clean_act
        if self.hidden_gates:
            gate = self.hidden_gates().view(1, 1, -1)
            gated_act = gate * clean_act + (1 - gate) * corrupted_act
        
        # 3. Get final outputs from the second linear layer
        clean_output = self.original_mlp.dropout(self.original_mlp.fc_out(gated_act))
        corrupted_output = self.original_mlp.dropout(self.original_mlp.fc_out(corrupted_act))

        # 4. Apply gates to the OUTPUT layer, if enabled
        gated_output = clean_output
        if self.output_gates:
            gate = self.output_gates().view(1, 1, -1)
            gated_output = gate * clean_output + (1 - gate) * corrupted_output

        return gated_output, corrupted_output

    def get_sparsity_loss(self) -> Dict[str, torch.Tensor]:
        """Calculates the sparsity loss for any gates present in this module."""
        losses = {}
        if self.hidden_gates:
            losses['mlp_hidden'] = self.hidden_gates.get_sparsity_loss()
        if self.output_gates:
            losses['mlp_output'] = self.output_gates.get_sparsity_loss()
        return losses


class PrunableGPTJBlock(nn.Module):
    def __init__(self, original_block, gptj_config: GPTJConfig, pruning_config: PruningConfig):
        super().__init__()
        self.ln_1 = original_block.ln_1
        
        self.attn = PrunableGPTJAttention(original_block.attn, gptj_config, pruning_config)
        self.mlp = PrunableGPTJMLP(original_block.mlp, gptj_config, pruning_config)
        
        self.attention_block_gate = None
        if pruning_config.prune_attention_blocks:
            self.attention_block_gate = HardConcreteGate(1)
            
        self.mlp_block_gate = None
        if pruning_config.prune_mlp_blocks:
            self.mlp_block_gate = HardConcreteGate(1)

    def forward(
        self,
        clean_states: torch.FloatTensor,
        corrupted_states: torch.FloatTensor,
        **kwargs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, tuple]:
        # Store residuals
        residual_clean = clean_states
        residual_corrupted = corrupted_states
        
        # Apply LayerNorm to both states
        hidden_states_clean = self.ln_1(clean_states)
        hidden_states_corrupted = self.ln_1(corrupted_states)
        
        # Pass both through the PrunableAttention wrapper
        attn_outputs, corrupted_attn_output = self.attn(
            hidden_states_clean, hidden_states_corrupted, **kwargs
        )
        attn_output = attn_outputs[0]
        
        # NEW: Apply attention block gate if enabled
        if self.attention_block_gate:
            gate = self.attention_block_gate()
            attn_output = gate * attn_output + (1 - gate) * corrupted_attn_output
        
        # MLP forward pass on normalized hidden states
        feed_forward_hidden_states, corrupted_feed_forward = self.mlp(
            hidden_states_clean, hidden_states_corrupted
        )
        
        # NEW: Apply MLP block gate if enabled
        if self.mlp_block_gate:
            gate = self.mlp_block_gate()
            feed_forward_hidden_states = gate * feed_forward_hidden_states + (1 - gate) * corrupted_feed_forward
        
        # GPT-J combines attention and MLP outputs with residual
        hidden_states_clean = attn_output + feed_forward_hidden_states + residual_clean
        hidden_states_corrupted = corrupted_attn_output + corrupted_feed_forward + residual_corrupted
        
        return hidden_states_clean, hidden_states_corrupted, attn_outputs

    def get_sparsity_loss(self) -> Dict[str, torch.Tensor]:
        """Get sparsity losses for block-level gates."""
        losses = {}
        if self.attention_block_gate:
            losses['attention_blocks'] = self.attention_block_gate.get_sparsity_loss()
        if self.mlp_block_gate:
            losses['mlp_blocks'] = self.mlp_block_gate.get_sparsity_loss()
        return losses


class PrunableGPTJForCausalLM(GPTJForCausalLM):
    @classmethod
    def from_pretrained_with_pruning(cls, model_name: str, pruning_config: PruningConfig, **kwargs):
        # Load the standard pre-trained model
        model = cls.from_pretrained(model_name, **kwargs)
        model.embedding_gate = HardConcreteGate(1)
        
        # Replace each block in the transformer with our prunable wrapper
        prunable_blocks = nn.ModuleList([
            PrunableGPTJBlock(block, model.config, pruning_config)
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
        print("Model successfully adapted for pruning with block-level gates.")
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        corrupted_input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor]]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        corrupted_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        is_pruning_run = corrupted_input_ids is not None or corrupted_inputs_embeds is not None
        if not is_pruning_run:
            # Fallback to original model's logic
            return super().forward(
                input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
                token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,
                inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict, cache_position=cache_position
            )

        # All the logic below is for the dual-stream pruning run
        transformer = self.transformer

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is not None and inputs_embeds is not None) or \
           (corrupted_input_ids is not None and corrupted_inputs_embeds is not None):
            raise ValueError("You cannot specify both `input_ids` and `inputs_embeds` for the same stream.")

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either `input_ids` or `inputs_embeds` for the clean stream.")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        seq_length = input_shape[1]
        if cache_position is None:
            past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = transformer.wte(input_ids)
        if corrupted_inputs_embeds is None:
            corrupted_inputs_embeds = transformer.wte(corrupted_input_ids)

        hidden_states_clean = inputs_embeds
        hidden_states_corrupted = corrupted_inputs_embeds
        
        # Apply embedding gate
        gate = self.embedding_gate()
        hidden_states_clean = gate * hidden_states_clean + (1 - gate) * hidden_states_corrupted

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, seq_length)
            token_type_embeds = transformer.wte(token_type_ids)
            hidden_states_clean = hidden_states_clean + token_type_embeds
            hidden_states_corrupted = hidden_states_corrupted + token_type_embeds

        hidden_states_clean = transformer.drop(hidden_states_clean)
        hidden_states_corrupted = transformer.drop(hidden_states_corrupted)

        output_shape = (-1, seq_length, hidden_states_clean.size(-1))

        causal_mask = transformer._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        next_decoder_cache = None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, block in enumerate(transformer.h):
            # Model parallel handling
            if transformer.model_parallel:
                torch.cuda.set_device(hidden_states_clean.device)
                if past_key_values is not None:
                    past_key_values.key_cache = past_key_values.key_cache.to(hidden_states_clean.device)
                    past_key_values.value_cache = past_key_values.value_cache.to(hidden_states_clean.device)
                if causal_mask is not None:
                    causal_mask = causal_mask.to(hidden_states_clean.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states_clean.device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states_clean,)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            clean_states=inputs[0], corrupted_states=inputs[1],
                            layer_past=inputs[2], attention_mask=inputs[3],
                            position_ids=inputs[4], head_mask=inputs[5],
                            use_cache=inputs[6], output_attentions=inputs[7],
                            cache_position=inputs[8],
                        )
                    return custom_forward

                checkpointed_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states_clean, hidden_states_corrupted,
                    past_key_values, causal_mask,
                    position_ids, head_mask[i],
                    use_cache, output_attentions,
                    cache_position,
                    use_reentrant=False,
                )
                hidden_states_clean, hidden_states_corrupted, outputs = checkpointed_outputs
            else:
                hidden_states_clean, hidden_states_corrupted, outputs = block(
                    hidden_states_clean, hidden_states_corrupted,
                    layer_past=past_key_values, attention_mask=causal_mask,
                    position_ids=position_ids, head_mask=head_mask[i],
                    use_cache=use_cache, output_attentions=output_attentions,
                    cache_position=cache_position,
                )

            # NEW: Apply layer-level gate if enabled
            if self.layer_gates is not None:
                layer_gate = self.layer_gates[i]()
                # Mix the clean and corrupted states based on the layer gate
                hidden_states_clean = layer_gate * hidden_states_clean + (1 - layer_gate) * hidden_states_corrupted

            if use_cache is True:
                next_decoder_cache = outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

            # Model Parallel handling
            if transformer.model_parallel:
                for k, v in transformer.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != transformer.last_device:
                        next_device = "cuda:" + str(k + 1)
                        hidden_states_clean = hidden_states_clean.to(next_device)
                        hidden_states_corrupted = hidden_states_corrupted.to(next_device)

        hidden_states_clean = transformer.ln_f(hidden_states_clean)
        hidden_states_clean = hidden_states_clean.view(output_shape)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states_clean,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states_clean = hidden_states_clean.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states_clean).to(torch.float32)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Flatten the tokens
            loss = self.loss_function(
                lm_logits,
                labels,
                vocab_size=self.config.vocab_size,
            )
            loss = loss.to(hidden_states_clean.dtype)

        if not return_dict:
            output = (lm_logits,) + (next_cache, all_hidden_states, all_self_attentions)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

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

    def get_sparsity_loss(self, step: int = 0) -> Dict[str, torch.Tensor]:
        losses, total_loss = {}, torch.tensor(0.0, device=self.device)
        warmup_mult = min(1.0, step / self.pruning_config.sparsity_warmup_steps if self.pruning_config.sparsity_warmup_steps > 0 else 1.0)
        
        # Embedding loss
        if self.pruning_config.prune_embedding and hasattr(self, 'embedding_gate'):
            losses.setdefault('embedding', torch.tensor(0.0, device=self.device))
            losses['embedding'] += self.embedding_gate.get_sparsity_loss()
        
        # Layer-level losses
        if self.layer_gates is not None:
            losses.setdefault('full_layers', torch.tensor(0.0, device=self.device))
            for layer_gate in self.layer_gates:
                losses['full_layers'] += layer_gate.get_sparsity_loss()
        
        # Block and fine-grained losses
        for block in self.transformer.h:
            # Block-level losses
            if hasattr(block, 'get_sparsity_loss'):
                block_losses = block.get_sparsity_loss()
                for key, loss in block_losses.items():
                    losses.setdefault(key, torch.tensor(0.0, device=self.device))
                    losses[key] += loss
            
            # Fine-grained attention losses (heads)
            if hasattr(block.attn, 'head_gates') and block.attn.head_gates is not None:
                losses.setdefault('attention_heads', torch.tensor(0.0, device=self.device))
                losses['attention_heads'] += block.attn.head_gates.get_sparsity_loss()
                
            ### ADDED FOR ATTENTION NEURONS ###
            # Fine-grained attention losses (neurons)
            if hasattr(block.attn, 'neuron_gates') and block.attn.neuron_gates is not None:
                losses.setdefault('attention_neurons', torch.tensor(0.0, device=self.device))
                losses['attention_neurons'] += block.attn.neuron_gates.get_sparsity_loss()
            ### END ADDITION ###
                
            # Fine-grained MLP losses
            if hasattr(block.mlp, 'get_sparsity_loss'):
                mlp_losses = block.mlp.get_sparsity_loss()
                for key, loss in mlp_losses.items():
                    losses.setdefault(key, torch.tensor(0.0, device=self.device))
                    losses[key] += loss

        # Apply lambdas for each component
        if 'embedding' in losses:
            total_loss += self.pruning_config.lambda_embedding * warmup_mult * losses['embedding']
        if 'full_layers' in losses:
            total_loss += self.pruning_config.lambda_full_layers * warmup_mult * losses['full_layers']
        if 'attention_blocks' in losses:
            total_loss += self.pruning_config.lambda_attention_blocks * warmup_mult * losses['attention_blocks']
        if 'mlp_blocks' in losses:
            total_loss += self.pruning_config.lambda_mlp_blocks * warmup_mult * losses['mlp_blocks']
        if 'attention_heads' in losses: 
            total_loss += self.pruning_config.lambda_attention_heads * warmup_mult * losses['attention_heads']
            
        ### ADDED FOR ATTENTION NEURONS ###
        if 'attention_neurons' in losses:
            total_loss += self.pruning_config.lambda_attention_neurons * warmup_mult * losses['attention_neurons']
        ### END ADDITION ###
            
        if 'mlp_hidden' in losses: 
            total_loss += self.pruning_config.lambda_mlp_hidden * warmup_mult * losses['mlp_hidden']
        if 'mlp_output' in losses:
            total_loss += self.pruning_config.lambda_mlp_output * warmup_mult * losses['mlp_output']

        losses['total_sparsity'] = total_loss
        return losses

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask