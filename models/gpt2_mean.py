# models/gpt2_circuit_mean_interp.py

import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
from transformers import GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, Cache
from transformers.utils import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union

# Make sure this import path is correct for your project structure
from models.l0 import HardConcreteGate

logger = logging.get_logger(__name__)

# This config defines which components can be pruned and their regularization strengths (lambdas).
# It's used to dynamically build the prunable model.
@dataclass
class PruningConfig:
    init_value: float = 1.0
    sparsity_warmup_steps: int = 1000
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 0.001 * 5
    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 0.00005 * 5
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 0.00005 * 5
    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 0.0002 * 5
    prune_embedding: bool = False
    lambda_embedding: float = 1 * 5
    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 0.05 * 5
    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 0.05 * 5
    prune_full_layers: bool = True
    lambda_full_layers: float = 0.05 * 5


class PrunableAttention(nn.Module):
    """
    Attention module that interpolates between clean output and a pre-computed
    mean activation vector based on gate values.
    """
    def __init__(self, original_attention, gpt_config: GPT2Config, pruning_config: PruningConfig):
        super().__init__()
        self.original_attention = original_attention
        self.num_heads = gpt_config.n_head
        self.head_dim = gpt_config.hidden_size // self.num_heads
        self.head_gates = HardConcreteGate(self.num_heads) if pruning_config.prune_attention_heads else None
        self.neuron_gates = HardConcreteGate(self.num_heads * self.head_dim) if pruning_config.prune_attention_neurons else None

    def forward(self, hidden_states, layer_idx, mean_activations, **kwargs):
        clean_attn_outputs = self.original_attention(hidden_states, **kwargs)
        clean_output = clean_attn_outputs[0]
        gated_output = clean_output

        if self.head_gates or self.neuron_gates:
            b, s, d = clean_output.shape
            mean_key = f'h.{layer_idx}.attn_output'
            mean_act = mean_activations[mean_key].to(clean_output.device)
            expanded_mean = mean_act.unsqueeze(0).unsqueeze(0).expand_as(clean_output)
            output_reshaped = clean_output.view(b, s, self.num_heads, self.head_dim)
            mean_reshaped = expanded_mean.view(b, s, self.num_heads, self.head_dim)
            gated_output_reshaped = output_reshaped

            if self.head_gates:
                head_gate = self.head_gates().view(1, 1, self.num_heads, 1)
                gated_output_reshaped = head_gate * output_reshaped + (1 - head_gate) * mean_reshaped
            if self.neuron_gates:
                neuron_gate = self.neuron_gates().view(1, 1, self.num_heads, self.head_dim)
                gated_output_reshaped = gated_output_reshaped * neuron_gate

            gated_output = gated_output_reshaped.view(b, s, d)

        return (gated_output,) + clean_attn_outputs[1:]


class PrunableMLP(nn.Module):
    """
    MLP module that interpolates its internal and final activations with
    pre-computed mean activation vectors.
    """
    def __init__(self, original_mlp, gpt_config: GPT2Config, pruning_config: PruningConfig):
        super().__init__()
        self.original_mlp = original_mlp
        intermediate_size = gpt_config.n_inner if gpt_config.n_inner is not None else 4 * gpt_config.hidden_size
        self.hidden_gates = HardConcreteGate(intermediate_size) if pruning_config.prune_mlp_hidden else None
        self.output_gates = HardConcreteGate(gpt_config.hidden_size) if pruning_config.prune_mlp_output else None

    def forward(self, hidden_states, layer_idx, mean_activations):
        clean_act = self.original_mlp.act(self.original_mlp.c_fc(hidden_states))
        gated_act = clean_act
        if self.hidden_gates:
            gate = self.hidden_gates().view(1, 1, -1)
            mean_hidden_act = mean_activations[f'h.{layer_idx}.mlp_hidden_act'].to(gated_act.device)
            expanded_mean_act = mean_hidden_act.unsqueeze(0).unsqueeze(0).expand_as(gated_act)
            gated_act = gate * clean_act + (1 - gate) * expanded_mean_act

        final_output = self.original_mlp.dropout(self.original_mlp.c_proj(gated_act))
        gated_output = final_output
        if self.output_gates:
            gate = self.output_gates().view(1, 1, -1)
            mean_output = mean_activations[f'h.{layer_idx}.mlp_output'].to(gated_output.device)
            expanded_mean_output = mean_output.unsqueeze(0).unsqueeze(0).expand_as(gated_output)
            gated_output = gate * final_output + (1 - gate) * expanded_mean_output
        return gated_output


class PrunableGPT2Block(nn.Module):
    """
    A GPT-2 block where attention and MLP sub-modules are prunable via mean activation patching.
    """
    def __init__(self, original_block, gpt_config: GPT2Config, pruning_config: PruningConfig):
        super().__init__()
        self.ln_1 = original_block.ln_1
        self.attn = PrunableAttention(original_block.attn, gpt_config, pruning_config)
        self.ln_2 = original_block.ln_2
        self.mlp = PrunableMLP(original_block.mlp, gpt_config, pruning_config)
        self.attention_block_gate = HardConcreteGate(1) if pruning_config.prune_attention_blocks else None
        self.mlp_block_gate = HardConcreteGate(1) if pruning_config.prune_mlp_blocks else None

    def forward(self, hidden_states, layer_idx, mean_activations, **kwargs):
        # The 'outputs' from the original block are (self_attn_weights, cross_attn_weights)
        # We will return the same structure.
        residual = hidden_states
        hidden_states_ln1 = self.ln_1(hidden_states)
        
        attn_outputs = self.attn(
            hidden_states_ln1, layer_idx=layer_idx, mean_activations=mean_activations, **kwargs
        )
        attn_output = attn_outputs[0]
        other_attn_outputs = attn_outputs[1:] # e.g., (attn_weights, present_key_value)

        if self.attention_block_gate:
            gate = self.attention_block_gate()
            attn_output = gate * attn_output
        
        hidden_states = residual + attn_output
        
        residual = hidden_states
        hidden_states_ln2 = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states_ln2, layer_idx=layer_idx, mean_activations=mean_activations)
        
        if self.mlp_block_gate:
            gate = self.mlp_block_gate()
            mlp_output = gate * mlp_output
            
        hidden_states = residual + mlp_output

        return (hidden_states,) + other_attn_outputs


class PrunableGPT2LMHeadModel(GPT2LMHeadModel):
    @classmethod
    def from_pretrained_with_pruning(cls, model_name: str, pruning_config: PruningConfig, **kwargs):
        model = cls.from_pretrained(model_name, **kwargs)
        model.embedding_gate = HardConcreteGate(1) if pruning_config.prune_embedding else None

        prunable_blocks = nn.ModuleList([
            PrunableGPT2Block(block, model.config, pruning_config)
            for block in model.transformer.h
        ])
        model.transformer.h = prunable_blocks

        if pruning_config.prune_full_layers:
            model.layer_gates = nn.ModuleList([
                HardConcreteGate(1) for _ in range(len(model.transformer.h))
            ])
        else:
            model.layer_gates = None

        model.pruning_config = pruning_config
        model.mean_activations = None
        print("Model successfully adapted for mean activation patching.")
        return model

    def register_mean_activations(self, mean_activations: Dict[str, torch.Tensor]):
        print("Registering mean activations with the circuit model...")
        self.mean_activations = {k: v.to(self.device) for k, v in mean_activations.items()}
        print("Mean activations registered.")

    # REPLACE the old forward method with this one:
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs, # Keep this to handle the streamlined evaluation call
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        # This is now the ONLY forward path. It is used for both training and evaluation.
        # The behavior of `self.training` (set by model.train() or model.eval())
        # automatically controls whether the HardConcreteGates are stochastic or deterministic.
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache:
            logger.warning_once("`use_cache=True` is not supported in this custom forward pass and will be ignored.")
        
        transformer = self.transformer

        # --- Input Processing (adapted from Hugging Face) ---
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        input_shape = input_ids.size() if input_ids is not None else inputs_embeds.size()[:-1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        past_length = past_key_values[0][0].size(-2) if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None: inputs_embeds = transformer.wte(input_ids)
        position_embeds = transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        if token_type_ids is not None:
            hidden_states = hidden_states + transformer.wte(token_type_ids)
            
        # --- Gating and Patching Logic ---
        # The gates' behavior (stochastic vs deterministic) is controlled by `self.training`.
        # The patching logic remains the same, but the gate values will be different.
        if self.embedding_gate and self.mean_activations:
            gate = self.embedding_gate()
            mean_embedding = self.mean_activations['embedding_output'].to(hidden_states.device)
            expanded_mean = mean_embedding.unsqueeze(0).unsqueeze(0).expand_as(hidden_states)
            hidden_states = gate * hidden_states + (1 - gate) * expanded_mean
            
        hidden_states = transformer.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        # Prepare attention mask and head mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        head_mask = transformer.get_head_mask(head_mask, transformer.config.n_layer)

        # --- Transformer Blocks Loop ---
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        for i, block in enumerate(transformer.h):
            if output_hidden_states: all_hidden_states = all_hidden_states + (hidden_states,)
            
            # This call works because it always provides the arguments your custom block needs.
            outputs = block(
                hidden_states,
                layer_idx=i,
                mean_activations=self.mean_activations,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]
            
            if self.layer_gates and self.mean_activations:
                layer_gate = self.layer_gates[i]()
                mean_layer_output = self.mean_activations[f'h.{i}.block_output'].to(hidden_states.device)
                expanded_mean = mean_layer_output.unsqueeze(0).unsqueeze(0).expand_as(hidden_states)
                hidden_states = layer_gate * hidden_states + (1 - layer_gate) * expanded_mean
            
            if output_attentions: all_self_attentions = all_self_attentions + (outputs[1],)

        hidden_states = transformer.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states: all_hidden_states = all_hidden_states + (hidden_states,)
        
        # --- Final Head ---
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits, None, all_hidden_states, all_self_attentions)

        return CausalLMOutputWithCrossAttentions(
            loss=None, # Loss is handled outside
            logits=lm_logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


    def get_sparsity_loss(self, step: int = 0) -> Dict[str, torch.Tensor]:
        """
        Calculates the total L0 regularization loss for all gates in the model.
        """
        losses, total_loss = {}, torch.tensor(0.0, device=self.device)
        warmup_mult = min(1.0, step / self.pruning_config.sparsity_warmup_steps if self.pruning_config.sparsity_warmup_steps > 0 else 1.0)
        
        loss_keys = [
            'embedding', 'full_layers', 'attention_blocks', 'mlp_blocks',
            'attention_heads', 'attention_neurons', 'mlp_hidden', 'mlp_output'
        ]
        for key in loss_keys: losses[key] = torch.tensor(0.0, device=self.device)

        if self.pruning_config.prune_embedding and self.embedding_gate:
            losses['embedding'] += self.embedding_gate.get_sparsity_loss()

        if self.pruning_config.prune_full_layers and self.layer_gates:
            for gate in self.layer_gates:
                losses['full_layers'] += gate.get_sparsity_loss()
                
        for block in self.transformer.h:
            if self.pruning_config.prune_attention_blocks and block.attention_block_gate:
                losses['attention_blocks'] += block.attention_block_gate.get_sparsity_loss()
            if self.pruning_config.prune_mlp_blocks and block.mlp_block_gate:
                losses['mlp_blocks'] += block.mlp_block_gate.get_sparsity_loss()
            if self.pruning_config.prune_attention_heads and block.attn.head_gates:
                losses['attention_heads'] += block.attn.head_gates.get_sparsity_loss()
            if self.pruning_config.prune_attention_neurons and block.attn.neuron_gates:
                losses['attention_neurons'] += block.attn.neuron_gates.get_sparsity_loss()
            if self.pruning_config.prune_mlp_hidden and block.mlp.hidden_gates:
                losses['mlp_hidden'] += block.mlp.hidden_gates.get_sparsity_loss()
            if self.pruning_config.prune_mlp_output and block.mlp.output_gates:
                losses['mlp_output'] += block.mlp.output_gates.get_sparsity_loss()

        # Apply lambdas
        total_loss += self.pruning_config.lambda_embedding * losses['embedding']
        total_loss += self.pruning_config.lambda_full_layers * losses['full_layers']
        total_loss += self.pruning_config.lambda_attention_blocks * losses['attention_blocks']
        total_loss += self.pruning_config.lambda_mlp_blocks * losses['mlp_blocks']
        total_loss += self.pruning_config.lambda_attention_heads * losses['attention_heads']
        total_loss += self.pruning_config.lambda_attention_neurons * losses['attention_neurons']
        total_loss += self.pruning_config.lambda_mlp_hidden * losses['mlp_hidden']
        total_loss += self.pruning_config.lambda_mlp_output * losses['mlp_output']
        
        total_loss *= warmup_mult
        
        losses['total_sparsity'] = total_loss
        return losses
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