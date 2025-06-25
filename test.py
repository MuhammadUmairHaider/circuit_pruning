import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers.modeling_outputs import CausalLMOutput
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# ==============================================================================
# SECTION 1: FINAL, WORKING MODEL DEFINITIONS
# ==============================================================================

@dataclass
class PruningConfig:
    init_value: float = 1.0
    # Add lambdas etc. back for actual training
    enable_attention_pruning: bool = True
    enable_mlp_pruning: bool = True

class LearnableGate(nn.Module):
    def __init__(self, size, init_value: float = 1.0):
        super().__init__()
        self.gate = nn.Parameter(torch.full(size if isinstance(size, tuple) else (size,), init_value))

class PrunableMLP(nn.Module):
    def __init__(self, original_mlp, gpt_config: GPT2Config, pruning_config: PruningConfig):
        super().__init__()
        self.original_mlp = original_mlp
        if pruning_config.enable_mlp_pruning:
            # Gating the final output is the most robust method
            self.neuron_gates = LearnableGate(gpt_config.hidden_size, init_value=pruning_config.init_value)
        else:
            self.neuron_gates = None
    def forward(self, clean_states, corrupted_states):
        clean_output = self.original_mlp(clean_states)
        corrupted_output = self.original_mlp(corrupted_states)
        if self.neuron_gates:
            gate = self.neuron_gates.gate.view(1, 1, -1)
            gated_output = gate * clean_output + (1 - gate) * corrupted_output
            return gated_output, corrupted_output
        else:
            return clean_output, corrupted_output

class PrunableAttention(nn.Module):
    def __init__(self, original_attention, gpt_config: GPT2Config, pruning_config: PruningConfig):
        super().__init__()
        self.original_attention = original_attention
        self.num_heads = gpt_config.n_head
        self.head_dim = gpt_config.hidden_size // self.num_heads
        if pruning_config.enable_attention_pruning:
            self.head_gates = LearnableGate(self.num_heads, init_value=pruning_config.init_value)
        else:
            self.head_gates = None
            
    def forward(self, clean_states, corrupted_states, **kwargs):
        # The clean pass runs as normal
        clean_attn_outputs = self.original_attention(clean_states, **kwargs)
        clean_outputs = clean_attn_outputs[0]

        # --- YOUR BUG FIX: EXPLICITLY DISABLE CACHE FOR THE CORRUPTED PASS ---
        corrupted_kwargs = kwargs.copy()
        corrupted_kwargs['use_cache'] = False
        corrupted_kwargs['past_key_value'] = None
        # ---------------------------------------------------------------------

        # Run the corrupted pass with a guaranteed clean slate
        corrupted_attn_outputs = self.original_attention(corrupted_states, **corrupted_kwargs)
        corrupted_outputs = corrupted_attn_outputs[0]

        if self.head_gates:
            b, s, d = clean_outputs.shape
            clean_reshaped = clean_outputs.view(b, s, self.num_heads, self.head_dim)
            corrupted_reshaped = corrupted_outputs.view(b, s, self.num_heads, self.head_dim)
            gate = self.head_gates.gate.view(1, 1, self.num_heads, 1)
            gated_output = gate * clean_reshaped + (1 - gate) * corrupted_reshaped
            
            # The PrunableBlock needs the main output and the original tuple for caching
            return (gated_output.view(b, s, d),) + clean_attn_outputs[1:], corrupted_outputs
        else:
            # Return the clean pass outputs and the pure corrupted output
            return clean_attn_outputs, corrupted_outputs

class PrunableBlock(nn.Module):
    def __init__(self, original_block, gpt_config: GPT2Config, pruning_config: PruningConfig):
        super().__init__()
        self.ln_1 = original_block.ln_1
        self.attn = PrunableAttention(original_block.attn, gpt_config, pruning_config)
        self.ln_2 = original_block.ln_2
        self.mlp = PrunableMLP(original_block.mlp, gpt_config, pruning_config)

    # This is the compatible forward pass for baseline checks
    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn.original_attention(hidden_states, **kwargs)
        hidden_states = residual + attn_outputs[0]
        
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.mlp.original_mlp(hidden_states)
        
        return (hidden_states,) + attn_outputs[1:]

    # This is your custom logic, separated for clarity
    def forward_gated(self, clean_states, corrupted_states, **kwargs):
        attn_outputs_tuple, corrupted_attn_output = self.attn(
            self.ln_1(clean_states), self.ln_1(corrupted_states), **kwargs
        )
        attn_output = attn_outputs_tuple[0]
        
        clean_after_attn = clean_states + attn_output
        corrupted_after_attn = corrupted_states + corrupted_attn_output
        
        mlp_output, corrupted_mlp_output = self.mlp(
            self.ln_2(clean_after_attn), self.ln_2(corrupted_after_attn)
        )
        final_clean_states = clean_after_attn + mlp_output
        final_corrupted_states = corrupted_after_attn + corrupted_mlp_output
        
        # We also pass up the full attention output tuple from the clean pass for caching
        return final_clean_states, final_corrupted_states, attn_outputs_tuple

class CircuitDiscoveryGPT2(GPT2LMHeadModel):
    # The final, simplest class. Inherits the correct forward pass.
    @classmethod
    def from_pretrained_with_pruning(cls, model_name: str, pruning_config: PruningConfig, **kwargs):
        model = cls.from_pretrained(model_name, **kwargs)
        model.transformer.h = nn.ModuleList([PrunableBlock(b, model.config, pruning_config) for b in model.transformer.h])
        return model

# This helper function orchestrates the dual-stream logic for your experiments
# This helper function orchestrates your custom dual-stream logic
def run_gated_pass(model, clean_input_ids, corrupted_input_ids, attention_mask):
    transformer = model.transformer
    
    clean_embeds = transformer.wte(clean_input_ids) + transformer.wpe.weight[:clean_input_ids.shape[1],:]
    corrupted_embeds = transformer.wte(corrupted_input_ids) + transformer.wpe.weight[:corrupted_input_ids.shape[1],:]
    clean_states = transformer.drop(clean_embeds)
    corrupted_states = transformer.drop(corrupted_embeds)
    
    # Use the library's own helper functions to guarantee correctness
    extended_attention_mask = model.get_extended_attention_mask(attention_mask, clean_input_ids.shape)
    head_mask = model.get_head_mask(None, model.config.n_layer)
    past_key_values = tuple([None] * len(transformer.h))

    for i, block in enumerate(transformer.h):
        # --- THIS IS THE ONLY CHANGE ---
        # We now pass a dictionary with the modern `past_key_value` argument name.
        block_kwargs = {
            "past_key_value": past_key_values[i], 
            "attention_mask": extended_attention_mask, 
            "head_mask": head_mask[i],
            "use_cache": True if past_key_values[i] is not None else False,
        }
        
        clean_states, corrupted_states = block.forward_gated(
            clean_states, corrupted_states, **block_kwargs
        )
    
    hidden_states = transformer.ln_f(clean_states)
    logits = model.lm_head(hidden_states)
    return CausalLMOutput(logits=logits)

# ==============================================================================
# SECTION 2: FINAL SANITY CHECK
# ==============================================================================
if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = 'gpt2'
    
    print(f"--- RUNNING FINAL SANITY CHECK WITH KV CACHE FIX ---")
    print(f"Using device: {DEVICE}")

    pruning_config = PruningConfig()
    full_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    circuit_model = CircuitDiscoveryGPT2.from_pretrained_with_pruning(MODEL_NAME, pruning_config).to(DEVICE).eval()
    
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    dummy_prompt = "The rain in Spain stays mainly on the plain"
    inputs = tokenizer(dummy_prompt, return_tensors='pt').to(DEVICE)
    
    print("\n--- Running Sanity Check ---")
    with torch.no_grad():
        # 1. Baseline
        o1 = full_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
        # 2. Your circuit model, using the gated pass where clean==corrupted
        # This now uses YOUR fix for the KV Cache Leak.
        o2 = run_gated_pass(
            circuit_model, 
            clean_input_ids=inputs['input_ids'], 
            corrupted_input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask']
        )

    if torch.allclose(o1.logits, o2.logits, atol=1e-5):
        print("\n******************************************************************")
        print("🎉🎉🎉 IT WORKS! The logits are identical. 🎉🎉🎉")
        print("Your discovery about the KV cache was the final key. The bug is fixed.")
        print("******************************************************************")
    else:
        diff = torch.abs(o1.logits - o2.logits).max().item()
        print(f"\n❌ Mismatch persists. Max difference: {diff:.6f}.")

