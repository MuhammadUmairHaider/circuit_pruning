import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

import torch
import torch.nn as nn

# def analyze_and_finalize_circuit(model: nn.Module, verbose: bool = True):
#     """
#     Analyzes a trained model with HardConcreteGates, pedantically enforces hierarchical
#     consistency, prints detailed pruning statistics, and keeps the model in its
#     final, hard-pruned inference mode.
#     """
#     print("\n" + "="*80)
#     print("  ANALYZING AND FINALIZING THE DISCOVERED CIRCUIT (PEDANTIC MODE)")
#     print("="*80)

#     model.eval()
#     # Set to final mode to read and enforce deterministic 0/1 gate values
#     model.set_final_circuit_mode(True)

#     # --- 1. INITIALIZATION ---
#     config = model.config
#     hidden_size = config.hidden_size
#     num_heads = config.n_head
#     head_dim = hidden_size // num_heads if num_heads > 0 else 0
#     num_layers = config.n_layer
#     intermediate_size = config.n_inner if config.n_inner is not None else 4 * hidden_size

#     granularity_stats = {
#         'embedding': {'total': 1, 'active': 0},
#         'layer_level': {'total': num_layers, 'active': 0},
#         'attention_blocks': {'total': 0, 'active': 0},
#         'mlp_blocks': {'total': 0, 'active': 0},
#         'attention_heads': {'total': 0, 'active': 0},
#         'attention_neurons': {'total': 0, 'active': 0},
#         'mlp_hidden': {'total': 0, 'active': 0},
#         'mlp_output': {'total': 0, 'active': 0}
#     }
#     layer_report_data = []

#     with torch.no_grad():
#         # --- 2. HIERARCHICAL CONSISTENCY ENFORCEMENT ---
#         # This section modifies the gates in-place to ensure strict hierarchy.
        
#         layer_gates_status = [True] * num_layers
#         if hasattr(model, 'layer_gates') and model.layer_gates is not None:
#             for i, layer_gate in enumerate(model.layer_gates):
#                 if (layer_gate() < 0.5).item():
#                     layer_gates_status[i] = False

#         for i, block in enumerate(model.transformer.h):
#             if not layer_gates_status[i]:
#                 # If layer is pruned, force everything inside it to be pruned
#                 if hasattr(block, 'attention_block_gate'): block.attention_block_gate.log_alpha.data.fill_(-1e6)
#                 if hasattr(block, 'mlp_block_gate'): block.mlp_block_gate.log_alpha.data.fill_(-1e6)
#                 if hasattr(block.attn, 'head_gates'): block.attn.head_gates.log_alpha.data.fill_(-1e6)
#                 if hasattr(block.attn, 'neuron_gates'): block.attn.neuron_gates.log_alpha.data.fill_(-1e6)
#                 if hasattr(block.mlp, 'hidden_gates'): block.mlp.hidden_gates.log_alpha.data.fill_(-1e6)
#                 if hasattr(block.mlp, 'output_gates'): block.mlp.output_gates.log_alpha.data.fill_(-1e6)
#                 continue

#             # Top-Down: Block -> Children
#             if hasattr(block, 'attention_block_gate') and (block.attention_block_gate() < 0.5).item():
#                 if hasattr(block.attn, 'head_gates'): block.attn.head_gates.log_alpha.data.fill_(-1e6)
#             if hasattr(block, 'mlp_block_gate') and (block.mlp_block_gate() < 0.5).item():
#                 if hasattr(block.mlp, 'hidden_gates'): block.mlp.hidden_gates.log_alpha.data.fill_(-1e6)
#                 if hasattr(block.mlp, 'output_gates'): block.mlp.output_gates.log_alpha.data.fill_(-1e6)

#             # Top-Down: Head -> Neurons
#             if hasattr(block.attn, 'head_gates') and hasattr(block.attn, 'neuron_gates'):
#                 head_gates_mask = block.attn.head_gates() < 0.5
#                 if head_gates_mask.any():
#                     neuron_log_alpha = block.attn.neuron_gates.log_alpha.view(num_heads, head_dim)
#                     neuron_log_alpha[head_gates_mask, :] = -1e6

#             # Bottom-Up: Neurons -> Head -> Block
#             if hasattr(block.attn, 'head_gates') and hasattr(block.attn, 'neuron_gates'):
#                 neuron_mask_by_head = (block.attn.neuron_gates() < 0.5).view(num_heads, head_dim)
#                 all_neurons_pruned_mask = neuron_mask_by_head.all(dim=1)
#                 if all_neurons_pruned_mask.any():
#                     block.attn.head_gates.log_alpha.data[all_neurons_pruned_mask] = -1e6
            
#             if hasattr(block.attn, 'head_gates') and (block.attn.head_gates() < 0.5).all().item():
#                  if hasattr(block, 'attention_block_gate'): block.attention_block_gate.log_alpha.data.fill_(-1e6)

#             # Bottom-Up: MLP Neurons -> Block
#             if hasattr(block, 'mlp_block_gate') and hasattr(block.mlp, 'hidden_gates') and hasattr(block.mlp, 'output_gates'):
#                 all_mlp_pruned = (block.mlp.hidden_gates() < 0.5).all() and (block.mlp.output_gates() < 0.5).all()
#                 if all_mlp_pruned:
#                     block.mlp_block_gate.log_alpha.data.fill_(-1e6)

#         # --- 3. STATISTICS GATHERING (POST-ENFORCEMENT) ---
#         if hasattr(model, 'embedding_gate') and (model.embedding_gate() > 0.5).item():
#             granularity_stats['embedding']['active'] = 1
#         embedding_gate_status = "Active" if granularity_stats['embedding']['active'] > 0 else "Pruned"
        
#         granularity_stats['layer_level']['active'] = int(sum(layer_gates_status))

#         for i, block in enumerate(model.transformer.h):
#             layer_stats = {'layer': i, 'layer_active': layer_gates_status[i]}
            
#             if hasattr(block, 'attention_block_gate'):
#                 granularity_stats['attention_blocks']['total'] += 1
#                 is_active = (block.attention_block_gate() > 0.5).item()
#                 layer_stats['attn_block'] = "Active" if is_active else "Pruned"
#                 if is_active: granularity_stats['attention_blocks']['active'] += 1

#             if hasattr(block, 'mlp_block_gate'):
#                 granularity_stats['mlp_blocks']['total'] += 1
#                 is_active = (block.mlp_block_gate() > 0.5).item()
#                 layer_stats['mlp_block'] = "Active" if is_active else "Pruned"
#                 if is_active: granularity_stats['mlp_blocks']['active'] += 1
            
#             if hasattr(block.attn, 'head_gates'):
#                 active_count = (block.attn.head_gates() > 0.5).sum().item()
#                 layer_stats['attn_heads'] = f"{int(active_count)}/{num_heads}"
#                 granularity_stats['attention_heads']['total'] += num_heads
#                 if layer_gates_status[i]: granularity_stats['attention_heads']['active'] += active_count
            
#             if hasattr(block.attn, 'neuron_gates'):
#                 active_count = (block.attn.neuron_gates() > 0.5).sum().item()
#                 total_count = len(block.attn.neuron_gates.log_alpha)
#                 layer_stats['attn_neurons'] = f"{int(active_count)}/{total_count}"
#                 granularity_stats['attention_neurons']['total'] += total_count
#                 if layer_gates_status[i]: granularity_stats['attention_neurons']['active'] += active_count
            
#             if hasattr(block.mlp, 'hidden_gates'):
#                 active_count = (block.mlp.hidden_gates() > 0.5).sum().item()
#                 layer_stats['mlp_hidden'] = f"{int(active_count)}/{intermediate_size}"
#                 granularity_stats['mlp_hidden']['total'] += intermediate_size
#                 if layer_gates_status[i]: granularity_stats['mlp_hidden']['active'] += active_count

#             if hasattr(block.mlp, 'output_gates'):
#                 active_count = (block.mlp.output_gates() > 0.5).sum().item()
#                 layer_stats['mlp_output'] = f"{int(active_count)}/{hidden_size}"
#                 granularity_stats['mlp_output']['total'] += hidden_size
#                 if layer_gates_status[i]: granularity_stats['mlp_output']['active'] += active_count
            
#             layer_report_data.append(layer_stats)

#     # --- 4. REPORTING ---
#     if verbose:
#         print("\n" + "="*80)
#         print("  HIERARCHICAL PRUNING REPORT (Consistency Enforced)")
#         print("="*80)
        
#         print(f"\n📍 GLOBAL COMPONENTS:")
#         print(f"  - Embedding Gate: {embedding_gate_status}")
        
#         if granularity_stats['layer_level']['total'] > 0:
#             active = granularity_stats['layer_level']['active']
#             total = granularity_stats['layer_level']['total']
#             print(f"\n📍 LAYER-LEVEL PRUNING:")
#             print(f"  - Active Layers: {active}/{total} ({(active/total)*100:.1f}%)")
#             if active < total:
#                 print(f"  - Pruned Layer Indices: {' '.join([str(i) for i, act in enumerate(layer_gates_status) if not act])}")
        
#         print(f"\n📍 DETAILED LAYER REPORT:")
#         header = f"{'Layer':<6} | {'Status':<8} | {'Attn Block':<11} | {'MLP Block':<10} | {'Attn Heads':<12} | {'Attn Neurons':<15} | {'MLP Hidden':<15} | {'MLP Output':<15}"
#         print(header)
#         print("-" * len(header))
        
#         for stats in layer_report_data:
#             layer_status = "Active" if stats.get('layer_active', True) else "PRUNED"
#             attn_block = stats.get('attn_block', 'N/A')
#             mlp_block = stats.get('mlp_block', 'N/A')
#             attn_heads = stats.get('attn_heads', 'N/A')
#             attn_neurons = stats.get('attn_neurons', 'N/A')
#             mlp_hidden = stats.get('mlp_hidden', 'N/A')
#             mlp_output = stats.get('mlp_output', 'N/A')
            
#             if layer_status == "PRUNED":
#                 print(f"\033[90m{stats['layer']:<6} | {layer_status:<8} | {'---':<11} | {'---':<10} | {'---':<12} | {'---':<15} | {'---':<15} | {'---':<15}\033[0m")
#             else:
#                 print(f"{stats['layer']:<6} | {layer_status:<8} | {attn_block:<11} | {mlp_block:<10} | {attn_heads:<12} | {attn_neurons:<15} | {mlp_hidden:<15} | {mlp_output:<15}")

#     print("\n" + "="*80)
#     print("  PRUNING SUMMARY BY GRANULARITY")
#     print("="*80)
    
#     for G, S in granularity_stats.items():
#         if S['total'] > 0:
#             name = G.replace('_', ' ').title()
#             pruned_pct = (S['total'] - S['active']) / S['total'] * 100 if S['total'] > 0 else 0
#             print(f"\n{name}:")
#             print(f"  - Active: {S['active']:,} / {S['total']:,}  ({(100-pruned_pct):.1f}%)")
#             print(f"  - Pruned: {S['total'] - S['active']:,} ({pruned_pct:.1f}%)")

#     print("\n" + "="*80)
#     print("  OVERALL STATISTICS")
#     print("="*80)
    
#     total_params = sum(p.numel() for p in model.parameters())
#     active_params = 0
    
#     # Add non-transformer params like embeddings and final layer norm
#     active_params += model.transformer.wpe.weight.numel()
#     active_params += model.transformer.ln_f.weight.numel() + model.transformer.ln_f.bias.numel()
#     if granularity_stats['embedding']['active']:
#         active_params += model.transformer.wte.weight.numel()

#     for i, report in enumerate(layer_report_data):
#         if not report['layer_active']: continue
#         block = model.transformer.h[i]
        
#         active_params += block.ln_1.weight.numel() + block.ln_1.bias.numel()
#         active_params += block.ln_2.weight.numel() + block.ln_2.bias.numel()
        
#         if report.get('attn_block') == 'Active':
#             ### FIX: Access layers through .original_attention ###
#             active_params += block.attn.original_attention.c_proj.weight.numel() + block.attn.original_attention.c_proj.bias.numel()
#             active_params += block.attn.original_attention.c_attn.weight.numel() + block.attn.original_attention.c_attn.bias.numel()
            
#         if report.get('mlp_block') == 'Active':
#              active_params += block.mlp.original_mlp.c_proj.weight.numel() + block.mlp.original_mlp.c_proj.bias.numel()
#              active_params += block.mlp.original_mlp.c_fc.weight.numel() + block.mlp.original_mlp.c_fc.bias.numel()
            
        
             
#     pruned_params = total_params - active_params
#     compression = total_params / active_params if active_params > 0 else float('inf')

#     print(f"\nEstimated Active Parameters: {int(active_params):,} / {total_params:,}")
#     print(f"Model Compression Ratio (Parameters): {compression:.2f}x")
#     print(f"Reduction in Parameters: {pruned_params/total_params*100:.1f}%")
    
#     print("="*80)
#     print("\n✅ Model remains in 'Final Circuit Mode' for evaluation.")
    
#     return {
#         'granularity_stats': granularity_stats,
#         'layer_report': layer_report_data,
#         'active_parameters': active_params,
#         'total_parameters': total_params,
#         'compression_ratio': compression
#     }



import torch
import torch.nn as nn

def analyze_prunable_compression(model, layer_report_data, config, verbose=True):
    """
    Calculate compression based ONLY on prunable parameters:
    - Exclude: embeddings, positional embeddings, layer norms, LM head
    - Include: Only attention and MLP weight matrices that can be pruned
    """
    
    hidden_size = config.hidden_size  # 768
    num_heads = config.n_head  # 12
    head_dim = hidden_size // num_heads  # 64
    intermediate_size = config.n_inner if config.n_inner is not None else 4 * hidden_size  # 3072
    num_layers = config.n_layer
    
    # --- CALCULATE TOTAL PRUNABLE PARAMETERS ---
    total_prunable_params = 0
    
    # Per layer prunable parameters
    for layer_idx in range(num_layers):
        # Attention prunable parameters
        # c_attn: hidden_size → 3 * hidden_size (Q, K, V projections)
        attention_qkv_params = hidden_size * 3 * hidden_size + 3 * hidden_size  # weights + bias
        
        # c_proj: hidden_size → hidden_size (output projection)  
        attention_proj_params = hidden_size * hidden_size + hidden_size  # weights + bias
        
        total_attention_params = attention_qkv_params + attention_proj_params
        
        # MLP prunable parameters
        # c_fc: hidden_size → intermediate_size
        mlp_fc_params = hidden_size * intermediate_size + intermediate_size  # weights + bias
        
        # c_proj: intermediate_size → hidden_size
        mlp_proj_params = intermediate_size * hidden_size + hidden_size  # weights + bias
        
        total_mlp_params = mlp_fc_params + mlp_proj_params
        
        total_prunable_params += total_attention_params + total_mlp_params
    
    if verbose:
        print(f"\n📍 PRUNABLE PARAMETER BREAKDOWN:")
        print(f"  - Layers: {num_layers}")
        print(f"  - Attention params per layer: {total_attention_params:,}")
        print(f"  - MLP params per layer: {total_mlp_params:,}")
        print(f"  - Total per layer: {total_attention_params + total_mlp_params:,}")
        print(f"  - TOTAL PRUNABLE: {total_prunable_params:,}")
    
    # --- CALCULATE ACTIVE PRUNABLE PARAMETERS ---
    active_prunable_params = 0
    
    for i, report in enumerate(layer_report_data):
        if not report['layer_active']:
            continue  # Skip entirely pruned layers
            
        block = model.transformer.h[i]
        layer_active_params = 0
        
        # Active attention parameters
        if report.get('attn_block') == 'Active':
            if hasattr(block.attn, 'neuron_gates') and block.attn.neuron_gates!=None:
                active_attention_neurons = (block.attn.neuron_gates() > 0.5).sum().item()
            else:
                active_attention_neurons = hidden_size
            
            # QKV: hidden_size → 3 * active_attention_neurons
            active_qkv_params = hidden_size * 3 * active_attention_neurons + 3 * active_attention_neurons
            
            # Output projection: active_attention_neurons → hidden_size
            active_proj_params = active_attention_neurons * hidden_size + hidden_size
            
            layer_attention_params = active_qkv_params + active_proj_params
            layer_active_params += layer_attention_params
            
            if verbose:
                print(f"  Layer {i} Attention: {active_attention_neurons}/{hidden_size} neurons → {layer_attention_params:,} params")
        
        # Active MLP parameters  
        if report.get('mlp_block') == 'Active':
            if hasattr(block.mlp, 'hidden_gates') and block.mlp.hidden_gates!=None:
                active_hidden_neurons = (block.mlp.hidden_gates() > 0.5).sum().item()
            else:
                active_hidden_neurons = intermediate_size
                
            if hasattr(block.mlp, 'output_gates') and block.mlp.output_gates!=None:
                active_output_neurons = (block.mlp.output_gates() > 0.5).sum().item()
            else:
                active_output_neurons = hidden_size
            
            # FC: hidden_size → active_hidden_neurons
            active_fc_params = hidden_size * active_hidden_neurons + active_hidden_neurons
            
            # Projection: active_hidden_neurons → active_output_neurons
            active_mlp_proj_params = active_hidden_neurons * active_output_neurons + active_output_neurons
            
            layer_mlp_params = active_fc_params + active_mlp_proj_params
            layer_active_params += layer_mlp_params
            
            if verbose:
                print(f"  Layer {i} MLP: {active_hidden_neurons}/{intermediate_size} hidden, {active_output_neurons}/{hidden_size} output → {layer_mlp_params:,} params")
        
        active_prunable_params += layer_active_params
        
        if verbose and layer_active_params > 0:
            print(f"  Layer {i} TOTAL: {layer_active_params:,} params")
    
    # --- CALCULATE COMPRESSION METRICS ---
    pruned_params = total_prunable_params - active_prunable_params
    compression_ratio = total_prunable_params / active_prunable_params if active_prunable_params > 0 else float('inf')
    reduction_percentage = (pruned_params / total_prunable_params) * 100
    
    # Calculate effective model compression including fixed params
    total_model_params = sum(p.numel() for p in model.parameters())
    fixed_params = total_model_params - total_prunable_params
    effective_model_params = fixed_params + active_prunable_params
    effective_compression = total_model_params / effective_model_params if effective_model_params > 0 else float('inf')
    
    if verbose:
        print(f"\n" + "="*80)
        print("  PRUNABLE PARAMETER COMPRESSION ANALYSIS")
        print("="*80)
        print(f"Total prunable parameters: {total_prunable_params:,}")
        print(f"Active prunable parameters: {active_prunable_params:,}")
        print(f"Pruned parameters: {pruned_params:,}")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Parameter reduction: {reduction_percentage:.1f}%")
        
        print(f"\n📍 MODEL BREAKDOWN:")
        print(f"  - Total model parameters: {total_model_params:,}")
        print(f"  - Fixed parameters (embeddings, norms, etc.): {fixed_params:,} ({fixed_params/total_model_params*100:.1f}%)")
        print(f"  - Prunable parameters: {total_prunable_params:,} ({total_prunable_params/total_model_params*100:.1f}%)")
        print(f"  - Active prunable parameters: {active_prunable_params:,} ({active_prunable_params/total_model_params*100:.1f}%)")
        
        print(f"\n🎯 EFFECTIVE MODEL COMPRESSION:")
        print(f"  - Effective model size: {effective_model_params:,} parameters")
        print(f"  - Overall compression: {effective_compression:.2f}x")
        print(f"  - Overall reduction: {(total_model_params - effective_model_params)/total_model_params*100:.1f}%")
    
    return {
        'total_prunable_params': total_prunable_params,
        'active_prunable_params': active_prunable_params,
        'compression_ratio': compression_ratio,
        'reduction_percentage': reduction_percentage,
        'total_model_params': total_model_params,
        'fixed_params': fixed_params,
        'effective_model_params': effective_model_params,
        'effective_compression': effective_compression
    }


def analyze_and_finalize_circuit(model: nn.Module, verbose: bool = True):
    """
    Analyzes a trained model with HardConcreteGates, pedantically enforces hierarchical
    consistency, prints detailed pruning statistics, and keeps the model in its
    final, hard-pruned inference mode. FOCUSED ON PRUNABLE PARAMETERS.
    """
    print("\n" + "="*80)
    print("  ANALYZING AND FINALIZING THE DISCOVERED CIRCUIT (PEDANTIC MODE)")
    print("="*80)

    model.eval()
    # Set to final mode to read and enforce deterministic 0/1 gate values
    model.set_final_circuit_mode(True)

    # --- 1. INITIALIZATION ---
    config = model.config
    hidden_size = config.hidden_size
    num_heads = config.n_head
    head_dim = hidden_size // num_heads if num_heads > 0 else 0
    num_layers = config.n_layer
    intermediate_size = config.n_inner if config.n_inner is not None else 4 * hidden_size

    granularity_stats = {
        'embedding': {'total': 1, 'active': 0},
        'layer_level': {'total': num_layers, 'active': 0},
        'attention_blocks': {'total': 0, 'active': 0},
        'mlp_blocks': {'total': 0, 'active': 0},
        'attention_heads': {'total': 0, 'active': 0},
        'attention_neurons': {'total': 0, 'active': 0},
        'mlp_hidden': {'total': 0, 'active': 0},
        'mlp_output': {'total': 0, 'active': 0}
    }
    layer_report_data = []

    with torch.no_grad():
        # --- 2. HIERARCHICAL CONSISTENCY ENFORCEMENT ---
        # This section modifies the gates in-place to ensure strict hierarchy.
        
        layer_gates_status = [True] * num_layers
        if hasattr(model, 'layer_gates') and model.layer_gates is not None:
            for i, layer_gate in enumerate(model.layer_gates):
                if layer_gate is not None and (layer_gate() < 0.5).item():
                    layer_gates_status[i] = False

        for i, block in enumerate(model.transformer.h):
            if not layer_gates_status[i]:
                # If layer is pruned, force everything inside it to be pruned
                if hasattr(block, 'attention_block_gate') and block.attention_block_gate is not None:
                    block.attention_block_gate.log_alpha.data.fill_(-1e6)
                if hasattr(block, 'mlp_block_gate') and block.mlp_block_gate is not None:
                    block.mlp_block_gate.log_alpha.data.fill_(-1e6)
                if hasattr(block.attn, 'head_gates') and block.attn.head_gates is not None:
                    block.attn.head_gates.log_alpha.data.fill_(-1e6)
                if hasattr(block.attn, 'neuron_gates') and block.attn.neuron_gates is not None:
                    block.attn.neuron_gates.log_alpha.data.fill_(-1e6)
                if hasattr(block.mlp, 'hidden_gates') and block.mlp.hidden_gates is not None:
                    block.mlp.hidden_gates.log_alpha.data.fill_(-1e6)
                if hasattr(block.mlp, 'output_gates') and block.mlp.output_gates is not None:
                    block.mlp.output_gates.log_alpha.data.fill_(-1e6)
                continue

            # Top-Down: Block -> Children
            if hasattr(block, 'attention_block_gate') and block.attention_block_gate is not None and (block.attention_block_gate() < 0.5).item():
                if hasattr(block.attn, 'head_gates') and block.attn.head_gates is not None:
                    block.attn.head_gates.log_alpha.data.fill_(-1e6)
                if hasattr(block.attn, 'neuron_gates') and block.attn.neuron_gates is not None:
                    block.attn.neuron_gates.log_alpha.data.fill_(-1e6)
            
            if hasattr(block, 'mlp_block_gate') and block.mlp_block_gate is not None and (block.mlp_block_gate() < 0.5).item():
                if hasattr(block.mlp, 'hidden_gates') and block.mlp.hidden_gates is not None:
                    block.mlp.hidden_gates.log_alpha.data.fill_(-1e6)
                if hasattr(block.mlp, 'output_gates') and block.mlp.output_gates is not None:
                    block.mlp.output_gates.log_alpha.data.fill_(-1e6)

            # Top-Down: Head -> Neurons
            if (hasattr(block.attn, 'head_gates') and block.attn.head_gates is not None and 
                hasattr(block.attn, 'neuron_gates') and block.attn.neuron_gates is not None):
                head_gates_mask = block.attn.head_gates() < 0.5
                if head_gates_mask.any():
                    neuron_log_alpha = block.attn.neuron_gates.log_alpha.view(num_heads, head_dim)
                    neuron_log_alpha[head_gates_mask, :] = -1e6

            # Bottom-Up: Neurons -> Head -> Block
            if (hasattr(block.attn, 'head_gates') and block.attn.head_gates is not None and 
                hasattr(block.attn, 'neuron_gates') and block.attn.neuron_gates is not None):
                neuron_mask_by_head = (block.attn.neuron_gates() < 0.5).view(num_heads, head_dim)
                all_neurons_pruned_mask = neuron_mask_by_head.all(dim=1)
                if all_neurons_pruned_mask.any():
                    block.attn.head_gates.log_alpha.data[all_neurons_pruned_mask] = -1e6
            
            if (hasattr(block.attn, 'head_gates') and block.attn.head_gates is not None and 
                (block.attn.head_gates() < 0.5).all().item()):
                if hasattr(block, 'attention_block_gate') and block.attention_block_gate is not None:
                    block.attention_block_gate.log_alpha.data.fill_(-1e6)

            # Bottom-Up: MLP Neurons -> Block
            if (hasattr(block, 'mlp_block_gate') and block.mlp_block_gate is not None and 
                hasattr(block.mlp, 'hidden_gates') and block.mlp.hidden_gates is not None and 
                hasattr(block.mlp, 'output_gates') and block.mlp.output_gates is not None):
                all_mlp_pruned = (block.mlp.hidden_gates() < 0.5).all() and (block.mlp.output_gates() < 0.5).all()
                if all_mlp_pruned:
                    block.mlp_block_gate.log_alpha.data.fill_(-1e6)

        # --- 2.5. LAYER-LEVEL BOTTOM-UP CONSISTENCY ---
        # Check if both attention and MLP blocks are pruned, then prune the entire layer
        if hasattr(model, 'layer_gates') and model.layer_gates is not None:
            for i, block in enumerate(model.transformer.h):
                if i >= len(model.layer_gates) or model.layer_gates[i] is None:
                    continue
                    
                attention_block_pruned = False
                mlp_block_pruned = False
                
                # Check if attention block is pruned
                if hasattr(block, 'attention_block_gate') and block.attention_block_gate is not None:
                    attention_block_pruned = (block.attention_block_gate() < 0.5).item()
                else:
                    # If no attention block gate, consider it active (fallback)
                    attention_block_pruned = False
                
                # Check if MLP block is pruned  
                if hasattr(block, 'mlp_block_gate') and block.mlp_block_gate is not None:
                    mlp_block_pruned = (block.mlp_block_gate() < 0.5).item()
                else:
                    # If no MLP block gate, consider it active (fallback)
                    mlp_block_pruned = False
                
                # If both major components are pruned, prune the entire layer
                if attention_block_pruned and mlp_block_pruned:
                    model.layer_gates[i].log_alpha.data.fill_(-1e6)
                    layer_gates_status[i] = False
                    if verbose:
                        print(f"  🔧 Layer {i}: Both attention and MLP blocks pruned → Pruning entire layer")

        # --- 3. STATISTICS GATHERING (POST-ENFORCEMENT) ---
        if (hasattr(model, 'embedding_gate') and model.embedding_gate is not None and 
            (model.embedding_gate() > 0.5).item()):
            granularity_stats['embedding']['active'] = 1
        embedding_gate_status = "Active" if granularity_stats['embedding']['active'] > 0 else "Pruned"
        
        granularity_stats['layer_level']['active'] = int(sum(layer_gates_status))

        for i, block in enumerate(model.transformer.h):
            layer_stats = {'layer': i, 'layer_active': layer_gates_status[i]}
            
            if hasattr(block, 'attention_block_gate') and block.attention_block_gate is not None:
                granularity_stats['attention_blocks']['total'] += 1
                is_active = (block.attention_block_gate() > 0.5).item()
                layer_stats['attn_block'] = "Active" if is_active else "Pruned"
                if is_active: granularity_stats['attention_blocks']['active'] += 1

            if hasattr(block, 'mlp_block_gate') and block.mlp_block_gate is not None:
                granularity_stats['mlp_blocks']['total'] += 1
                is_active = (block.mlp_block_gate() > 0.5).item()
                layer_stats['mlp_block'] = "Active" if is_active else "Pruned"
                if is_active: granularity_stats['mlp_blocks']['active'] += 1
            
            if hasattr(block.attn, 'head_gates') and block.attn.head_gates is not None:
                active_count = (block.attn.head_gates() > 0.5).sum().item()
                layer_stats['attn_heads'] = f"{int(active_count)}/{num_heads}"
                granularity_stats['attention_heads']['total'] += num_heads
                if layer_gates_status[i]: granularity_stats['attention_heads']['active'] += active_count
            
            if hasattr(block.attn, 'neuron_gates') and block.attn.neuron_gates is not None:
                active_count = (block.attn.neuron_gates() > 0.5).sum().item()
                total_count = len(block.attn.neuron_gates.log_alpha)
                layer_stats['attn_neurons'] = f"{int(active_count)}/{total_count}"
                granularity_stats['attention_neurons']['total'] += total_count
                if layer_gates_status[i]: granularity_stats['attention_neurons']['active'] += active_count
            
            if hasattr(block.mlp, 'hidden_gates') and block.mlp.hidden_gates is not None:
                active_count = (block.mlp.hidden_gates() > 0.5).sum().item()
                layer_stats['mlp_hidden'] = f"{int(active_count)}/{intermediate_size}"
                granularity_stats['mlp_hidden']['total'] += intermediate_size
                if layer_gates_status[i]: granularity_stats['mlp_hidden']['active'] += active_count

            if hasattr(block.mlp, 'output_gates') and block.mlp.output_gates is not None:
                active_count = (block.mlp.output_gates() > 0.5).sum().item()
                layer_stats['mlp_output'] = f"{int(active_count)}/{hidden_size}"
                granularity_stats['mlp_output']['total'] += hidden_size
                if layer_gates_status[i]: granularity_stats['mlp_output']['active'] += active_count
            
            layer_report_data.append(layer_stats)

    # --- 4. PRUNABLE PARAMETER COMPRESSION ANALYSIS ---
    prunable_stats = analyze_prunable_compression(model, layer_report_data, config, verbose)

    # --- 5. REPORTING ---
    if verbose:
        print("\n" + "="*80)
        print("  HIERARCHICAL PRUNING REPORT (Consistency Enforced)")
        print("="*80)
        
        print(f"\n📍 GLOBAL COMPONENTS:")
        print(f"  - Embedding Gate: {embedding_gate_status}")
        
        if granularity_stats['layer_level']['total'] > 0:
            active = granularity_stats['layer_level']['active']
            total = granularity_stats['layer_level']['total']
            print(f"\n📍 LAYER-LEVEL PRUNING:")
            print(f"  - Active Layers: {active}/{total} ({(active/total)*100:.1f}%)")
            if active < total:
                print(f"  - Pruned Layer Indices: {' '.join([str(i) for i, act in enumerate(layer_gates_status) if not act])}")
        
        print(f"\n📍 DETAILED LAYER REPORT:")
        header = f"{'Layer':<6} | {'Status':<8} | {'Attn Block':<11} | {'MLP Block':<10} | {'Attn Heads':<12} | {'Attn Neurons':<15} | {'MLP Hidden':<15} | {'MLP Output':<15}"
        print(header)
        print("-" * len(header))
        
        for stats in layer_report_data:
            layer_status = "Active" if stats.get('layer_active', True) else "PRUNED"
            attn_block = stats.get('attn_block', 'N/A')
            mlp_block = stats.get('mlp_block', 'N/A')
            attn_heads = stats.get('attn_heads', 'N/A')
            attn_neurons = stats.get('attn_neurons', 'N/A')
            mlp_hidden = stats.get('mlp_hidden', 'N/A')
            mlp_output = stats.get('mlp_output', 'N/A')
            
            if layer_status == "PRUNED":
                print(f"\033[90m{stats['layer']:<6} | {layer_status:<8} | {'---':<11} | {'---':<10} | {'---':<12} | {'---':<15} | {'---':<15} | {'---':<15}\033[0m")
            else:
                print(f"{stats['layer']:<6} | {layer_status:<8} | {attn_block:<11} | {mlp_block:<10} | {attn_heads:<12} | {attn_neurons:<15} | {mlp_hidden:<15} | {mlp_output:<15}")

    print("\n" + "="*80)
    print("  PRUNING SUMMARY BY GRANULARITY")
    print("="*80)
    
    for G, S in granularity_stats.items():
        if S['total'] > 0:
            name = G.replace('_', ' ').title()
            pruned_pct = (S['total'] - S['active']) / S['total'] * 100 if S['total'] > 0 else 0
            print(f"\n{name}:")
            print(f"  - Active: {S['active']:,} / {S['total']:,}  ({(100-pruned_pct):.1f}%)")
            print(f"  - Pruned: {S['total'] - S['active']:,} ({pruned_pct:.1f}%)")

    print(f"\n🎯 PRUNABLE PARAMETER BOTTOM LINE:")
    print(f"  - Circuit uses {prunable_stats['active_prunable_params']:,} out of {prunable_stats['total_prunable_params']:,} prunable parameters")
    print(f"  - Prunable compression: {prunable_stats['compression_ratio']:.1f}x")
    print(f"  - Prunable reduction: {prunable_stats['reduction_percentage']:.1f}%")
    
    # Compare with component statistics for sanity check
    # print(f"\n🔍 SANITY CHECK (should align):")
    # print(f"  - Active layers: {granularity_stats['layer_level']['active']}/{granularity_stats['layer_level']['total']} ({granularity_stats['layer_level']['active']/granularity_stats['layer_level']['total']*100:.1f}%)")
    # print(f"  - Active attention neurons: {granularity_stats['attention_neurons']['active']}/{granularity_stats['attention_neurons']['total']} ({granularity_stats['attention_neurons']['active']/granularity_stats['attention_neurons']['total']*100:.1f}%)")
    # total_mlp_neurons = granularity_stats['mlp_hidden']['total'] + granularity_stats['mlp_output']['total']
    # active_mlp_neurons = granularity_stats['mlp_hidden']['active'] + granularity_stats['mlp_output']['active']
    # print(f"  - Active MLP neurons: {active_mlp_neurons}/{total_mlp_neurons} ({active_mlp_neurons/total_mlp_neurons*100:.1f}%)")
    
    print("="*80)
    print("\n✅ Model remains in 'Final Circuit Mode' for evaluation.")
    
    return {
        'granularity_stats': granularity_stats,
        'layer_report': layer_report_data,
        'prunable_compression': prunable_stats
    }
def disable_dropout(model: nn.Module):
    """
    Recursively finds all nn.Dropout layers in a model and sets their
    dropout probability to 0.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    