import torch
import torch.nn as nn
import torch.nn.functional as F
def analyze_and_finalize_circuit(model: nn.Module, verbose: bool = True):
    """
    Analyzes a trained model with HardConcreteGates, prints detailed pruning
    statistics for each component type, and switches the model into its
    final, hard-pruned inference mode.
    """
    print("\n" + "="*80)
    print("  ANALYZING AND FINALIZING THE DISCOVERED CIRCUIT")
    print("="*80)
    
    model.eval()
    
    config = model.config
    hidden_size = config.hidden_size
    num_heads = config.n_head
    intermediate_size = config.n_inner if config.n_inner is not None else 4 * hidden_size
    params_per_head = (hidden_size * (hidden_size // num_heads)) * 4
    params_per_mlp_neuron = hidden_size * 2

    report_data = []
    totals = {
        'total_gates': 0, 'active_gates': 0,
        'total_params': 0, 'active_params': 0
    }

    with torch.no_grad():
        # === NEW: Process the global embedding gate first ===
        embedding_gate_status = "Not Present"
        if hasattr(model, 'embedding_gate') and model.embedding_gate is not None:
            gate_value = model.embedding_gate()
            is_active = (gate_value > 0.5).item()
            embedding_gate_status = "Active" if is_active else "Pruned"
            
            # Add its count to the totals
            totals['total_gates'] += 1
            if is_active:
                totals['active_gates'] += 1
        # ======================================================

        # Iterate through the PrunableBlocks to find all layer-specific gates
        for i, block in enumerate(model.transformer.h):
            layer_stats = {'layer': i}

            # --- Process Attention Head Gates ---
            if hasattr(block.attn, 'head_gates') and block.attn.head_gates is not None:
                gates = block.attn.head_gates()
                hard_mask = (gates > 0.5).float()
                num_active = hard_mask.sum().item()
                layer_stats['attn_heads'] = f"{int(num_active)}/{num_heads}"
                totals['total_gates'] += num_heads
                totals['active_gates'] += num_active
                totals['total_params'] += num_heads * params_per_head
                totals['active_params'] += num_active * params_per_head
            
            # --- Process MLP Gates ---
            if hasattr(block.mlp, 'hidden_gates') and block.mlp.hidden_gates is not None:
                gates = block.mlp.hidden_gates()
                hard_mask = (gates > 0.5).float()
                num_active = hard_mask.sum().item()
                layer_stats['mlp_hidden'] = f"{int(num_active)}/{intermediate_size}"
                totals['total_gates'] += intermediate_size
                totals['active_gates'] += num_active
                totals['total_params'] += intermediate_size * params_per_mlp_neuron
                totals['active_params'] += num_active * params_per_mlp_neuron

            if hasattr(block.mlp, 'output_gates') and block.mlp.output_gates is not None:
                gates = block.mlp.output_gates()
                hard_mask = (gates > 0.5).float()
                num_active = hard_mask.sum().item()
                layer_stats['mlp_output'] = f"{int(num_active)}/{hidden_size}"
                totals['total_gates'] += hidden_size
                totals['active_gates'] += num_active
            
            report_data.append(layer_stats)

    # --- Print the detailed report table (unchanged) ---
    if verbose:
        print(f"{'Layer':<7} | {'Active Attn Heads':<20} | {'Active MLP Hidden':<20} | {'Active MLP Output':<20}")
        print("-" * 80)
        for stats in report_data:
            attn_str = stats.get('attn_heads', 'N/A')
            mlp_hidden_str = stats.get('mlp_hidden', 'N/A')
            mlp_output_str = stats.get('mlp_output', 'N/A')
            print(f"{stats['layer']:<7} | {attn_str:<20} | {mlp_hidden_str:<20} | {mlp_output_str:<20}")

    # --- Print the final summary ---
    pruned_gates = totals['total_gates'] - totals['active_gates']
    pruned_params = totals['total_params'] - totals['active_params']
    
    print("-" * 80)
    print(" SUMMARY:")
    # === NEW: Report the status of the embedding gate ===
    print(f"  - Embedding Gate Status:      {embedding_gate_status}")
    # ===================================================
    print(f"  - Pruned Gates (All Types): {pruned_gates:,.0f} / {totals['total_gates']:,} ({(pruned_gates / totals['total_gates'] * 100):.1f}%)")
    if totals['total_params'] > 0:
        print(f"  - Est. Pruned Weight Params: {pruned_params:,.0f} / {totals['total_params']:,} ({(pruned_params / totals['total_params'] * 100):.1f}%)")
    print("="*80)
    
    model.set_final_circuit_mode(True)
    print("\nModel has been switched to 'Final Circuit Mode' for evaluation.")

def disable_dropout(model: nn.Module):
    """
    Recursively finds all nn.Dropout layers in a model and sets their
    dropout probability to 0.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    