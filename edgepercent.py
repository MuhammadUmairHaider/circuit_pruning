import re

def parse_summary_table(summary_table: str) -> dict:
    """Parses the string-based summary table into a structured dictionary."""
    pruning_summary = {}
    lines = summary_table.strip().split('\n')
    
    # Skip header and separator lines
    for line in lines[2:]:
        parts = [p.strip() for p in line.split('|')]
        if not parts or not parts[0].isdigit():
            continue
            
        layer_idx = int(parts[0])
        mlp_status = parts[3]
        attn_heads_str = parts[4]
        
        # Check if MLP block is active
        is_mlp_active = (mlp_status == 'Active')
        
        # Parse 'X/12' format for attention heads
        match = re.match(r'(\d+)/\d+', attn_heads_str)
        if match:
            num_heads = int(match.group(1))
        else:
            num_heads = 0
            
        pruning_summary[layer_idx] = {
            'heads': num_heads,
            'mlp_active': is_mlp_active
        }
    return pruning_summary

def calculate_pruned_edges(pruning_summary: dict):
    """
    Calculates remaining edges based on the parsed summary.
    """
    NUM_LAYERS = 12
    NUM_HEADS = 12

    # --- 1. Calculate stats for the full, unpruned model ---
    total_components_per_layer = NUM_HEADS + 1
    full_output_edges = NUM_LAYERS * total_components_per_layer
    
    full_inter_layer_edges = 0
    full_mlp_edges = 0
    full_qkv_edges = 0
    for j in range(1, NUM_LAYERS):
        num_sources = j * total_components_per_layer
        full_mlp_edges += num_sources
        full_qkv_edges += NUM_HEADS * 3 * num_sources
    
    total_full_original_edges = full_output_edges + full_mlp_edges + full_qkv_edges
    total_full_extra_edges = 1 + NUM_LAYERS + (NUM_LAYERS * NUM_HEADS * 6)
    total_full_edges = total_full_original_edges + total_full_extra_edges

    # --- 2. Calculate stats for the pruned model ---
    active_head_counts = {layer: data['heads'] for layer, data in pruning_summary.items()}
    active_mlp_layers = {layer for layer, data in pruning_summary.items() if data['mlp_active']}

    # Calculate active source components before each layer
    num_active_sources_before_layer = {}
    current_source_count = 0
    for i in range(NUM_LAYERS):
        num_active_sources_before_layer[i] = current_source_count
        if i in pruning_summary:
            if pruning_summary[i]['mlp_active']:
                current_source_count += 1
            current_source_count += pruning_summary[i]['heads']

    # Calculate remaining edges by category
    rem_output_mlp = len(active_mlp_layers)
    rem_output_heads = sum(active_head_counts.values())
    rem_output_edges = rem_output_mlp + rem_output_heads

    rem_mlp_edges, rem_q_edges, rem_k_edges, rem_v_edges = 0, 0, 0, 0
    for j in range(NUM_LAYERS):
        num_sources = num_active_sources_before_layer[j]
        num_active_heads_j = active_head_counts.get(j, 0)
        
        if j in active_mlp_layers:
            rem_mlp_edges += num_sources
        
        rem_q_edges += num_active_heads_j * num_sources
        rem_k_edges += num_active_heads_j * num_sources
        rem_v_edges += num_active_heads_j * num_sources

    total_rem_original_edges = rem_output_edges + rem_mlp_edges + rem_q_edges + rem_k_edges + rem_v_edges

    # Extra (Internal) Edges
    rem_extra_edges = 1
    rem_extra_edges += len(active_mlp_layers)
    rem_extra_edges += sum(active_head_counts.values()) * 6

    total_rem_edges = total_rem_original_edges + rem_extra_edges

    # --- 3. Format and print results ---
    print("--- Pruned Model Edge Count ---")
    print(f"Total Remaining Edges: {total_rem_edges} / {total_full_edges} ({total_rem_edges / total_full_edges:.2%})")
    print("-" * 30)
    print(f"Original Edges: {total_rem_original_edges} / {total_full_original_edges} ({total_rem_original_edges / total_full_original_edges:.2%})")
    print(f"  ➞ To Final Output: {rem_output_edges} / {full_output_edges}")
    print(f"  ➞ To MLP Blocks:   {rem_mlp_edges} / {full_mlp_edges}")
    print(f"  ➞ To Q Projections: {rem_q_edges} / {full_qkv_edges // 3}")
    print(f"  ➞ To K Projections: {rem_k_edges} / {full_qkv_edges // 3}")
    print(f"  ➞ To V Projections: {rem_v_edges} / {full_qkv_edges // 3}")
    print("-" * 30)
    print(f"Extra (Internal) Edges: {rem_extra_edges} / {total_full_extra_edges} ({rem_extra_edges / total_full_extra_edges:.2%})")

# Paste your summary table here
summary_table = """
Layer  | Status   | Attn Block  | MLP Block  | Attn Heads   | Attn Neurons    | MLP Hidden      | MLP Output     
-----------------------------------------------------------------------------------------------------------------
0      | Active   | Active      | Active     | 5/12         | 198/768         | 2209/3072       | 702/768        
1      | Active   | Pruned      | Pruned     | 0/12         | 0/768           | 0/3072          | 0/768          
2      | Active   | Pruned      | Pruned     | 0/12         | 0/768           | 0/3072          | 0/768          
3      | Active   | Pruned      | Pruned     | 0/12         | 0/768           | 0/3072          | 0/768          
4      | Active   | Active      | Pruned     | 12/12        | 560/768         | 0/3072          | 0/768          
5      | Active   | Pruned      | Pruned     | 0/12         | 0/768           | 0/3072          | 0/768          
6      | Active   | Active      | Pruned     | 3/12         | 142/768         | 0/3072          | 0/768          
7      | Active   | Pruned      | Active     | 0/12         | 0/768           | 1454/3072       | 698/768        
8      | Active   | Pruned      | Pruned     | 0/12         | 0/768           | 0/3072          | 0/768          
9      | Active   | Active      | Pruned     | 8/12         | 237/768         | 0/3072          | 0/768          
10     | Active   | Active      | Pruned     | 12/12        | 380/768         | 0/3072          | 0/768          
11     | Active   | Active      | Active     | 5/12         | 141/768         | 805/3072        | 596/768   
"""

# Parse the table and run the calculation
parsed_summary = parse_summary_table(summary_table)
calculate_pruned_edges(parsed_summary)