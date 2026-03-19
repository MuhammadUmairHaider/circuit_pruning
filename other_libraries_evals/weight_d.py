import re

def calculate_discogp_density(report_text):
    # --- GPT-2 Small Architecture Constants ---
    N_LAYERS = 12
    D_MODEL = 768
    N_HEADS = 12
    HEAD_DIM = 64  # 768 / 12
    D_MLP_HIDDEN = 3072
    
    # --- 1. Define the "Full" Baseline Weights ---
    # According to DiscoGPTransformer.py, 'emb', 'unembed', and 'ln' are EXCLUDED.
    # We only count the weights inside the blocks.
    
    # Attention: 
    #   W_Q, W_K, W_V: Input (768) -> Output (12 * 64) each.
    #   W_Proj:        Input (12 * 64) -> Output (768).
    #   (Ignoring biases for density as they are negligible compared to matrices)
    full_attn_params = (D_MODEL * (N_HEADS * HEAD_DIM) * 3) + ((N_HEADS * HEAD_DIM) * D_MODEL)
    
    # MLP:
    #   W_In (Up):     Input (768) -> Hidden (3072)
    #   W_Out (Down):  Hidden (3072) -> Output (768)
    full_mlp_params = (D_MODEL * D_MLP_HIDDEN) + (D_MLP_HIDDEN * D_MODEL)
    
    # Total denominator for the model
    total_model_weights = N_LAYERS * (full_attn_params + full_mlp_params)

    active_weights = 0
    
    print(f"{'Layer':<5} | {'Attn Density':<15} | {'MLP Density':<15} | {'Layer Active Params':<20}")
    print("-" * 70)

    lines = report_text.strip().split('\n')
    
    for line in lines:
        if "Layer" in line or "---" in line or not line.strip():
            continue
            
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 8: continue

        # --- Parse Report Data ---
        layer_idx = int(parts[0])
        attn_status = parts[2]
        mlp_status = parts[3]
        
        # Format: "Active/Total" -> grab Active
        n_heads_active = int(parts[4].split('/')[0])
        n_attn_neurons_active = int(parts[5].split('/')[0]) # Affects W_Proj columns
        n_mlp_hidden_active = int(parts[6].split('/')[0])   # Affects W_In cols and W_Out rows
        n_mlp_output_active = int(parts[7].split('/')[0])   # Affects W_Out columns

        # --- Calculate Attention Active Weights ---
        # Logic matches DiscoGP: pruning a Head removes it from QKV and Proj.
        # Pruning an "Attn Neuron" removes that column from Proj.
        curr_attn_weights = 0
        if attn_status == "Active":
            # 1. W_QKV (3 matrices)
            # Input is Dense (768) because it comes from Residual Stream
            # Output depends on Active Heads
            w_qkv_active = D_MODEL * (n_heads_active * HEAD_DIM) * 3
            
            # 2. W_Proj
            # Input depends on Active Heads (Rows)
            # Output depends on Active Attn Neurons (Cols)
            w_proj_active = (n_heads_active * HEAD_DIM) * n_attn_neurons_active
            
            curr_attn_weights = w_qkv_active + w_proj_active

        # --- Calculate MLP Active Weights ---
        curr_mlp_weights = 0
        if mlp_status == "Active":
            # 1. W_In (Up Projection)
            # Input is Dense (768) because it comes from Residual Stream
            # Output depends on Active Hidden Neurons (Cols)
            w_in_active = D_MODEL * n_mlp_hidden_active
            
            # 2. W_Out (Down Projection)
            # Input depends on Active Hidden Neurons (Rows)
            # Output depends on Active MLP Output Neurons (Cols)
            w_out_active = n_mlp_hidden_active * n_mlp_output_active
            
            curr_mlp_weights = w_in_active + w_out_active

        # Sum Layer
        layer_total = curr_attn_weights + curr_mlp_weights
        active_weights += layer_total
        
        # Stats per layer
        print(f"{layer_idx:<5} | {curr_attn_weights/full_attn_params:>7.1%} ({attn_status[0]})   | {curr_mlp_weights/full_mlp_params:>7.1%} ({mlp_status[0]})   | {layer_total:,}")

    # --- Final Result ---
    density_pct = (active_weights / total_model_weights) * 100
    
    print("-" * 70)
    print(f"Total Model Weights (Baseline): {total_model_weights:,}")
    print(f"Active Weights (Calculated):    {active_weights:,}")
    print(f"Final Weight Density:           {density_pct:.4f}%")

# --- Paste your report below ---
report_data = """
Layer  | Status   | Attn Block  | MLP Block  | Attn Heads   | Attn Neurons    | MLP Hidden      | MLP Output      
-----------------------------------------------------------------------------------------------------------------
0      | Active   | Active      | Active     | 9/12         | 299/768         | 2120/3072       | 732/768        
1      | Active   | Pruned      | Active     | 0/12         | 0/768           | 1588/3072       | 635/768        
2      | Active   | Pruned      | Pruned     | 0/12         | 0/768           | 0/3072          | 0/768          
3      | Active   | Active      | Pruned     | 6/12         | 272/768         | 0/3072          | 0/768          
4      | Active   | Pruned      | Pruned     | 0/12         | 0/768           | 0/3072          | 0/768          
5      | Active   | Active      | Active     | 3/12         | 138/768         | 1513/3072       | 604/768        
6      | Active   | Pruned      | Active     | 0/12         | 0/768           | 1495/3072       | 595/768        
7      | Active   | Active      | Pruned     | 5/12         | 247/768         | 0/3072          | 0/768          
8      | Active   | Pruned      | Active     | 0/12         | 0/768           | 1454/3072       | 562/768        
9      | Active   | Active      | Pruned     | 4/12         | 189/768         | 0/3072          | 0/768          
10     | Active   | Active      | Pruned     | 7/12         | 276/768         | 0/3072          | 0/768          
11     | Active   | Pruned      | Active     | 0/12         | 0/768           | 1565/3072       | 571/768        
"""

if __name__ == "__main__":
    calculate_discogp_density(report_data)