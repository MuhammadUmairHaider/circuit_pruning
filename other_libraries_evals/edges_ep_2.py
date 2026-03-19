import json
import re
import argparse
import os
from collections import defaultdict

# Try to import graphviz
try:
    from graphviz import Digraph
except ImportError:
    Digraph = None

def analyze_circuit_strict(edges_data, num_layers=12, num_heads=12):
    """
    Analyzes an edge list with STRICT Value-Flow and Blind Node Pruning logic.
    
    Logic Parity with DiscoGP Script:
    1. No Weight Pruning (assumed all hardware is functional).
    2. Reachability: Only traces paths through Value (.v) or MLP inputs. Q/K edges are ignored for flow.
    3. Ghost Check: Iteratively removes nodes that have no valid inputs (parents).
    """

    # --- 1. PARSE EDGES & BUILD GRAPH ---
    # We need a map of Children -> Parents to check flow.
    # Node names are simplified to "a{L}.h{H}" or "m{L}" or "embed" or "output"
    
    # Regex to capture components. 
    # Matches: a10.h7.q, a10.h7, m5, tok_embeds, resid_post
    node_pattern = re.compile(r"(a(\d+)\.h(\d+)|m(\d+)|tok_embeds|resid_post)")
    
    parents_map = defaultdict(set)
    all_nodes = set()

    for source, dest in edges_data:
        # 1. Clean names (remove .q, .k, .v suffix for identification)
        src_clean = re.sub(r'\.[qkv]$', '', source)
        dst_clean = re.sub(r'\.[qkv]$', '', dest)
        
        all_nodes.add(src_clean)
        all_nodes.add(dst_clean)

        # 2. STRICT VALUE CHECK
        # If the edge points to a Query (.q) or Key (.k), we IGNORE it for the 
        # purposes of graph "liveness". Information flow requires writing to Value.
        # (MLP and Resid_Post edges are always valid flow).
        if ".q" in dest or ".k" in dest:
            continue
            
        parents_map[dst_clean].add(src_clean)

    # --- 2. REVERSE BFS (Flow Check) ---
    # Start from output, find everything that eventually writes to it.
    
    reachable_nodes = set()
    queue = ['resid_post']
    
    # Simple BFS
    while queue:
        child = queue.pop(0)
        # If we haven't processed this node yet
        if child not in reachable_nodes:
            reachable_nodes.add(child)
            # Add its parents to the queue
            for parent in parents_map.get(child, []):
                queue.append(parent)

    print(f"[Step 1] Nodes reachable from output (Value/MLP paths only): {len(reachable_nodes)}")

    # --- 3. ITERATIVE GHOST PRUNING (Blind Node Check) ---
    # A node might be reachable from output, but have NO valid inputs itself.
    # If we remove it, its children might now become blind. Repeat until stable.
    
    while True:
        nodes_to_remove = set()
        
        for node in reachable_nodes:
            # Base cases: Embeddings/Input are never blind
            if node == 'tok_embeds':
                continue
                
            # 'resid_post' is not a computing node, just a sink, so we check its parents normally
            
            current_parents = parents_map.get(node, set())
            
            # Filter parents: keep only those that are STILL in reachable_nodes
            active_parents = [p for p in current_parents if p in reachable_nodes]
            
            # If a node has 0 active parents, it is a Ghost/Blind node
            if len(active_parents) == 0:
                nodes_to_remove.add(node)
        
        if not nodes_to_remove:
            break
            
        # Prune
        reachable_nodes -= nodes_to_remove
        # print(f"   -> Pruned {len(nodes_to_remove)} blind nodes (e.g., {list(nodes_to_remove)[:3]})")

    print(f"[Step 2] Final Active Nodes after pruning blind spots: {len(reachable_nodes)}")


    # --- 4. COMPILE STATISTICS ---
    active_heads_by_layer = defaultdict(set)
    active_mlps = set()
    
    for node in reachable_nodes:
        # Check if it's a head
        match_head = re.match(r"a(\d+)\.h(\d+)", node)
        if match_head:
            layer, head = int(match_head.group(1)), int(match_head.group(2))
            active_heads_by_layer[layer].add(head)
            continue
            
        # Check if it's an MLP
        match_mlp = re.match(r"m(\d+)", node)
        if match_mlp:
            layer = int(match_mlp.group(1))
            active_mlps.add(layer)

    # Calculate Stats
    total_possible_heads = num_layers * num_heads
    total_possible_mlps = num_layers
    total_possible_components = total_possible_heads + total_possible_mlps

    total_active_heads = sum(len(h) for h in active_heads_by_layer.values())
    total_active_mlps = len(active_mlps)
    total_active_components = total_active_heads + total_active_mlps
    
    component_sparsity = (1 - total_active_components / total_possible_components) * 100

    return {
        "active_heads_by_layer": {k: sorted(list(v)) for k, v in active_heads_by_layer.items()},
        "active_mlps": sorted(list(active_mlps)),
        "active_nodes_set": reachable_nodes, # For visualization filter
        "stats": {
            "total_active_heads": total_active_heads,
            "total_active_mlps": total_active_mlps,
            "total_active_components": total_active_components,
            "component_sparsity_percent": component_sparsity,
            "total_possible_components": total_possible_components,
        }
    }

def print_summary(analysis_results, num_layers=12, num_heads=12):
    print("--- Edge Circuit Summary (Strict Value Flow) ---")
    header = f"| {'Layer':^5} | {'Status':^8} | {'Attention Heads Active':^25} | {'MLP Block Active':^18} |"
    print(header)
    print(f"|{'-'*7}|{'-'*10}|{'-'*27}|{'-'*20}|")

    for i in range(num_layers):
        heads = analysis_results["active_heads_by_layer"].get(i, [])
        mlp_active = i in analysis_results["active_mlps"]
        
        status = "Active" if heads or mlp_active else "PRUNED"
        heads_str = f"{len(heads)} / {num_heads}" if status == "Active" else f"0 / {num_heads}"
        mlp_str = "✅" if mlp_active else "❌"
        
        row = f"| {i:^5} | {status:^8} | {heads_str:^25} | {mlp_str:^18} |"
        print(row)

    print("\n--- Statistics ---")
    stats = analysis_results['stats']
    print(f"Attention Heads: {stats['total_active_heads']} / {num_layers * num_heads}")
    print(f"MLP Blocks:      {stats['total_active_mlps']} / {num_layers}")
    print(f"Sparsity:        {stats['component_sparsity_percent']:.2f}%")

def visualize_circuit(edges_data, analysis_results, output_filename="circuit_graph"):
    if not Digraph:
        print("\nSkipping visualization (Graphviz not found).")
        return

    dot = Digraph(comment='Transformer Circuit')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.3', ranksep='1.0')
    
    # Styles
    node_styles = {
        "embed": {"shape": "box", "style": "filled", "fillcolor": "#fde7b0"},
        "post": {"shape": "box", "style": "filled", "fillcolor": "#e0bbe4"},
        "mlp": {"shape": "ellipse", "style": "filled", "fillcolor": "#d8e2dc"},
        "head": {"shape": "box", "style": "rounded,filled", "fillcolor": "#a2d2ff"},
    }

    # Only visualize nodes that survived the Strict Checks
    valid_nodes = analysis_results['active_nodes_set']

    dot.node('tok_embeds', 'Embeddings', **node_styles['embed'])
    dot.node('resid_post', 'Final Output', **node_styles['post'])

    # Add Nodes
    active_layers = sorted(list(set(analysis_results['active_mlps']) | set(analysis_results['active_heads_by_layer'].keys())))
    
    for layer in active_layers:
        with dot.subgraph(name=f'cluster_{layer}') as c:
            c.attr(label=f'Layer {layer}', style='rounded', color='lightgrey')
            
            # MLP
            mlp_node = f'm{layer}'
            if mlp_node in valid_nodes:
                c.node(mlp_node, f'MLP {layer}', **node_styles['mlp'])
            
            # Heads
            for head in analysis_results['active_heads_by_layer'].get(layer, []):
                head_node = f'a{layer}.h{head}'
                if head_node in valid_nodes:
                    c.node(head_node, f'H {layer}.{head}', **node_styles['head'])

    # Add Edges (Only if BOTH source and dest are valid)
    for source, dest in edges_data:
        src_clean = re.sub(r'\.[qkv]$', '', source)
        dst_clean = re.sub(r'\.[qkv]$', '', dest)
        
        # 1. Check if nodes survived
        if src_clean in valid_nodes and dst_clean in valid_nodes:
            # 2. Strict Visual: Differentiate Q/K/V edges visually?
            # Or just show them. Let's just show them, but dotted if they are Q/K
            # to indicate they didn't contribute to "liveness"
            
            style = 'solid'
            color = 'black'
            if '.q' in dest or '.k' in dest:
                style = 'dashed'
                color = 'gray'
            
            dot.edge(src_clean, dst_clean, style=style, color=color)

    try:
        dot.render(output_filename, format='png', view=False, cleanup=True)
        print(f"\n✅ Graph generated: {output_filename}.png")
    except Exception as e:
        print(f"Error generating graph: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=str, help="Path to JSON edges.")
    parser.add_argument("-o", "--output", type=str, default="circuit_strict", help="Output filename.")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: {args.json_file} not found.")
        return

    with open(args.json_file, 'r') as f:
        edges = json.load(f)
    
    analysis = analyze_circuit_strict(edges, num_layers=args.num_layers, num_heads=args.num_heads)
    print_summary(analysis, num_layers=args.num_layers, num_heads=args.num_heads)
    visualize_circuit(edges, analysis, args.output)

if __name__ == "__main__":
    main()