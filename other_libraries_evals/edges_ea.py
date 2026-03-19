import json
import re
from collections import defaultdict
import argparse
import os

# Try to import graphviz, and provide a helpful error message if it's not installed.
try:
    from graphviz import Digraph
except ImportError:
    print("Graphviz library not found. Please install it to generate visualizations:")
    print("pip install graphviz")
    print("You may also need to install the Graphviz system package:")
    print("  - On Debian/Ubuntu: sudo apt-get install graphviz")
    print("  - On MacOS (Homebrew): brew install graphviz")
    Digraph = None

def normalize_node_name(node_str):
    """
    Convert node names from the new format to the old format.
    Examples:
        'head.10.9' -> 'a10.h9'
        'head.4.3.v' -> 'a4.h3.v'
        'mlp.7' -> 'm7'
        'resid_post.11' -> 'resid_post'
    """
    # Handle resid_post
    if node_str.startswith('resid_post'):
        return 'resid_post'
    
    # Handle MLP: mlp.7 -> m7
    if node_str.startswith('mlp.'):
        layer = node_str.split('.')[1]
        return f'm{layer}'
    
    # Handle attention heads: head.10.9.v -> a10.h9.v or head.10.9 -> a10.h9
    if node_str.startswith('head.'):
        parts = node_str.split('.')
        layer = parts[1]
        head = parts[2]
        suffix = f'.{parts[3]}' if len(parts) > 3 else ''
        return f'a{layer}.h{head}{suffix}'
    
    return node_str

def analyze_circuit(edges_data, num_layers=12, num_heads=12):
    """
    Analyzes a list of circuit edges to determine active components and calculate sparsity.

    Args:
        edges_data (list): A list of edge dictionaries with 'from', 'to', and 'score' fields.
        num_layers (int): The total number of layers in the base model.
        num_heads (int): The number of attention heads per layer in the base model.

    Returns:
        dict: A dictionary containing the analysis results, including active components and stats.
    """
    active_heads_by_layer = defaultdict(set)
    active_mlps = set()

    # Regex to parse component names like 'a10.h7.q' or 'm5'
    node_pattern = re.compile(r"(a(\d+)\.h(\d+)|m(\d+))")
    sparsity = 0.9372
    num_edges = int(32046*(1-sparsity))
    edges_data = edges_data[:num_edges]

    for edge in edges_data:
        # Get source and destination from the new format
        source = normalize_node_name(edge['from'])
        dest = normalize_node_name(edge['to'])
        
        for node_str in [source, dest]:
            match = node_pattern.match(node_str)
            if match:
                if node_str.startswith('a'):
                    layer, head = int(match.group(2)), int(match.group(3))
                    active_heads_by_layer[layer].add(head)
                elif node_str.startswith('m'):
                    layer = int(match.group(4))
                    active_mlps.add(layer)

    # --- Calculate Statistics ---
    total_possible_heads = num_layers * num_heads
    total_possible_mlps = num_layers
    total_possible_components = total_possible_heads + total_possible_mlps

    total_active_heads = sum(len(heads) for heads in active_heads_by_layer.values())
    total_active_mlps = len(active_mlps)
    total_active_components = total_active_heads + total_active_mlps
    
    component_sparsity = (1 - total_active_components / total_possible_components) * 100

    return {
        "active_heads_by_layer": {k: sorted(list(v)) for k, v in active_heads_by_layer.items()},
        "active_mlps": sorted(list(active_mlps)),
        "stats": {
            "total_active_heads": total_active_heads,
            "total_active_mlps": total_active_mlps,
            "total_active_components": total_active_components,
            "component_sparsity_percent": component_sparsity,
            "total_possible_components": total_possible_components,
        }
    }

def print_summary(analysis_results, num_layers=12, num_heads=12):
    """Prints a formatted summary of the circuit analysis."""
    print("--- Pruning Summary ---")
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

    print("\n--- Component Sparsity Calculation ---")
    stats = analysis_results['stats']
    print(f"Attention Heads: {stats['total_active_heads']} / {num_layers * num_heads} active")
    print(f"MLP Blocks:      {stats['total_active_mlps']} / {num_layers} active")
    print("-" * 35)
    print(f"Total Active Components: {stats['total_active_components']} / {stats['total_possible_components']}")
    print(f"Component Sparsity:      {stats['component_sparsity_percent']:.2f}%")


def visualize_circuit(edges_data, analysis_results, output_filename="circuit_graph"):
    """Generates a visual graph of the circuit using Graphviz."""
    if not Digraph:
        print("\nSkipping visualization because Graphviz is not installed.")
        return

    dot = Digraph(comment='Transformer Circuit')
    dot.attr(rankdir='TB', splines='ortho', ranksep='1.5', nodesep='0.3')

    # --- Define Styles ---
    node_styles = {
        "embed": {"shape": "box", "style": "filled", "fillcolor": "#fde7b0"},
        "post": {"shape": "box", "style": "filled", "fillcolor": "#e0bbe4"},
        "mlp": {"shape": "ellipse", "style": "filled", "fillcolor": "#d8e2dc"},
        "head": {"shape": "box", "style": "rounded,filled", "fillcolor": "#a2d2ff"},
    }

    # --- Add Nodes ---
    dot.node('tok_embeds', 'Embeddings', **node_styles['embed'])
    dot.node('resid_post', 'Final Output', **node_styles['post'])

    active_layers = sorted(list(set(analysis_results['active_mlps']) | set(analysis_results['active_heads_by_layer'].keys())))

    for layer in active_layers:
        # Create a subgraph for each layer to group nodes visually
        with dot.subgraph(name=f'cluster_{layer}') as c:
            c.attr(label=f'Layer {layer}', style='rounded', color='lightgrey')
            if layer in analysis_results['active_mlps']:
                c.node(f'm{layer}', f'MLP {layer}', **node_styles['mlp'])
            for head in analysis_results['active_heads_by_layer'].get(layer, []):
                c.node(f'a{layer}.h{head}', f'Head {layer}.{head}', **node_styles['head'])

    # --- Add Edges ---
    for edge in edges_data:
        source = normalize_node_name(edge['from'])
        dest = normalize_node_name(edge['to'])
        
        # Simplify node names for graphing (e.g., a0.h1.q -> a0.h1)
        source_simple = re.sub(r'\.[qkv]$', '', source)
        dest_simple = re.sub(r'\.[qkv]$', '', dest)
        dot.edge(source_simple, dest_simple)

    # --- Render Graph ---
    try:
        dot.render(output_filename, format='png', view=False, cleanup=True)
        print(f"\n✅ Successfully generated graph: {output_filename}.png")
    except Exception as e:
        print(f"\n❌ Failed to generate graph. Is Graphviz installed correctly?")
        print(f"   Error: {e}")


def main():
    """Main function to run the analysis from the command line."""
    parser = argparse.ArgumentParser(description="Analyze and visualize a transformer circuit from a JSON file of edges.")
    parser.add_argument("json_file", type=str, nargs='?', default="edges_eap/gp-200.json", help="Path to the JSON file containing the circuit edges.")
    parser.add_argument("-o", "--output", type=str, default="circuit_graph", help="Output filename for the graph visualization (without extension).")
    # Add arguments for model architecture to make the script more flexible
    parser.add_argument("--num-layers", type=int, default=12, help="Total number of layers in the model.")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads per layer.")
    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: File not found at '{args.json_file}'")
        return

    print(f"Loading edges from {args.json_file}...")
    with open(args.json_file, 'r') as f:
        edges = json.load(f)
    
    analysis = analyze_circuit(edges, num_layers=args.num_layers, num_heads=args.num_heads)
    print_summary(analysis, num_layers=args.num_layers, num_heads=args.num_heads)
    visualize_circuit(edges, analysis, args.output)


if __name__ == "__main__":
    # To run this script, save it as a Python file (e.g., analyze.py) and execute from your terminal:
    # python analyze.py your_edges_file.json
    #
    # Example:
    # If your edge data is in a file named `my_circuit.json`, you would run:
    # python analyze.py my_circuit.json -o my_circuit_visualization
    main()