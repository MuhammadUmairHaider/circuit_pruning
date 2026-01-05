import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich.layout import Layout
from rich import box

def disable_dropout(model: nn.Module):
    """
    Recursively finds all nn.Dropout layers in a model and sets their
    dropout probability to 0.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

def analyze_prunable_compression(model, layer_report_data, config):
    """
    Calculates compression based ONLY on prunable parameters (weights/biases of Attn/MLP).
    Excludes embeddings and LayerNorms.
    """
    hidden_size = config.hidden_size
    num_heads = config.n_head
    intermediate_size = config.n_inner if config.n_inner is not None else 4 * hidden_size
    num_layers = config.n_layer
    
    # --- 1. Total Prunable Params (Theoretical Max) ---
    # Attn: QKV (h->3h) + Proj (h->h) + biases
    attn_params_per_layer = (hidden_size * 3 * hidden_size + 3 * hidden_size) + \
                            (hidden_size * hidden_size + hidden_size)
    # MLP: FC (h->int) + Proj (int->h) + biases
    mlp_params_per_layer = (hidden_size * intermediate_size + intermediate_size) + \
                           (intermediate_size * hidden_size + hidden_size)
    
    total_prunable = (attn_params_per_layer + mlp_params_per_layer) * num_layers

    # --- 2. Active Prunable Params ---
    active_prunable = 0
    
    for i, report in enumerate(layer_report_data):
        if not report['layer_active']: continue
        
        block = model.transformer.h[i]
        
        # --- Attention ---
        if report.get('attn_block') == 'Active':
            # Count active neurons (head_dim * num_heads)
            if getattr(block.attn, 'neuron_gates', None) is not None:
                active_attn_neurons = (block.attn.neuron_gates() > 0.5).sum().item()
            else:
                active_attn_neurons = hidden_size

            # QKV weights depend on active neurons
            # Proj weights depend on active neurons
            layer_attn_params = (hidden_size * 3 * active_attn_neurons + 3 * active_attn_neurons) + \
                                (active_attn_neurons * hidden_size + hidden_size)
            active_prunable += layer_attn_params

        # --- MLP ---
        if report.get('mlp_block') == 'Active':
            # Hidden Neurons
            if getattr(block.mlp, 'hidden_gates', None) is not None:
                active_hidden = (block.mlp.hidden_gates() > 0.5).sum().item()
            else:
                active_hidden = intermediate_size
            
            # Output Neurons
            if getattr(block.mlp, 'output_gates', None) is not None:
                active_output = (block.mlp.output_gates() > 0.5).sum().item()
            else:
                active_output = hidden_size

            # FC: h -> hidden
            # Proj: hidden -> output
            layer_mlp_params = (hidden_size * active_hidden + active_hidden) + \
                               (active_hidden * active_output + active_output)
            active_prunable += layer_mlp_params

    # Stats
    compression = total_prunable / active_prunable if active_prunable > 0 else float('inf')
    reduction = (1 - active_prunable / total_prunable) * 100

    return {
        'total': total_prunable,
        'active': active_prunable,
        'compression': compression,
        'reduction': reduction
    }

def analyze_and_finalize_circuit(model: nn.Module, console: Console = None):
    """
    Analyzes model, enforces strict hierarchy, and renders a beautiful report via Rich.
    """
    if console is None:
        console = Console()

    # console.print(Panel("[bold yellow]Finalizing Circuit & Enforcing Hierarchy...[/]", border_style="yellow"))
    
    model.eval()
    model.set_final_circuit_mode(True)

    config = model.config
    num_layers = config.n_layer
    
    # Track stats
    stats = {
        'layer_level': {'total': num_layers, 'active': 0},
        'blocks': {'attn': 0, 'mlp': 0},
        'elements': {'heads': 0, 'neurons': 0, 'mlp_hidden': 0, 'mlp_output': 0}
    }
    
    layer_report = []
    
    # ==========================================================================
    # 1. HIERARCHICAL ENFORCEMENT & DATA GATHERING
    # ==========================================================================
    with torch.no_grad():
        # Determine Layer Status
        layer_active_mask = [True] * num_layers
        if getattr(model, 'layer_gates', None) is not None:
            for i, lg in enumerate(model.layer_gates):
                if (lg() < 0.5).item(): layer_active_mask[i] = False
        
        stats['layer_level']['active'] = sum(layer_active_mask)

        for i, block in enumerate(model.transformer.h):
            l_stat = {'id': i, 'active': layer_active_mask[i]}
            
            # Helper to kill a gate
            def kill(gate):
                if gate is not None: gate.log_alpha.data.fill_(-1e6)

            # --- Logic: If layer pruned, kill children ---
            if not layer_active_mask[i]:
                kill(getattr(block, 'attention_block_gate', None))
                kill(getattr(block, 'mlp_block_gate', None))
                kill(getattr(block.attn, 'head_gates', None))
                kill(getattr(block.attn, 'neuron_gates', None))
                kill(getattr(block.mlp, 'hidden_gates', None))
                kill(getattr(block.mlp, 'output_gates', None))
                layer_report.append(l_stat)
                continue

            # --- Logic: Top-Down (Block -> Components) ---
            # Attn Block
            attn_gate = getattr(block, 'attention_block_gate', None)
            attn_active = True
            if attn_gate is not None and (attn_gate() < 0.5).item():
                attn_active = False
                kill(getattr(block.attn, 'head_gates', None))
                kill(getattr(block.attn, 'neuron_gates', None))
            
            # MLP Block
            mlp_gate = getattr(block, 'mlp_block_gate', None)
            mlp_active = True
            if mlp_gate is not None and (mlp_gate() < 0.5).item():
                mlp_active = False
                kill(getattr(block.mlp, 'hidden_gates', None))
                kill(getattr(block.mlp, 'output_gates', None))

            # --- Logic: Bottom-Up (Components -> Block) ---
            # If all heads/neurons dead -> Kill Block
            head_gates = getattr(block.attn, 'head_gates', None)
            if head_gates is not None and (head_gates() < 0.5).all().item():
                 attn_active = False
                 kill(attn_gate)
            
            # --- Gather Stats for Report ---
            l_stat['attn_block'] = "Active" if attn_active else "Pruned"
            l_stat['mlp_block'] = "Active" if mlp_active else "Pruned"
            
            if attn_active: stats['blocks']['attn'] += 1
            if mlp_active: stats['blocks']['mlp'] += 1

            # Heads & Neurons
            if attn_active and head_gates is not None:
                n_heads = (head_gates() > 0.5).sum().item()
                l_stat['heads'] = int(n_heads)
                stats['elements']['heads'] += int(n_heads)
            else:
                l_stat['heads'] = 0 if not attn_active else config.n_head

            # MLP Neurons
            hidden_gates = getattr(block.mlp, 'hidden_gates', None)
            if mlp_active and hidden_gates is not None:
                n_hid = (hidden_gates() > 0.5).sum().item()
                l_stat['mlp_hid'] = int(n_hid)
                stats['elements']['mlp_hidden'] += int(n_hid)
            else:
                l_stat['mlp_hid'] = 0 if not mlp_active else (config.n_inner or 4*config.hidden_size)

            layer_report.append(l_stat)
            
            # Safety: If both blocks died during processing, kill the layer for next time
            if not attn_active and not mlp_active and getattr(model, 'layer_gates', None) is not None:
                 kill(model.layer_gates[i])
                 l_stat['active'] = False # Visual update only

    # ==========================================================================
    # 2. VISUALIZATION (THE PRETTY PART)
    # ==========================================================================
    
    # A. Compression Panel
    comp_stats = analyze_prunable_compression(model, [{'layer_active': x['active'], 
                                                       'attn_block': x.get('attn_block'), 
                                                       'mlp_block': x.get('mlp_block')} for x in layer_report], config)
    
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="center", ratio=1)
    
    grid.add_row(
        Panel(f"[bold white]{comp_stats['active']:,}[/]", title="Active Params", border_style="green"),
        Panel(f"[bold white]{comp_stats['total']:,}[/]", title="Original Params", border_style="grey50"),
        Panel(f"[bold cyan]{comp_stats['compression']:.2f}x[/]", title="Compression Ratio", border_style="cyan")
    )
    
    console.print(Panel(grid, title="[bold magenta]Circuit Finalization & Analysis[/]", border_style="magenta"))

    # B. Detailed Layer Table
    table = Table(title="Layer-wise Circuit Architecture", box=box.ROUNDED, expand=True)
    table.add_column("Lyr", justify="right", style="cyan", no_wrap=True)
    table.add_column("State", justify="center")
    table.add_column("Attn Block", justify="center")
    table.add_column("MLP Block", justify="center")
    table.add_column("Active Heads", justify="right")
    table.add_column("MLP Neurons", justify="right")

    for row in layer_report:
        if not row['active']:
            table.add_row(
                str(row['id']),
                "[bold red]PRUNED[/]",
                "[dim]---[/]", "[dim]---[/]", "[dim]-[/]", "[dim]-[/]",
                style="dim"
            )
        else:
            # Attn Status
            attn_style = "[bold green]Active[/]" if row['attn_block'] == "Active" else "[dim red]Pruned[/]"
            heads_val = f"{row['heads']}" if row['attn_block'] == "Active" else "[dim]-[/]"
            
            # MLP Status
            mlp_style = "[bold green]Active[/]" if row['mlp_block'] == "Active" else "[dim red]Pruned[/]"
            mlp_val = f"{row['mlp_hid']}" if row['mlp_block'] == "Active" else "[dim]-[/]"
            
            table.add_row(
                str(row['id']),
                "[bold green]Active[/]",
                attn_style,
                mlp_style,
                heads_val,
                mlp_val
            )

    console.print(table)
    
    # C. Final Footer
    footer = Text.assemble(
        (" Model is now in ", "dim"),
        ("Final Circuit Mode", "bold yellow"),
        (" (Binary Gates Enforced)", "dim")
    )
    console.print(Panel(footer, border_style="yellow"))