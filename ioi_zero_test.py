import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
import time
from dataclasses import dataclass
from collections import deque
import io
import contextlib

# --- TUI Imports ---
import plotext as plt
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.ansi import AnsiDecoder
from rich.text import Text

# --- Custom Modules ---
from models.gpt2_zero import PrunableGPT2LMHeadModel as CircuitDiscoveryGPT2
from dataset.ioi_t import IOIDataset, load_or_generate_ioi_data, run_evaluation, filter_dataset_by_model_correctness
from utils_cli import disable_dropout, analyze_and_finalize_circuit

# ==============================================================================
# PRUNING CONFIGURATION
# ==============================================================================
PRUNING_FACTOR = 15.0

@dataclass
class PruningConfig:
    init_value: float = 1.0
    sparsity_warmup_steps: int = 50
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 2.0 * PRUNING_FACTOR
    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 10 * PRUNING_FACTOR
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 10 * PRUNING_FACTOR
    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 10 * PRUNING_FACTOR
    prune_embedding: bool = False
    lambda_embedding: float = 1 * PRUNING_FACTOR
    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 1.0 * PRUNING_FACTOR
    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 1.0 * PRUNING_FACTOR
    prune_full_layers: bool = False
    lambda_full_layers: float = 0.000000005 * PRUNING_FACTOR

# ==============================================================================
# TUI HELPER CLASSES
# ==============================================================================
class RichGraph:
    """Widget that renders a plotext graph into a Rich renderable."""
    def __init__(self):
        self.decoder = AnsiDecoder()

    def __rich_console__(self, console, options):
        self.width = options.max_width or console.width
        self.height = options.height or console.height
        
        # Configure plotext
        plt.plotsize(self.width, self.height)
        plt.title("Training Loss")
        plt.theme("dark")
        plt.frame(True)
        plt.grid(True, True)
        plt.ylim(0, None) # Auto scale Y, start at 0
        
        # Build canvas
        canvas = plt.build()
        yield from self.decoder.decode(canvas)

def make_layout():
    """Define the grid layout for the dashboard."""
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=8)
    )
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=2)
    )
    return layout

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    MODEL_NAME = 'gpt2'
    NUM_EPOCHS = 500
    LEARNING_RATE = 3e-2
    BATCH_SIZE = 32
    ACCURACY_BUDGET = 0.05
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Setup Console ---
    console = Console()
    console.clear()
    
    with console.status("[bold cyan]Initializing Models and Datasets...[/]", spinner="dots"):
        # 1. Models
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        pruning_config = PruningConfig()
        circuit_model = CircuitDiscoveryGPT2.from_pretrained_with_pruning(MODEL_NAME, pruning_config).to(DEVICE)
        circuit_model.eval()
        
        full_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
        for param in full_model.parameters(): param.requires_grad = False
        
        disable_dropout(circuit_model)
        
        # Freezing
        for name, param in circuit_model.named_parameters():
            if 'gate' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # 2. Datasets
        # Silence dataset loading prints
        with contextlib.redirect_stdout(io.StringIO()):
            test_data = load_or_generate_ioi_data(split="test", num_samples=200) 
            train_data = load_or_generate_ioi_data(split="train", num_samples=200)
            val_data = load_or_generate_ioi_data(split="validation", num_samples=200)

            train_data = filter_dataset_by_model_correctness(train_data, full_model, tokenizer, DEVICE, batch_size=BATCH_SIZE)
            val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, DEVICE, batch_size=BATCH_SIZE)
            test_data = filter_dataset_by_model_correctness(test_data, full_model, tokenizer, DEVICE, batch_size=BATCH_SIZE)

        train_dataset = IOIDataset(train_data, tokenizer)
        val_dataset = IOIDataset(val_data, tokenizer)
        test_dataset = IOIDataset(test_data, tokenizer)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # 3. Baseline Eval
        # We run this quickly to capture baseline acc
        with contextlib.redirect_stdout(io.StringIO()):
             baseline_results = run_evaluation(full_model, "Baseline", None, test_dataloader, DEVICE, tokenizer)
        base_accuracy = baseline_results.get("accuracy", 0.0)

    # --- Prepare for Training ---
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=LEARNING_RATE)
    
    # --- Dashboard State ---
    loss_history = deque(maxlen=100)
    kl_history = deque(maxlen=100)
    log_messages = deque(maxlen=10)
    log_messages.append(f"[bold cyan]Setup Complete.[/] Baseline Acc: [green]{base_accuracy:.4f}[/]")

    # Create UI
    layout = make_layout()
    layout["header"].update(Panel(f"[bold magenta]IOI Circuit Discovery[/] | Target Acc: [bold yellow]>{(base_accuracy-ACCURACY_BUDGET):.4f}[/]", style="white on black"))
    
    # Progress Bar
    job_progress = Progress(
        "{task.description}",
        BarColumn(),
        "{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        TextColumn("[cyan]Epoch {task.completed}/{task.total}")
    )
    epoch_task = job_progress.add_task("[green]Training", total=NUM_EPOCHS)
    
    # Graph
    rich_graph = RichGraph()
    
    circuit_model.train()
    total_steps = 0
    
    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    with Live(layout, refresh_per_second=4, screen=True) as live:
        for epoch in range(NUM_EPOCHS):
            epoch_loss_accum = 0
            
            # --- Batch Loop ---
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                # Move to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor): batch[k] = v.to(DEVICE)
                
                # Forward
                circuit_outputs = circuit_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                with torch.no_grad():
                    target_outputs = full_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                
                # KL Loss
                batch_size = circuit_outputs.logits.size(0)
                total_kl = 0
                for i in range(batch_size):
                    t_start = batch['T_Start'][i].item() - 1 
                    t_end = batch['T_End'][i].item() - 1
                    valid_len = batch['attention_mask'][i].sum().item()
                    end_pos = min(t_end, valid_len)
                    
                    if t_start < end_pos:
                        kl = F.kl_div(
                            F.log_softmax(circuit_outputs.logits[i, t_start:end_pos], dim=-1),
                            F.log_softmax(target_outputs.logits[i, t_start:end_pos], dim=-1),
                            reduction='batchmean', log_target=True
                        )
                        total_kl += kl
                kl_loss = total_kl / batch_size

                # Task Loss
                pos_good = batch['T_Start'] - 1 
                token_good = batch['target_tokens'][:, 0]
                token_bad = batch['distractor_tokens'][:, 0]
                batch_idx = torch.arange(batch_size, device=DEVICE)
                
                logit_good = circuit_outputs.logits[batch_idx, pos_good, token_good]
                logit_bad = circuit_outputs.logits[batch_idx, pos_good, token_bad]
                task_loss = F.relu(1.0 - (logit_good - logit_bad)).mean()
                
                # Sparsity Loss
                sparsity_stats = circuit_model.get_sparsity_loss(step=total_steps)
                sparsity_loss = sparsity_stats['total_sparsity']
                
                loss = kl_loss + sparsity_loss + task_loss
                loss.backward()
                optimizer.step()
                
                # Update Stats
                epoch_loss_accum += loss.item()
                loss_history.append(loss.item())
                kl_history.append(kl_loss.item())
                total_steps += 1
                
                # --- Update Dashboard ---
                # 1. Update Graph
                plt.clt()
                plt.cld()
                plt.plot(list(loss_history), color="cyan", label="Total Loss")
                plt.plot(list(kl_history), color="magenta", label="KL")
                
                # 2. Update Metric Grid
                table = Table.grid(expand=True)
                table.add_column(justify="center", ratio=1)
                table.add_column(justify="center", ratio=1)
                table.add_row(
                    Panel(f"[bold cyan]{loss.item():.4f}[/]", title="Total Loss"),
                    Panel(f"[bold magenta]{kl_loss.item():.4f}[/]", title="KL Divergence")
                )
                table.add_row(
                    Panel(f"[bold green]{task_loss.item():.4f}[/]", title="Task Loss"),
                    Panel(f"[bold yellow]{sparsity_loss.item():.4f}[/]", title="Sparsity")
                )
                
                # 3. Assemble Left Panel
                left_panel = Group(
                    Panel(table, title="Real-time Metrics", border_style="green"),
                    Panel(job_progress, title="Progress", border_style="blue")
                )
                
                layout["left"].update(left_panel)
                layout["right"].update(Panel(rich_graph, title="Loss History", border_style="cyan"))
                layout["footer"].update(Panel("\n".join(log_messages), title="Event Log", style="grey50"))

            # --- Epoch End ---
            job_progress.advance(epoch_task)
            
            # Validation
            if ((epoch + 1) % 10 == 0) or (epoch == NUM_EPOCHS - 1):
                circuit_model.eval()
                # Capture output to prevent screen mess
                with contextlib.redirect_stdout(io.StringIO()):
                    val_results = run_evaluation(
                        model_to_eval=circuit_model,
                        model_name=f"Val_{epoch}",
                        full_model_for_faithfulness=full_model,
                        dataloader=val_dataloader,
                        device=DEVICE,
                        tokenizer=tokenizer
                    )
                circuit_model.train()
                
                val_acc = val_results.get("accuracy", 0.0)
                diff = val_acc - base_accuracy
                color = "green" if abs(diff) < ACCURACY_BUDGET else "red"
                
                msg = f"[bold white]Epoch {epoch+1}[/]: Val Acc = [{color}]{val_acc:.4f}[/] (Base: {base_accuracy:.4f})"
                log_messages.append(msg)

    # ==========================================================================
    # FINALIZATION
    # ==========================================================================
    console.clear()
    console.print(Panel("[bold green]Training Complete! Finalizing Circuit...[/]", style="bold green"))
    
    analyze_and_finalize_circuit(circuit_model)
    
    circuit_model.eval()
    console.print("[bold]Running Final Test Set Evaluation...[/]")
    final_results = run_evaluation(circuit_model, "Final", full_model, test_dataloader, DEVICE, tokenizer)
    
    # Print Final Summary
    table = Table(title="Final Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Baseline Accuracy", f"{base_accuracy:.4f}")
    table.add_row("Final Accuracy", f"{final_results['accuracy']:.4f}")
    table.add_row("Accuracy Drop", f"{base_accuracy - final_results['accuracy']:.4f}")
    table.add_row("KL Divergence", f"{final_results['kl_div']:.4f}")
    
    console.print(table)