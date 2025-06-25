import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer
from models.gpt2_masks import PrunableGPT2LMHeadModel as CircuitDiscoveryGPT2
class CircuitTrainer:
    def __init__(self, model: CircuitDiscoveryGPT2, tokenizer: GPT2Tokenizer, device: str = 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def train(self, dataloader, optimizer, num_epochs=3):
        self.model.train()
        print("Starting training...")
        for epoch in range(num_epochs):
            total_data_loss, total_sparsity_loss = 0, 0
            
            for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                data_loss = outputs.loss
                
                global_step = epoch * len(dataloader) + step
                sparsity_losses = self.model.get_sparsity_loss(step=global_step)
                sparsity_loss = sparsity_losses['total_sparsity']
                
                total_loss = data_loss + sparsity_loss
                total_loss.backward()
                optimizer.step()

                total_data_loss += data_loss.item()
                total_sparsity_loss += sparsity_loss.item()
            
            avg_data_loss = total_data_loss / len(dataloader)
            avg_sparsity_loss = total_sparsity_loss / len(dataloader)
            print(f"Epoch {epoch+1} Summary: Avg Data Loss: {avg_data_loss:.4f}, Avg Sparsity Loss: {avg_sparsity_loss:.4f}")

    def visualize_circuit_stats(self, save_path: str = None):
        stats = self.model.get_circuit_statistics()
        num_layers = self.model.config.n_layer
        layer_indices = range(num_layers)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Discovered Circuit Structure Post-Training', fontsize=16)
        sns.set_style("whitegrid")

        plot_configs = [
            ('attention_heads', 'Active Attention Heads per Block', axes[0, 0], "viridis"),
            ('mlp_neurons', 'Active MLP Neurons per Block', axes[0, 1], "viridis"),
            ('attention_components', 'Active Attention Components per Block', axes[1, 0], "mako"),
            ('blocks', 'Active Blocks', axes[1, 1], "mako")
        ]
        
        for key, title, ax, palette in plot_configs:
            if stats[key]['active_ratio_per_layer']:
                # The data for components and blocks might be duplicated, let's average it per layer.
                if key in ['attention_components', 'mlp_components', 'blocks']:
                    data_to_plot = stats[key]['active_ratio_per_layer']
                else:
                    data_to_plot = stats[key]['active_ratio_per_layer']
                
                sns.barplot(x=list(layer_indices), y=data_to_plot, ax=ax, palette=palette)
                ax.set_title(title)
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("Active Ratio")
                ax.set_xlabel("Layer Index")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path: plt.savefig(save_path, dpi=300)
        plt.show()
