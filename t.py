import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

# ==============================================================================
# 1. PRUNING CONFIGURATION AND CORE MODULES
# ==============================================================================

@dataclass
class PruningConfig:
    """Configuration for different granularity pruning."""
    init_value: float = 1.0
    lambda_attention: float = 0.001
    lambda_mlp: float = 0.001
    attention_threshold: float = 0.001
    mlp_threshold: float = 0.001
    sparsity_warmup_steps: int = 50
    enable_attention_pruning: bool = True
    enable_mlp_pruning: bool = True

class LearnableGate(nn.Module):
    """A learnable scaling factor gate for pruning model components."""
    def __init__(self, size: Union[int, Tuple[int, ...]], init_value: float = 1.0):
        super().__init__()
        if isinstance(size, int):
            size = (size,)
        self.gate = nn.Parameter(torch.full(size, init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate

    def get_sparsity_loss(self) -> torch.Tensor:
        return torch.norm(self.gate, p=1)

    def get_active_ratio(self, threshold: float) -> float:
        with torch.no_grad():
            return (self.gate.abs() > threshold).float().mean().item()

class PrunableAttention(nn.Module):
    """GPT-2 attention layer wrapped for ablation-by-mean-replacement."""
    def __init__(self, original_attention, gpt_config: GPT2Config, pruning_config: PruningConfig):
        super().__init__()
        self.original_attention = original_attention
        self.pruning_config = pruning_config
        self.embed_dim = gpt_config.hidden_size
        self.num_heads = gpt_config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.register_buffer('mean_head_activation', torch.zeros(self.num_heads, self.head_dim))
        if self.pruning_config.enable_attention_pruning:
            self.head_gates = LearnableGate(self.num_heads, init_value=self.pruning_config.init_value)
        else:
            self.head_gates = None

    def forward(self, hidden_states, **kwargs):
        attn_outputs = self.original_attention(hidden_states, **kwargs)
        attn_output = attn_outputs[0]
        batch_size, seq_len, _ = attn_output.shape
        if self.head_gates:
            reshaped_output = attn_output.view(batch_size, seq_len, self.num_heads, self.head_dim)
            gates_reshaped = self.head_gates.gate.view(1, 1, self.num_heads, 1)
            gated_out = reshaped_output * gates_reshaped
            mean_out = self.mean_head_activation.view(1, 1, self.num_heads, self.head_dim) * (1 - gates_reshaped)
            attn_output = (gated_out + mean_out).view(batch_size, seq_len, self.embed_dim)
        return (attn_output,) + attn_outputs[1:]

class PrunableMLP(nn.Module):
    """GPT-2 MLP wrapped for ablation-by-mean-replacement."""
    def __init__(self, original_mlp, gpt_config: GPT2Config, pruning_config: PruningConfig):
        super().__init__()
        self.original_mlp = original_mlp
        self.pruning_config = pruning_config
        self.intermediate_size = gpt_config.n_inner if gpt_config.n_inner is not None else 4 * gpt_config.hidden_size
        self.register_buffer('mean_neuron_activation', torch.zeros(self.intermediate_size))
        if self.pruning_config.enable_mlp_pruning:
            self.neuron_gates = LearnableGate(self.intermediate_size, init_value=self.pruning_config.init_value)
        else:
            self.neuron_gates = None

    def forward(self, hidden_states):
        hidden_states_fc = self.original_mlp.c_fc(hidden_states)
        hidden_states_act = self.original_mlp.act(hidden_states_fc)
        if self.neuron_gates:
            gate = self.neuron_gates.gate.view(1, 1, -1)
            gated_act = hidden_states_act * gate
            mean_act = self.mean_neuron_activation.view(1, 1, -1) * (1 - gate)
            hidden_states_act = gated_act + mean_act
        hidden_states_proj = self.original_mlp.c_proj(hidden_states_act)
        return self.original_mlp.dropout(hidden_states_proj)

class PrunableBlock(nn.Module):
    """A prunable Transformer block integrating the prunable components."""
    def __init__(self, original_block, gpt_config: GPT2Config, pruning_config: PruningConfig, layer_idx: int):
        super().__init__()
        self.ln_1 = original_block.ln_1
        self.ln_2 = original_block.ln_2
        self.attn = PrunableAttention(original_block.attn, gpt_config, pruning_config)
        self.mlp = PrunableMLP(original_block.mlp, gpt_config, pruning_config)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states_ln1 = self.ln_1(hidden_states)
        attn_outputs = self.attn(hidden_states_ln1, **kwargs)
        hidden_states = residual + attn_outputs[0]
        residual = hidden_states
        hidden_states_ln2 = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states_ln2)
        hidden_states = residual + mlp_output
        return (hidden_states,) + attn_outputs[1:]

class CircuitDiscoveryGPT2(GPT2LMHeadModel):
    """GPT-2 model with prunable components and specific constructor."""
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.pruning_config = None

    @classmethod
    def from_pretrained_with_pruning(cls, model_name: str, pruning_config: PruningConfig, **kwargs):
        model = cls.from_pretrained(model_name, **kwargs)
        model.pruning_config = pruning_config
        config = model.config
        original_blocks = list(model.transformer.h)
        model.transformer.h = nn.ModuleList([
            PrunableBlock(original_blocks[i], config, pruning_config, i)
            for i in range(len(original_blocks))
        ])
        return model

    def get_sparsity_loss(self, step: int = 0) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=self.device)
        warmup_mult = min(1.0, step / self.pruning_config.sparsity_warmup_steps if self.pruning_config.sparsity_warmup_steps > 0 else 1.0)
        for block in self.transformer.h:
            if self.pruning_config.enable_attention_pruning and block.attn.head_gates:
                losses.setdefault('attention', torch.tensor(0.0, device=self.device))
                losses['attention'] += block.attn.head_gates.get_sparsity_loss()
            if self.pruning_config.enable_mlp_pruning and block.mlp.neuron_gates:
                losses.setdefault('mlp', torch.tensor(0.0, device=self.device))
                losses['mlp'] += block.mlp.neuron_gates.get_sparsity_loss()
        if 'attention' in losses: total_loss += self.pruning_config.lambda_attention * warmup_mult * losses['attention']
        if 'mlp' in losses: total_loss += self.pruning_config.lambda_mlp * warmup_mult * losses['mlp']
        losses['total_sparsity'] = total_loss
        return losses

    def apply_circuit_mask(self, verbose: bool = True):
        if verbose: print("\nApplying final circuit mask...")
        with torch.no_grad():
            for i, block in enumerate(self.transformer.h):
                if self.pruning_config.enable_attention_pruning and block.attn.head_gates:
                    gates = block.attn.head_gates.gate
                    mask = gates.abs() > self.pruning_config.attention_threshold
                    gates.copy_((gates.abs() > self.pruning_config.attention_threshold).float())
                    if verbose: print(f"  Block {i} Attention Heads: {mask.sum().item()}/{len(mask)} active")
                if self.pruning_config.enable_mlp_pruning and block.mlp.neuron_gates:
                    gates = block.mlp.neuron_gates.gate
                    mask = gates.abs() > self.pruning_config.mlp_threshold
                    gates.copy_((gates.abs() > self.pruning_config.mlp_threshold).float())
                    if verbose: print(f"  Block {i} MLP Neurons: {mask.sum().item()}/{len(mask)} active")

    def get_circuit_statistics(self) -> Dict:
        stats = {'attention_heads': {'active_ratio_per_layer': []}, 'mlp_neurons': {'active_ratio_per_layer': []}}
        for block in self.transformer.h:
            if self.pruning_config.enable_attention_pruning and block.attn.head_gates:
                stats['attention_heads']['active_ratio_per_layer'].append(block.attn.head_gates.get_active_ratio(self.pruning_config.attention_threshold))
            if self.pruning_config.enable_mlp_pruning and block.mlp.neuron_gates:
                stats['mlp_neurons']['active_ratio_per_layer'].append(block.mlp.neuron_gates.get_active_ratio(self.pruning_config.mlp_threshold))
        return stats

# ==============================================================================
# 2. DATASET AND CALIBRATION
# ==============================================================================

def generate_ioi_sample(names, locations, objects):
    giver_name, receiver_name = random.sample(names, 2)
    location = random.choice(locations)
    obj_item = random.choice(objects)
    prompt = f"When {receiver_name} and {giver_name} went to the {location}, {giver_name} gave a {obj_item} to"
    full_sentence = f"{prompt} {receiver_name}"
    return {"prompt": prompt, "completion": receiver_name, "full_sentence": full_sentence}

class IOIDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data, self.tokenizer = data, tokenizer
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized_full = self.tokenizer(item['full_sentence'], truncation=True, max_length=50, padding='max_length', return_tensors='pt')
        input_ids = tokenized_full.input_ids.squeeze(0)
        tokenized_prompt = self.tokenizer(item['prompt'], return_tensors='pt').input_ids
        prompt_len = tokenized_prompt.shape[1] if tokenized_prompt.shape[1] < 50 else 49
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        return {"input_ids": input_ids, "attention_mask": tokenized_full.attention_mask.squeeze(0), "labels": labels}

def calibrate_model(model: CircuitDiscoveryGPT2, dataloader: DataLoader, device: str):
    print("\nStarting model calibration to compute mean activations...")
    model.eval()
    summed_activations = {}
    counts = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calibration"):
            input_ids = batch['input_ids'].to(device)
            hidden_states = model.transformer.wte(input_ids) + model.transformer.wpe.weight[:input_ids.shape[1], :]
            for i, block in enumerate(model.transformer.h):
                attn_input_states = block.ln_1(hidden_states)
                attn_output = block.attn.original_attention(attn_input_states)[0]
                reshaped_attn = attn_output.view(-1, block.attn.num_heads, block.attn.head_dim)
                summed_activations[f'attn_head_{i}'] = summed_activations.get(f'attn_head_{i}', 0) + reshaped_attn.sum(dim=0)
                
                mlp_input_states = block.ln_2(hidden_states + attn_output)
                mlp_intermediate = block.mlp.original_mlp.act(block.mlp.original_mlp.c_fc(mlp_input_states))
                summed_activations[f'mlp_neuron_{i}'] = summed_activations.get(f'mlp_neuron_{i}', 0) + mlp_intermediate.view(-1, block.mlp.intermediate_size).sum(dim=0)
                
                count = input_ids.shape[0] * input_ids.shape[1]
                counts[f'attn_head_{i}'] = counts.get(f'attn_head_{i}', 0) + count
                counts[f'mlp_neuron_{i}'] = counts.get(f'mlp_neuron_{i}', 0) + count
                hidden_states = hidden_states + attn_output + block.mlp.original_mlp(mlp_input_states)
    print("Calibration complete. Setting mean activation buffers...")
    for i, block in enumerate(model.transformer.h):
        block.attn.mean_head_activation.copy_(summed_activations[f'attn_head_{i}'] / counts[f'attn_head_{i}'])
        block.mlp.mean_neuron_activation.copy_(summed_activations[f'mlp_neuron_{i}'] / counts[f'mlp_neuron_{i}'])
    print("Mean activations have been set in model buffers.\n")

# ==============================================================================
# 3. TRAINER AND VALIDATION
# ==============================================================================

class CircuitTrainer:
    def __init__(self, model: CircuitDiscoveryGPT2, tokenizer: GPT2Tokenizer, device: str):
        self.model, self.tokenizer, self.device = model.to(device), tokenizer, device
    def train(self, dataloader, optimizer, num_epochs=3):
        self.model.train()
        print("Starting pruning-specific training on IOI task...")
        for epoch in range(num_epochs):
            total_data_loss, total_sparsity_loss = 0, 0
            for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                optimizer.zero_grad()
                outputs = self.model(input_ids=batch['input_ids'].to(self.device), attention_mask=batch['attention_mask'].to(self.device), labels=batch['labels'].to(self.device))
                data_loss = outputs.loss
                sparsity_loss = self.model.get_sparsity_loss(step=epoch * len(dataloader) + step)['total_sparsity']
                (data_loss + sparsity_loss).backward()
                optimizer.step()
                total_data_loss += data_loss.item()
                total_sparsity_loss += sparsity_loss.item()
            print(f"Epoch {epoch+1} Summary: Avg Data Loss: {total_data_loss/len(dataloader):.4f}, Avg Sparsity Loss: {total_sparsity_loss/len(dataloader):.4f}")
    def visualize_circuit_stats(self, save_path: str = None):
        stats = self.model.get_circuit_statistics()
        num_layers = self.model.config.n_layer
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
        fig.suptitle('Discovered Circuit Structure Post-Training', fontsize=16)
        sns.set_style("whitegrid")
        if stats['attention_heads']['active_ratio_per_layer']:
            sns.barplot(x=list(range(num_layers)), y=stats['attention_heads']['active_ratio_per_layer'], ax=axes[0], palette="viridis")
            axes[0].set_title('Active Attention Heads per Block'); axes[0].set_xlabel("Layer Index"); axes[0].set_ylabel("Active Ratio")
        if stats['mlp_neurons']['active_ratio_per_layer']:
            sns.barplot(x=list(range(num_layers)), y=stats['mlp_neurons']['active_ratio_per_layer'], ax=axes[1], palette="plasma")
            axes[1].set_title('Active MLP Neurons per Block'); axes[1].set_xlabel("Layer Index")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

def test_ioi_circuit(model: CircuitDiscoveryGPT2, tokenizer: GPT2Tokenizer, test_data: List[Dict], device: str):
    """Tests the pruned model's ability to perform the IOI task on unseen examples."""
    print("\n" + "="*50 + "\n      PERFORMING INFERENCE ON THE PRUNED CIRCUIT\n" + "="*50 + "\n")
    model.eval()
    model.to(device)
    correct_predictions = 0
    with torch.no_grad():
        for i, item in enumerate(tqdm(test_data, desc="Testing Circuit")):
            prompt, expected_completion = item['prompt'], item['completion']
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            # The 'early_stopping' flag has been removed from this call
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=6,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_text = tokenizer.decode(output_sequences[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
            is_correct = generated_text.startswith(expected_completion)
            if is_correct: 
                correct_predictions += 1
            
    accuracy = (correct_predictions / len(test_data)) * 100
    print("\n" + "="*50 + f"\nCircuit Performance Summary:\nCorrect Predictions: {correct_predictions} / {len(test_data)}\nAccuracy on IOI task: {accuracy:.2f}%\n" + "="*50)

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    # --- Configuration ---
    MODEL_NAME = 'gpt2'
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-2
    PRUNING_BATCH_SIZE = 16
    CALIBRATION_BATCH_SIZE = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Setup Model and Tokenizer ---
    pruning_config = PruningConfig()
    model = CircuitDiscoveryGPT2.from_pretrained_with_pruning(MODEL_NAME, pruning_config)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model.to(DEVICE)

    # --- 1. CALIBRATION on general text (WikiText-2) ---
    print("Loading WikiText-2 for calibration...")
    wikitext = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    all_text = "\n".join(filter(lambda x: len(x.strip()) > 0, wikitext['text']))
    tokenized_ids = tokenizer(all_text, return_tensors='pt')['input_ids'].squeeze()
    
    seq_length = 128
    calibration_data = [tokenized_ids[i:i+seq_length] for i in range(0, tokenized_ids.size(0), seq_length) if i+seq_length < tokenized_ids.size(0)]
    
    class GeneralTextDataset(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return {"input_ids": self.data[idx]}

    calibration_dataset = GeneralTextDataset(calibration_data[:500]) # Use 500 chunks for stable calibration
    calibration_dataloader = DataLoader(calibration_dataset, batch_size=CALIBRATION_BATCH_SIZE)
    calibrate_model(model, calibration_dataloader, DEVICE)

    # --- 2. PRUNING on specific task (IOI) ---
    gate_params = []
    for name, param in model.named_parameters():
        if 'gate' in name:
            gate_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    print(f"Trainable Gate Parameters: {sum(p.numel() for p in gate_params)}")

    optimizer = AdamW(gate_params, lr=LEARNING_RATE)
    trainer = CircuitTrainer(model, tokenizer, device=DEVICE)

    names = ["Mary", "John", "Alice", "Bob", "Patricia", "James", "Linda", "Robert"]
    locations = ["store", "park", "cafe", "office", "school", "hospital"]
    objects = ["drink", "book", "gift", "letter", "key", "report"]
    ioi_data = [generate_ioi_sample(names, locations, objects) for _ in range(2000)]
    ioi_dataset = IOIDataset(ioi_data, tokenizer)

    ioi_test_data = [generate_ioi_sample(names, locations, objects) for _ in range(500)]

    test_ioi_circuit(model, tokenizer, ioi_test_data, DEVICE)
    pruning_dataloader = DataLoader(ioi_dataset, batch_size=PRUNING_BATCH_SIZE, shuffle=True)
    trainer.train(pruning_dataloader, optimizer, num_epochs=NUM_EPOCHS)

    # --- 3. ANALYSIS and VALIDATION ---
    model.apply_circuit_mask(verbose=True)
    trainer.visualize_circuit_stats()
    

    test_ioi_circuit(model, tokenizer, ioi_test_data, DEVICE)