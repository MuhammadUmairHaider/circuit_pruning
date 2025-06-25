import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

# ==============================================================================
# 1. PRUNING CONFIGURATION AND CORE MODULES
# ==============================================================================

@dataclass
class PruningConfig:
    init_value: float = 1.0; lambda_attention: float = 0.9; lambda_mlp: float = 0.01
    attention_threshold: float = 0.3; mlp_threshold: float = 1.0;  # Thresholds for pruning
    sparsity_warmup_steps: int = 50; enable_attention_pruning: bool = True; enable_mlp_pruning: bool = True

class LearnableGate(nn.Module):
    # THE FIX: Simplified to just be a parameter container. The logic is moved to the modules that use it.
    def __init__(self, size, init_value: float = 1.0):
        super().__init__()
        if isinstance(size, int): size = (size,)
        self.gate = nn.Parameter(torch.full(size, init_value))
    def get_sparsity_loss(self) -> torch.Tensor:
        return torch.norm(self.gate, p=1)
    def get_active_ratio(self, threshold: float) -> float:
        with torch.no_grad():
            return (self.gate.abs() > threshold).float().mean().item()

class PrunableAttention(nn.Module):
    def __init__(self, original_attention, gpt_config: GPT2Config, pruning_config: PruningConfig):
        super().__init__()
        self.original_attention = original_attention
        self.num_heads = gpt_config.n_head
        self.head_dim = gpt_config.hidden_size // self.num_heads
        if pruning_config.enable_attention_pruning:
            self.head_gates = LearnableGate(self.num_heads, init_value=pruning_config.init_value)
        else: self.head_gates = None

    def forward(self, clean_states, corrupted_states, attention_mask=None):
        clean_outputs = self.original_attention(clean_states, attention_mask=attention_mask)[0]
        corrupted_outputs = self.original_attention(corrupted_states, attention_mask=attention_mask)[0]
        if self.head_gates:
            b, s, d = clean_outputs.shape
            clean_reshaped = clean_outputs.view(b, s, self.num_heads, self.head_dim)
            corrupted_reshaped = corrupted_outputs.view(b, s, self.num_heads, self.head_dim)
            
            # THE FIX: Reshape gate for broadcasting and apply interpolation logic here.
            gate = self.head_gates.gate.view(1, 1, self.num_heads, 1)
            gated_output = gate * clean_reshaped + (1 - gate) * corrupted_reshaped
            return gated_output.view(b, s, d)
        
        return clean_outputs

class PrunableMLP(nn.Module):
    def __init__(self, original_mlp, gpt_config: GPT2Config, pruning_config: PruningConfig):
        super().__init__()
        self.original_mlp = original_mlp
        self.intermediate_size = gpt_config.n_inner if gpt_config.n_inner is not None else 4 * gpt_config.hidden_size
        if pruning_config.enable_mlp_pruning:
            self.neuron_gates = LearnableGate(self.intermediate_size, init_value=pruning_config.init_value)
        else: self.neuron_gates = None

    def forward(self, clean_states, corrupted_states):
        clean_act = self.original_mlp.act(self.original_mlp.c_fc(clean_states))
        corrupted_act = self.original_mlp.act(self.original_mlp.c_fc(corrupted_states))
        
        if self.neuron_gates:
            # THE FIX: Reshape gate for broadcasting and apply interpolation logic here.
            gate = self.neuron_gates.gate.view(1, 1, -1)
            gated_act = gate * clean_act + (1 - gate) * corrupted_act
            return self.original_mlp.dropout(self.original_mlp.c_proj(gated_act))

        return self.original_mlp.dropout(self.original_mlp.c_proj(clean_act))

class PrunableBlock(nn.Module):
    def __init__(self, original_block, gpt_config: GPT2Config, pruning_config: PruningConfig):
        super().__init__()
        self.ln_1, self.ln_2 = original_block.ln_1, original_block.ln_2
        self.attn = PrunableAttention(original_block.attn, gpt_config, pruning_config)
        self.mlp = PrunableMLP(original_block.mlp, gpt_config, pruning_config)

    def forward(self, clean_states, corrupted_states, attention_mask=None):
        attn_output = self.attn(self.ln_1(clean_states), self.ln_1(corrupted_states), attention_mask=attention_mask)
        clean_after_attn = clean_states + attn_output
        corrupted_after_attn = corrupted_states + self.attn.original_attention(self.ln_1(corrupted_states), attention_mask=attention_mask)[0]
        mlp_output = self.mlp(self.ln_2(clean_after_attn), self.ln_2(corrupted_after_attn))
        final_clean = clean_after_attn + mlp_output
        final_corrupted = corrupted_after_attn + self.mlp.original_mlp(self.ln_2(corrupted_after_attn))
        return final_clean, final_corrupted

class CircuitDiscoveryGPT2(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.pruning_config = None
    @classmethod
    def from_pretrained_with_pruning(cls, model_name: str, pruning_config: PruningConfig, **kwargs):
        model = cls.from_pretrained(model_name, **kwargs)
        model.pruning_config = pruning_config
        model.transformer.h = nn.ModuleList([PrunableBlock(b, model.config, pruning_config) for b in model.transformer.h])
        return model

    def forward(self, clean_input_ids, corrupted_input_ids, attention_mask=None):
        clean_states = self.transformer.wte(clean_input_ids) + self.transformer.wpe.weight[:clean_input_ids.shape[1], :]
        corrupted_states = self.transformer.wte(corrupted_input_ids) + self.transformer.wpe.weight[:corrupted_input_ids.shape[1], :]
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        for block in self.transformer.h:
            clean_states, corrupted_states = block(clean_states, corrupted_states, attention_mask=attention_mask)
        return self.lm_head(self.transformer.ln_f(clean_states))

    def get_sparsity_loss(self, step: int = 0) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=self.device)
        warmup_mult = min(1.0, step / self.pruning_config.sparsity_warmup_steps if self.pruning_config.sparsity_warmup_steps > 0 else 1.0)
        for block in self.transformer.h:
            if self.pruning_config.enable_attention_pruning and block.attn.head_gates:
                losses.setdefault('attention', torch.tensor(0.0, device=self.device)); losses['attention'] += block.attn.head_gates.get_sparsity_loss()
            if self.pruning_config.enable_mlp_pruning and block.mlp.neuron_gates:
                losses.setdefault('mlp', torch.tensor(0.0, device=self.device)); losses['mlp'] += block.mlp.neuron_gates.get_sparsity_loss()
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
                    gates.copy_((mask).float()); print(f"  Block {i} Attention Heads: {mask.sum().item()}/{len(mask)} active")
                if self.pruning_config.enable_mlp_pruning and block.mlp.neuron_gates:
                    gates = block.mlp.neuron_gates.gate
                    mask = gates.abs() > self.pruning_config.mlp_threshold
                    gates.copy_((mask).float()); print(f"  Block {i} MLP Neurons: {mask.sum().item()}/{len(mask)} active")

# ==============================================================================
# 2. DATASET AND EVALUATION
# ==============================================================================
def generate_ioi_sample_pair(names, locations, objects):
    s_name, io_name = random.sample(names, 2)
    loc, obj = random.choice(locations), random.choice(objects)
    clean_prompt = f"When {io_name} and {s_name} went to the {loc}, {s_name} gave the {obj} to"
    corrupted_prompt = f"When {s_name} and {io_name} went to the {loc}, {io_name} gave the {obj} to"
    return {"clean_prompt": clean_prompt, "clean_completion": io_name, "corrupted_prompt": corrupted_prompt, "incorrect_completion": s_name}

class IOIDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: GPT2Tokenizer):
        self.data, self.tokenizer = data, tokenizer
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        clean_inputs = self.tokenizer(item['clean_prompt'], padding='max_length', max_length=40, truncation=True, return_tensors='pt')
        corrupted_inputs = self.tokenizer(item['corrupted_prompt'], padding='max_length', max_length=40, truncation=True, return_tensors='pt')
        correct_token_id = self.tokenizer.encode(f" {item['clean_completion']}")[0]
        incorrect_token_id = self.tokenizer.encode(f" {item['incorrect_completion']}")[0]
        return {
            "clean_input_ids": clean_inputs['input_ids'].squeeze(0), "clean_attention_mask": clean_inputs['attention_mask'].squeeze(0),
            "corrupted_input_ids": corrupted_inputs['input_ids'].squeeze(0), "correct_token_id": torch.tensor(correct_token_id, dtype=torch.long),
            "incorrect_token_id": torch.tensor(incorrect_token_id, dtype=torch.long)
        }

def evaluate_circuit(circuit_model, full_model, dataloader, device):
    """
    Evaluates the circuit on Faithfulness (KL Div), Performance (Logit Diff), 
    and two types of Accuracy (Full Vocab and Binary).
    """
    circuit_model.eval()
    full_model.eval()
    total_kl_div, total_logit_diff = 0, 0
    full_vocab_correct, binary_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Circuit"):
            for key, val in batch.items():
                batch[key] = val.to(device)

            circuit_logits = circuit_model(
                clean_input_ids=batch['clean_input_ids'],
                corrupted_input_ids=batch['corrupted_input_ids'],
                attention_mask=batch['clean_attention_mask']
            )
            full_model_logits = full_model(
                input_ids=batch['clean_input_ids'],
                attention_mask=batch['clean_attention_mask']
            ).logits

            # --- Metrics Calculation ---
            last_token_logits = circuit_logits[:, -1, :]

            # 1. Faithfulness: KL Divergence
            last_token_circuit_log_probs = F.log_softmax(last_token_logits, dim=-1)
            last_token_full_log_probs = F.log_softmax(full_model_logits[:, -1, :], dim=-1)
            kl_div = F.kl_div(last_token_circuit_log_probs, last_token_full_log_probs, log_target=True, reduction='batchmean')
            total_kl_div += kl_div.item()

            # --- Get logits for the two names of interest ---
            correct_logits = last_token_logits.gather(1, batch['correct_token_id'].unsqueeze(-1))
            incorrect_logits = last_token_logits.gather(1, batch['incorrect_token_id'].unsqueeze(-1))

            # 2. Performance: Logit Difference
            total_logit_diff += (correct_logits - incorrect_logits).mean().item()

            # 3. Performance: Full Vocabulary Accuracy
            predicted_token_ids = torch.argmax(last_token_logits, dim=-1)
            full_vocab_correct += (predicted_token_ids == batch['correct_token_id']).sum().item()
            
            # 4. Performance: Binary Accuracy (forced-choice)
            binary_correct += (correct_logits > incorrect_logits).sum().item()

            total_samples += batch['clean_input_ids'].size(0)

    avg_kl_div = total_kl_div / len(dataloader)
    avg_logit_diff = total_logit_diff / len(dataloader)
    full_vocab_accuracy = (full_vocab_correct / total_samples) * 100
    binary_accuracy = (binary_correct / total_samples) * 100

    print("\n" + "="*50)
    print("Circuit Evaluation Summary:")
    print(f"  - Faithfulness (KL Div):        {avg_kl_div:.4f} (lower is better)")
    print(f"  - Performance (Logit Diff):      {avg_logit_diff:.4f} (higher is better)")
    print(f"  - Performance (Binary Accuracy):   {binary_accuracy:.2f}%")
    print(f"  - Performance (Full Vocab Acc):  {full_vocab_accuracy:.2f}%")
    print("="*50)
    
    
def evaluate_baseline_model(full_model, dataloader, device):
    """
    Performs a sanity check on the original, unpruned model to get baseline performance.
    """
    print("\n" + "="*50)
    print("  SANITY CHECK: Evaluating Baseline Full Model")
    print("="*50)
    
    full_model.eval()
    total_logit_diff = 0
    full_vocab_correct, binary_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Baseline Model"):
            for key, val in batch.items():
                batch[key] = val.to(device)

            # Get logits from the full model on the clean inputs
            outputs = full_model(
                input_ids=batch['clean_input_ids'],
                attention_mask=batch['clean_attention_mask']
            )
            last_token_logits = outputs.logits[:, -1, :]

            # --- Metrics Calculation ---
            correct_logits = last_token_logits.gather(1, batch['correct_token_id'].unsqueeze(-1))
            incorrect_logits = last_token_logits.gather(1, batch['incorrect_token_id'].unsqueeze(-1))
            
            # 1. Performance: Logit Difference
            total_logit_diff += (correct_logits - incorrect_logits).mean().item()

            # 2. Performance: Full Vocabulary Accuracy
            predicted_token_ids = torch.argmax(last_token_logits, dim=-1)
            full_vocab_correct += (predicted_token_ids == batch['correct_token_id']).sum().item()

            # 3. Performance: Binary Accuracy (forced-choice)
            binary_correct += (correct_logits > incorrect_logits).sum().item()

            total_samples += batch['clean_input_ids'].size(0)

    avg_logit_diff = total_logit_diff / len(dataloader)
    full_vocab_accuracy = (full_vocab_correct / total_samples) * 100
    binary_accuracy = (binary_correct / total_samples) * 100

    print("\n" + "="*50)
    print("Baseline Model Performance:")
    print(f"  - Logit Difference:      {avg_logit_diff:.4f}")
    print(f"  - Binary Accuracy:       {binary_accuracy:.2f}%")
    print(f"  - Full Vocab Accuracy:   {full_vocab_accuracy:.2f}%")
    print("="*50 + "\n")
# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    MODEL_NAME = 'gpt2'
    NUM_EPOCHS = 5
    LEARNING_RATE = 5e-3
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Setup Models and Tokenizer ---
    pruning_config = PruningConfig()
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    circuit_model = CircuitDiscoveryGPT2.from_pretrained_with_pruning(MODEL_NAME, pruning_config).to(DEVICE)
    full_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    for param in full_model.parameters():
        param.requires_grad = False

    # --- Prepare Data ---
    names = ["Mary", "John", "Alice", "Bob", "Patricia", "James", "Linda", "Robert"]
    locations = ["store", "park", "cafe", "office", "school", "hospital"]
    objects = ["drink", "book", "gift", "letter", "key", "report"]
    ioi_data = [generate_ioi_sample_pair(names, locations, objects) for _ in range(4000)]
    train_dataset = IOIDataset(ioi_data[:3000], tokenizer)
    val_dataset = IOIDataset(ioi_data[3000:], tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # --- SANITY CHECK: Evaluate the baseline model first ---
    evaluate_baseline_model(full_model, val_dataloader, DEVICE)

    # --- Isolate Gate Parameters for Optimizer ---
    gate_params = [p for name, p in circuit_model.named_parameters() if 'gate' in name]
    optimizer = AdamW(gate_params, lr=LEARNING_RATE)
    
    # --- Training Loop ---
    print("Starting training with interchange ablation...")
    circuit_model.train()
    for epoch in range(NUM_EPOCHS):
        total_kl_loss, total_sparsity_loss = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            optimizer.zero_grad()
            for key, val in batch.items():
                batch[key] = val.to(DEVICE)
            
            circuit_logits = circuit_model(
                clean_input_ids=batch['clean_input_ids'],
                corrupted_input_ids=batch['corrupted_input_ids'],
                attention_mask=batch['clean_attention_mask']
            )
            with torch.no_grad():
                target_logits = full_model(
                    input_ids=batch['clean_input_ids'],
                    attention_mask=batch['clean_attention_mask']
                ).logits

            kl_loss = F.kl_div(
                F.log_softmax(circuit_logits / 1.0, dim=-1),
                F.log_softmax(target_logits / 1.0, dim=-1),
                reduction='batchmean',
                log_target=True
            )
            sparsity_loss = circuit_model.get_sparsity_loss(step=epoch * len(train_dataloader) + step)['total_sparsity']
            loss = kl_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            total_kl_loss += kl_loss.item()
            total_sparsity_loss += sparsity_loss.item()

        avg_kl, avg_sparsity = total_kl_loss / len(train_dataloader), total_sparsity_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Summary: Avg KL Loss: {avg_kl:.4f}, Avg Sparsity Loss: {avg_sparsity:.4f}")
        evaluate_circuit(circuit_model, full_model, val_dataloader, DEVICE)
        circuit_model.train()

    # --- Final Analysis ---
    print("\nFinal evaluation on validation set after training:")
    circuit_model.apply_circuit_mask(verbose=True)
    evaluate_circuit(circuit_model, full_model, val_dataloader, DEVICE)