import torch
import os
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch.nn.functional as F
import argparse
from collections import Counter

# --- UTILITY/HELPER FUNCTIONS ---

class bcolors:
    """A class to color terminal output for better readability."""
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def info(text):
    """Prints informational text in blue."""
    print(f"{bcolors.OKBLUE}{text}{bcolors.ENDC}")

def good(text):
    """Prints success text in green."""
    print(f"{bcolors.OKGREEN}{text}{bcolors.ENDC}")

def bad(text):
    """Prints error or warning text in red."""
    print(f"{bcolors.FAIL}{text}{bcolors.ENDC}")

# --- PYTORCH DATASET CLASS ---

class IOIDataset(TorchDataset):
    """
    PyTorch Dataset for the Indirect Object Identification (IOI) task.
    This class tokenizes clean and corrupted prompts from the pre-processed
    dataset and prepares them for model evaluation.
    """
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        clean_prompt = item['ioi_sentences']
        corrupted_prompt = item['corr_ioi_sentences']
        io_token_str = item['a'] 
        s_token_str = item['b']  

        clean_inputs = self.tokenizer(clean_prompt, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        corrupted_inputs = self.tokenizer(corrupted_prompt, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')

        try:
            io_token_id = self.tokenizer.encode(" " + io_token_str, add_special_tokens=False)[0]
            s_token_id = self.tokenizer.encode(" " + s_token_str, add_special_tokens=False)[0]
        except IndexError:
            bad(f"Warning: Could not encode names '{io_token_str}' or '{s_token_str}'. Skipping item {idx}.")
            io_token_id = self.tokenizer.pad_token_id
            s_token_id = self.tokenizer.pad_token_id

        # *** FIX ***: The prediction target is the last token in the sequence.
        # An autoregressive model predicts the token at position `t` using the output from position `t-1`.
        # Therefore, we need the logits from the second-to-last token position.
        # The original `...sum().item() - 1` was incorrect as it pointed to the last token,
        # checking the model's prediction for what comes *after* the sentence.
        last_token_idx = clean_inputs['attention_mask'].squeeze().sum().item() - 2

        return {
            "clean_input_ids": clean_inputs['input_ids'].squeeze(0),
            "clean_attention_mask": clean_inputs['attention_mask'].squeeze(0),
            "corrupted_input_ids": corrupted_inputs['input_ids'].squeeze(0),
            "io_token_id": torch.tensor(io_token_id, dtype=torch.long),
            "s_token_id": torch.tensor(s_token_id, dtype=torch.long),
            "last_token_idx": torch.tensor(last_token_idx, dtype=torch.long),
        }

# --- PRIMARY EVALUATION FUNCTION ---

@torch.no_grad()
def run_ioi_evaluation(model, model_name, dataloader, device, full_model_for_faithfulness=None):
    """
    Runs a comprehensive evaluation on the IOI task.
    Calculates task performance (logit diff, accuracy) and optional faithfulness metrics.
    """
    info("\n" + "="*50 + f"\n  EVALUATING: {model_name} on IOI Task\n" + "="*50)
    model.eval()
    if full_model_for_faithfulness:
        full_model_for_faithfulness.eval()

    all_logit_diffs, total_kl_div, correct_predictions, exact_matches, total_samples = [], 0.0, 0, 0, 0
    
    for batch in tqdm(dataloader, desc=f"Evaluating {model_name}", leave=False):
        for key, val in batch.items():
            if isinstance(val, torch.Tensor): batch[key] = val.to(device)
        
        try:
            outputs = model(input_ids=batch['clean_input_ids'], attention_mask=batch['clean_attention_mask'], corrupted_input_ids=batch['corrupted_input_ids'])
        except TypeError:
            outputs = model(input_ids=batch['clean_input_ids'], attention_mask=batch['clean_attention_mask'])

        last_token_logits = outputs.logits.gather(1, batch['last_token_idx'].view(-1, 1, 1).expand(-1, -1, outputs.logits.size(-1))).squeeze(1)
        
        logit_correct = last_token_logits.gather(1, batch['io_token_id'].unsqueeze(1))
        logit_incorrect = last_token_logits.gather(1, batch['s_token_id'].unsqueeze(1))
        all_logit_diffs.extend((logit_correct - logit_incorrect).cpu().squeeze().tolist())
        
        predicted_tokens = torch.argmax(last_token_logits, dim=-1)
        correct_predictions += (predicted_tokens == batch['io_token_id']).sum().item()
        total_samples += batch['io_token_id'].size(0)

        if full_model_for_faithfulness:
            full_model_outputs = full_model_for_faithfulness(input_ids=batch['clean_input_ids'], attention_mask=batch['clean_attention_mask'])
            full_model_logits = full_model_outputs.logits.gather(1, batch['last_token_idx'].view(-1, 1, 1).expand(-1, -1, full_model_outputs.logits.size(-1))).squeeze(1)
            
            kl_div = F.kl_div(F.log_softmax(last_token_logits, dim=-1), F.log_softmax(full_model_logits, dim=-1), reduction='batchmean', log_target=True)
            total_kl_div += kl_div.item() * batch['io_token_id'].size(0)

            full_model_predicted_tokens = torch.argmax(full_model_logits, dim=-1)
            exact_matches += (predicted_tokens == full_model_predicted_tokens).sum().item()

    avg_logit_diff = sum(all_logit_diffs) / len(all_logit_diffs) if all_logit_diffs else 0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    good(f"--- {model_name} Evaluation Summary ---")
    print(f"  - Task Performance:")
    print(f"    - Average Logit Difference: {avg_logit_diff:.4f}")
    print(f"    - Accuracy: {accuracy:.2%}")

    results = {"logit_diff": avg_logit_diff, "accuracy": accuracy}
    if full_model_for_faithfulness:
        avg_kl_div = total_kl_div / total_samples if total_samples > 0 else 0
        exact_match_pct = exact_matches / total_samples if total_samples > 0 else 0
        print(f"  - Circuit Faithfulness:")
        print(f"    - KL Divergence from Full Model: {avg_kl_div:.4f}")
        print(f"    - Exact Match with Full Model: {exact_match_pct:.2%}")
        results.update({"kl_div": avg_kl_div, "exact_match": exact_match_pct})
    
    print("="*50)
    return results

# --- ADDITIONAL DIAGNOSTIC/TESTING FUNCTIONS ---

@torch.no_grad()
def run_corrupted_only_test(model, model_name, dataloader, device):
    """Tests the model's performance on corrupted inputs only."""
    info(f"\n--- Running Corrupted-Input-Only Test for {model_name} ---")
    model.eval()
    correct_predictions = 0
    total_samples = 0
    for batch in tqdm(dataloader, desc=f"Corrupted Test", leave=False):
        for key, val in batch.items():
            if isinstance(val, torch.Tensor): batch[key] = val.to(device)

        outputs = model(input_ids=batch['corrupted_input_ids'])
        
        # *** FIX ***: Apply the same off-by-one correction here.
        # We need the logits from the position *before* the final token.
        last_token_idx = batch['corrupted_input_ids'].ne(dataloader.dataset.tokenizer.pad_token_id).sum(-1) - 2
        last_token_logits = outputs.logits.gather(1, last_token_idx.view(-1, 1, 1).expand(-1, -1, outputs.logits.size(-1))).squeeze(1)
        
        # A "correct" prediction here is picking the S token, which is the name from the corrupted context
        predicted_tokens = torch.argmax(last_token_logits, dim=-1)
        correct_predictions += (predicted_tokens == batch['s_token_id']).sum().item()
        total_samples += batch['s_token_id'].size(0)

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    good(f"Corrupted Input Test Accuracy (predicting S-token): {accuracy:.2%}")
    print("This metric shows if the model defaults to the name present in a corrupted context.")
    return accuracy


@torch.no_grad()
def test_pure_token_bias(model, dataloader, tokenizer, device):
    """Analyzes if the model has a strong intrinsic bias for certain names."""
    info("\n--- Testing for Pure Token Bias ---")
    model.eval()
    biases = Counter()
    for batch in dataloader:
        io_tokens = batch['io_token_id']
        s_tokens = batch['s_token_id']
        for t in io_tokens:
            biases[t.item()] += 1
        for t in s_tokens:
            biases[t.item()] -= 1

    top_5_io = biases.most_common(5)
    bottom_5_io = biases.most_common()[:-6:-1]

    print("Bias Score (IO count - S count) across the dataset:")
    print("Top 5 IO-biased tokens:")
    for token_id, score in top_5_io:
        print(f"  - '{tokenizer.decode(token_id)}': {score}")
    print("Top 5 S-biased tokens (least biased towards IO):")
    for token_id, score in bottom_5_io:
        print(f"  - '{tokenizer.decode(token_id)}': {score}")

def analyze_dataset_balance(dataloader, tokenizer):
    """Checks the distribution of names in the dataset."""
    info("\n--- Analyzing Dataset Balance ---")
    io_counts = Counter()
    s_counts = Counter()
    dataset = dataloader.dataset.data 
    for item in dataset:
        io_counts[item['a']] += 1
        s_counts[item['b']] += 1
        
    print(f"Total examples: {len(dataset)}")
    print(f"Unique IO names: {len(io_counts)}")
    print(f"Unique S names: {len(s_counts)}")
    print("\nMost common IO names:")
    for name, count in io_counts.most_common(5):
        print(f"  - {name}: {count} times")
    print("\nMost common S names:")
    for name, count in s_counts.most_common(5):
        print(f"  - {name}: {count} times")


def test_swapped_roles(model, dataloader, tokenizer, device):
    """
    Checks if model performance differs on ABBA vs BABA templates,
    which can indicate reliance on positional cues vs grammatical understanding.
    """
    info("\n--- Swapped Roles Test ---")
    
    abba_correct, abba_total = 0, 0
    baba_correct, baba_total = 0, 0
    
    model.eval()
    with torch.no_grad():
      for i, batch in enumerate(dataloader):
          raw_data_indices = range(i * dataloader.batch_size, (i + 1) * dataloader.batch_size)

          for j, idx in enumerate(raw_data_indices):
              if idx >= len(dataloader.dataset.data): continue
              item = dataloader.dataset.data[idx]
              order = item.get('order') 

              for key, val in batch.items():
                  if isinstance(val, torch.Tensor): batch[key] = val.to(device)

              # Evaluate one item at a time
              single_item_batch = {k: v[j].unsqueeze(0) for k, v in batch.items()}
              
              outputs = model(input_ids=single_item_batch['clean_input_ids'])
              last_token_idx = single_item_batch['last_token_idx']
              last_token_logits = outputs.logits.gather(1, last_token_idx.view(-1, 1, 1).expand(-1, -1, outputs.logits.size(-1))).squeeze(1)
              prediction = torch.argmax(last_token_logits, dim=-1)
              
              is_correct = (prediction == single_item_batch['io_token_id']).item()

              if order == 'abba':
                  abba_correct += is_correct
                  abba_total += 1
              elif order == 'baba':
                  baba_correct += is_correct
                  baba_total += 1

    if abba_total > 0:
        good(f"Accuracy on ABBA templates: {abba_correct/abba_total:.2%}")
    if baba_total > 0:
        good(f"Accuracy on BABA templates: {baba_correct/baba_total:.2%}")
    if abba_total == 0 and baba_total == 0:
        bad("Could not run swapped roles test: 'order' field not found in dataset.")

# --- DATA PREPARATION FUNCTION ---

def prepare_ioi_dataset(dataset_path: str, split: str = 'train', num_examples: int = None):
    """
    Loads a specified split of the IOI dataset from disk.
    """
    info(f"Loading dataset from disk: '{dataset_path}' using split '{split}'...")
    try:
        full_dataset = load_from_disk(dataset_path)
        if split not in full_dataset:
            bad(f"Split '{split}' not found. Available: {list(full_dataset.keys())}")
            return None
        
        dataset = full_dataset[split]
        if num_examples:
            dataset = dataset.select(range(min(num_examples, len(dataset))))
        good(f"Loaded {len(dataset)} examples.")
        return dataset
    except FileNotFoundError:
        bad(f"Dataset path not found: {dataset_path}")
        return None