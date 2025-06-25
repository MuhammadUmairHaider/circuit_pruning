import random
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_from_disk
import os

# ==============================================================================
# DATASET AND EVALUATION (ALIGNED WITH HANNA ET AL., 2023)
# ==============================================================================
NOUN_POOL = [
    'abduction', 'accord', 'affair', 'agreement', 'appraisal', 'assaults', 'assessment', 'attack',
    'attempts', 'campaign', 'captivity', 'case', 'challenge', 'chaos', 'clash', 'collaboration', 'coma',
    'competition', 'confrontation', 'consequence', 'conspiracy', 'construction', 'consultation', 'contact',
    'contract', 'convention', 'cooperation', 'custody', 'deal', 'decline', 'decrease', 'demonstrations',
    'development', 'disagreement', 'disorder', 'dispute', 'domination', 'dynasty', 'effect', 'effort',
    'employment', 'endeavor', 'engagement', 'epidemic', 'evaluation', 'exchange', 'existence', 'expansion',
    'expedition', 'experiments', 'fall', 'fame', 'flights', 'friendship', 'growth', 'hardship', 'hostility',
    'illness', 'impact', 'imprisonment', 'improvement', 'incarceration', 'increase', 'insurgency', 'invasion',
    'investigation', 'journey', 'kingdom', 'marriage', 'modernization', 'negotiation', 'notoriety',
    'obstruction', 'operation', 'order', 'outbreak', 'outcome', 'overhaul', 'patrols', 'pilgrimage',
    'plague', 'plan', 'practice', 'process', 'program', 'progress', 'project', 'pursuit', 'quest',
    'raids', 'reforms', 'reign', 'relationship', 'retaliation', 'riot', 'rise', 'rivalry', 'romance',
    'rule', 'sanctions', 'shift', 'siege', 'slump', 'stature', 'stint', 'strikes', 'study', 'test',
    'testing', 'tests', 'therapy', 'tour', 'tradition', 'treaty', 'trial', 'trip', 'unemployment',
    'voyage', 'warfare', 'work'
]

def convert_disk_sample_to_gt_format(disk_sample):
    """Convert a sample from the disk dataset format to the GT format expected by the code"""
    # Direct mapping: prefix -> clean_prompt, corr_prefix -> corrupted_prompt, digits -> threshold_suffix
    return {
        "clean_prompt": disk_sample['prefix'],
        "corrupted_prompt": disk_sample['corr_prefix'],
        "threshold_suffix": int(disk_sample['digits'])
    }

def load_or_generate_gt_data(
    dataset_path: str = "/u/amo-d1/grad/mha361/work/circuits/data/edge_pruning/datasets/gt",
    split: str = "train",
    num_samples: Optional[int] = None
) -> List[Dict]:
    """
    Try to load GT data from disk, fall back to generation if not available.
    
    Args:
        dataset_path: Path to the saved dataset
        split: Which split to load ('train', 'train_90k', 'validation', 'test')
        num_samples: Number of samples to generate if loading fails (None = use all from disk)
    
    Returns:
        List of dictionaries with GT sample pairs
    """
    try:
        print(f"Attempting to load dataset from: {dataset_path}")
        dataset_dict = load_from_disk(dataset_path)
        
        if split not in dataset_dict:
            raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset_dict.keys())}")
        
        dataset = dataset_dict[split]
        print(f"Successfully loaded {split} split with {len(dataset)} samples")
        
        # Convert all samples to the expected format
        gt_samples = []
        for sample in dataset:
            gt_samples.append(convert_disk_sample_to_gt_format(sample))
        
        # If num_samples specified and less than available, sample randomly
        if num_samples is not None and num_samples < len(gt_samples):
            gt_samples = random.sample(gt_samples, num_samples)
            print(f"Sampled {num_samples} from {len(dataset)} available samples")
        
        return gt_samples
        
    except Exception as e:
        print(f"Failed to load dataset from disk: {e}")
        print(f"Falling back to generating {num_samples or 1000} samples...")
        
        if num_samples is None:
            num_samples = 1000  # Default number of samples to generate
        
        return [generate_gt_sample_pair() for _ in range(num_samples)]

def generate_gt_sample_pair():
    """Original generation function as fallback"""
    noun = random.choice(NOUN_POOL)
    template = "The {noun} lasted from the year {year} to the year {prefix}"
    XX = random.randint(11, 17)
    YY = random.randint(2, 98)
    year1 = XX * 100 + YY
    clean_prompt = template.format(noun=noun, year=year1, prefix=str(XX))
    corrupted_year = XX * 100 + 1
    corrupted_prompt = template.format(noun=noun, year=corrupted_year, prefix=str(XX))
    return {"clean_prompt": clean_prompt, "corrupted_prompt": corrupted_prompt, "threshold_suffix": YY}

class GTDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: GPT2Tokenizer, max_length: int = 32):
        self.data, self.tokenizer, self.max_length = data, tokenizer, max_length
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        clean_inputs = self.tokenizer(item['clean_prompt'], padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        corrupted_inputs = self.tokenizer(item['corrupted_prompt'], padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        last_token_idx = clean_inputs['attention_mask'].squeeze().sum().item() - 1
        return {"clean_input_ids": clean_inputs['input_ids'].squeeze(0), "clean_attention_mask": clean_inputs['attention_mask'].squeeze(0),
                "corrupted_input_ids": corrupted_inputs['input_ids'].squeeze(0), "threshold_suffix": torch.tensor(item['threshold_suffix'], dtype=torch.long),
                "last_token_idx": torch.tensor(last_token_idx, dtype=torch.long)}

def create_two_digit_token_mapping(tokenizer):
    """Create a robust mapping of two-digit numbers to their token IDs"""
    two_digit_tokens = {}
    print("Creating two-digit token mapping...")
    for i in range(2, 99):  # Only include numbers that can be thresholds (2-98)
        candidates = [f"{i:02d}", f" {i:02d}", str(i), f" {i}"]
        for candidate in candidates:
            tokens = tokenizer.encode(candidate, add_special_tokens=False)
            if len(tokens) == 1:
                two_digit_tokens[i] = tokens[0]
                break
    print(f"Successfully mapped {len(two_digit_tokens)} two-digit numbers to tokens")
    return two_digit_tokens

def run_evaluation(model_to_eval, model_name: str, full_model_for_faithfulness: Optional[nn.Module], dataloader, device, two_digit_tokens, verbose=True, tokenizer=None):
    if verbose: print("\n" + "="*50 + f"\n  EVALUATING: {model_name} (with Re-Normalization)\n" + "="*50) # <-- Updated title
    model_to_eval.eval()
    if full_model_for_faithfulness: full_model_for_faithfulness.eval()
    if not two_digit_tokens: return {}

    # --- SETUP FOR RE-NORMALIZATION ---
    # Create a sorted list of the numbers and their corresponding token IDs
    # This is crucial for indexing the re-normalized probabilities correctly.
    sorted_tokens = sorted(two_digit_tokens.items()) # <-- NEW: Sort by number
    sorted_nums = [item[0] for item in sorted_tokens] # <-- NEW: Just the numbers [2, 3, ...]
    num_to_idx = {num: i for i, num in enumerate(sorted_nums)} # <-- NEW: Map number to its index in the sorted list
    digit_token_ids = torch.tensor([item[1] for item in sorted_tokens], device=device) # <-- NEW: Tensor of token IDs on the correct device

    all_prob_diffs, all_cutoff_sharpness, total_kl = [], [], 0.0
    valid_samples = 0
    desc = f"Evaluating {model_name}" if verbose else "Binary Searching"

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor): batch[key] = val.to(device)
            
            outputs = model_to_eval(input_ids=batch['clean_input_ids'], corrupted_input_ids=batch.get('corrupted_input_ids'), attention_mask=batch['clean_attention_mask'])
            
            # This part is the same: get the logits for the last token position
            last_token_logits = outputs.logits[torch.arange(outputs.logits.size(0)), batch['last_token_idx']-1, :]

            # --- RE-NORMALIZATION LOGIC ---
            # 1. Filter logits to only include our ~97 two-digit number tokens
            digit_logits = torch.gather( # <-- NEW
                last_token_logits,
                1, # Dimension to gather from (the vocabulary dimension)
                digit_token_ids.unsqueeze(0).expand(last_token_logits.shape[0], -1)
            )

            # 2. Apply softmax to this smaller, filtered logit tensor
            eval_probs = F.softmax(digit_logits, dim=-1) # <-- CHANGED: Now applied to digit_logits

            # --- UPDATED METRIC CALCULATION ---
            for i in range(eval_probs.size(0)): # <-- This loop iterates through items in the batch
                YY = batch['threshold_suffix'][i].item()
                if not (2 <= YY <= 98 and YY in num_to_idx): continue
                
                probs = eval_probs[i] # This is now a vector of ~97 probabilities that sum to 1
                
                yy_index = num_to_idx[YY] # Find the index corresponding to our threshold YY
                
                # Calculate prob_diff using tensor slicing for efficiency
                p_greater = probs[yy_index + 1:].sum() # <-- CHANGED
                p_less_equal = probs[:yy_index + 1].sum() # <-- CHANGED
                
                all_prob_diffs.append((p_greater - p_less_equal).item())

                # Cutoff sharpness logic also needs to use the new index mapping
                p_yy_plus_1 = probs[num_to_idx[YY + 1]].item() if (YY + 1) in num_to_idx else 0.0 # <-- CHANGED
                p_yy_minus_1 = probs[num_to_idx[YY - 1]].item() if (YY - 1) in num_to_idx else 0.0 # <-- CHANGED
                all_cutoff_sharpness.append(p_yy_plus_1 - p_yy_minus_1)
                valid_samples += 1

            # --- UPDATED FAITHFULNESS (KL DIVERGENCE) CALCULATION ---
            if full_model_for_faithfulness:
                full_model_outputs = full_model_for_faithfulness(input_ids=batch['clean_input_ids'], attention_mask=batch['clean_attention_mask'])
                last_full_logits = full_model_outputs.logits[torch.arange(full_model_outputs.logits.size(0)), batch['last_token_idx'], :]
                
                # Also filter the full model's logits to ensure a fair comparison
                full_digit_logits = torch.gather( # <-- NEW
                    last_full_logits, 1, digit_token_ids.unsqueeze(0).expand(last_full_logits.shape[0], -1)
                )
                
                # Compare the log_softmax of the two re-normalized distributions
                total_kl += F.kl_div( # <-- CHANGED
                    F.log_softmax(digit_logits, dim=-1), 
                    F.log_softmax(full_digit_logits, dim=-1), 
                    log_target=True, reduction='batchmean'
                ).item()

    avg_pd = sum(all_prob_diffs) / len(all_prob_diffs) if all_prob_diffs else 0
    avg_cs = sum(all_cutoff_sharpness) / len(all_cutoff_sharpness) if all_cutoff_sharpness else 0
    avg_kl = total_kl / len(dataloader) if len(dataloader) > 0 else 0
    
    if verbose:
        print(f"\nProcessed {valid_samples} valid samples.")
        print("\n" + "="*50)
        print(f"{model_name} Evaluation Summary (Re-Normalized):") # <-- Updated title
        if full_model_for_faithfulness: print(f"  - Faithfulness (KL Div):        {avg_kl:.4f}")
        print(f"  - Performance (Prob Diff):      {avg_pd:.4f}")
        print(f"  - Performance (Cutoff Sharpness): {avg_cs:.4f}")
        print("="*50)
    
    return {"prob_diff": avg_pd, "cutoff_sharpness": avg_cs, "kl_div": avg_kl}