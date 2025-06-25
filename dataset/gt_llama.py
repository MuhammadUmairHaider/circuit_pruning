import random
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_from_disk
import os

# ==============================================================================
# DATASET AND EVALUATION FOR LLAMA MODELS
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
    few_shot_examples = '''Example completions:
The war lasted from the year 1325 to the year 1367
The period lasted from the year 1156 to the year 1187

Now complete: '''
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
            num_samples = 1000
        
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
    """GT Dataset specifically designed for Llama models with proper special token handling"""
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Identify special tokens for Llama
        self.special_token_ids = set()
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            self.special_token_ids.add(tokenizer.bos_token_id)
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            self.special_token_ids.add(tokenizer.eos_token_id)
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            self.special_token_ids.add(tokenizer.pad_token_id)
            
        print(f"Initialized Llama dataset with special tokens: {self.special_token_ids}")
        
    def __len__(self):
        return len(self.data)
        
    def find_last_content_token_position(self, input_ids, attention_mask):
        """Find the position of the last actual content token (not special token)"""
        # Start from the last attended position and work backwards
        last_attended = attention_mask.sum().item() - 1
        
        for i in range(last_attended, -1, -1):
            token_id = input_ids[i].item()
            
            # Skip special tokens
            if token_id in self.special_token_ids:
                continue
                
            # Verify this is actual content by decoding
            decoded = self.tokenizer.decode([token_id], skip_special_tokens=True)
            if decoded.strip():  # Non-empty content
                return i
                
        # Fallback (shouldn't happen with valid inputs)
        return last_attended
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize the prompts
        clean_inputs = self.tokenizer(
            item['clean_prompt'],
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        corrupted_inputs = self.tokenizer(
            item['corrupted_prompt'],
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        # Get tensors
        clean_input_ids = clean_inputs['input_ids'].squeeze()
        clean_attention_mask = clean_inputs['attention_mask'].squeeze()
        corrupted_input_ids = corrupted_inputs['input_ids'].squeeze()
        
        # Find the last content token position
        last_token_idx = self.find_last_content_token_position(clean_input_ids, clean_attention_mask)
        
        # Debug first few samples
        if idx < 3:
            print(f"\n=== Sample {idx} ===")
            print(f"Original text: {item['clean_prompt']}")
            print(f"Threshold YY: {item['threshold_suffix']}")
            print(f"Tokenized (with special tokens): {self.tokenizer.decode(clean_input_ids, skip_special_tokens=False)}")
            print(f"Last content token position: {last_token_idx}")
            print(f"Token at that position: '{self.tokenizer.decode([clean_input_ids[last_token_idx].item()])}'")
            
            # Show a few tokens around the position
            start = max(0, last_token_idx - 2)
            end = min(len(clean_input_ids), last_token_idx + 3)
            print(f"Tokens around position:")
            for i in range(start, end):
                token = clean_input_ids[i].item()
                decoded = self.tokenizer.decode([token])
                is_special = token in self.special_token_ids
                print(f"  [{i}] {token}: '{decoded}' {'(SPECIAL)' if is_special else ''}")
        
        return {
            "clean_input_ids": clean_input_ids,
            "clean_attention_mask": clean_attention_mask,
            "corrupted_input_ids": corrupted_input_ids,
            "threshold_suffix": torch.tensor(item['threshold_suffix'], dtype=torch.long),
            "last_token_idx": torch.tensor(last_token_idx, dtype=torch.long)
        }

def create_two_digit_token_mapping(tokenizer):
    """Create a robust mapping of two-digit numbers to their token IDs"""
    two_digit_tokens = {}
    print("Creating two-digit token mapping for Llama...")
    
    for i in range(2, 99):  # Only include numbers that can be thresholds (2-98)
        candidates = [str(i), f" {i}", f"{i:02d}", f" {i:02d}"]
        
        for candidate in candidates:
            tokens = tokenizer.encode(candidate, add_special_tokens=False)
            if len(tokens) == 1:
                two_digit_tokens[i] = tokens[0]
                break
                
        # Debug: show what didn't map
        if i not in two_digit_tokens and i < 10:
            print(f"  Warning: Could not find single token for {i}")
            
    print(f"Successfully mapped {len(two_digit_tokens)} two-digit numbers to tokens")
    
    # Show some examples
    examples = [10, 25, 50, 75, 90]
    print("Sample mappings:")
    for ex in examples:
        if ex in two_digit_tokens:
            decoded = tokenizer.decode([two_digit_tokens[ex]])
            print(f"  {ex} -> token_id {two_digit_tokens[ex]} -> '{decoded}'")
    
    return two_digit_tokens

def run_evaluation(model_to_eval, model_name: str, full_model_for_faithfulness: Optional[nn.Module], 
                        dataloader, device, two_digit_tokens, verbose=True, tokenizer=None):
    """Evaluation function specifically for Llama models"""
    if verbose: 
        print("\n" + "="*50 + f"\n  EVALUATING: {model_name} (Llama-specific with Re-Normalization)\n" + "="*50)
    
    model_to_eval.eval()
    if full_model_for_faithfulness: 
        full_model_for_faithfulness.eval()
    
    if not two_digit_tokens: 
        print("ERROR: No two-digit token mappings found!")
        return {}

    # Setup for re-normalization
    sorted_tokens = sorted(two_digit_tokens.items())
    sorted_nums = [item[0] for item in sorted_tokens]
    num_to_idx = {num: i for i, num in enumerate(sorted_nums)}
    digit_token_ids = torch.tensor([item[1] for item in sorted_tokens], device=device)

    all_prob_diffs, all_cutoff_sharpness, total_kl = [], [], 0.0
    valid_samples = 0
    desc = f"Evaluating {model_name}" if verbose else "Binary Searching"
    
    # Debug flag - set to True to see detailed output for first few batches
    debug_mode = verbose and len(dataloader) > 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc, leave=False)):
            # Move batch to device
            for key, val in batch.items():
                if isinstance(val, torch.Tensor): 
                    batch[key] = val.to(device)
            
            # Get model outputs
            outputs = model_to_eval(
                input_ids=batch['clean_input_ids'], 
                attention_mask=batch['clean_attention_mask']
            )
            
            # Get logits for the last token position
            last_token_logits = outputs.logits[torch.arange(outputs.logits.size(0)), batch['last_token_idx'], :]

            # Filter logits to only include two-digit number tokens
            digit_logits = torch.gather(
                last_token_logits,
                1,
                digit_token_ids.unsqueeze(0).expand(last_token_logits.shape[0], -1)
            )

            # Apply softmax to get probabilities
            eval_probs = F.softmax(digit_logits, dim=-1)

            # Calculate metrics for each sample in batch
            for i in range(eval_probs.size(0)):
                YY = batch['threshold_suffix'][i].item()
                if not (2 <= YY <= 98 and YY in num_to_idx): 
                    continue
                
                probs = eval_probs[i]
                yy_index = num_to_idx[YY]
                
                # Calculate probability differences
                p_greater = probs[yy_index + 1:].sum() if yy_index + 1 < len(probs) else 0.0
                p_less_equal = probs[:yy_index + 1].sum()
                prob_diff = (p_greater - p_less_equal).item()
                
                all_prob_diffs.append(prob_diff)

                # Calculate cutoff sharpness
                p_yy_plus_1 = probs[num_to_idx[YY + 1]].item() if (YY + 1) in num_to_idx else 0.0
                p_yy_minus_1 = probs[num_to_idx[YY - 1]].item() if (YY - 1) in num_to_idx else 0.0
                all_cutoff_sharpness.append(p_yy_plus_1 - p_yy_minus_1)
                
                valid_samples += 1
                
                # Debug output for first few samples
                if debug_mode and batch_idx == 0 and i < 3:
                    print(f"\n--- Batch {batch_idx}, Sample {i} ---")
                    print(f"Input: {tokenizer.decode(batch['clean_input_ids'][i], skip_special_tokens=True)}")
                    print(f"Threshold YY: {YY}")
                    print(f"P(>{YY}): {p_greater:.4f}")
                    print(f"P(≤{YY}): {p_less_equal:.4f}")
                    print(f"Prob diff: {prob_diff:.4f}")
                    
                    # Show top 5 predictions
                    top_probs, top_indices = torch.topk(probs, min(5, len(probs)))
                    print("Top 5 predictions:")
                    for j, (p, idx) in enumerate(zip(top_probs, top_indices)):
                        num = sorted_nums[idx]
                        print(f"  {num}: {p:.4f}")

            # Calculate faithfulness (KL divergence) if full model provided
            if full_model_for_faithfulness:
                full_model_outputs = full_model_for_faithfulness(
                    input_ids=batch['clean_input_ids'], 
                    attention_mask=batch['clean_attention_mask']
                )
                last_full_logits = full_model_outputs.logits[torch.arange(full_model_outputs.logits.size(0)), batch['last_token_idx'], :]
                
                # Filter full model's logits too
                full_digit_logits = torch.gather(
                    last_full_logits, 1, 
                    digit_token_ids.unsqueeze(0).expand(last_full_logits.shape[0], -1)
                )
                
                # Calculate KL divergence
                total_kl += F.kl_div(
                    F.log_softmax(digit_logits, dim=-1), 
                    F.log_softmax(full_digit_logits, dim=-1), 
                    log_target=True, reduction='batchmean'
                ).item()

    # Calculate averages
    avg_pd = sum(all_prob_diffs) / len(all_prob_diffs) if all_prob_diffs else 0
    avg_cs = sum(all_cutoff_sharpness) / len(all_cutoff_sharpness) if all_cutoff_sharpness else 0
    avg_kl = total_kl / len(dataloader) if len(dataloader) > 0 else 0
    
    if verbose:
        print(f"\nProcessed {valid_samples} valid samples.")
        print("\n" + "="*50)
        print(f"{model_name} Evaluation Summary (Re-Normalized):")
        if full_model_for_faithfulness: 
            print(f"  - Faithfulness (KL Div):         {avg_kl:.4f}")
        print(f"  - Performance (Prob Diff):       {avg_pd:.4f}")
        print(f"  - Performance (Cutoff Sharpness): {avg_cs:.4f}")
        print("="*50)
    
    return {"prob_diff": avg_pd, "cutoff_sharpness": avg_cs, "kl_div": avg_kl}