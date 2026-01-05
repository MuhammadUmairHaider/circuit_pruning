import random
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_from_disk
import os

# ==============================================================================
# DATASET AND EVALUATION (ALIGNED WITH WANG ET AL., 2023)
# ==============================================================================

# IOI Templates from the original paper
BABA_TEMPLATES = [
    "Then, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} had a lot of fun at the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} were working at the {PLACE}. {B} decided to give a {OBJECT} to {A}",
    "Then, {B} and {A} were thinking about going to the {PLACE}. {B} wanted to give a {OBJECT} to {A}",
    "Then, {B} and {A} had a long argument, and afterwards {B} said to {A}",
    "After {B} and {A} went to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "When {B} and {A} got a {OBJECT} at the {PLACE}, {B} decided to give it to {A}",
    "When {B} and {A} got a {OBJECT} at the {PLACE}, {B} decided to give the {OBJECT} to {A}",
    "While {B} and {A} were working at the {PLACE}, {B} gave a {OBJECT} to {A}",
    "While {B} and {A} were commuting to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "After the lunch, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Afterwards, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} had a long argument. Afterwards {B} said to {A}",
    "The {PLACE} {B} and {A} went to had a {OBJECT}. {B} gave it to {A}",
    "Friends {B} and {A} found a {OBJECT} at the {PLACE}. {B} gave it to {A}",
]

ABBA_TEMPLATES = [
    "Then, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {A} and {B} had a lot of fun at the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {A} and {B} were working at the {PLACE}. {B} decided to give a {OBJECT} to {A}",
    "Then, {A} and {B} were thinking about going to the {PLACE}. {B} wanted to give a {OBJECT} to {A}",
    "Then, {A} and {B} had a long argument, and afterwards {B} said to {A}",
    "After {A} and {B} went to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "When {A} and {B} got a {OBJECT} at the {PLACE}, {B} decided to give it to {A}",
    "When {A} and {B} got a {OBJECT} at the {PLACE}, {B} decided to give the {OBJECT} to {A}",
    "While {A} and {B} were working at the {PLACE}, {B} gave a {OBJECT} to {A}",
    "While {A} and {B} were commuting to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "After the lunch, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Afterwards, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {A} and {B} had a long argument. Afterwards {B} said to {A}",
    "The {PLACE} {A} and {B} went to had a {OBJECT}. {B} gave it to {A}",
    "Friends {A} and {B} found a {OBJECT} at the {PLACE}. {B} gave it to {A}",
]

def convert_disk_sample_to_ioi_format(disk_sample):
    """Convert a sample from the disk dataset format to the IOI format expected by the code"""
    # print("keys in disk_sample:", disk_sample.keys())
    return {
        **disk_sample,
        "sentence": disk_sample['ioi_sentences'],
        "corrupted_sentence": disk_sample['corr_ioi_sentences'],
        # Parse target and distractor from the sentence if not directly available
        "target": None,  # Will be computed during processing
        "distractor": None,  # Will be computed during processing
        
    }

def try_fit_template(string: str, template: str) -> Optional[Dict[str, str]]:
    """Try to fit a sentence to a template and extract placeholders"""
    pieces_s, pieces_t = string.strip().split(), template.strip().split()
    
    if len(pieces_s) != len(pieces_t):
        return None
    
    mapping = {}
    
    for s, t in zip(pieces_s, pieces_t):
        if s == t:
            continue
        # Handle punctuation
        if s[-1] == t[-1] and s[-1] in [',', '.']:
            s, t = s[:-1], t[:-1]
        if t not in ['{A}', '{B}', '{PLACE}', '{OBJECT}']:
            return None
        elif t[1:-1].lower() in mapping:
            if mapping[t[1:-1].lower()] != s:
                return None
        else:
            mapping[t[1:-1].lower()] = s
    
    # Add None for missing optional placeholders
    if 'place' not in mapping:
        mapping['place'] = None
    if 'object' not in mapping:
        mapping['object'] = None
    
    return mapping

def find_template(string: str) -> Optional[Dict[str, str]]:
    """Find which template matches the given sentence"""
    # Try BABA templates first
    for template in BABA_TEMPLATES:
        mapping = try_fit_template(string, template)
        if mapping is not None:
            mapping.update({
                'template': template,
                'order': 'baba'
            })
            return mapping
    
    # Try ABBA templates
    for template in ABBA_TEMPLATES:
        mapping = try_fit_template(string, template)
        if mapping is not None:
            mapping.update({
                'template': template,
                'order': 'abba'
            })
            return mapping
    
    return None

def load_or_generate_ioi_data(
    dataset_path: str = "/u/amo-d1/grad/mha361/work/circuits/filtered_datasets/ioi",
    split: str = "train",
    num_samples: Optional[int] = None
) -> List[Dict]:
    """
    Try to load IOI data from disk, fall back to generation if not available.
    
    Args:
        dataset_path: Path to the saved dataset
        split: Which split to load ('train', 'validation', 'test')
        num_samples: Number of samples to use (None = use all from disk)
    
    Returns:
        List of dictionaries with IOI sample pairs
    """
    try:
        print(f"Attempting to load dataset from: {dataset_path}")
        dataset_dict = load_from_disk(dataset_path)
        
        if split not in dataset_dict:
            raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset_dict.keys())}")
        
        dataset = dataset_dict[split]
        print(f"Successfully loaded {split} split with {len(dataset)} samples")
        
        # Convert all samples to the expected format
        ioi_samples = []
        for sample in dataset:
            ioi_samples.append(convert_disk_sample_to_ioi_format(sample))
        
        # If num_samples specified and less than available, sample randomly
        if num_samples is not None and num_samples < len(ioi_samples):
            ioi_samples = random.sample(ioi_samples, num_samples)
            print(f"Sampled {num_samples} from {len(dataset)} available samples")
        
        return ioi_samples
        
    except Exception as e:
        print(f"Failed to load dataset from disk: {e}")
        print(f"Please ensure the IOI dataset is available at {dataset_path}")
        raise

class IOIDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: GPT2Tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Pre-processing and tokenizing {len(data)} samples...")
        
        self.input_ids = []
        self.attention_mask = []
        self.corrupted_input_ids = []
        self.T_Start = []
        self.T_End = []
        self.D_Start = []
        self.D_End = []      # Added
        self.T_len = []      # Added
        self.D_len = []      # Added
        self.target_tokens = []
        self.distractor_tokens = []
        
        valid_count = 0
        for item in tqdm(data, desc="Pre-processing"):
            sentence = item['sentence']
            corr_sentence = item['corrupted_sentence']
            
            # --- 1. Robust Target/Distractor Extraction ---
            target = item.get('target')
            distractor = item.get('distractor')
            
            if target is None:
                target = sentence.strip().split()[-1]
            
            if distractor is None:
                if 'a' in item and 'b' in item:
                    distractor = item["b"] if item["a"] == target else item["a"]
                else:
                    continue 
            
            # --- 2. Tokenize Names ---
            tgt_tokens = self.tokenizer.encode(" " + target, add_special_tokens=False)
            dist_tokens = self.tokenizer.encode(" " + distractor, add_special_tokens=False)
            
            if len(tgt_tokens) == 0 or len(dist_tokens) == 0:
                continue

            # --- 3. Tokenize Sentences ---
            inputs = self.tokenizer(
                sentence, padding='max_length', max_length=self.max_length, 
                truncation=True, return_tensors='pt'
            )
            corrupted_inputs = self.tokenizer(
                corr_sentence, padding='max_length', max_length=self.max_length, 
                truncation=True, return_tensors='pt'
            )
            
            # --- 4. Calculate Indices (Matches original logic) ---
            sentence_prefix = sentence[:sentence.rfind(" ")]
            prefix_tokens = self.tokenizer.encode(sentence_prefix, add_special_tokens=False)
            
            # T_Start is the length of the prefix.
            # In your original code, T_Start was effectively 1-based index of the target
            t_start_val = len(prefix_tokens)
            
            # We enforce single-token evaluation for speed, but store lengths for compatibility
            t_len_val = 1 
            d_len_val = 1
            
            t_end_val = t_start_val + t_len_val
            d_start_val = t_start_val  # Distractor evaluated at same position as target
            d_end_val = t_end_val
            
            # Append all
            self.input_ids.append(inputs['input_ids'].squeeze(0))
            self.attention_mask.append(inputs['attention_mask'].squeeze(0))
            self.corrupted_input_ids.append(corrupted_inputs['input_ids'].squeeze(0))
            
            self.T_Start.append(torch.tensor(t_start_val, dtype=torch.long))
            self.T_End.append(torch.tensor(t_end_val, dtype=torch.long))
            self.D_Start.append(torch.tensor(d_start_val, dtype=torch.long))
            self.D_End.append(torch.tensor(d_end_val, dtype=torch.long)) # Added
            
            self.T_len.append(torch.tensor(t_len_val, dtype=torch.long)) # Added
            self.D_len.append(torch.tensor(d_len_val, dtype=torch.long)) # Added
            
            self.target_tokens.append(torch.tensor(tgt_tokens[0], dtype=torch.long))
            self.distractor_tokens.append(torch.tensor(dist_tokens[0], dtype=torch.long))
            
            valid_count += 1

        print(f"Kept {valid_count} valid samples.")

        # Stack tensors
        self.input_ids = torch.stack(self.input_ids)
        self.attention_mask = torch.stack(self.attention_mask)
        self.corrupted_input_ids = torch.stack(self.corrupted_input_ids)
        self.T_Start = torch.stack(self.T_Start)
        self.T_End = torch.stack(self.T_End)
        self.D_Start = torch.stack(self.D_Start)
        self.D_End = torch.stack(self.D_End)
        self.T_len = torch.stack(self.T_len)
        self.D_len = torch.stack(self.D_len)
        self.target_tokens = torch.stack(self.target_tokens)
        self.distractor_tokens = torch.stack(self.distractor_tokens)

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "corrupted_input_ids": self.corrupted_input_ids[idx],
            "target_tokens": self.target_tokens[idx].unsqueeze(0),
            "distractor_tokens": self.distractor_tokens[idx].unsqueeze(0),
            "T_Start": self.T_Start[idx],
            "T_End": self.T_End[idx],
            "D_Start": self.D_Start[idx],
            "D_End": self.D_End[idx],  # Added
            "T_len": self.T_len[idx],  # Added
            "D_len": self.D_len[idx],  # Added
        }

def run_evaluation(
    model_to_eval, 
    model_name: str, 
    full_model_for_faithfulness: Optional[nn.Module], 
    dataloader, 
    device, 
    verbose=True, 
    tokenizer=None
):
    """Run evaluation on IOI task"""
    if verbose:
        print("\n" + "="*50 + f"\n  EVALUATING: {model_name}\n" + "="*50)
    
    model_to_eval.eval()
    if full_model_for_faithfulness:
        full_model_for_faithfulness.eval()
    
    total_accuracy = 0
    total_logit_diff = 0
    total_kl = 0.0
    total_exact_match = 0
    valid_samples = 0
    
    desc = f"Evaluating {model_name}" if verbose else "Evaluating"
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            # Move batch to device
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)
            
            # Get model outputs
            outputs = model_to_eval(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask'],
                corrupted_input_ids=batch.get('corrupted_input_ids')
            )
            
            batch_size = outputs.logits.size(0)
            
            for i in range(batch_size):
                # Get positions for target and distractor
                t_start = batch['T_Start'][i].item()-1
                t_end = batch['T_End'][i].item()-1
                d_start = batch['D_Start'][i].item()-1
                d_end = batch['D_End'][i].item()-1
                
                # Get target and distractor token IDs
                target_tokens = batch['target_tokens'][i][:batch['T_len'][i].item()]  # Can be multiple tokens
                distractor_tokens = batch['distractor_tokens'][i][:batch['D_len'][i].item()]  # Can be multiple tokens

                # Calculate average logit difference across all target/distractor positions
                target_logits = []
                distractor_logits = []
                
                # Collect logits for target tokens at their positions
                for pos_idx, pos in enumerate(range(t_start, t_end)):
                    if pos < outputs.logits.size(1):  # Check bounds
                        token_id = target_tokens[pos_idx] if pos_idx < len(target_tokens) else target_tokens[0]
                        logit = outputs.logits[i, pos, token_id].item()
                        target_logits.append(logit)
                
                # Collect logits for distractor tokens at target positions 
                # (what would the model assign to distractor tokens at target positions)
                for pos_idx, pos in enumerate(range(d_start, d_end)):
                    if pos < outputs.logits.size(1):  # Check bounds
                        token_id = distractor_tokens[pos_idx] if pos_idx < len(distractor_tokens) else distractor_tokens[0]
                        logit = outputs.logits[i, pos, token_id].item()
                        distractor_logits.append(logit)
                
                avg_target_logit = target_logits[0]
                avg_distractor_logit = distractor_logits[0]
                
                # Calculate average logit difference
                if target_logits and distractor_logits:
                    # avg_target_logit = sum(target_logits) / len(target_logits)
                    # avg_distractor_logit = sum(distractor_logits) / len(distractor_logits)
                    logit_diff = avg_target_logit - avg_distractor_logit
                    total_logit_diff += logit_diff
                    
                    # Calculate accuracy (model prefers target over distractor on average)
                    if avg_target_logit >= avg_distractor_logit:
                        total_accuracy += 1
                
                valid_samples += 1
            
            # Calculate faithfulness (KL divergence) if full model provided
            if full_model_for_faithfulness:
                full_outputs = full_model_for_faithfulness(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                for i in range(batch_size):
                    # Calculate KL from T_Start to end of sequence (or T_End + some margin)
                    t_start = batch['T_Start'][i].item()-1
                    t_end = batch['T_End'][i].item()-1
                    
                    # Get valid sequence length (before padding)
                    valid_length = batch['attention_mask'][i].sum().item()
                    
                    # Calculate KL from target start position to end of valid sequence
                    if t_start < valid_length:
                        model_logits = outputs.logits[i, t_start:t_end, :]
                        full_logits = full_outputs.logits[i, t_start:t_end, :]
                        
                        kl = F.kl_div(
                            F.log_softmax(model_logits, dim=-1),
                            F.log_softmax(full_logits, dim=-1),
                            log_target=True,
                            reduction='sum'
                        ).item()
                        
                        # Normalize by number of tokens
                        num_tokens = t_end - t_start
                        kl = kl / num_tokens if num_tokens > 0 else kl
                        
                        total_kl += kl
                    
                    # Check exact match at target positions
                    model_pred = torch.argmax(outputs.logits[i, t_start, :])
                    full_pred = torch.argmax(full_outputs.logits[i, t_start, :])
                    if model_pred == full_pred:
                        total_exact_match += 1
    
    # Calculate averages
    avg_accuracy = total_accuracy / valid_samples if valid_samples > 0 else 0
    avg_logit_diff = total_logit_diff / valid_samples if valid_samples > 0 else 0
    avg_kl = total_kl / valid_samples if valid_samples > 0 else 0
    exact_match_rate = total_exact_match / valid_samples if valid_samples > 0 else 0
    
    if verbose:
        print(f"\nProcessed {valid_samples} valid samples.")
        print("\n" + "="*50)
        print(f"{model_name} Evaluation Summary:")
        print(f"  - Accuracy:              {avg_accuracy:.4f}")
        print(f"  - Logit Difference:      {avg_logit_diff:.4f}")
        if full_model_for_faithfulness:
            print(f"  - Faithfulness (KL Div): {avg_kl:.4f}")
            print(f"  - Exact Match Rate:      {exact_match_rate:.4f}")
        print("="*50)
    
    return {
        "accuracy": avg_accuracy,
        "logit_diff": avg_logit_diff,
        "kl_div": avg_kl,
        "exact_match": exact_match_rate
    }
    
    
from torch.utils.data import DataLoader
def filter_dataset_by_model_correctness(data_list, model, tokenizer, device, batch_size=32):
    """
    Filters a list of raw IOI data samples, keeping only those where the base model 
    assigns a higher logit to the target than the distractor (matching run_evaluation logic).
    """
    if not data_list:
        return []

    print(f"Filtering {len(data_list)} samples for base model correctness...")
    
    # Create temporary dataset/loader
    # Assumes your dataset class is named IOIDataset
    temp_dataset = IOIDataset(data_list, tokenizer) 
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)
    
    valid_indices = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(temp_loader, desc="Checking model predictions")):
            # Move batch to device
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            
            current_batch_size = outputs.logits.size(0)
            
            for i in range(current_batch_size):
                # Replicating the logic from your run_evaluation exactly
                t_start = batch['T_Start'][i].item() - 1
                d_start = batch['D_Start'][i].item() - 1
                
                # Get the specific target and distractor tokens used in eval
                # Note: Your eval code takes the first token of the target/distractor span
                target_token_id = batch['target_tokens'][i][0].item()
                distractor_token_id = batch['distractor_tokens'][i][0].item()

                # Extract logits
                # Using [t_start] because your eval code essentially does target_logits[0]
                target_logit = outputs.logits[i, t_start, target_token_id].item()
                
                # Using [d_start] logic from your code implies comparing target vs distractor strength
                # However, usually IOI compares (Target - Distractor) at the PREDICTION position (End of sentence).
                # Your run_evaluation snippet seems to look at t_start/d_start. 
                # I will strictly follow your `avg_target_logit` vs `avg_distractor_logit` logic:
                
                # Re-reading your snippet:
                # It collects logits at t_start...t_end. Then takes index 0.
                target_logit = outputs.logits[i, t_start, target_token_id].item()
                
                # For distractor logic in your snippet:
                # It iterates d_start...d_end. Then takes index 0.
                distractor_logit = outputs.logits[i, d_start, distractor_token_id].item()

                # Check "Correctness" defined by your eval: Target > Distractor
                if target_logit >= distractor_logit:
                    global_idx = (batch_idx * batch_size) + i
                    valid_indices.append(global_idx)

    # Reconstruct the list
    filtered_data = [data_list[i] for i in valid_indices]
    
    print(f"  -> Retained: {len(filtered_data)}/{len(data_list)} "
          f"({len(filtered_data)/len(data_list)*100:.2f}%)")
    
    return filtered_data