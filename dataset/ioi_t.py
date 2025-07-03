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
    return {
        "sentence": disk_sample['ioi_sentences'],
        "corrupted_sentence": disk_sample['corr_ioi_sentences'],
        # Parse target and distractor from the sentence if not directly available
        "target": None,  # Will be computed during processing
        "distractor": None  # Will be computed during processing
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
    dataset_path: str = "/u/amo-d1/grad/mha361/work/circuits/data/datasets/ioi",
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
        
        # Process data to extract targets and distractors
        self.processed_data = []
        for item in data:
            sentence = item['sentence']
            corr_sentence = item['corrupted_sentence']
            
            # Find template to extract names
            template_info = find_template(sentence)
            if template_info is None:
                continue
            
            # The target is the last word in the sentence (should be name A)
            target = sentence.strip().split()[-1]
            # The distractor is the other name (B)
            distractor = template_info["b"] if template_info["a"] == target else template_info["a"]
            
            # Tokenize names with space prefix for consistency
            target_tokens = tokenizer.encode(" " + target, add_special_tokens=False)
            distractor_tokens = tokenizer.encode(" " + distractor, add_special_tokens=False)
            
            if len(target_tokens) == 1 and len(distractor_tokens) == 1:
                self.processed_data.append({
                    'sentence': sentence,
                    'corrupted_sentence': corr_sentence,
                    'target': target,
                    'distractor': distractor,
                    'target_token': target_tokens[0],
                    'distractor_token': distractor_tokens[0],
                    'template_order': template_info['order']
                })
        
        print(f"Processed {len(self.processed_data)} valid samples from {len(data)} total")
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        
        # Tokenize sentences
        inputs = self.tokenizer(
            item['sentence'], 
            padding='max_length', 
            max_length=self.max_length, 
            truncation=True, 
            return_tensors='pt'
        )
        
        corrupted_inputs = self.tokenizer(
            item['corrupted_sentence'], 
            padding='max_length', 
            max_length=self.max_length, 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Find the position before the last token (where we predict)
        # We need to find where the sentence actually ends (before padding)
        sentence_prefix = item['sentence'][:item['sentence'].rfind(" ")]
        prefix_length = len(self.tokenizer.encode(sentence_prefix, add_special_tokens=True))
        
        return {
            "input_ids": inputs['input_ids'].squeeze(0),
            "attention_mask": inputs['attention_mask'].squeeze(0),
            "corrupted_input_ids": corrupted_inputs['input_ids'].squeeze(0),
            "corrupted_attention_mask": corrupted_inputs['attention_mask'].squeeze(0),
            "target_token": torch.tensor(item['target_token'], dtype=torch.long),
            "distractor_token": torch.tensor(item['distractor_token'], dtype=torch.long),
            "prefix_length": torch.tensor(prefix_length, dtype=torch.long),
            "template_order": item['template_order']
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
                # Get the position where we make prediction (before last token)
                prefix_length = batch['prefix_length'][i].item()
                pred_position = prefix_length - 1
                
                # Get logits at prediction position
                logits = outputs.logits[i, pred_position, :]
                
                # Get target and distractor logits
                target_logit = logits[batch['target_token'][i]].item()
                distractor_logit = logits[batch['distractor_token'][i]].item()
                
                # Calculate logit difference
                logit_diff = target_logit - distractor_logit
                total_logit_diff += logit_diff
                
                # Calculate accuracy (model chooses target over distractor)
                if target_logit > distractor_logit:
                    total_accuracy += 1
                
                valid_samples += 1
            
            # Calculate faithfulness (KL divergence) if full model provided
            if full_model_for_faithfulness:
                full_outputs = full_model_for_faithfulness(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                for i in range(batch_size):
                    prefix_length = batch['prefix_length'][i].item()
                    pred_position = prefix_length - 1
                    
                    # Calculate KL divergence between distributions
                    model_logits = outputs.logits[i, pred_position, :]
                    full_logits = full_outputs.logits[i, pred_position, :]
                    
                    kl = F.kl_div(
                        F.log_softmax(model_logits, dim=-1),
                        F.log_softmax(full_logits, dim=-1),
                        log_target=True,
                        reduction='sum'
                    ).item()
                    total_kl += kl
                    
                    # Check exact match
                    model_choice = torch.argmax(model_logits)
                    full_choice = torch.argmax(full_logits)
                    if model_choice == full_choice:
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