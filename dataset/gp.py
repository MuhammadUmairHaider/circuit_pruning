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
# DATASET AND EVALUATION FOR GENDER PRONOUNS TASK
# ==============================================================================

def convert_disk_sample_to_gp_format(disk_sample):
    """Convert a sample from the disk dataset format to the GP format expected by the code"""
    return {
        "sentence": disk_sample['prefix'] + " " + disk_sample['pronoun'],
        "corrupted_sentence": disk_sample['corr_prefix'] + " " + disk_sample['corr_pronoun'],
        "target": disk_sample['pronoun'].lower().strip(),
        "prefix": disk_sample['prefix'],
        "corr_prefix": disk_sample['corr_prefix']
    }

def load_or_generate_gp_data(
    dataset_path: str = "./data/datasets/gp/",
    split: str = "test",
    num_samples: Optional[int] = None
) -> List[Dict]:
    """
    Load GP data from disk.
    
    Args:
        dataset_path: Path to the saved dataset
        split: Which split to load (typically 'test' for GP)
        num_samples: Number of samples to use (None = use all from disk)
    
    Returns:
        List of dictionaries with GP sample pairs
    """
    try:
        print(f"Attempting to load dataset from: {dataset_path}")
        dataset_dict = load_from_disk(dataset_path)
        
        if split not in dataset_dict:
            raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset_dict.keys())}")
        
        dataset = dataset_dict[split]
        print(f"Successfully loaded {split} split with {len(dataset)} samples")
        
        # Convert all samples to the expected format
        gp_samples = []
        for i in range(len(dataset)):
            if num_samples is not None and i >= num_samples:
                break
            gp_samples.append(convert_disk_sample_to_gp_format(dataset[i]))
        
        print(f"Loaded {len(gp_samples)} samples")
        return gp_samples
        
    except Exception as e:
        print(f"Failed to load dataset from disk: {e}")
        print(f"Please ensure the GP dataset is available at {dataset_path}")
        raise

class GPDataset(Dataset):
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
            target = item['target']
            
            # Determine distractor based on target
            distractor = "he" if target == "she" else "she"
            
            # Tokenize pronouns with space prefix
            target_tokens = tokenizer.encode(" " + target, add_special_tokens=False)
            distractor_tokens = tokenizer.encode(" " + distractor, add_special_tokens=False)
            
            # Only keep samples where both pronouns tokenize to single tokens
            if len(target_tokens) == 1 and len(distractor_tokens) == 1:
                self.processed_data.append({
                    'sentence': sentence,
                    'corrupted_sentence': corr_sentence,
                    'target': target,
                    'distractor': distractor,
                    'target_token': target_tokens[0],
                    'distractor_token': distractor_tokens[0],
                    'prefix': item['prefix']
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
        sentence_prefix = item['sentence'][:item['sentence'].rfind(" ")]
        prefix_length = len(self.tokenizer.encode(sentence_prefix, add_special_tokens=True))
        
        return {
            "input_ids": inputs['input_ids'].squeeze(0),
            "attention_mask": inputs['attention_mask'].squeeze(0),
            "corrupted_input_ids": corrupted_inputs['input_ids'].squeeze(0),
            "corrupted_attention_mask": corrupted_inputs['attention_mask'].squeeze(0),
            "target_token": torch.tensor(item['target_token'], dtype=torch.long),
            "distractor_token": torch.tensor(item['distractor_token'], dtype=torch.long),
            "prefix_length": torch.tensor(prefix_length, dtype=torch.long)
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
    """Run evaluation on Gender Pronouns task"""
    if verbose:
        print("\n" + "="*50 + f"\n  EVALUATING: {model_name}\n" + "="*50)
    
    model_to_eval.eval()
    if full_model_for_faithfulness:
        full_model_for_faithfulness.eval()
    
    accuracy = 0
    logit_difference = 0
    kl_divergence = 0
    exact_match = 0
    outputs_ = []
    
    # Get total number of samples
    total_samples = len(dataloader.dataset)
    
    desc = f"Evaluating {model_name}" if verbose else "Evaluating"
    bar = tqdm(range(0, total_samples, dataloader.batch_size), desc=desc)
    
    sample_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_size = batch['input_ids'].shape[0]
            
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            corr_input_ids = batch['corrupted_input_ids'].to(device)
            
            # Get prefix lengths for this batch
            prefix_lengths = batch['prefix_length'].tolist()
            targets = batch['target_token'].to(device)
            distractors = batch['distractor_token'].to(device)
            
            # Get control model outputs
            control_outputs = full_model_for_faithfulness(input_ids, attention_mask=attention_mask) if full_model_for_faithfulness else None
            control_logits = control_outputs.logits if control_outputs else None
            
            # Get model outputs
            outputs = model_to_eval(
                input_ids=input_ids,
                corrupted_input_ids=corr_input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            
            # Process each item in batch
            for j in range(batch_size):
                prefix_length = prefix_lengths[j]
                
                # Get logits at prediction position
                # logit_target = logits[j, prefix_length-1, targets[i+j]].detach().cpu().item()
                logit_target = logits[j, prefix_length-1, targets[j]].detach().cpu().item()
                logit_distractor = logits[j, prefix_length-1, distractors[j]].detach().cpu().item()
                logit_difference += logit_target - logit_distractor
                
                # Get chosen word
                chosen_word = tokenizer.decode(torch.argmax(logits[j, prefix_length-1]).item())
                
                # Calculate KL divergence if control model available
                if control_logits is not None:
                    logits_ = F.log_softmax(logits[j, prefix_length-1], dim=-1)
                    control_logits_ = F.log_softmax(control_logits[j, prefix_length-1], dim=-1)
                    kld = F.kl_div(logits_, control_logits_, reduction="sum", log_target=True)
                    kl_divergence += kld.detach().cpu().item()
                
                # Check accuracy (exact match with argmax)
                choice = torch.argmax(logits[j, prefix_length-1])
                accuracy += (choice == targets[j]).int().detach().cpu().item()
                
                # Check exact match with control model
                if control_logits is not None:
                    control_choice = torch.argmax(control_logits[j, prefix_length-1])
                    exact_match += (choice == control_choice).int().detach().cpu().item()
                
                # Store outputs
                outputs_.append({
                    "sentence": dataloader.dataset.processed_data[sample_idx]['sentence'],
                    "target": tokenizer.decode(targets[j].item()),
                    "distractor": tokenizer.decode(distractors[j].item()),
                    "chosen_word": chosen_word,
                    "logit_target": logit_target,
                    "logit_distractor": logit_distractor,
                    "logit_difference": logit_target - logit_distractor,
                    "choice": choice.item(),
                })
                
                sample_idx += 1
            
            # Update progress bar
            bar.update(batch_size)
            current_total = min(sample_idx, total_samples)
            bar.set_description(f"Acc: {accuracy/current_total:.3f}, LD: {logit_difference/current_total:.3f}")
    
    bar.close()
    
    # Calculate final averages
    accuracy /= total_samples
    logit_difference /= total_samples
    kl_divergence /= total_samples
    exact_match /= total_samples
    
    if verbose:
        print(f"\nProcessed {total_samples} valid samples.")
        print("\n" + "="*50)
        print(f"{model_name} Evaluation Summary:")
        print(f"  - Accuracy:              {accuracy:.4f}")
        print(f"  - Logit Difference:      {logit_difference:.4f}")
        if full_model_for_faithfulness:
            print(f"  - KL Divergence:         {kl_divergence:.4f}")
            print(f"  - Exact Match:           {exact_match:.4f}")
        print("="*50)
    
    return {
        "accuracy": accuracy,
        "logit_diff": logit_difference,
        "kl_div": kl_divergence,
        "exact_match": exact_match,
        "outputs": outputs_
    }