import random
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_from_disk
import os

# ==============================================================================
# DATASET AND EVALUATION FOR GREATER-THAN TASK (EDGE PRUNING)
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

templates = [
    "The {noun} lasted from {start} to {end_century}",
    "The {noun} stretched from {start} to {end_century}",
    "The {noun} spanned the years {start} to {end_century}",
    "The {noun} unfolded from {start} to {end_century}",
    "The {noun} took place between {start} and {end_century}",
    "The {noun} persisted from {start} to {end_century}",
]


def convert_disk_sample_to_gt_format(disk_sample):
    """Convert a sample from the disk dataset format to the GT format expected by the code"""
    return {
        "clean_prompt": disk_sample['prefix'],
        "corrupted_prompt": disk_sample['corr_prefix'],
        "threshold_suffix": int(disk_sample['digits']),
        **disk_sample,
    }


def load_or_generate_gt_data(
    dataset_path: str = "/u/amo-d1/grad/mha361/work/circuits/data/datasets/gt_gen",
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

        gt_samples = []
        for sample in dataset:
            gt_samples.append(convert_disk_sample_to_gt_format(sample))

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
    template = random.choice(templates)
    XX = random.randint(11, 17)
    YY = random.randint(2, 98)
    year1 = XX * 100 + YY
    clean_prompt = template.format(noun=noun, start=year1, end_century=str(XX))
    corrupted_year = XX * 100 + 1
    corrupted_prompt = template.format(noun=noun, start=corrupted_year, end_century=str(XX))
    return {'clean_prompt': clean_prompt, 'corrupted_prompt': corrupted_prompt, 'threshold_suffix': YY}


class GTDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: GPT2Tokenizer, max_length: int = 32):
        self.data, self.tokenizer, self.max_length = data, tokenizer, max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        clean_text = item.get('prefix', item.get('clean_prompt'))
        clean_inputs = self.tokenizer(clean_text, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        corrupted_inputs = self.tokenizer(item['corrupted_prompt'], padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        last_token_idx = clean_inputs['attention_mask'].squeeze().sum().item() - 1
        return {
            "clean_input_ids": clean_inputs['input_ids'].squeeze(0),
            "clean_attention_mask": clean_inputs['attention_mask'].squeeze(0),
            "corrupted_input_ids": corrupted_inputs['input_ids'].squeeze(0),
            "threshold_suffix": torch.tensor(item['threshold_suffix'], dtype=torch.long),
            "last_token_idx": torch.tensor(last_token_idx, dtype=torch.long),
        }


def create_two_digit_token_mapping(tokenizer):
    """Create a robust mapping of two-digit numbers to their token IDs"""
    two_digit_tokens = {}
    print("Creating two-digit token mapping...")
    for i in range(0, 100):
        s = f" {i:02d}"
        enc = tokenizer.encode(s, add_special_tokens=False)
        assert len(enc) == 1, f"{s!r} does not map to a single token: {enc}"
        two_digit_tokens[i] = enc[0]
    print(f"Successfully mapped {len(two_digit_tokens)} two-digit numbers to tokens")
    return two_digit_tokens


def run_evaluation(model_to_eval, model_name: str, full_model_for_faithfulness: Optional[nn.Module],
                   dataloader, device, two_digit_tokens, verbose=True, tokenizer=None):
    if verbose:
        print("\n" + "="*50 + f"\n  EVALUATING: {model_name} (with Re-Normalization)\n" + "="*50)
    model_to_eval.eval()
    if full_model_for_faithfulness:
        full_model_for_faithfulness.eval()
    if not two_digit_tokens:
        return {}

    sorted_tokens = sorted(two_digit_tokens.items())
    sorted_nums = [item[0] for item in sorted_tokens]
    num_to_idx = {num: i for i, num in enumerate(sorted_nums)}
    digit_token_ids = torch.tensor([item[1] for item in sorted_tokens], device=device)

    all_prob_diffs, all_cutoff_sharpness, all_kl_divs, all_prob_diffs_global = [], [], [], []
    valid_samples = 0
    desc = f"Evaluating {model_name}" if verbose else "Evaluating"
    accuracy = 0.0
    n = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)

            outputs = model_to_eval(
                input_ids=batch['clean_input_ids'],
                corrupted_input_ids=batch.get('corrupted_input_ids'),
                attention_mask=batch['clean_attention_mask']
            )
            last_token_logits = outputs.logits[torch.arange(outputs.logits.size(0)), batch['last_token_idx'], :]

            digit_logits = torch.gather(
                last_token_logits, 1,
                digit_token_ids.unsqueeze(0).expand(last_token_logits.shape[0], -1)
            )

            eval_probs = F.softmax(digit_logits, dim=-1)
            W = 10
            N = eval_probs.size(1)

            if full_model_for_faithfulness:
                full_model_outputs = full_model_for_faithfulness(
                    input_ids=batch['clean_input_ids'],
                    attention_mask=batch['clean_attention_mask']
                )
                last_full_logits = full_model_outputs.logits[torch.arange(full_model_outputs.logits.size(0)), batch['last_token_idx'], :]
                full_digit_logits = torch.gather(
                    last_full_logits, 1,
                    digit_token_ids.unsqueeze(0).expand(last_full_logits.shape[0], -1)
                )

            for i in range(eval_probs.size(0)):
                YY = batch['threshold_suffix'][i].item()
                if YY not in num_to_idx:
                    continue

                probs = eval_probs[i]
                yy_index = num_to_idx[YY]

                if yy_index + W + 1 <= N:
                    p_greater = probs[yy_index + 1: yy_index + W + 1].sum()
                else:
                    p_greater = probs[yy_index + 1: N].sum()

                if yy_index - W >= 0:
                    p_less_equal = probs[yy_index - W: yy_index].sum()
                else:
                    p_less_equal = probs[0: yy_index].sum()

                n += 1

                all_prob_diffs.append((p_greater - p_less_equal).item())

                p_yy_plus = probs[num_to_idx[YY + 1]].item() if (YY + 1) in num_to_idx else 0.0
                p_yy_minus = probs[num_to_idx[YY - 1]].item() if (YY - 1) in num_to_idx else 0.0
                all_cutoff_sharpness.append(p_yy_plus - p_yy_minus)

                p_greater_global = probs[yy_index + 1:].sum()
                p_less_global = probs[:yy_index].sum()

                if p_greater > p_less_equal:
                    accuracy += 1.0
                all_prob_diffs_global.append((p_greater_global - p_less_global).item())

                if full_model_for_faithfulness:
                    window_start = max(0, yy_index - W)
                    window_end = min(N, yy_index + W + 1)

                    sparse_window_logits = digit_logits[i, window_start:window_end]
                    full_window_logits = full_digit_logits[i, window_start:window_end]

                    kl_div = F.kl_div(
                        F.log_softmax(sparse_window_logits, dim=-1),
                        F.log_softmax(full_window_logits, dim=-1),
                        log_target=True,
                        reduction='sum'
                    ).item()

                    all_kl_divs.append(kl_div)

                valid_samples += 1

    avg_pd = sum(all_prob_diffs) / len(all_prob_diffs) if all_prob_diffs else 0
    avg_cs = sum(all_cutoff_sharpness) / len(all_cutoff_sharpness) if all_cutoff_sharpness else 0
    avg_kl = sum(all_kl_divs) / len(all_kl_divs) if all_kl_divs else 0

    if verbose:
        print(f"\nProcessed {valid_samples} valid samples.")
        print("\n" + "="*50)
        print(f"{model_name} Evaluation Summary (Re-Normalized):")
        if full_model_for_faithfulness:
            print(f"  - Faithfulness (Windowed KL Div): {avg_kl:.4f}")
        print(f"  - Performance (Prob Diff):         {avg_pd:.4f}")
        print(f"  - Performance (Cutoff Sharpness):  {avg_cs:.4f}")
        print(f"  - Accuracy:                         {accuracy / n:.4f}" if n > 0 else "  - Accuracy:                         N/A")
        print(f"Non-Windowed Avg Prob Diff:        {sum(all_prob_diffs_global) / len(all_prob_diffs_global) if all_prob_diffs_global else 0:.4f}")
        print("="*50)

    avg_accuracy = accuracy / n if n > 0 else 0.0
    return {"prob_diff": avg_pd, "cutoff_sharpness": avg_cs, "kl_div": avg_kl, "accuracy": avg_accuracy}


def filter_dataset_by_model_correctness(data_list, model, tokenizer, device, two_digit_tokens, batch_size=32):
    """
    Filters the GT dataset, keeping only samples where the base model correctly
    predicts the year range (P(> year) > P(<= year)) using re-normalized probabilities.
    """
    if not data_list:
        return []

    print(f"Filtering {len(data_list)} samples for base model correctness...")

    sorted_tokens = sorted(two_digit_tokens.items())
    sorted_nums = [item[0] for item in sorted_tokens]
    num_to_idx = {num: i for i, num in enumerate(sorted_nums)}
    digit_token_ids = torch.tensor([item[1] for item in sorted_tokens], device=device)

    temp_dataset = GTDataset(data_list, tokenizer, max_length=32)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

    valid_indices = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(temp_loader, desc="Checking model predictions")):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)

            outputs = model(
                input_ids=batch['clean_input_ids'],
                attention_mask=batch['clean_attention_mask']
            )

            last_token_logits = outputs.logits[torch.arange(outputs.logits.size(0)), batch['last_token_idx'], :]

            digit_logits = torch.gather(
                last_token_logits, 1,
                digit_token_ids.unsqueeze(0).expand(last_token_logits.shape[0], -1)
            )

            eval_probs = F.softmax(digit_logits, dim=-1)

            W = 10
            N = eval_probs.size(1)
            current_batch_size = eval_probs.size(0)

            for i in range(current_batch_size):
                YY = batch['threshold_suffix'][i].item()

                if YY not in num_to_idx:
                    continue

                probs = eval_probs[i]
                yy_index = num_to_idx[YY]

                if yy_index + W + 1 <= N:
                    p_greater = probs[yy_index + 1: yy_index + W + 1].sum()
                else:
                    p_greater = probs[yy_index + 1: N].sum()

                if yy_index - W >= 0:
                    p_less_equal = probs[yy_index - W: yy_index].sum()
                else:
                    p_less_equal = probs[0: yy_index].sum()

                if p_greater > p_less_equal:
                    global_idx = (batch_idx * batch_size) + i
                    valid_indices.append(global_idx)

    filtered_data = [data_list[i] for i in valid_indices]

    print(f"  -> Retained: {len(filtered_data)}/{len(data_list)} "
          f"({len(filtered_data)/len(data_list)*100:.2f}%)")

    return filtered_data
