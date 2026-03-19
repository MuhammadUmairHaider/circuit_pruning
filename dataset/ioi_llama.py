"""
IOI (Indirect Object Identification) Dataset and Evaluation for Llama Models.

Generates IOI data on-the-fly using templates and names, adapted for Llama's
tokenizer. Filters names to those that are single-token under the Llama tokenizer.
"""

import os
import json
import random
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ==============================================================================
# IOI TEMPLATES
# ==============================================================================

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

PLACES = [
    "store", "market", "hospital", "school", "park", "library", "beach",
    "restaurant", "airport", "station", "office", "church", "gym", "zoo",
    "museum", "theater", "garden", "mall", "hotel", "cafe"
]

OBJECTS = [
    "ring", "kiss", "bone", "basketball", "book", "drink", "necklace",
    "computer", "letter", "ball", "guitar", "pen", "phone", "cake",
    "flower", "hat", "bottle", "toy", "watch", "key"
]


# ==============================================================================
# NAME FILTERING FOR LLAMA TOKENIZER
# ==============================================================================

def load_names(names_path: Optional[str] = None) -> List[str]:
    """Load the names list from names.json."""
    if names_path is None:
        names_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "names.json")

    with open(names_path, 'r') as f:
        names_data = json.load(f)

    all_names = names_data.get("girls", []) + names_data.get("boys", [])
    return list(set(all_names))


def filter_single_token_names(names: List[str], tokenizer) -> List[str]:
    """
    Filter names to only those that tokenize as a single token with the given tokenizer.
    Checks " Name" (with space prefix) since that's how names appear mid-sentence.
    """
    single_token_names = []
    for name in names:
        tokens = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(tokens) == 1:
            single_token_names.append(name)

    print(f"Filtered {len(single_token_names)} single-token names from {len(names)} total names")
    return single_token_names


# ==============================================================================
# IOI DATA GENERATION
# ==============================================================================

def generate_ioi_data_llama(
    num_samples: int,
    tokenizer,
    names_path: Optional[str] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate IOI data samples on-the-fly, filtering for names that are
    single-token under the Llama tokenizer.

    Corruption strategy: replace name A with name B so the corrupted sentence
    has B appearing twice. This removes the indirect object signal, forcing
    the circuit to rely on the "find the non-repeated name" computation.

    Returns list of dicts with keys:
        - sentence, corrupted_sentence, a, b, ioi_sentences, corr_ioi_sentences
    """
    random.seed(seed)

    all_names = load_names(names_path)
    valid_names = filter_single_token_names(all_names, tokenizer)

    if len(valid_names) < 2:
        raise ValueError(f"Not enough single-token names for IOI task. Found {len(valid_names)}")

    all_templates = BABA_TEMPLATES + ABBA_TEMPLATES
    samples = []

    for _ in range(num_samples):
        # Pick two different names
        name_a, name_b = random.sample(valid_names, 2)

        # Pick template, place, object
        template = random.choice(all_templates)
        place = random.choice(PLACES)
        obj = random.choice(OBJECTS)

        # Generate clean sentence: A and B are distinct names
        sentence = template.format(A=name_a, B=name_b, PLACE=place, OBJECT=obj)

        # Generate corrupted sentence: replace A with B so B appears twice.
        # This is the standard IOI corruption — the model can no longer
        # distinguish indirect object from subject since both are B.
        corrupted_sentence = template.format(A=name_b, B=name_b, PLACE=place, OBJECT=obj)

        # Determine order (BABA or ABBA)
        order = "baba" if template in BABA_TEMPLATES else "abba"

        samples.append({
            "sentence": sentence,
            "corrupted_sentence": corrupted_sentence,
            "ioi_sentences": sentence,
            "corr_ioi_sentences": corrupted_sentence,
            "a": name_a,
            "b": name_b,
            "template_order": order,
        })

    print(f"Generated {len(samples)} IOI samples")
    return samples


# ==============================================================================
# IOI DATASET CLASS FOR LLAMA
# ==============================================================================

class IOIDatasetLlama(Dataset):
    """IOI Dataset adapted for Llama tokenizer."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.processed_data = []

        for item in data:
            sentence = item['sentence']
            corr_sentence = item['corrupted_sentence']

            # Target is the last word (name A) — what the model should predict
            target = sentence.strip().split()[-1]
            # Distractor is the repeated name (B)
            distractor = item["b"]

            # Tokenize names with space prefix
            target_tokens = tokenizer.encode(" " + target, add_special_tokens=False)
            distractor_tokens = tokenizer.encode(" " + distractor, add_special_tokens=False)

            self.processed_data.append({
                **item,
                'sentence': sentence,
                'corrupted_sentence': corr_sentence,
                'target': target,
                'distractor': distractor,
                'target_tokens': target_tokens,
                'distractor_tokens': distractor_tokens,
                'template_order': item.get('template_order', 'unknown'),
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

        # Find position of the target (last real token before padding)
        # IMPORTANT: use add_special_tokens=True so that T_Start accounts for BOS token
        sentence_prefix = item['sentence'][:item['sentence'].rfind(" ")]
        T_Start = len(self.tokenizer.encode(sentence_prefix, add_special_tokens=True))
        T_End = T_Start + len(item['target_tokens'])
        T_len = T_End - T_Start

        D_Start = T_Start
        D_End = D_Start + len(item['distractor_tokens'])
        D_len = D_End - D_Start

        # Pad target/distractor tokens to fixed size
        target_tokens = item['target_tokens'][:5]
        distractor_tokens = item['distractor_tokens'][:5]
        target_tokens = target_tokens + [self.tokenizer.pad_token_id] * (5 - len(target_tokens))
        distractor_tokens = distractor_tokens + [self.tokenizer.pad_token_id] * (5 - len(distractor_tokens))

        return {
            "input_ids": inputs['input_ids'].squeeze(0),
            "attention_mask": inputs['attention_mask'].squeeze(0),
            "corrupted_input_ids": corrupted_inputs['input_ids'].squeeze(0),
            "corrupted_attention_mask": corrupted_inputs['attention_mask'].squeeze(0),
            "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
            "distractor_tokens": torch.tensor(distractor_tokens, dtype=torch.long),
            "T_Start": torch.tensor(T_Start, dtype=torch.long),
            "T_End": torch.tensor(T_End, dtype=torch.long),
            "D_Start": torch.tensor(D_Start, dtype=torch.long),
            "D_End": torch.tensor(D_End, dtype=torch.long),
            "T_len": torch.tensor(T_len, dtype=torch.long),
            "D_len": torch.tensor(D_len, dtype=torch.long),
            "template_order": item['template_order'],
        }


# ==============================================================================
# EVALUATION
# ==============================================================================

def run_evaluation(
    model_to_eval,
    model_name: str,
    full_model_for_faithfulness: Optional[nn.Module],
    dataloader,
    device,
    verbose=True,
    tokenizer=None,
):
    """Run evaluation on IOI task."""
    if verbose:
        print("\n" + "=" * 50 + f"\n  EVALUATING: {model_name}\n" + "=" * 50)

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
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)

            outputs = model_to_eval(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                corrupted_input_ids=batch.get('corrupted_input_ids'),
            )

            batch_size = outputs.logits.size(0)

            for i in range(batch_size):
                t_start = batch['T_Start'][i].item() - 1
                t_end = batch['T_End'][i].item() - 1
                d_start = batch['D_Start'][i].item() - 1
                d_end = batch['D_End'][i].item() - 1

                target_tokens = batch['target_tokens'][i][:batch['T_len'][i].item()]
                distractor_tokens = batch['distractor_tokens'][i][:batch['D_len'][i].item()]

                target_logits = []
                distractor_logits = []

                for pos_idx, pos in enumerate(range(t_start, t_end)):
                    if pos < outputs.logits.size(1):
                        token_id = target_tokens[pos_idx] if pos_idx < len(target_tokens) else target_tokens[0]
                        logit = outputs.logits[i, pos, token_id].item()
                        target_logits.append(logit)

                for pos_idx, pos in enumerate(range(d_start, d_end)):
                    if pos < outputs.logits.size(1):
                        token_id = distractor_tokens[pos_idx] if pos_idx < len(distractor_tokens) else distractor_tokens[0]
                        logit = outputs.logits[i, pos, token_id].item()
                        distractor_logits.append(logit)

                if target_logits and distractor_logits:
                    avg_target_logit = target_logits[0]
                    avg_distractor_logit = distractor_logits[0]
                    logit_diff = avg_target_logit - avg_distractor_logit
                    total_logit_diff += logit_diff

                    if avg_target_logit >= avg_distractor_logit:
                        total_accuracy += 1

                valid_samples += 1

            # Faithfulness (KL divergence)
            if full_model_for_faithfulness:
                full_outputs = full_model_for_faithfulness(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                )

                for i in range(batch_size):
                    t_start = batch['T_Start'][i].item() - 1
                    t_end = batch['T_End'][i].item() - 1
                    valid_length = batch['attention_mask'][i].sum().item()

                    end_pos = min(t_end, valid_length)

                    if t_start < end_pos:
                        model_logits = outputs.logits[i, t_start:end_pos, :]
                        full_logits = full_outputs.logits[i, t_start:end_pos, :]

                        kl = F.kl_div(
                            F.log_softmax(model_logits, dim=-1),
                            F.log_softmax(full_logits, dim=-1),
                            log_target=True,
                            reduction='batchmean'
                        ).item()

                        total_kl += kl

                    model_pred = torch.argmax(outputs.logits[i, t_start, :])
                    full_pred = torch.argmax(full_outputs.logits[i, t_start, :])
                    if model_pred == full_pred:
                        total_exact_match += 1

    avg_accuracy = total_accuracy / valid_samples if valid_samples > 0 else 0
    avg_logit_diff = total_logit_diff / valid_samples if valid_samples > 0 else 0
    avg_kl = total_kl / valid_samples if valid_samples > 0 else 0
    exact_match_rate = total_exact_match / valid_samples if valid_samples > 0 else 0

    if verbose:
        print(f"\nProcessed {valid_samples} valid samples.")
        print("\n" + "=" * 50)
        print(f"{model_name} Evaluation Summary:")
        print(f"  - Accuracy:              {avg_accuracy:.4f}")
        print(f"  - Logit Difference:      {avg_logit_diff:.4f}")
        if full_model_for_faithfulness:
            print(f"  - Faithfulness (KL Div): {avg_kl:.4f}")
            print(f"  - Exact Match Rate:      {exact_match_rate:.4f}")
        print("=" * 50)

    return {
        "accuracy": avg_accuracy,
        "logit_diff": avg_logit_diff,
        "kl_div": avg_kl,
        "exact_match": exact_match_rate,
    }


def filter_dataset_by_model_correctness(data_list, model, tokenizer, device, batch_size=32):
    """
    Filter IOI data samples, keeping only those where the base model
    assigns a higher logit to the target than the distractor.
    """
    if not data_list:
        return []

    print(f"Filtering {len(data_list)} samples for base model correctness...")

    temp_dataset = IOIDatasetLlama(data_list, tokenizer)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

    valid_indices = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(temp_loader, desc="Checking model predictions")):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )

            current_batch_size = outputs.logits.size(0)

            for i in range(current_batch_size):
                t_start = batch['T_Start'][i].item() - 1
                d_start = batch['D_Start'][i].item() - 1

                target_token_id = batch['target_tokens'][i][0].item()
                distractor_token_id = batch['distractor_tokens'][i][0].item()

                target_logit = outputs.logits[i, t_start, target_token_id].item()
                distractor_logit = outputs.logits[i, d_start, distractor_token_id].item()

                if target_logit >= distractor_logit:
                    global_idx = (batch_idx * batch_size) + i
                    valid_indices.append(global_idx)

    filtered_data = [data_list[i] for i in valid_indices]

    print(f"  -> Retained: {len(filtered_data)}/{len(data_list)} "
          f"({len(filtered_data) / len(data_list) * 100:.2f}%)")

    return filtered_data