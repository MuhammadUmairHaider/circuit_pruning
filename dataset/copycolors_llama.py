"""
CopyColors MCQA Dataset and Evaluation for Llama Models.

Loads the MIB-bench copycolors_mcqa dataset from HuggingFace and adapts it
for circuit discovery with the dual-stream (clean/corrupted) architecture.

Task: Given a description of colored objects, answer a multiple-choice question
about which color an object has.

Example:
    "A box is brown. What color is a box? A. gray B. black C. white D. brown Answer:"
    Target: " D"

Dataset: mib-bench/copycolors_mcqa (4_answer_choices config)
"""

import os
import random
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_copycolors_data(
    split: str = "test",
    num_samples: Optional[int] = None,
    num_choices: int = 4,
    dataset_path: Optional[str] = None,
) -> List[Dict]:
    """
    Load CopyColors MCQA data from HuggingFace or disk.

    Each sample contains a clean prompt and a corrupted prompt (counterfactual)
    where both the answer position and color are changed.

    Args:
        split: 'train', 'validation', or 'test'
        num_samples: Number of samples to use (None = use all)
        num_choices: Number of answer choices (default 4)
        dataset_path: Optional path to pre-saved dataset on disk

    Returns:
        List of dicts with keys: prompt, corrupted_prompt, answer_key,
        corrupted_answer_key, choices, corrupted_choices
    """
    from datasets import load_dataset, load_from_disk

    config_name = f"{num_choices}_answer_choices"

    if dataset_path and os.path.exists(dataset_path):
        print(f"Loading CopyColors from disk: {dataset_path}")
        dataset_dict = load_from_disk(dataset_path)
        dataset = dataset_dict[split]
    else:
        print(f"Loading CopyColors from HuggingFace: mib-bench/copycolors_mcqa ({config_name})")
        dataset_dict = load_dataset("mib-bench/copycolors_mcqa", config_name)
        dataset = dataset_dict[split]

    print(f"Loaded {len(dataset)} samples from {split} split")

    samples = []
    for i in range(len(dataset)):
        if num_samples is not None and i >= num_samples:
            break

        item = dataset[i]

        # Clean prompt
        clean_prompt = item["prompt"]
        clean_answer_key = item["answerKey"]
        clean_choices = item["choices"]

        # Use answerPosition_color_counterfactual as the corrupted input
        # This changes both the answer position and the color for strongest signal
        counterfactual = item.get("answerPosition_color_counterfactual")
        if counterfactual is None:
            # Fallback to color_counterfactual or answerPosition_counterfactual
            counterfactual = item.get("color_counterfactual") or item.get("answerPosition_counterfactual")

        if counterfactual is None:
            continue

        corrupted_prompt = counterfactual["prompt"]
        corrupted_answer_key = counterfactual["answerKey"]
        corrupted_choices = counterfactual["choices"]

        # print("clean_prompt: ", clean_prompt, flush=True)
        # print("corrupted_prompt: ", corrupted_prompt, flush=True)
        # print("clean_answer_key: ", clean_answer_key, flush=True)
        # print("corrupted_answer_key: ", corrupted_answer_key, flush=True)
        # print("clean_choices: ", clean_choices, flush=True)
        # print("corrupted_choices: ", corrupted_choices, flush=True)
        # exit(1)

        samples.append({
            "prompt": clean_prompt,
            "corrupted_prompt": corrupted_prompt,
            "answer_key": clean_answer_key,
            "corrupted_answer_key": corrupted_answer_key,
            "choices": clean_choices,
            "corrupted_choices": corrupted_choices,
        })

    print(f"Prepared {len(samples)} valid samples")
    return samples


# ==============================================================================
# DATASET CLASS FOR LLAMA
# ==============================================================================

class CopyColorsDatasetLlama(Dataset):
    """CopyColors MCQA Dataset adapted for Llama tokenizer and dual-stream pruning."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Pre-tokenize answer choice letters to verify they're single tokens
        self.choice_letter_tokens = {}
        for letter in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
            tokens = tokenizer.encode(" " + letter, add_special_tokens=False)
            if len(tokens) == 1:
                self.choice_letter_tokens[letter] = tokens[0]

        self.processed_data = []

        for item in data:
            clean_prompt = item["prompt"]
            corrupted_prompt = item["corrupted_prompt"]
            answer_key = item["answer_key"]
            corrupted_answer_key = item["corrupted_answer_key"]
            choices = item["choices"]
            corrupted_choices = item.get("corrupted_choices", choices)

            # Get the correct answer letter
            choice_labels = choices["label"]
            correct_label = choice_labels[answer_key]
            corrupted_label = corrupted_choices["label"][corrupted_answer_key]

            # Verify answer letters are single tokens
            if correct_label not in self.choice_letter_tokens:
                continue
            if corrupted_label not in self.choice_letter_tokens:
                continue

            target_token = self.choice_letter_tokens[correct_label]
            corrupted_target_token = self.choice_letter_tokens[corrupted_label]

            # Get all distractor tokens (incorrect answer letters)
            distractor_tokens = []
            for label in choice_labels:
                if label != correct_label and label in self.choice_letter_tokens:
                    distractor_tokens.append(self.choice_letter_tokens[label])

            if not distractor_tokens:
                continue

            self.processed_data.append({
                "prompt": clean_prompt,
                "corrupted_prompt": corrupted_prompt,
                "target_token": target_token,
                "corrupted_target_token": corrupted_target_token,
                "distractor_tokens": distractor_tokens,
                "correct_label": correct_label,
                "all_choice_labels": choice_labels,
                "answer_key": answer_key,
            })

        print(f"Processed {len(self.processed_data)} valid samples from {len(data)} total")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]

        # Tokenize clean prompt
        inputs = self.tokenizer(
            item["prompt"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize corrupted prompt
        corrupted_inputs = self.tokenizer(
            item["corrupted_prompt"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Find prediction position: the position of the last real token
        # The prompt ends with "Answer:" and we predict the next token
        # So prefix_length = number of tokens in the prompt (including BOS)
        prompt_tokens = self.tokenizer.encode(item["prompt"], add_special_tokens=True)
        prefix_length = len(prompt_tokens)

        # Use first distractor as the primary distractor for logit diff
        distractor_token = item["distractor_tokens"][0]

        # Pad all choice token IDs to fixed size (max 10 choices)
        all_choice_tokens = [item["target_token"]] + item["distractor_tokens"]
        padded_choice_tokens = all_choice_tokens[:10]
        padded_choice_tokens += [self.tokenizer.pad_token_id] * (10 - len(padded_choice_tokens))

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "corrupted_input_ids": corrupted_inputs["input_ids"].squeeze(0),
            "corrupted_attention_mask": corrupted_inputs["attention_mask"].squeeze(0),
            "target_token": torch.tensor(item["target_token"], dtype=torch.long),
            "distractor_token": torch.tensor(distractor_token, dtype=torch.long),
            "prefix_length": torch.tensor(prefix_length, dtype=torch.long),
            "num_choices": torch.tensor(len(item["all_choice_labels"]), dtype=torch.long),
            "all_choice_tokens": torch.tensor(padded_choice_tokens, dtype=torch.long),
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
    """Run evaluation on CopyColors MCQA task."""
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
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                corrupted_input_ids=batch.get("corrupted_input_ids"),
            )

            batch_size = outputs.logits.size(0)

            for i in range(batch_size):
                pred_pos = batch["prefix_length"][i].item() - 1
                if pred_pos >= outputs.logits.size(1):
                    continue

                target_token = batch["target_token"][i].item()
                distractor_token = batch["distractor_token"][i].item()
                num_choices = batch["num_choices"][i].item()
                all_choice_tokens = batch["all_choice_tokens"][i][:num_choices]

                # Logit at prediction position for target vs distractor
                logit_target = outputs.logits[i, pred_pos, target_token].item()
                logit_distractor = outputs.logits[i, pred_pos, distractor_token].item()
                total_logit_diff += logit_target - logit_distractor

                # Accuracy: target should have highest logit among all choices
                choice_logits = outputs.logits[i, pred_pos, all_choice_tokens]
                predicted_choice = all_choice_tokens[torch.argmax(choice_logits).item()].item()
                if predicted_choice == target_token:
                    total_accuracy += 1

                valid_samples += 1

            # KL divergence (faithfulness)
            if full_model_for_faithfulness:
                full_outputs = full_model_for_faithfulness(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                for i in range(batch_size):
                    pred_pos = batch["prefix_length"][i].item() - 1
                    if pred_pos >= outputs.logits.size(1):
                        continue

                    model_logits = outputs.logits[i, pred_pos, :]
                    full_logits = full_outputs.logits[i, pred_pos, :]

                    kl = F.kl_div(
                        F.log_softmax(model_logits.float(), dim=-1),
                        F.log_softmax(full_logits.float(), dim=-1),
                        log_target=True,
                        reduction="sum",
                    ).item()
                    total_kl += kl

                    # Exact match
                    model_pred = torch.argmax(outputs.logits[i, pred_pos, :])
                    full_pred = torch.argmax(full_outputs.logits[i, pred_pos, :])
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


# ==============================================================================
# FILTERING
# ==============================================================================

def filter_dataset_by_model_correctness(data_list, model, tokenizer, device, batch_size=32):
    """
    Filter CopyColors data samples, keeping only those where the base model
    predicts the correct answer letter as the top choice among all options.
    """
    if not data_list:
        return []

    print(f"Filtering {len(data_list)} samples for base model correctness...")

    temp_dataset = CopyColorsDatasetLlama(data_list, tokenizer)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

    valid_indices = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(temp_loader, desc="Checking model predictions")):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            current_batch_size = outputs.logits.size(0)

            for i in range(current_batch_size):
                pred_pos = batch["prefix_length"][i].item() - 1
                if pred_pos >= outputs.logits.size(1):
                    continue

                target_token = batch["target_token"][i].item()
                num_choices = batch["num_choices"][i].item()
                all_choice_tokens = batch["all_choice_tokens"][i][:num_choices]

                # Check if the model's top choice among answer letters is correct
                choice_logits = outputs.logits[i, pred_pos, all_choice_tokens]
                predicted_choice = all_choice_tokens[torch.argmax(choice_logits).item()].item()

                if predicted_choice == target_token:
                    global_idx = (batch_idx * batch_size) + i
                    valid_indices.append(global_idx)

    filtered_data = [data_list[i] for i in valid_indices]

    print(
        f"  -> Retained: {len(filtered_data)}/{len(data_list)} "
        f"({len(filtered_data) / len(data_list) * 100:.2f}%)"
    )

    return filtered_data
