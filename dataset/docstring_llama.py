"""
Docstring Task Dataset and Evaluation for Llama Circuit Discovery.

Reuses the prompt generation from dataset/docstring.py and adapts the dataset
class, evaluation, and filtering for Llama models (AutoTokenizer, bfloat16).

Task: Given a Python function def with :param docstrings, predict the next
argument name at the last token position.
"""

from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Reuse prompt generation from the GPT-2 docstring module
from dataset.docstring import generate_docstring_data, generate_docstring_prompt


# ==============================================================================
# DATASET CLASS FOR LLAMA
# ==============================================================================

class DocstringDatasetLlama(Dataset):
    """Docstring task dataset adapted for Llama tokenizer and dual-stream pruning."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.processed_data = []

        for item in data:
            clean_prompt = item["clean_prompt"]
            corrupt_prompt = item["corrupt_prompt"]
            correct_answer = item["correct_answer"]  # space-prefixed
            wrong_answers = item["wrong_answers"]  # space-prefixed list

            # Tokenize the correct answer -- must be single token
            correct_tokens = tokenizer.encode(correct_answer, add_special_tokens=False)
            if len(correct_tokens) != 1:
                continue

            # Tokenize wrong answers -- keep only single-token ones
            wrong_token_ids = []
            for wa in wrong_answers:
                wa_tokens = tokenizer.encode(wa, add_special_tokens=False)
                if len(wa_tokens) == 1:
                    wrong_token_ids.append(wa_tokens[0])

            if not wrong_token_ids:
                continue

            self.processed_data.append({
                "clean_prompt": clean_prompt,
                "corrupt_prompt": corrupt_prompt,
                "target_token": correct_tokens[0],
                "distractor_tokens": wrong_token_ids,
            })

        print(f"Processed {len(self.processed_data)} valid samples from {len(data)} total")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]

        # Tokenize clean prompt
        inputs = self.tokenizer(
            item["clean_prompt"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize corrupted prompt
        corrupted_inputs = self.tokenizer(
            item["corrupt_prompt"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Prediction position: last real token in the prompt
        prompt_tokens = self.tokenizer.encode(item["clean_prompt"], add_special_tokens=True)
        prefix_length = min(len(prompt_tokens), self.max_length)

        # Use first distractor as the primary one for logit diff
        distractor_token = item["distractor_tokens"][0]

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "corrupted_input_ids": corrupted_inputs["input_ids"].squeeze(0),
            "corrupted_attention_mask": corrupted_inputs["attention_mask"].squeeze(0),
            "target_token": torch.tensor(item["target_token"], dtype=torch.long),
            "distractor_token": torch.tensor(distractor_token, dtype=torch.long),
            "prefix_length": torch.tensor(prefix_length, dtype=torch.long),
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
    """Run evaluation on the docstring task for Llama models."""
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

                logit_target = outputs.logits[i, pred_pos, target_token].float().item()
                logit_distractor = outputs.logits[i, pred_pos, distractor_token].float().item()
                total_logit_diff += logit_target - logit_distractor

                # Accuracy: target should beat distractor
                if logit_target > logit_distractor:
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

                    model_logits = outputs.logits[i, pred_pos, :].float()
                    full_logits = full_outputs.logits[i, pred_pos, :].float()

                    kl = F.kl_div(
                        F.log_softmax(model_logits, dim=-1),
                        F.log_softmax(full_logits, dim=-1),
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

def filter_dataset_by_model_correctness(data_list, model, tokenizer, device, max_length=64, batch_size=32):
    """
    Filter docstring samples, keeping only those where the base model's argmax
    at the prediction position matches the correct answer token.
    """
    if not data_list:
        return []

    print(f"Filtering {len(data_list)} samples for base model correctness...")

    temp_dataset = DocstringDatasetLlama(data_list, tokenizer, max_length=max_length)
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

                predicted_token_id = torch.argmax(outputs.logits[i, pred_pos]).item()
                target_token_id = batch["target_token"][i].item()

                if predicted_token_id == target_token_id:
                    global_idx = (batch_idx * batch_size) + i
                    valid_indices.append(global_idx)

    filtered_data = [data_list[i] for i in valid_indices]

    print(
        f"  -> Retained: {len(filtered_data)}/{len(data_list)} "
        f"({len(filtered_data) / len(data_list) * 100:.2f}%)"
    )

    return filtered_data
