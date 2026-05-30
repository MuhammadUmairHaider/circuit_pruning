"""
Docstring Task Dataset and Evaluation for GPT-2 Circuit Discovery.

Implements the docstring task from the ACDC paper (Conmy et al., NeurIPS 2023).
The task generates Python function definitions with :param docstrings, and the
model must predict the next argument name at position -1.

Example:
    def method_name(self, prefix1, prefix2, arg1, arg2, arg3, suffix1):
        \"\"\"desc1 desc2 desc3
        :param arg1: desc1 desc2
        :param arg2: desc1 desc2
        :param                          <-- predict "arg3" here

Corruption (random_random): Both def args AND doc args are replaced with
completely different random variable names, removing all signal.

Based on: github.com/ArthurConmy/Automatic-Circuit-Discovery
  - acdc/docstring/prompts.py
  - acdc/docstring/utils.py
"""

import random
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ==============================================================================
# VARIABLE AND DESCRIPTION POOLS (from ACDC prompts.py)
# ==============================================================================

VARIABLE_NAMES = "data name file value test new result line user key default request path output node item url model response text version function log string field start number values sub index error current context image message check json create size event state row obj parser end files form query fields instance label run action target array code num table source msg config first required options ret update module status group content client filename results last read command val base format color settings order found count match tag title note daughter sun box river profit division stone post client help image oil sector attack direction seat employment goal sign ability campaign fish item medium show version drug library press surface blood culture memory return bar talk access deal star text cause mouth payment context reference second article chair earth object agency card collection communication public document weight bird talk".split(" ")

# Remove duplicates while preserving order
_seen = set()
_unique_vars = []
for v in VARIABLE_NAMES:
    if v not in _seen:
        _seen.add(v)
        _unique_vars.append(v)
VARIABLE_NAMES = _unique_vars

COMMON_NOUNS = "price control action cost issue process position course minute education type research change company order support point member information time body side group state class part model field program operation area office room market power money process system system company state right product course night number value right issue name place study government class body point home type area product business action right hand family company fact case state kind week head change line power country people problem school day right service area order group company family state group system course community research story world information industry question report person development house result change system country life company type plan family side area right issue policy service research life year program point state result part government area group change day school process system country type industry life part order story course home plan power company area study night state result fact change body kind week head service type group system action company life plan world school side state fact order process change research development story system area result country number type power family right service information question study school process system area type part power country group people change action order home state result plan government process type service area point state group research world question story life company side development course plan industry country result power school system order study value state change area home name type process number action place country result fact state body company group plan change system hand kind type process country right research week issue question area service point family school study type group process country part change system result order area number model government fact state value right question action group country area point power company study change type world plan research family service side industry course information school system story type".split(" ")

_seen2 = set()
_unique_nouns = []
for n in COMMON_NOUNS:
    if n not in _seen2:
        _seen2.add(n)
        _unique_nouns.append(n)
COMMON_NOUNS = _unique_nouns


# ==============================================================================
# PROMPT GENERATION (from ACDC prompts.py)
# ==============================================================================

def docstring_prompt_templ(
    met_name: str,
    met_desc_words: List[str],
    def_args: List[str],
    doc_args: List[str],
    doc_args_desc_words: List[List[str]],
):
    """
    Generate a Python function definition with :param docstring (reStructuredText style).

    Args:
        met_name: Method name
        met_desc_words: Words for the method description
        def_args: All arguments in the def line (prefix + matching + suffix)
        doc_args: Arguments documented in docstring (subset that appears in :param lines)
        doc_args_desc_words: Description words for each documented arg

    Returns:
        prompt string ending with ":param " (model predicts next arg name)
    """
    # Build def line
    all_args = ", ".join(["self"] + def_args)
    def_line = f"def {met_name}({all_args}):"

    # Build docstring
    met_desc = " ".join(met_desc_words)
    doc_lines = [f'    """{met_desc}']

    for arg, desc_words in zip(doc_args, doc_args_desc_words):
        desc = " ".join(desc_words)
        doc_lines.append(f"    :param {arg}: {desc}")

    # The prompt ends with ":param " -- model predicts the next arg name
    doc_lines.append("    :param ")

    prompt = def_line + "\n" + "\n".join(doc_lines)
    return prompt


def generate_docstring_prompt(
    n_matching_args: int = 3,
    n_def_prefix_args: int = 2,
    n_def_suffix_args: int = 1,
    n_doc_prefix_args: int = 0,
    met_desc_len: int = 3,
    arg_desc_len: int = 2,
    rng: Optional[random.Random] = None,
):
    """
    Generate a single docstring prompt with clean and corrupted versions.

    Returns:
        dict with keys:
            clean_prompt: Proper function with matching arg names
            corrupt_prompt: random_random corruption (both def and doc args randomized)
            correct_answer: The correct arg name (space-prefixed)
            wrong_answers: Other matching arg names (space-prefixed)
            all_def_args: All args in the def line
            matching_args: The matching args between def and doc
    """
    if rng is None:
        rng = random.Random()

    # Total unique variable names needed for clean prompt
    total_args = n_def_prefix_args + n_matching_args + n_def_suffix_args
    total_doc_args = n_doc_prefix_args + n_matching_args  # doc prefix + matching (before prediction)

    # We need: method name, all def args, doc prefix args (if any extra), descriptions
    # For clean: pick unique variable names for all roles
    available_vars = list(VARIABLE_NAMES)
    rng.shuffle(available_vars)

    met_name = available_vars.pop()

    # Pick def args
    def_prefix_args = [available_vars.pop() for _ in range(n_def_prefix_args)]
    matching_args = [available_vars.pop() for _ in range(n_matching_args)]
    def_suffix_args = [available_vars.pop() for _ in range(n_def_suffix_args)]

    all_def_args = def_prefix_args + matching_args + def_suffix_args

    # Doc args: doc_prefix (extra args not in def, if any) + matching args already documented
    # n_doc_prefix_args=0 in ACDC default, so doc args = matching args
    doc_prefix_args = [available_vars.pop() for _ in range(n_doc_prefix_args)]

    # The documented args before the prediction point
    # We document (n_matching_args - 1) matching args, then predict the last one
    doc_args = doc_prefix_args + matching_args[:-1]

    # The correct answer is the last matching arg
    correct_answer = matching_args[-1]
    # Wrong answers are the other matching args
    wrong_answers = matching_args[:-1]

    # Pick description words
    met_desc_words = rng.sample(COMMON_NOUNS, min(met_desc_len, len(COMMON_NOUNS)))
    doc_args_desc_words = [
        rng.sample(COMMON_NOUNS, min(arg_desc_len, len(COMMON_NOUNS)))
        for _ in doc_args
    ]

    # Build clean prompt
    clean_prompt = docstring_prompt_templ(
        met_name=met_name,
        met_desc_words=met_desc_words,
        def_args=all_def_args,
        doc_args=doc_args,
        doc_args_desc_words=doc_args_desc_words,
    )

    # Build corrupted prompt (random_random): replace BOTH def args AND doc args
    # with completely different random names
    corrupt_available = [v for v in VARIABLE_NAMES if v not in set(all_def_args + doc_prefix_args + [met_name])]
    rng.shuffle(corrupt_available)

    corrupt_def_args = [corrupt_available.pop() for _ in range(total_args)]
    corrupt_doc_prefix = [corrupt_available.pop() for _ in range(n_doc_prefix_args)]
    # For the documented matching args in corruption, use random names
    corrupt_matching_for_doc = [corrupt_available.pop() for _ in range(n_matching_args - 1)]
    corrupt_doc_args = corrupt_doc_prefix + corrupt_matching_for_doc

    # New description words for corruption
    corrupt_met_desc = rng.sample(COMMON_NOUNS, min(met_desc_len, len(COMMON_NOUNS)))
    corrupt_doc_desc = [
        rng.sample(COMMON_NOUNS, min(arg_desc_len, len(COMMON_NOUNS)))
        for _ in corrupt_doc_args
    ]

    corrupt_prompt = docstring_prompt_templ(
        met_name=met_name,  # Keep method name the same
        met_desc_words=corrupt_met_desc,
        def_args=corrupt_def_args,
        doc_args=corrupt_doc_args,
        doc_args_desc_words=corrupt_doc_desc,
    )

    print("\n" + "="*60)
    print("Generated Docstring Samples")
    print("="*60)
    print("Clean prompt: ", clean_prompt)
    print("Corrupt prompt: ", corrupt_prompt)
    print("Correct answer: ", correct_answer)
    print("Wrong answers: ", wrong_answers)
    print("Matching args: ", matching_args)
    print("All def args: ", all_def_args)


    return {
        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt,
        "correct_answer": " " + correct_answer,
        "wrong_answers": [" " + a for a in wrong_answers],
        "matching_args": matching_args,
        "all_def_args": all_def_args,
    }


def generate_docstring_data(
    num_samples: int = 500,
    seed: int = 42,
    **kwargs,
) -> List[Dict]:
    """Generate a list of docstring prompt samples."""
    rng = random.Random(seed)
    samples = []
    for _ in range(num_samples):
        sample = generate_docstring_prompt(rng=rng, **kwargs)
        samples.append(sample)
    return samples


# ==============================================================================
# DATASET CLASS
# ==============================================================================

class DocstringDataset(Dataset):
    """Docstring task dataset for GPT-2 dual-stream circuit discovery."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
    """Run evaluation on the docstring task."""
    if verbose:
        print("\n" + "=" * 50 + f"\n  EVALUATING: {model_name}\n" + "=" * 50)

    model_to_eval.eval()
    if full_model_for_faithfulness:
        full_model_for_faithfulness.eval()

    accuracy = 0
    logit_difference = 0
    kl_divergence = 0
    exact_match = 0
    total_samples = len(dataloader.dataset)

    desc = f"Evaluating {model_name}" if verbose else "Evaluating"
    bar = tqdm(range(0, total_samples, dataloader.batch_size), desc=desc)

    sample_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_size = batch["input_ids"].shape[0]

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            corr_input_ids = batch["corrupted_input_ids"].to(device)
            prefix_lengths = batch["prefix_length"].tolist()
            targets = batch["target_token"].to(device)
            distractors = batch["distractor_token"].to(device)

            # Full model outputs for faithfulness
            control_outputs = full_model_for_faithfulness(input_ids, attention_mask=attention_mask) if full_model_for_faithfulness else None
            control_logits = control_outputs.logits if control_outputs else None

            # Model outputs
            outputs = model_to_eval(
                input_ids=input_ids,
                corrupted_input_ids=corr_input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

            for j in range(batch_size):
                pred_pos = prefix_lengths[j] - 1

                logit_target = logits[j, pred_pos, targets[j]].detach().cpu().item()
                logit_distractor = logits[j, pred_pos, distractors[j]].detach().cpu().item()
                logit_difference += logit_target - logit_distractor

                # Accuracy: correct > max distractor
                if logit_target > logit_distractor:
                    accuracy += 1

                # KL divergence
                if control_logits is not None:
                    logits_ = F.log_softmax(logits[j, pred_pos], dim=-1)
                    control_logits_ = F.log_softmax(control_logits[j, pred_pos], dim=-1)
                    kld = F.kl_div(logits_, control_logits_, reduction="sum", log_target=True)
                    kl_divergence += kld.detach().cpu().item()

                # Exact match with full model
                if control_logits is not None:
                    choice = torch.argmax(logits[j, pred_pos])
                    control_choice = torch.argmax(control_logits[j, pred_pos])
                    exact_match += (choice == control_choice).int().detach().cpu().item()

                sample_idx += 1

            bar.update(batch_size)
            current_total = min(sample_idx, total_samples)
            bar.set_description(f"Acc: {accuracy/current_total:.3f}, LD: {logit_difference/current_total:.3f}")

    bar.close()

    accuracy /= total_samples
    logit_difference /= total_samples
    kl_divergence /= total_samples
    exact_match /= total_samples

    if verbose:
        print(f"\nProcessed {total_samples} valid samples.")
        print("\n" + "=" * 50)
        print(f"{model_name} Evaluation Summary:")
        print(f"  - Accuracy:              {accuracy:.4f}")
        print(f"  - Logit Difference:      {logit_difference:.4f}")
        if full_model_for_faithfulness:
            print(f"  - KL Divergence:         {kl_divergence:.4f}")
            print(f"  - Exact Match:           {exact_match:.4f}")
        print("=" * 50)

    return {
        "accuracy": accuracy,
        "logit_diff": logit_difference,
        "kl_div": kl_divergence,
        "exact_match": exact_match,
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

    temp_dataset = DocstringDataset(data_list, tokenizer, max_length=max_length)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

    valid_indices = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(temp_loader, desc="Checking model predictions")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target_token"].to(device)
            prefix_lengths = batch["prefix_length"].tolist()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            current_batch_size = input_ids.size(0)
            for i in range(current_batch_size):
                pred_pos = prefix_lengths[i] - 1
                predicted_token_id = torch.argmax(logits[i, pred_pos]).item()
                target_token_id = targets[i].item()

                if predicted_token_id == target_token_id:
                    global_idx = (batch_idx * batch_size) + i
                    valid_indices.append(global_idx)

    filtered_data = [data_list[i] for i in valid_indices]

    print(
        f"  -> Retained: {len(filtered_data)}/{len(data_list)} "
        f"({len(filtered_data) / len(data_list) * 100:.2f}%)"
    )

    return filtered_data
