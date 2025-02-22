import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm  # Import tqdm for progress bars

# Load dataset
dataset = load_dataset("ag_news")

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# BERT uses [PAD] token by default, no need to set pad_token explicitly

# Preprocess function


def preprocess(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    return {
        "input_ids": tokens["input_ids"].squeeze(0).numpy(),
        "attention_mask": tokens["attention_mask"].squeeze(0).numpy(),
        "label": np.array(example["label"], dtype=np.int64),
    }


# Apply preprocessing with progress bar
for split in dataset.keys():
    print(f"Processing {split} dataset...")
    dataset[split] = list(tqdm(dataset[split].map(
        preprocess), total=len(dataset[split])))

# Save as raw .bin files
for split in ["train", "test"]:
    print(f"Saving {split} dataset...")

    input_ids = np.array([x["input_ids"] for x in tqdm(
        dataset[split], desc="Input IDs")], dtype=np.int32)
    attention_masks = np.array([x["attention_mask"] for x in tqdm(
        dataset[split], desc="Attention Masks")], dtype=np.int32)
    labels = np.array([x["label"] for x in tqdm(
        dataset[split], desc="Labels")], dtype=np.int64)

    # Save each component separately
    input_ids.tofile(f"ag_news_{split}_input_ids.bin")
    attention_masks.tofile(f"ag_news_{split}_attention_masks.bin")
    labels.tofile(f"ag_news_{split}_labels.bin")

print("✅ Saved dataset as raw .bin files!")
