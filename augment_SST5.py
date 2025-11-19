# =====================================================
# SST-5 Back-Translation Augmentation
# - Fine-grained (5-way)
# - Ternary (3-way)
# =====================================================
import time
import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer

start = time.time()

# -----------------------------------------------------
# 1. Device Setup (no reproducibility for randomness)
# -----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------
# 2. Load SST-5 dataset
# -----------------------------------------------------
ds = load_dataset("SetFit/sst5")

df_all = pd.concat([
    pd.DataFrame(ds["train"]),
    pd.DataFrame(ds["validation"]),
    pd.DataFrame(ds["test"])
], ignore_index=True)

print("Original class counts (ALL splits):")
print(df_all["label"].value_counts())

# -----------------------------------------------------
# 3. Load MarianMT models (EN→FR→EN)
# -----------------------------------------------------
en_to_fr = "Helsinki-NLP/opus-mt-en-fr"
fr_to_en = "Helsinki-NLP/opus-mt-fr-en"

en_to_fr_tok = MarianTokenizer.from_pretrained(en_to_fr)
en_to_fr_model = MarianMTModel.from_pretrained(en_to_fr).to(device)

fr_to_en_tok = MarianTokenizer.from_pretrained(fr_to_en)
fr_to_en_model = MarianMTModel.from_pretrained(fr_to_en).to(device)


def batch_translate(texts, tokenizer, model, batch_size=16):
    """Translate a list of texts using MarianMT with randomness enabled."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,       #  enable randomness
                top_k=50,             #  choose from top 50 words
                top_p=0.95,           #  nucleus sampling
                temperature=1.0,      #  controls diversity
                max_length=256
            )

        results.extend(
            [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
        )
    return results


def back_translate(texts, batch_size=16):
    """Perform back-translation EN→FR→EN."""
    fr_texts = batch_translate(texts, en_to_fr_tok, en_to_fr_model, batch_size)
    en_texts = batch_translate(fr_texts, fr_to_en_tok, fr_to_en_model, batch_size)
    return en_texts


# -----------------------------------------------------
# 4. Fine-grained SST-5 (5k per class)
# -----------------------------------------------------
TARGET_PER_CLASS = 5000
augmented_data = []
log_records = []

output_file_fine = "finegrainedsst5_5k_each_class.csv"
log_file_fine = "finegrainedsst5_log.csv"

os.makedirs("data", exist_ok=True)

for label in range(5):
    class_texts = df_all[df_all["label"] == label]["text"].tolist()
    current_count = len(class_texts)
    needed = max(TARGET_PER_CLASS - current_count, 0)

    print(f"\n[Fine-grained] Class {label} | Current: {current_count} | Needed: {needed}")

    # Keep original samples
    augmented_data.extend([{"label": label, "text": t} for t in class_texts])

    # Generate augmented samples if needed
    if needed > 0:
        idx, used_count = 0, 0
        batch, batch_orig = [], []

        with tqdm(total=needed, desc=f"Class {label}") as pbar:
            while used_count < needed:
                text = class_texts[idx % len(class_texts)]
                batch.append(text)
                batch_orig.append(text)

                if len(batch) == 16 or used_count + len(batch) >= needed:
                    bt_texts = back_translate(batch)
                    for orig, aug in zip(batch_orig, bt_texts):
                        if used_count >= needed:
                            break
                        augmented_data.append({"label": label, "text": aug})
                        log_records.append({
                            "label": label,
                            "original": orig,
                            "augmented": aug,
                            "method": "back_translation"
                        })
                        used_count += 1
                        pbar.update(1)
                    batch, batch_orig = [], []
                idx += 1

# Save fine-grained dataset
df_fine = pd.DataFrame(augmented_data)
df_fine.to_csv(output_file_fine, index=False)
pd.DataFrame(log_records).to_csv(log_file_fine, index=False)

print("\nFine-grained dataset saved:", output_file_fine)
print("Log saved:", log_file_fine)
print(df_fine["label"].value_counts())

# -----------------------------------------------------
# 5. Ternary SST (0=negative, 1=neutral, 2=positive)
# -----------------------------------------------------
# Map original SST labels → ternary labels
label_map = {
    0: 0,  # very neg → negative
    1: 0,  # neg → negative
    2: 1,  # neutral → neutral
    3: 2,  # pos → positive
    4: 2,  # very pos → positive
}

df_fine["label_ternary"] = df_fine["label"].map(label_map)

print("\nClass counts before neutral augmentation:")
print(df_fine["label_ternary"].value_counts())

# Augment Neutral → 10k
neutral_texts = df_fine[df_fine["label_ternary"] == 1]["text"].tolist()
current_neutral = len(neutral_texts)
needed_neutral = max(10000 - current_neutral, 0)

neutral_augmented = []
neutral_log = []

if needed_neutral > 0:
    idx, used_count = 0, 0
    batch, batch_orig = [], []

    with tqdm(total=needed_neutral, desc="Neutral Augmentation") as pbar:
        while used_count < needed_neutral:
            text = neutral_texts[idx % len(neutral_texts)]
            batch.append(text)
            batch_orig.append(text)

            if len(batch) == 16 or used_count + len(batch) >= needed_neutral:
                bt_texts = back_translate(batch)
                for orig, aug in zip(batch_orig, bt_texts):
                    if used_count >= needed_neutral:
                        break
                    neutral_augmented.append({"label": 1, "text": aug})
                    neutral_log.append({
                        "original": orig,
                        "augmented": aug,
                        "method": "back_translation"
                    })
                    used_count += 1
                    pbar.update(1)
                batch, batch_orig = [], []
            idx += 1

# Combine ternary dataset
df_ternary = df_fine[["label_ternary", "text"]].rename(columns={"label_ternary": "label"}).copy()
df_ternary = pd.concat([df_ternary, pd.DataFrame(neutral_augmented)], ignore_index=True)

# Save ternary dataset
output_file_ternary = "ternarysst5_10k_each_class.csv"
log_file_ternary = "ternarysst5_neutral_log.csv"

df_ternary.to_csv(output_file_ternary, index=False)
pd.DataFrame(neutral_log).to_csv(log_file_ternary, index=False)

print("\nTernary dataset saved:", output_file_ternary)
print("Neutral augmentation log saved:", log_file_ternary)
print(df_ternary["label"].value_counts())


end = time.time()
total_seconds = int(end - start)
hours = total_seconds // 3600
minutes = (total_seconds % 3600) // 60
seconds = total_seconds % 60

print(f"Execution finished in {hours}h {minutes}m {seconds}s")
