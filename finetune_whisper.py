"""
WHISPER-SMALL HINDI ASR FINE-TUNING PIPELINE (macOS Optimized)
================================================================
Task: Fine-tune openai/whisper-small on ~10 hours of Hindi audio data
Evaluation: Compare baseline vs fine-tuned on FLEURS Hindi test set
Metric: Word Error Rate (WER) - lower is better

macOS-Specific Optimizations:
- Apple Silicon (M1/M2/M3) MPS acceleration support
- Intel Mac CPU fallback with optimized threading
- Memory-efficient processing for 8-16GB RAM Macs
- Fork vs spawn multiprocessing handling
"""

import os
import json
import pandas as pd
import torch
import numpy as np
import random
from datasets import Dataset, Audio, load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
# Add these standard imports if missing
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from google.cloud import storage
import evaluate
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================================================================
# MACOS-SPECIFIC CONFIGURATION
# ==============================================================================


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 1. Extract Audio Features
        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]

        # 2. Extract Labels (Text)
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]

        # 3. Pad Audio (Results in 'input_features')
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")

        # 4. Pad Labels (Results in 'input_ids' and 'attention_mask')
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        # 5. Handle Label Padding (-100 ignore index)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # 6. Remove BOS token if present (Whisper specific)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        # --- THE FIX: DEFENSIVE CLEANUP ---
        # Whisper's forward pass crashes if 'input_ids' or 'attention_mask'
        # are passed to the encoder. We explicitly remove them from the batch.
        if "input_ids" in batch:
            del batch["input_ids"]
        if "attention_mask" in batch:
            del batch["attention_mask"]
        # ----------------------------------

        return batch


def create_data_collator(processor, model):
    return DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )


def setup_macos_environment():
    """
    Configure optimal settings for macOS

    WHY THESE SETTINGS:
    1. MPS (Metal Performance Shaders) = Apple's GPU acceleration
    2. Fork multiprocessing = prevents spawn issues on macOS
    3. OMP threads = prevents oversubscription on M-series chips
    4. Memory limits = prevents swap thrashing on 8GB Macs
    """
    # Set multiprocessing start method (critical for macOS)
    # WHY: macOS defaults to 'spawn' which causes issues with HF datasets
    # 'fork' is faster and more memory-efficient
    import multiprocessing
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # Already set

    # Optimize CPU threading for M-series chips
    # WHY: M1/M2/M3 have performance + efficiency cores
    # Prevent thread oversubscription which kills performance
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    # Disable MPS fallback warnings
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    print("🖥️  macOS Environment Configuration")
    print("-" * 60)

    # Detect device (MPS for Apple Silicon, CPU for Intel)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Apple Silicon detected - Using MPS acceleration")
        print("   Note: MPS is 3-5x faster than CPU for training")
    else:
        device = torch.device("cpu")
        print("✅ Intel Mac detected - Using CPU")
        print("   Note: Training will be slower (~6-8 hours)")

    # Memory detection
    import psutil
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"💾 Available RAM: {total_ram_gb:.1f} GB")

    if total_ram_gb < 16:
        print("⚠️  Warning: <16GB RAM detected")
        print("   Recommendation: Close other apps during training")

    print("-" * 60)
    return device

# ==============================================================================
# SECTION 1: GOOGLE CLOUD STORAGE AUTHENTICATION & DATA ACCESS
# ==============================================================================


def setup_gcs_access(credentials_path):
    """
    WHY THIS METHOD:
    - Uses official GCS Python SDK (reliable, handles auth/retries)
    - Downloads files on-demand (only when needed during processing)
    - Automatic cleanup after processing each file
    - Supports parallel processing with minimal disk usage

    macOS NOTE: No special handling needed - GCS SDK works identically
    """
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"GCS credentials not found at {credentials_path}\n"
            f"Please download from Google Cloud Console:\n"
            f"1. Go to IAM & Admin > Service Accounts\n"
            f"2. Create/select service account\n"
            f"3. Keys > Add Key > JSON\n"
            f"4. Save as {credentials_path}"
        )

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    return storage.Client()


def download_from_gcs(storage_client, gcs_url, local_path):
    """
    Handles both:
    1. gs://bucket/path/file.wav
    2. https://storage.googleapis.com/bucket/path/file.wav
    """
    if pd.isna(gcs_url):
        raise ValueError("GCS URL is NaN/Empty")

    url = str(gcs_url).strip()

    bucket_name = None
    blob_path = None

    # Case 1: HTTPS URL (Your data format)
    if url.startswith("https://storage.googleapis.com/"):
        # Remove the prefix
        path_without_prefix = url.replace(
            "https://storage.googleapis.com/", "")
        # Split into [bucket, path/to/file]
        parts = path_without_prefix.split("/", 1)
        if len(parts) == 2:
            bucket_name = parts[0]
            blob_path = parts[1]

    # Case 2: GS URL
    elif url.startswith("gs://"):
        # Remove the prefix
        path_without_prefix = url[5:]
        # Split into [bucket, path/to/file]
        parts = path_without_prefix.split("/", 1)
        if len(parts) == 2:
            bucket_name = parts[0]
            blob_path = parts[1]

    # Validation
    if not bucket_name or not blob_path:
        raise ValueError(f"Could not parse bucket/path from URL: {url}")

    # Download
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_path)
    except Exception as e:
        print(f"⚠️ Failed to download {url}: {e}")
        # Return None so we can skip this file later
        if os.path.exists(local_path):
            os.remove(local_path)
        return None

    return local_path

# ==============================================================================
# SECTION 2: DATASET PREPROCESSING (macOS Optimized)
# ==============================================================================


def create_hf_dataset(csv_path, storage_client=None, cache_dir="./audio_cache"):
    # Basic check
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    df = pd.read_csv(csv_path)
    data_list = []

    print(f"📂 Reading from {cache_dir}...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading local files"):
        # Handle potential filename variations (with or without _audio suffix)
        # We check standard ID first
        audio_path = os.path.join(cache_dir, f"{row['recording_id']}.wav")
        if not os.path.exists(audio_path):
            # Fallback to _audio suffix
            audio_path = os.path.join(
                cache_dir, f"{row['recording_id']}_audio.wav")

        trans_path = os.path.join(
            cache_dir, f"{row['recording_id']}_trans.json")
        if not os.path.exists(trans_path):
            trans_path = os.path.join(
                cache_dir, f"{row['recording_id']}_transcription.json")

        # If still missing, skip
        if not os.path.exists(audio_path) or not os.path.exists(trans_path):
            continue

        try:
            with open(trans_path, 'r', encoding='utf-8') as f:
                trans_data = json.load(f)

                text = ""
                # --- FIX: HANDLE LIST VS DICT ---
                if isinstance(trans_data, list):
                    # Case: [{"transcription": "..."}]
                    if len(trans_data) > 0:
                        item = trans_data[0]
                        if isinstance(item, dict):
                            text = item.get('transcription') or item.get(
                                'text') or ""
                        else:
                            text = str(item)  # Case: ["Just the text"]
                elif isinstance(trans_data, dict):
                    # Case: {"transcription": "..."}
                    text = trans_data.get(
                        'transcription') or trans_data.get('text') or ""

                if not text:
                    # Skip empty transcripts to prevent training errors
                    continue

            data_list.append({
                'audio': audio_path,
                'transcription': text,
                'duration': row['duration'],
                'recording_id': row['recording_id']
            })

        except json.JSONDecodeError:
            print(f"⚠️ Skipping empty/corrupt JSON: {trans_path}")
            continue
        except Exception as e:
            print(f"⚠️ Error reading {trans_path}: {e}")
            continue

    if len(data_list) == 0:
        raise ValueError(
            "No valid audio/transcript pairs found in cache folder! Check file naming patterns.")

    dataset = Dataset.from_list(data_list)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def prepare_dataset_for_training(dataset, processor, use_augmentation=True):

    # 1. Verify 'transcription' exists before processing
    if "transcription" not in dataset.column_names:
        # Debug output to help identify why data is missing
        print(f"⚠️  DATASET ERROR: 'transcription' column is missing.")
        print(f"   Current columns: {dataset.column_names}")
        if len(dataset) > 0:
            # Print keys of the first item to verify data structure
            print(f"   First item keys: {list(dataset[0].keys())}")
        raise ValueError(
            "Dataset missing 'transcription' column. Check create_hf_dataset function.")

    def augment_audio(audio_array, sample_rate=16000):
        if not use_augmentation or random.random() > 0.5:
            return audio_array

        augmented = audio_array.copy()

        # Speed perturbation
        if random.random() < 0.3:
            speed_factor = random.uniform(0.9, 1.1)
            indices = np.round(
                np.arange(0, len(augmented), speed_factor)).astype(int)
            indices = indices[indices < len(augmented)]
            augmented = augmented[indices]

        # Additive noise
        if random.random() < 0.3:
            noise_amp = random.uniform(0.001, 0.01)
            noise = np.random.randn(len(augmented)) * noise_amp
            augmented = augmented + noise

        return augmented

    def preprocess_function(batch):
        audio = batch["audio"]
        audio_array = audio["array"]

        # Augment
        audio_array = augment_audio(audio_array)

        batch["input_features"] = processor.feature_extractor(
            audio_array,
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    # 2. DYNAMIC COLUMN REMOVAL (The Fix)
    # Only remove columns that actually exist in the dataset
    cols_to_remove = [c for c in ["audio", "transcription"]
                      if c in dataset.column_names]

    processed = dataset.map(
        preprocess_function,
        remove_columns=cols_to_remove,
        num_proc=2,
        desc="Extracting features",
        load_from_cache_file=False  # Set to False temporarily to force data reload
    )

    return processed


def filter_and_validate(dataset):
    def is_valid(example):
        # Ensure text exists
        if len(example["labels"]) == 0:
            return False
        # Relaxed filter: Only drop if < 1s.
        # We will truncate >30s in the tokenizer/feature extractor automatically
        if example["duration"] < 1:
            return False
        return True

    filtered = dataset.filter(is_valid, desc="Filtering invalid samples")
    return filtered


def split_dataset(dataset, test_size=0.1, seed=42):
    """
    WHY 90/10 SPLIT:
    - 90% training = enough data to learn patterns
    - 10% validation = enough samples for reliable WER estimates
    - seed=42 = reproducible splits
    """
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]

# ==============================================================================
# SECTION 3: MODEL FINE-TUNING (macOS Optimized)
# ==============================================================================


def setup_lora_model(base_model_name="openai/whisper-small", device=None):
    """
    WHY LORA (Parameter-Efficient Fine-Tuning):
    - Freezes original 244M parameters (they stay unchanged)
    - Adds tiny "adapter" matrices (~1.5M params)
    - Only trains 0.6% of original model size

    MACOS OPTIMIZATIONS:
    - Use float32 instead of float16 for MPS compatibility
    - MPS doesn't fully support float16 operations yet
    - Automatic device placement (MPS or CPU)
    """

    # MACOS: Use float32 instead of float16
    # WHY: MPS backend has limited float16 support
    # Trade-off: 2x memory but stable training
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name,
        dtype=torch.float32,  # Changed from float16 for macOS
    )

    # Move to device manually (device_map="auto" not reliable on MPS)
    if device is not None:
        model = model.to(device)

    # Configure LoRA adapters
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none"
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    # Verify parameter efficiency
    model.print_trainable_parameters()

    return model


def create_data_collator(processor, model):
    """
    WHY DYNAMIC PADDING:
    - Audio lengths vary - dynamic padding saves computation

    WHY DECODER_START_TOKEN_ID:
    - Tells Whisper to start generation properly
    """
    return DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )


def compute_metrics(pred, processor):
    """
    WHY WER (not accuracy or BLEU):
    - Industry standard for ASR evaluation
    - Directly measures transcription errors
    """
    wer_metric = evaluate.load("wer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Handle padding tokens
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode to text
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def train_model(model, processor, train_dataset, eval_dataset, device, output_dir="./whisper-hi-lora"):
    data_collator = create_data_collator(processor, model)

    # Detect if using MPS
    use_mps = str(device) == "mps"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,  # Kept low for stability
        gradient_accumulation_steps=8,  # INCREASED for stable gradient update
        learning_rate=5e-6,             # CRITICAL FIX: REDUCED by 20x for low-data stability
        warmup_steps=100,               # INCREASED for gentle start
        max_steps=2000,                 # INCREASED to allow lower LR to converge
        gradient_checkpointing=False,   # Keep False if memory allows, True if crashing
        fp16=False,                     # Critical for MPS
        eval_strategy="steps",
        eval_steps=100,                 # Check performance more often
        save_steps=200,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to=["none"],
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor,  # <-- pass full WhisperProcessor here
        compute_metrics=lambda pred: compute_metrics(pred, processor)
    )

    print("🚀 Starting fine-tuning...")
    if use_mps:
        print("⚡ Using Apple Silicon MPS acceleration")
    else:
        print("⚙️  Using CPU (this will take longer)")

    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    print(f"✅ Model saved to {output_dir}")

    return trainer

# ==============================================================================
# SECTION 4: EVALUATION ON FLEURS HINDI TEST SET
# ==============================================================================


def load_fleurs_hindi():
    """
    WHY FLEURS:
    - Standardized benchmark across 102 languages
    - ~12 hours per language (sufficient test set size)

    WHY "hi_in":
    - Hindi as spoken in India
    - Includes regional variations
    """
    print("📥 Loading FLEURS Hindi test set...")
    fleurs = load_dataset("google/fleurs", "hi_in",
                          split="test", trust_remote_code=True)
    return fleurs


def evaluate_model(model, processor, test_dataset, device, dataset_name="FLEURS Hindi"):
    """
    WHY SEPARATE EVALUATION FUNCTION:
    - Reusable for baseline and fine-tuned models
    - Consistent evaluation settings
    """
    # Preprocess FLEURS with same pipeline as training data
    def preprocess_fleurs(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(
            batch["raw_transcription"]).input_ids
        return batch

    processed = test_dataset.map(
        preprocess_fleurs,
        remove_columns=test_dataset.column_names,
        num_proc=2  # Reduced for macOS
    )

    # Move model to device
    model = model.to(device)

    # Evaluation-only trainer
    eval_args = Seq2SeqTrainingArguments(
        output_dir="./eval-temp",
        per_device_eval_batch_size=4,  # Reduced for macOS
        predict_with_generate=True,
        generation_max_length=225,
        fp16=False,  # Disabled for macOS
        dataloader_num_workers=0,  # Critical for macOS
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        tokenizer=processor.feature_extractor,
        compute_metrics=lambda pred: compute_metrics(pred, processor)
    )

    print(f"🔍 Evaluating on {dataset_name}...")
    results = trainer.evaluate(processed)
    wer = results['eval_wer'] * 100
    print(f"📊 {dataset_name} WER: {wer:.2f}%")

    return wer

# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================


def main():
    """
    COMPLETE WORKFLOW (macOS Optimized):
    1. Setup macOS environment
    2. Authenticate with GCS
    3. Load and preprocess training data
    4. Fine-tune Whisper-small with LoRA
    5. Evaluate baseline and fine-tuned models
    6. Report results
    """

    # Configuration
    GCS_CREDENTIALS = "gcs-creds.json"
    TRAINING_CSV = "FT Data - data.csv"
    OUTPUT_DIR = "./whisper-small-hi-lora"

    print("=" * 60)
    print("WHISPER-SMALL HINDI FINE-TUNING PIPELINE (macOS)")
    print("=" * 60)

    # Step 1: Setup macOS environment
    device = setup_macos_environment()

    # Validate files exist
    if not os.path.exists(GCS_CREDENTIALS):
        print(f"\n❌ Error: {GCS_CREDENTIALS} not found")
        print("Please download GCS credentials and save as 'gcs-creds.json'")
        return

    if not os.path.exists(TRAINING_CSV):
        print(f"\n❌ Error: {TRAINING_CSV} not found")
        print("Please ensure the dataset CSV is in the current directory")
        return

    # Step 2: Setup GCS and processor
    storage_client = setup_gcs_access(GCS_CREDENTIALS)
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        language="hindi",
        task="transcribe"
    )

    # Step 3: Preprocess training data
    print("\n📂 Loading training dataset...")
    raw_dataset = create_hf_dataset(TRAINING_CSV, storage_client)
    print(f"✅ Loaded {len(raw_dataset)} samples")

    print("\n🔄 Preprocessing with data augmentation...")
    processed_dataset = prepare_dataset_for_training(
        raw_dataset, processor, use_augmentation=True)
    filtered_dataset = filter_and_validate(processed_dataset)
    train_dataset, eval_dataset = split_dataset(filtered_dataset)

    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(eval_dataset)}")

    # Step 4: Fine-tune model
    print("\n🔧 Setting up LoRA model...")
    model = setup_lora_model(device=device)

    print("\n🚀 Starting training...")
    print("⏱️  Estimated time:")
    if str(device) == "mps":
        print("   - Apple Silicon: 2-4 hours")
    else:
        print("   - Intel Mac: 6-8 hours")

    trainer = train_model(model, processor, train_dataset,
                          eval_dataset, device, OUTPUT_DIR)

    # Step 5: Load FLEURS test set
    fleurs_test = load_fleurs_hindi()

    # Step 6: Evaluate baseline
    print("\n📊 Evaluating BASELINE (pretrained Whisper-small)...")
    baseline_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small",
        torch_dtype=torch.float32
    )
    baseline_wer = evaluate_model(
        baseline_model, processor, fleurs_test, device, "Baseline")

    # Step 7: Evaluate fine-tuned model
    print("\n📊 Evaluating FINE-TUNED model...")
    finetuned_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small",
        torch_dtype=torch.float32
    )
    finetuned_model = PeftModel.from_pretrained(finetuned_model, OUTPUT_DIR)
    finetuned_wer = evaluate_model(
        finetuned_model, processor, fleurs_test, device, "Fine-tuned")

    # Step 8: Report results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Model':<30} {'WER (%)':<10} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'Baseline (Pretrained)':<30} {baseline_wer:<10.2f} {'—':<15}")
    print(f"{'Fine-tuned (LoRA)':<30} {finetuned_wer:<10.2f} {(baseline_wer - finetuned_wer):<15.2f}")
    print("=" * 60)

    # Save results to CSV
    results_df = pd.DataFrame({
        'Model': ['Pretrained Whisper-small', 'Fine-tuned Whisper-small (LoRA)'],
        'WER (%)': [baseline_wer, finetuned_wer],
        'Notes': ['Baseline', f'Improvement: {baseline_wer - finetuned_wer:.2f}%']
    })
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\n💾 Results saved to evaluation_results.csv")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
