import os
import torch
import pandas as pd
import evaluate
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import PeftModel
# Import your custom collator class
from finetune_whisper import (
    setup_macos_environment,
    compute_metrics,
    DataCollatorSpeechSeq2SeqWithPadding  # Ensure this is imported
)

# --- CONFIGURATION ---
OUTPUT_DIR = "./whisper-small-hi-lora"
DEVICE = setup_macos_environment()


def evaluate_model(model, processor, test_dataset, device, dataset_name="FLEURS Hindi"):
    # Preprocess FLEURS
    def preprocess_fleurs(batch):
        audio = batch["audio"]
        # Ensure we return input_features key correctly
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        # Ensure labels are tokenized
        batch["labels"] = processor.tokenizer(
            batch["raw_transcription"]).input_ids
        return batch

    print(f"🔄 Preprocessing {dataset_name}...")
    # Remove all original columns to prevent conflicts
    processed = test_dataset.map(
        preprocess_fleurs,
        remove_columns=test_dataset.column_names,
        num_proc=2
    )

    model = model.to(device)

    # Instantiate the Data Collator (CRITICAL FIX)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Eval Arguments
    eval_args = Seq2SeqTrainingArguments(
        output_dir="./eval-temp",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=False,
        dataloader_num_workers=0,
        report_to=["none"],
        remove_unused_columns=False  # Prevent accidental removal of 'input_features'
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,  # Pass the custom collator here
        compute_metrics=lambda pred: compute_metrics(pred, processor)
    )

    print(f"🔍 Evaluating on {dataset_name}...")
    results = trainer.evaluate(processed)
    wer = results['eval_wer'] * 100
    print(f"📊 {dataset_name} WER: {wer:.2f}%")
    return wer


def main():
    # 1. Setup Processor
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        language="hindi",
        task="transcribe"
    )

    # 2. Load Test Data
    print("📥 Loading FLEURS Hindi test set...")
    try:
        fleurs_test = load_dataset(
            "google/fleurs",
            "hi_in",
            split="test",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Ensure you have run: pip install datasets==2.21.0")
        return

    # 3. Evaluate Baseline
    print("\n📊 Evaluating BASELINE (Pretrained)...")
    baseline_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small",
        torch_dtype=torch.float32
    )
    baseline_wer = evaluate_model(
        baseline_model, processor, fleurs_test, DEVICE, "Baseline")

    # 4. Evaluate Fine-Tuned
    print("\n📊 Evaluating FINE-TUNED (LoRA)...")
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ Error: Fine-tuned model not found at {OUTPUT_DIR}")
        return

    finetuned_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small",
        torch_dtype=torch.float32
    )
    finetuned_model = PeftModel.from_pretrained(finetuned_model, OUTPUT_DIR)
    finetuned_wer = evaluate_model(
        finetuned_model, processor, fleurs_test, DEVICE, "Fine-tuned")

    # 5. Save Results
    print("\n" + "="*60)
    print(f"{'Model':<30} {'WER (%)':<10} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'Baseline':<30} {baseline_wer:<10.2f} {'—':<15}")
    print(f"{'Fine-tuned':<30} {finetuned_wer:<10.2f} {(baseline_wer - finetuned_wer):<15.2f}")
    print("="*60)

    pd.DataFrame({
        'Model': ['Baseline', 'Fine-tuned'],
        'WER': [baseline_wer, finetuned_wer]
    }).to_csv('final_results.csv', index=False)


if __name__ == "__main__":
    main()
