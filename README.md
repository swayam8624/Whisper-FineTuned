# Whisper-Small Hindi Fine-tuning

This project fine-tunes `openai/whisper-small` on Hindi ASR data using LoRA (Low-Rank Adaptation) for efficient training.

## Prerequisites

1.  **Python 3.8+**
2.  **GPU** (Recommended: NVIDIA T4 or better with at least 8GB VRAM)
3.  **Google Cloud Storage Credentials**: A `credentials.json` file with access to the audio buckets.
4.  **Dataset**: `FT_Data.csv` exported from the provided Google Sheet.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Place your `credentials.json` and `FT_Data.csv` in the root directory.

## Usage

Run the fine-tuning script:

```bash
python finetune_whisper.py
```

## Output

-   **Model**: Saved in `./whisper-small-hi-lora`
-   **Results**: `evaluation_results.csv` containing WER metrics for baseline vs. fine-tuned models.
