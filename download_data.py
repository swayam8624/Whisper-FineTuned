import os
import pandas as pd
import subprocess
from tqdm import tqdm

# CONFIGURATION
CSV_PATH = "FT Data - data.csv"
CACHE_DIR = "./audio_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def fix_url(url):
    """
    Patches the broken URLs from the CSV.
    1. Converts HTTPS to gs://
    2. Fixes the folder name: 'hq_data/hi' -> 'hq/hi'
    """
    if pd.isna(url):
        return None

    url = str(url).strip()

    # Convert HTTPS to GS URI
    if "storage.googleapis.com" in url:
        url = url.replace("https://storage.googleapis.com/", "gs://")

    # PATCH: Fix the folder name mismatch
    if "/hq_data/hi/" in url:
        url = url.replace("/hq_data/hi/", "/hq/hi/")

    return url


def download_via_cli(gs_uri, local_path):
    try:
        # Use gcloud CLI for robust authentication
        subprocess.run(
            ["gcloud", "storage", "cp", gs_uri, local_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    print("🚀 Starting Download with URL Patching...")
    df = pd.read_csv(CSV_PATH)

    tasks = []
    for _, row in df.iterrows():
        tasks.append({
            "url": fix_url(row['rec_url_gcp']),
            "path": os.path.join(CACHE_DIR, f"{row['recording_id']}.wav")
        })
        tasks.append({
            "url": fix_url(row['transcription_url_gcp']),
            "path": os.path.join(CACHE_DIR, f"{row['recording_id']}_trans.json")
        })

    success_count = 0
    pbar = tqdm(tasks, desc="Downloading")
    for task in pbar:
        if not task['url']:
            continue
        if os.path.exists(task['path']):
            success_count += 1
            continue

        if download_via_cli(task['url'], task['path']):
            success_count += 1
        else:
            pbar.write(f"❌ Failed: {task['url']}")

    print(f"\n✅ Download complete. {success_count}/{len(tasks)} files ready.")


if __name__ == "__main__":
    main()
