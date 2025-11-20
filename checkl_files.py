import os
import pandas as pd

CACHE_DIR = "./audio_cache"
CSV_PATH = "FT Data - data.csv"


def check_data():
    print(f"📂 Checking {CACHE_DIR}...")

    if not os.path.exists(CACHE_DIR):
        print(f"❌ Error: Directory {CACHE_DIR} does not exist.")
        return

    # 1. List actual files on disk
    files = os.listdir(CACHE_DIR)
    wav_files = [f for f in files if f.endswith('.wav')]
    print(f"found {len(wav_files)} .wav files in cache.")

    if len(wav_files) > 0:
        print(f"   Example file on disk: '{wav_files[0]}'")
    else:
        print("❌ No .wav files found! Did the download finish?")
        return

    # 2. Check what the CSV expects
    print(f"\n📄 Reading {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
        first_id = df.iloc[0]['recording_id']
        print(f"   First ID in CSV: {first_id}")

        expected_name = f"{first_id}.wav"
        print(f"   Script expects:  '{expected_name}'")

        # 3. Check for match
        if expected_name in files:
            print("\n✅ MATCH! The file exists.")
        else:
            print("\n❌ MISMATCH!")
            print(f"   The script looks for: '{expected_name}'")
            # Check for common variations
            variation = f"{first_id}_audio.wav"
            if variation in files:
                print(f"   BUT found this instead: '{variation}'")
                print("   👉 SOLUTION: The files have '_audio' in the name.")
            else:
                print("   The file is missing completely.")

    except Exception as e:
        print(f"❌ Error reading CSV: {e}")


if __name__ == "__main__":
    check_data()
