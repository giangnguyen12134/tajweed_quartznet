from pathlib import Path

data_dir = Path("data/everyayah_without_verifier")

if not data_dir.exists():
    print("The 'data' directory does not exist.")
else:
    # Use rglob("*.json") to match .json files in subdirectories,
    # or glob("*.json") for only the top-level data/ folder.
    json_files = list(data_dir.rglob("*.json"))

    print(f"Found {len(json_files)} .json file(s) in '{data_dir}'. Deleting...")

    for f in json_files:
        try:
            f.unlink()
        except Exception as e:
            print(f"Error deleting {f}: {e}")

    print("Cleanup complete!")