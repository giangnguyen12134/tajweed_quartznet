import os

TARGET_DIR = r"f:\VMX\AiProjects\Quran_ASR\data\cache"

os.environ["HF_HOME"] = TARGET_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(TARGET_DIR, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(TARGET_DIR, "datasets")

os.environ["TMP"] = os.path.join(TARGET_DIR, "tmp")
os.environ["TEMP"] = os.path.join(TARGET_DIR, "tmp")
os.environ["TMPDIR"] = os.path.join(TARGET_DIR, "tmp")

os.makedirs(os.path.join(TARGET_DIR, "tmp"), exist_ok=True)

from datasets import load_dataset

LOCAL_DIR = "./data/everyayah"
LOCAL_CACHE_DIR = "./data/everyayah_cache"

os.makedirs(LOCAL_DIR, exist_ok=True)
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

print("Downloading and caching tarteel-ai/everyayah locally...")
dataset = load_dataset("tarteel-ai/everyayah", cache_dir=LOCAL_CACHE_DIR)

dataset.save_to_disk(LOCAL_DIR)

print("Download complete! Dataset cached successfully.")
