# Prerequirement

>**Run once: Create a virtual environment and enable Jupyter Notebook support.**

```bat
conda create -n vmx_quran_asr_env python=3.10
python -m ipykernel install --user --name=vmx_quran_asr_env --display-name "Python (vmx_quran_asr_env)"
```

> **Install the required packages.**

```bat
conda activate vmx_quran_asr_env
pip install -r requirements.txt
```

> **Or install each required package separately**
```bat
pip install torch torchvision torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu126
pip install datasets
pip install soundfile tqdm
pip install quranic-phonemizer
```

# Model Architecture

```text
Input Spectrogram
(B, 80, T) @ 100 FPS
        |
        v

+------------------------------------------------------------------------------------+
| PROLOGUE                                                                           |
|                                                                                    |
| C1: Conv1d (k=13, s=2) -----------------------------> 80 -> 256 ch  [50 FPS]       |
+------------------------------------------------------------------------------------+
        |
        v

+------------------------------------------------------------------------------------+
| ENCODER BODY (Jasper Residual Blocks with Depthwise Separable Convs)               |
|                                                                                    |
| B1: 3x SeparableConv1d (k=17) + Residual ------------> 256 -> 256 ch               |
| B2: 3x SeparableConv1d (k=21) + Residual ------------> 256 -> 384 ch               |
| B3: 3x SeparableConv1d (k=25) + Residual ------------> 384 -> 512 ch               |
| B4: 3x SeparableConv1d (k=29) + Residual ------------> 512 -> 512 ch               |
+------------------------------------------------------------------------------------+
        |
        v

+------------------------------------------------------------------------------------+
| EPILOGUE                                                                           |
|                                                                                    |
| C2: SeparableConv1d (k=33) -------------------------> 512 -> 640 ch                |
| C3: Dense Conv1d (k=1) -----------------------------> 640 -> 768 ch                |
+------------------------------------------------------------------------------------+
        |
        +------------------------------------------+
        |                                          |
        v                                          v

+----------------------------------+    +--------------------------------------------+
| CTC HEAD                         |    | TAJWEED VERIFIER HEAD (Optional)           |
|                                  |    |                                            |
| Conv1d 1x1 (768 -> 73)           |    | Linear Bottleneck (768 -> 128)             |
| LogSoftmax                       |    | BiGRU (hidden=96) + Extra Metadata         |
| 
```

 # HowTo
 > **1. Download dataset to local PC**
 ```bat
 python scripts\download_everyayah.py
 python scripts\extract_dataset.py
 ```
  > **2. Build JSON metadata for each file in the dataset. This step performs grapheme-to-phoneme (G2P) conversion using the quranic-phonemizer library**
  ```bat
 python scripts\build_dataset_without_verifier.py
 # The build_dataset.py also run the CTC forced aligment for each file in the dataset by using the seed model.
 # This is only needed for Tajweed verifier head.
 python scripts\build_dataset.py
 ```
> **3. Start the training process**
```bat
# Run the light mode to ensure that the model is working correctly
python src/training/train.py --data-dir data/everyayah_extracted --light --lr 5e-4  --epochs 500

# Then, run the full mode
python src/training/train.py --data-dir data/everyayah_extracted --full --epochs 120
```