# FakeVoiceFinder 🎙️🔍

**FakeVoiceFinder** is a Python library for the **detection and analysis of synthetic voices (fake voices / deepfakes)**.  
It provides tools to load models, prepare audio datasets, apply transformations, and evaluate results.

---

## 🚀 Installation

You can set up the environment either with **Conda** or with **pip + virtualenv**.  

---

### 🔹 Option 1: Using Conda
1. Clone the repository:
```bash
git clone https://github.com/DEEP-CGPS/FakeVoiceFinder.git
cd FakeVoiceFinder
```

2. Create the Conda environment:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate fakevoice
```

4. Install PyTorch and Torchvision according to your hardware (see section below).

---

### 🔹 Option 2: Using pip + virtualenv
1. Clone the repository:
```bash
git clone https://github.com/DEEP-CGPS/FakeVoiceFinder.git
cd FakeVoiceFinder
```

2. Create a virtual environment:
```bash
python -m venv venvtest
```

3. Activate the environment:

- **PowerShell (default in VS Code):**
  ```powershell
  .\venvtest\Scripts\Activate.ps1
  ```
  ⚠️ If you see *“execution of scripts is disabled”*, run this once as Administrator:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```

- **CMD (Command Prompt):**
  ```cmd
  venvtest\Scripts\activate.bat
  ```

- **Git Bash:**
  ```bash
  source venvtest/Scripts/activate
  ```

- **Linux / macOS:**
  ```bash
  source venvtest/bin/activate
  ```

When activated, your prompt will look like:
```
(venvtest) PS C:\Users\YourUser\FakeVoiceFinder>
```

4. Install the requirements:
```bash
pip install -r requirements.txt
```

5. Install PyTorch and Torchvision according to your hardware (see section below).

---

## 🔹 Installing PyTorch and Torchvision

**PyTorch is not included by default** in the environment files.  
Please install it manually according to your setup:

- **CPU only (portable and recommended if you don’t have an NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

- **GPU (CUDA 11.8, for older GPUs like GTX 10xx / RTX 20xx):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

- **GPU (CUDA 12.6, for recent NVIDIA GPUs):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

⚠️ **If installation fails**, try forcing a reinstall with:
```bash
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for other CUDA/ROCm options.
