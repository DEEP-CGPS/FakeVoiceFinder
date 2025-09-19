# FakeVoiceFinder 🎙️🔍

**FakeVoiceFinder** is a Python library for the **detection and analysis of synthetic voices (fake voices / deepfakes)**.  
It provides tools to load models, prepare audio datasets, apply transformations, and evaluate results.

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/DEEP-CGPS/FakeVoiceFinder.git
cd FakeVoiceFinder
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
```

This will create a folder called `venv` inside your project.

---

## ▶️ Activating the environment

The activation command depends on your operating system and terminal:

### 🔹 Windows

- **PowerShell** (default in VS Code):
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  ⚠️ If you see the error *"execution of scripts is disabled"*, run the following command **once** in PowerShell (as Administrator):
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```
  Then try activation again.

- **CMD (Command Prompt)**:
  ```cmd
  venv\Scripts\activate.bat
  ```

- **Git Bash**:
  ```bash
  source venv/Scripts/activate
  ```

### 🔹 Linux / macOS
```bash
source venv/bin/activate
```

When the environment is active, your terminal prompt will look like this:
```
(venvtest) PS C:\Users\YourUser\FakeVoiceFinder>
```

---

### 3. Install the library
```bash
pip install .
```

### 4. Install PyTorch and Torchvision
**PyTorch is not included in the default installation**. This is intentional, so you can install the correct version depending on your hardware.

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

See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for other CUDA/ROCm options.

---