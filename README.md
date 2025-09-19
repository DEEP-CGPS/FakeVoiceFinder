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
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install the library
```bash
pip install .
```

### 4. Install PyTorch and Torchvision
PyTorch depends on your hardware (CPU/GPU). Please install it manually according to your setup:

- **CPU only** (portable and recommended if you don’t have an NVIDIA GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

- **With CUDA 12.6 (for recent NVIDIA GPUs):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> See more options in the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

---

## 📦 Requirements

The [`requirements.txt`](requirements.txt) file lists the main dependencies:

- numpy  
- pandas  
- matplotlib  
- scikit-learn  
- librosa  
- pywavelets  
- pillow  
- torch (install separately according to hardware)  
- torchvision (install separately according to hardware)  

---

