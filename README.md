<div align="center">

# 🤟 ASL V3 — Real-Time Alphabet Translator

![Python](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![MediaPipe](https://imgshields.io/badge/MediaPipe-0.10.x-red?logo=googleneraldark&logoColor=white)
![GPU](https://img.shields.io/badge/GPU_Support-CUDA%2FCPU-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

## **Sign with your hand → Text appears instantly**

[Installation](#-installation-and-setup) · [How to Run](#%EF%B8%8F-how-to-run-the-translator) · [Results](#%F0%9F%93%8A-model-results)

---

</div>

## ✨ Project Overview

The **ASL Translator V3** is a robust real-time system designed to recognize the 26 letters of the American Sign Language (ASL) alphabet, plus the `SPACE` and `DELETE` commands.

It employs a pre-trained **ResNet-18** deep convolutional neural network for high-accuracy, low-latency classification, utilizing the webcam for input and MediaPipe for efficient hand landmark detection.

### Core Technology Stack

| Component | Role | Implementation |
| :--- | :--- | :--- |
| **Model** | Deep Learning Classifier | ResNet-18 (Custom PyTorch) |
| **Classification** | Output Layer | 29 classes (A-Z, DEL, NOTHING, SPACE) |
| **Vision Pipeline** | Hand Tracking & Preprocessing | OpenCV and MediaPipe |

---

## ⚙️ Installation and Setup

Follow these steps to get the translator running on your local machine.

### Prerequisites

* **Python 3.8+**
* A functional webcam.
* All project files (`translator.py`, `raw_model.pth`, etc.) must be in the same directory.

### Step 1: Clone or Download Files

Ensure all files are in the same local folder (e.g., `ASL_V3_Raw/`).

### Step 2: Create and Activate the Virtual Environment (VENV)

It is crucial to use a virtual environment to manage dependencies and avoid conflicts.

```bash
# 1. Create the virtual environment
python -m venv venv

# 2. Activate the VENV (Choose your OS command)
# On Windows (PowerShell/Command Prompt):
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

Step 3: Install Dependencies
With the environment activated, install the required libraries.
(venv) $ pip install torch torchvision torchaudio opencv-python mediapipe pillow

▶️ How to Run the Translator
Once installation is complete, you only need to perform two simple steps every time you want to use the application.

1. Activate VENV (Required for Every Session)
# Example for Windows:
.\venv\Scripts\activate

2. Execute the Script
(venv) $ python translator.py
