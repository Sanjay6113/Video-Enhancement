# Video-Enhancement
Amusement Park Ride Video Enhancer: A Python-based computer vision tool that removes severe motion blur from high-velocity ride footage using Generative AI (CodeFormer). Enhances rider faces in video streams for clear reaction captures.

# 🎢 Amusement Park Ride Video Enhancer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-orange)
![AI Model](https://img.shields.io/badge/Model-CodeFormer-green)

## 📋 Overview
This tool is designed for amusement park developers to solve the problem of **Motion Blur** in high-speed ride cameras. 

When capturing video on roller coasters or drop towers, the combination of high velocity and low shutter speed often results in unrecognizable faces. This project uses **Generative Facial Prior (CodeFormer)** to detect, crop, and reconstruct rider faces, hallucinating missing details to produce crystal-clear reaction videos.

## 🚀 Features
* **Motion Blur Removal:** Specifically tuned to fix "smearing" caused by high G-force movement.
* **Generative Restoration:** Uses Transformer-based prediction to reconstruct eyes, teeth, and skin texture.
* **Smart Pipeline:**
    * Detects faces using `RetinaFace`.
    * Aligns and warps faces for optimal restoration.
    * Pastes enhanced faces back into the original video frame seamlessly.
* **GPU Acceleration:** Optimized for NVIDIA CUDA to process frames efficiently.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Core Library:** PyTorch (TorchVision)
* **Computer Vision:** OpenCV (`cv2`)
* **AI Architecture:** [CodeFormer](https://github.com/sczhou/CodeFormer) (Transformer-based Face Restoration)

## ⚙️ Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/ride-enhancer.git](https://github.com/your-username/ride-enhancer.git)
    cd ride-enhancer
    ```

2.  **Install Dependencies**
    *(Requires a machine with NVIDIA GPU for reasonable performance)*
    ```bash
    # Install PyTorch with CUDA support
    pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    
    # Install project requirements
    pip install -r requirements.txt
    python basicsr/setup.py develop
    ```

3.  **Download Model Weights**
    ```bash
    python scripts/download_pretrained_models.py facelib
    python scripts/download_pretrained_models.py CodeFormer
    ```

## 💻 Usage

Place your raw ride video in the project folder and run:

```bash
python park_tool.py
