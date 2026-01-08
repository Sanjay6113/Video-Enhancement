# Save as diagnostic.py
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
except:
    print("❌ PyTorch not installed")

try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except:
    print("❌ OpenCV not installed")

try:
    import gfpgan
    print(f"✅ GFPGAN installed")
except:
    print("❌ GFPGAN not installed")