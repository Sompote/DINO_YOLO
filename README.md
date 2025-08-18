# YOLOv13 with DINO2 Backbone

Enhanced YOLOv13 object detection model integrated with Meta's DINO2 (DINOv2) pretrained vision transformer backbone for superior feature extraction and detection performance.

## 🚀 Features

- **🔬 DINO2 Integration**: Real pretrained weights from Meta's DINOv2 model
- **🧊 Transfer Learning**: Configurable weight freezing for DINO2 backbone
- **⚡ High Performance**: CNN + Vision Transformer hybrid architecture
- **🎯 Enhanced Detection**: Superior feature extraction for better accuracy
- **🛠️ Easy Training**: Simple command-line interface
- **📊 Production Ready**: Clean training output without debug warnings
- **🏗️ YOLOv13 Base**: Built on state-of-the-art YOLOv13 architecture

## 📋 Architecture Overview

```
YOLOv13-DINO2 Hybrid Architecture:
├── Input (3 channels RGB)
├── YOLOv13 Early Layers (Conv blocks)
├── DINO2 Backbone (Vision Transformer) ← NEW
│   ├── Patch Embeddings (14x14 patches)
│   ├── 12 Transformer Layers (frozen)
│   ├── Feature Adapter (768→512 channels)
│   └── Spatial Projection
├── YOLOv13 Neck (HyperACE + FullPAD)
└── YOLOv13 Head (Multi-scale Detection)

Model Statistics:
├── Total Parameters: ~95.8M
├── DINO2 Backbone: ~86M (frozen for transfer learning)
├── YOLOv13 Layers: ~9.8M (trainable)
└── FLOPs: ~18.1 GFLOPs
```

## 🔧 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Sompote/YOLO_Dino.git
cd YOLO_Dino
```

2. **Install dependencies:**

**Local Development:**
```bash
# Create conda environment
conda create -n yolov13-dino2 python=3.11
conda activate yolov13-dino2

# Install core packages
pip install ultralytics
pip install transformers>=4.21.0
pip install torch>=1.12.0

# Install from requirements
pip install -r requirements.txt
pip install -e .
```

**Cloud Deployment (AWS/GCP/Azure):**
```bash
# The requirements.txt is optimized for cloud environments
pip install -r requirements.txt

# Key features for cloud:
# - opencv-python-headless (no GUI dependencies)
# - Flexible version ranges for better compatibility
# - Automatic CUDA detection for GPU instances
```

## 🚀 Quick Start

### Training with DINO2 Backbone

Train YOLOv13 enhanced with DINO2 on your dataset:

```bash
# Basic training with frozen DINO2 (recommended for transfer learning)
python train_dino2.py --data path/to/data.yaml --epochs 100 --freeze-dino2

# Fast prototyping with smallest models
python train_dino2.py \
    --data data.yaml \
    --model yolov13-dino2-working \
    --size n \
    --dino-variant dinov2_vits14 \
    --epochs 50 \
    --batch-size 32 \
    --freeze-dino2

# Balanced performance
python train_dino2.py \
    --data data.yaml \
    --model yolov13-dino2-working \
    --size s \
    --dino-variant dinov2_vitb14 \
    --epochs 100 \
    --batch-size 16 \
    --freeze-dino2

# Maximum accuracy
python train_dino2.py \
    --data data.yaml \
    --model yolov13-dino2-working \
    --size x \
    --dino-variant dinov2_vitl14 \
    --epochs 200 \
    --batch-size 8 \
    --freeze-dino2
```

### Standard YOLOv13 Training

For standard YOLOv13 training without DINO2:

```bash
# Choose YOLOv13 size based on your needs
python train_dino2.py --data data.yaml --model yolov13n --epochs 100  # Fastest (2.5M params)
python train_dino2.py --data data.yaml --model yolov13s --epochs 100  # Balanced (9M params) 
python train_dino2.py --data data.yaml --model yolov13l --epochs 100  # Accurate (28M params)
python train_dino2.py --data data.yaml --model yolov13x --epochs 100  # Best (64M params)
```

**Alternative: Using Ultralytics API directly**
```python
from ultralytics import YOLO

# Choose model size: yolov13n, yolov13s, yolov13l, yolov13x
model = YOLO('ultralytics/cfg/models/v13/yolov13s.yaml')

results = model.train(
    data='coco.yaml',
    epochs=300,
    batch=64,
    imgsz=640,
    device="0,1,2,3"
)
```

## 📊 Training Arguments

### Model Selection Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--model` | YOLOv13 architecture variant | yolov13-dino2-working | yolov13n, yolov13s, yolov13l, yolov13x, yolov13-dino2-*, etc. |
| `--size` | YOLOv13 model size | None | n, s, l, x (auto-applied to base models) |
| `--dino-variant` | DINO2 model variant | dinov2_vitb14 | dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14 |
| `--freeze-dino2` | Freeze DINO2 weights | False | --freeze-dino2 |

### Training Configuration

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--data` | Dataset YAML file path | Required | Path to your data.yaml |
| `--epochs` | Number of training epochs | 100 | Integer |
| `--batch-size` | Training batch size | 16 | 4, 8, 16, 32, 64 |
| `--imgsz` | Input image size | 640 | 320, 480, 640, 1280 |
| `--name` | Experiment name | yolov13-dino2 | String |

## 🎯 Model Variants

### YOLOv13 Size Variants

Choose from 4 different YOLOv13 model sizes based on your speed/accuracy requirements:

| Size | Model | Parameters | Speed | Accuracy | Memory | Use Case |
|------|-------|------------|-------|----------|--------|----------|
| **Nano** | `yolov13n` | ~2.5M | ⚡ Fastest | Good | Low | Mobile/Edge devices |
| **Small** | `yolov13s` | ~9M | 🚀 Fast | Better | Medium | Real-time applications |
| **Large** | `yolov13l` | ~28M | 🐌 Slower | Best | High | High accuracy needs |
| **XLarge** | `yolov13x` | ~64M | 🐢 Slowest | Excellent | Very High | Maximum accuracy |

### DINO2 Model Variants

The implementation supports multiple DINO2 model sizes:

| Model | Parameters | Hidden Size | Patch Size | Performance | Speed | Memory |
|-------|------------|-------------|------------|-------------|-------|--------|
| `dinov2_vits14` | ~21M | 384 | 14×14 | Good | ⚡ Fastest | Low |
| `dinov2_vitb14` | ~86M | 768 | 14×14 | Better | 🚀 Fast | Medium |
| `dinov2_vitl14` | ~300M | 1024 | 14×14 | Best | 🐌 Slower | High |
| `dinov2_vitg14` | ~1.1B | 1536 | 14×14 | Excellent | 🐢 Slowest | Very High |

*Default: `dinov2_vitb14` (recommended balance of performance and speed)*

### Model Selection Options

```bash
# Method 1: Direct model selection
--model yolov13n           # YOLOv13 Nano
--model yolov13s           # YOLOv13 Small  
--model yolov13l           # YOLOv13 Large
--model yolov13x           # YOLOv13 XLarge

# Method 2: Base model + size modifier
--model yolov13-dino2-working --size n    # Creates yolov13-dino2-working-n
--model yolov13-dino2-working --size s    # Creates yolov13-dino2-working-s
--model yolov13-dino2-working --size l    # Creates yolov13-dino2-working-l
--model yolov13-dino2-working --size x    # Creates yolov13-dino2-working-x

# DINO2 variant selection (for DINO2-enabled models)
--dino-variant dinov2_vits14              # Small DINO2
--dino-variant dinov2_vitb14              # Base DINO2 (default)
--dino-variant dinov2_vitl14              # Large DINO2
--dino-variant dinov2_vitg14              # Giant DINO2
```

## 🎯 Recommended Model Combinations

### 🏃 For Speed-First Applications
```bash
# Ultra-fast: Nano YOLOv13 + Small DINO2
python train_dino2.py --model yolov13-dino2-working --size n --dino-variant dinov2_vits14

# Real-time: Small YOLOv13 only (no DINO2 overhead)
python train_dino2.py --model yolov13s
```

### ⚖️ For Balanced Performance  
```bash
# Recommended: Small YOLOv13 + Base DINO2
python train_dino2.py --model yolov13-dino2-working --size s --dino-variant dinov2_vitb14

# Alternative: Large YOLOv13 only
python train_dino2.py --model yolov13l
```

### 🎯 For Maximum Accuracy
```bash
# Best: XLarge YOLOv13 + Large DINO2
python train_dino2.py --model yolov13-dino2-working --size x --dino-variant dinov2_vitl14

# Research: XLarge YOLOv13 + Giant DINO2 (requires 24GB+ VRAM)
python train_dino2.py --model yolov13-dino2-working --size x --dino-variant dinov2_vitg14
```

### 💻 Hardware-Specific Recommendations

| Hardware | Recommended Combination | Command |
|----------|------------------------|---------|
| **Mobile/Edge** | YOLOv13n | `--model yolov13n` |
| **RTX 3060 (8GB)** | YOLOv13s + Small DINO2 | `--model yolov13-dino2-working --size s --dino-variant dinov2_vits14` |
| **RTX 3070 (8GB)** | YOLOv13s + Base DINO2 | `--model yolov13-dino2-working --size s --dino-variant dinov2_vitb14` |
| **RTX 4090 (24GB)** | YOLOv13l + Large DINO2 | `--model yolov13-dino2-working --size l --dino-variant dinov2_vitl14` |
| **A100 (40GB)** | YOLOv13x + Giant DINO2 | `--model yolov13-dino2-working --size x --dino-variant dinov2_vitg14` |

## 📁 Dataset Format

Your dataset should follow the YOLO format:

```yaml
# data.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # optional

nc: 5  # number of classes
names: ['class1', 'class2', 'class3', 'class4', 'class5']
```

Directory structure:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

## 🔬 Key Components

### 1. DINO2Backbone Module
- **Location**: `ultralytics/nn/modules/block.py`
- **Features**: 
  - Real pretrained weights from Meta
  - Configurable weight freezing
  - Dynamic channel adaptation
  - CNN-Transformer feature fusion

### 2. Model Configuration
- **File**: `ultralytics/cfg/models/v13/yolov13-dino2-working.yaml`
- **Integration**: DINO2 backbone at layer 4
- **Architecture**: Seamless fusion with YOLOv13 pipeline

### 3. Training Script
- **File**: `train_dino2.py`
- **Features**: Clean output, warning suppression, full functionality

## 📈 Performance Comparison

### Standard YOLOv13 Results (MS COCO)

| Model | Size | Parameters | FLOPs (G) | mAP50:95 | mAP50 | Latency (ms) | Use Case |
|-------|------|------------|-----------|----------|-------|--------------|----------|
| **YOLOv13n** | Nano | 2.5M | 6.4 | 41.6 | 57.8 | 1.97 | Mobile/Edge |
| **YOLOv13s** | Small | 9.0M | 20.8 | 48.0 | 65.2 | 2.98 | Real-time |
| **YOLOv13l** | Large | 27.6M | 88.4 | 53.4 | 70.9 | 8.63 | High accuracy |
| **YOLOv13x** | XLarge | 64.0M | 199.2 | 54.8 | 72.0 | 14.67 | Maximum accuracy |

### YOLOv13 + DINO2 Enhanced Performance

| Combination | Total Params | Expected mAP50 Boost | Expected mAP50:95 Boost | Training Time | Inference Time |
|-------------|--------------|---------------------|------------------------|---------------|----------------|
| **YOLOv13n + DINO2-Small** | ~23M | +2-4% | +1-3% | 1.5x | 1.8x |
| **YOLOv13s + DINO2-Base** | ~95M | +3-6% | +2-4% | 2.0x | 2.5x |
| **YOLOv13l + DINO2-Large** | ~328M | +4-8% | +3-6% | 3.5x | 4.0x |
| **YOLOv13x + DINO2-Giant** | ~1.16B | +5-10% | +4-7% | 6.0x | 8.0x |

*Boosts are compared to standard YOLOv13 models; Training/Inference times are relative to base YOLOv13*

### DINO2 Enhanced Benefits

- **🎯 Enhanced Feature Extraction**: Vision transformer captures global context and long-range dependencies
- **🔍 Better Small Object Detection**: Improved feature representation for detecting small and occluded objects  
- **🧠 Transfer Learning**: Leverages pretrained DINO2 knowledge from millions of images
- **💪 Robust Performance**: Superior handling of complex scenes, lighting, and challenging conditions
- **⚡ Flexible Scaling**: Choose optimal speed/accuracy trade-off for your specific use case

## 🔧 Training Output Example

```
YOLOv13 + DINO2 Training
Dataset: rail_defects/data.yaml
Epochs: 100, Batch: 16
DINO2 Frozen: True
==================================================
Loading DINO2 model: facebook/dinov2-base (from dinov2_vitb14)
✅ Successfully loaded pretrained DINO2: facebook/dinov2-base
DINO2 backbone weights frozen: dinov2_vitb14
✅ DINO2 backbone frozen

Starting training...
YOLOv13-dino2-working summary: 347 layers, 95,798,066 parameters

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances
        1/100      4.2G      3.45      2.31      1.82         42
        2/100      4.2G      3.12      2.18      1.76         38
        ...
Training Completed!
Best weights: runs/detect/yolov13-dino2/weights/best.pt
Final mAP50: 0.7245
Final mAP50-95: 0.6891
```

## ☁️ Cloud Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "train_dino2.py", "--data", "data.yaml", "--epochs", "100"]
```

### AWS SageMaker
```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train_dino2.py',
    source_dir='.',
    role=role,
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    framework_version='2.2.0',
    py_version='py311'
)
```

### Google Colab
```python
# Install requirements
!pip install -r requirements.txt

# Clone and run
!git clone https://github.com/Sompote/YOLO_Dino.git
%cd YOLO_Dino
!python train_dino2.py --data data.yaml --epochs 100 --batch-size 8
```

## 🛠️ Troubleshooting

### Memory Issues
```bash
# Reduce batch size
python train_dino2.py --data data.yaml --batch-size 8 --freeze-dino2

# Use smaller DINO2 variant (edit YAML config)
# Change 'dinov2_vitb14' to 'dinov2_vits14'
```

### CUDA Out of Memory
```bash
# Use mixed precision training (automatic in newer versions)
# Or use CPU training (slower)
python train_dino2.py --data data.yaml --device cpu
```

### Dataset Issues
```bash
# Check dataset paths in data.yaml
# Use absolute paths instead of relative paths
```

### Cloud-Specific Issues
```bash
# For headless environments, ensure you're using opencv-python-headless
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python-headless

# For permission issues in cloud containers
chmod +x train_dino2.py
```

## 📋 Requirements

### Hardware
- **RAM**: 8GB+ (16GB+ recommended)
- **GPU**: 8GB+ VRAM for base model (RTX 3070/V100+)
- **Storage**: 10GB+ free space

### Software
- **Python**: 3.8+
- **PyTorch**: 1.12+
- **CUDA**: 11.0+ (for GPU training)
- **Transformers**: 4.21.0+

## 📝 Key Files Structure

```
yolov13/
├── train_dino2.py                                     # Main DINO2 training script
├── ultralytics/
│   ├── nn/modules/block.py                           # DINO2Backbone implementation  
│   └── cfg/models/v13/yolov13-dino2-working.yaml     # DINO2 model configuration
├── requirements.txt                                   # Dependencies
└── README_DINO2.md                                   # This file
```

## 🏆 Citation

If you use this DINO2-enhanced implementation:

```bibtex
@misc{yolov13-dino2,
  title={YOLOv13 with DINO2 Backbone for Enhanced Object Detection},
  author={Sompote},
  year={2024},
  howpublished={\url{https://github.com/Sompote/YOLO_Dino}}
}

@article{yolov13,
  title={YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception},
  author={Lei, Mengqi and Li, Siqi and Wu, Yihong and et al.},
  journal={arXiv preprint arXiv:2506.17733},
  year={2025}
}
```

## 🙏 Acknowledgments

- **[YOLOv13](https://github.com/iMoonLab/yolov13)** - Base architecture with HyperACE and FullPAD
- **[Meta DINOv2](https://github.com/facebookresearch/dinov2)** - Vision transformer backbone
- **[Ultralytics](https://github.com/ultralytics/ultralytics)** - Training framework
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)** - Model loading utilities

## 📄 License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

---

**🎯 Ready to enhance your object detection with DINO2? Start training with `python train_dino2.py`!**