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

# Custom configuration
python train_dino2.py \
    --data custom_data.yaml \
    --epochs 200 \
    --batch-size 16 \
    --imgsz 640 \
    --freeze-dino2 \
    --name my_dino2_experiment
```

### Standard YOLOv13 Training

For standard YOLOv13 training without DINO2:

```python
from ultralytics import YOLO

model = YOLO('yolov13n.yaml')  # or yolov13s.yaml, yolov13l.yaml, yolov13x.yaml

results = model.train(
    data='coco.yaml',
    epochs=600,
    batch=256,
    imgsz=640,
    device="0,1,2,3"
)
```

## 📊 Training Arguments

### DINO2 Enhanced Training

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--data` | Dataset YAML file path | Required | Path to your data.yaml |
| `--epochs` | Number of training epochs | 100 | Integer |
| `--batch-size` | Training batch size | 16 | 4, 8, 16, 32, 64 |
| `--imgsz` | Input image size | 640 | 320, 480, 640, 1280 |
| `--freeze-dino2` | Freeze DINO2 weights | False | --freeze-dino2 |
| `--name` | Experiment name | yolov13-dino2 | String |

## 🎯 DINO2 Model Variants

The implementation supports multiple DINO2 model sizes:

| Model | Parameters | Hidden Size | Patch Size | Performance | Speed |
|-------|------------|-------------|------------|-------------|-------|
| `dinov2_vits14` | ~21M | 384 | 14×14 | Good | Fastest |
| `dinov2_vitb14` | ~86M | 768 | 14×14 | Better | Fast |
| `dinov2_vitl14` | ~300M | 1024 | 14×14 | Best | Slower |
| `dinov2_vitg14` | ~1.1B | 1536 | 14×14 | Excellent | Slowest |

*Default: `dinov2_vitb14` (recommended balance of performance and speed)*

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

| Model | FLOPs (G) | Parameters(M) | mAP50:95 | mAP50 | Latency (ms) |
|-------|-----------|---------------|----------|-------|--------------|
| YOLOv13-N | 6.4 | 2.5 | 41.6 | 57.8 | 1.97 |
| YOLOv13-S | 20.8 | 9.0 | 48.0 | 65.2 | 2.98 |
| YOLOv13-L | 88.4 | 27.6 | 53.4 | 70.9 | 8.63 |
| YOLOv13-X | 199.2 | 64.0 | 54.8 | 72.0 | 14.67 |

### DINO2 Enhanced Benefits

- **Enhanced Feature Extraction**: Vision transformer captures global context
- **Better Small Object Detection**: Improved feature representation  
- **Transfer Learning**: Leverages pretrained DINO2 knowledge
- **Robust Performance**: Superior handling of complex scenes

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