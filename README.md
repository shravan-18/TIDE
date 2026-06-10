# TIDE: Two-Stage Inverse Degradation Estimation with Guided Prior Disentanglement for Underwater Image Restoration

<div align="center">

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026%20Workshops-blue)](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/html/Venkatraman_TIDE_Two-Stage_Inverse_Degradation_Estimation_with_Guided_Prior_Disentanglement_for_CVPRW_2026_paper.html)
[![arXiv](https://img.shields.io/badge/arXiv-2512.07171-b31b1b.svg)](https://arxiv.org/abs/2512.07171)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.9+](https://img.shields.io/badge/pytorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

> **TIDE: Two-Stage Inverse Degradation Estimation with Guided Prior Disentanglement for Underwater Image Restoration**  
> *Shravan Venkatraman, Rakesh Raj Madavan, Pavan Kumar Sathya Venkatesh, Muthu Subash Kavitha*  
> Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops 2026 [NTIRE]

#### [project page](https://rakesh-123-cryp.github.io/TIDE/) | [paper](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/html/Venkatraman_TIDE_Two-Stage_Inverse_Degradation_Estimation_with_Guided_Prior_Disentanglement_for_CVPRW_2026_paper.html) | [arXiv](https://arxiv.org/abs/2512.07171) | [code](https://github.com/shravan-18/TIDE)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Ablation Studies](#ablation-studies)
<!-- - [Model Zoo](#model-zoo) -->
- [Code Structure](#code-structure)
- [Citation](#citation)
<!-- - [Acknowledgments](#acknowledgments) -->

---

## Overview

TIDE is a two-stage underwater image restoration framework that addresses the complex degradation patterns in underwater imagery through:

1. **Stage 1 (Base Model - DGMHRN)**: Degradation-Guided Multi-Hypothesis Restoration Network
   - Encoder **E** extracts shared features from degraded input
   - Four specialized decoders **F₁-F₄** generate restoration hypotheses **H₁-H₄** for different degradation types:
     - **F₁**: Color cast correction
     - **F₂**: Contrast enhancement  
     - **F₃**: Detail recovery
     - **F₄**: Noise suppression
   - Degradation maps **M₁-M₄** guide hypothesis fusion via safety gates
   - Safety-Gated Fusion produces initial restoration **I₁**

2. **Stage 2 (Refinement Network)**: Progressive restoration refinement
   - Takes initial restoration **I₁** and refines it to final output **I₂**
   - Focuses on residual degradation patterns
   - Maintains spatial consistency and prevents over-correction

### Key Features

- ✨ **Multi-hypothesis restoration** with specialized decoders for different degradation types
- 🎯 **Safety-gated fusion** mechanism for adaptive hypothesis combination
- 🔄 **Two-stage progressive refinement** for iterative quality improvement
- 🎛️ **Flexible training pipeline** supporting base training, refinement, and end-to-end fine-tuning
- 📊 **Comprehensive ablation study framework** for systematic component evaluation
- 🚀 **Mixed precision training** support for efficient GPU utilization

---

## Architecture

### Stage 1: Degradation-Guided Multi-Hypothesis Restoration Network (DGMHRN)

```
Input (I₀) → Encoder (E) → Shared Features (Z)
                              ↓
              ┌───────────────┼───────────────┬───────────┐
              ↓               ↓               ↓           ↓
          Decoder F₁      Decoder F₂      Decoder F₃   Decoder F₄
        (Color Cast)     (Contrast)       (Detail)     (Noise)
              ↓               ↓               ↓           ↓
         Hypothesis H₁   Hypothesis H₂   Hypothesis H₃  Hypothesis H₄
              ↓               ↓               ↓           ↓
            Map M₁          Map M₂          Map M₃       Map M₄
              └───────────────┴───────────────┴───────────┘
                              ↓
                    Safety-Gated Fusion
                              ↓
                    Initial Restoration (I₁)
```

### Stage 2: Refinement Network

```
Initial Restoration (I₁) → Refinement Network → Final Restoration (I₂)
                                ↓
                    Residual Degradation Modeling
```

### Model Components

| Component | Description | Parameters |
|-----------|-------------|------------|
| **Encoder (E)** | U-Net style encoder with skip connections | Configurable depth (default: 5 levels) |
| **Decoders (F₁-F₄)** | Specialized decoders for each degradation type | Symmetric to encoder |
| **Degradation Maps (M)** | Spatial attention maps indicating degradation severity | Single-channel per hypothesis |
| **Safety Gates (G)** | Learnable gating mechanism for fusion | Prevents over-correction |
| **Refinement Network** | Progressive restoration refinement | Lightweight encoder-decoder |

---

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA 10.2 or higher (for GPU training)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shravan-18/TIDE.git
   cd TIDE
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install numpy pillow scikit-image tqdm tensorboard opencv-python
   pip install pytorch-msssim lpips  # For perceptual losses
   ```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Dataset Preparation

### Directory Structure

Organize your dataset as follows:

```
data/
├── train/
│   ├── input/          # Degraded underwater images
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   └── target/         # Ground truth reference images
│       ├── image1.png
│       ├── image2.png
│       └── ...
├── val/
│   ├── input/
│   └── target/
└── test/
    ├── input/
    └── target/
```

### Supported Datasets

- **UIEB** (Underwater Image Enhancement Benchmark)
- **EUVP** (Enhancing Underwater Visual Perception)
- **RUIE** (Real-world Underwater Image Enhancement)
- Custom underwater image datasets

### Data Preprocessing

Images are automatically preprocessed during training:
- Resized to specified `--img_size` (default: 256×256)
- Random crops of size `--crop_size` during training
- Normalized to [-1, 1] range
- Data augmentation: random flips, rotations (optional)

---

## Training

### 1. Train Base Model (Stage 1 - DGMHRN)

Train the Degradation-Guided Multi-Hypothesis Restoration Network:

```bash
python main.py \
    --mode train \
    --data_dir ./data \
    --batch_size 16 \
    --num_epochs 300 \
    --lr 1e-4 \
    --base_channels 64 \
    --num_downs 5 \
    --save_dir ./checkpoints/base \
    --log_dir ./logs/base \
    --mixed_precision \
    --save_epoch_images
```

**Key Arguments:**
- `--mode train`: Base model training mode
- `--num_epochs`: Total training epochs (default: 300)
- `--lr`: Learning rate with cosine annealing (default: 1e-4)
- `--base_channels`: Base feature channels (default: 64)
- `--num_downs`: Encoder depth (default: 5)
- `--mixed_precision`: Enable automatic mixed precision for faster training

### 2. Train Refinement Network (Stage 2)

Train only the refinement stage using a pre-trained base model:

```bash
python main.py \
    --mode train_refinement \
    --data_dir ./data \
    --base_model_path ./checkpoints/base/best_model.pth \
    --batch_size 16 \
    --refinement_epochs 100 \
    --refinement_lr 5e-5 \
    --save_dir ./checkpoints/refinement \
    --log_dir ./logs/refinement \
    --mixed_precision
```

### 3. Train Full Progressive Model

Train the complete two-stage model from scratch:

```bash
python main.py \
    --mode train_progressive \
    --data_dir ./data \
    --batch_size 16 \
    --num_epochs 300 \
    --refinement_epochs 100 \
    --save_dir ./checkpoints/progressive \
    --log_dir ./logs/progressive \
    --mixed_precision
```

### 4. End-to-End Fine-tuning

Fine-tune the complete model end-to-end:

```bash
python main.py \
    --mode finetune \
    --data_dir ./data \
    --base_model_path ./checkpoints/base/best_model.pth \
    --batch_size 8 \
    --finetune_epochs 50 \
    --finetune_lr 1e-5 \
    --lambda_base 0.7 \
    --lambda_refinement 1.0 \
    --save_dir ./checkpoints/finetuned \
    --log_dir ./logs/finetuned
```

### Training Tips

1. **Learning Rate Scheduling**: Uses cosine annealing with warmup
   - Warmup epochs: `--warmup_epochs` (default: 10)
   - Cycle length: `--lr_cycle_epochs` (default: 50)

2. **Loss Weights**: Adjust loss component weights
   ```bash
   --lambda_l1 1.0 \
   --lambda_ssim 0.1 \
   --lambda_perceptual 0.1 \
   --lambda_diversity 0.05 \
   --lambda_degradation 0.1
   ```

3. **Resume Training**: Resume from checkpoint
   ```bash
   --resume ./checkpoints/base/checkpoint_epoch_100.pth \
   --start_epoch 100
   ```

4. **Multi-GPU Training**: Automatically uses DataParallel if multiple GPUs available

### Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir ./logs
```

---

## Evaluation

### Evaluate Base Model

```bash
python main.py \
    --mode eval \
    --data_dir ./data \
    --checkpoint ./checkpoints/base/best_model.pth \
    --output_dir ./results/base \
    --save_images \
    --save_hypotheses \
    --save_degradation_maps \
    --visualize
```

### Evaluate Progressive Model

```bash
python main.py \
    --mode eval \
    --data_dir ./data \
    --progressive_checkpoint ./checkpoints/progressive/best_model.pth \
    --output_dir ./results/progressive \
    --save_images \
    --save_refinement \
    --visualize
```

### Compare Base vs Progressive

```bash
python main.py \
    --mode eval \
    --data_dir ./data \
    --checkpoint ./checkpoints/base/best_model.pth \
    --progressive_checkpoint ./checkpoints/progressive/best_model.pth \
    --compare_with_base \
    --output_dir ./results/comparison \
    --save_images \
    --visualize
```

### Evaluate on Single Image

```bash
python eval_on_image.py \
    --model_path ./checkpoints/progressive/best_model.pth \
    --input_image ./test_images/underwater.png \
    --output_dir ./results/single \
    --visualize_all
```

### Evaluation Metrics

The evaluation script computes:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **LPIPS** (Learned Perceptual Image Patch Similarity)
- **MSE** (Mean Squared Error)
- **UIQM** (Underwater Image Quality Measure) - optional

Results are saved to `{output_dir}/metrics.json`

---

## Ablation Studies

Run comprehensive ablation studies to evaluate component contributions:

```bash
python main.py \
    --mode ablation \
    --data_dir ./data \
    --batch_size 16 \
    --save_dir ./checkpoints/ablation
```

### Automated Ablation Scripts

**Linux/Mac:**
```bash
bash run_ablations.sh
```

**Windows:**
```cmd
run_ablations.bat
```

### Available Ablation Configurations

The ablation study evaluates:

1. **Decoder Architecture**: Single vs. Multi-hypothesis decoders
2. **Fusion Mechanisms**: Direct average vs. Learned safety-gated fusion
3. **Loss Components**: Impact of each loss term
4. **Degradation Maps**: With and without explicit degradation estimation
5. **Refinement Stage**: Single-stage vs. Two-stage restoration
6. **Number of Hypotheses**: 2, 3, or 4 degradation-specific decoders

Results are saved in `./ablation_results/` with detailed metrics and visualizations.

---

<!-- ## Model Zoo

### Pre-trained Models

| Model | Dataset | PSNR | SSIM | Download |
|-------|---------|------|------|----------|
| TIDE-Base | UIEB | 24.5 | 0.89 | [Link](#) |
| TIDE-Progressive | UIEB | 25.8 | 0.91 | [Link](#) |
| TIDE-Base | EUVP | 23.2 | 0.87 | [Link](#) |
| TIDE-Progressive | EUVP | 24.6 | 0.89 | [Link](#) |

*Pre-trained models will be released upon paper acceptance.*

--- -->

## Code Structure

```
TIDE/
├── model.py                      # Network architectures (E, F₁-F₄, G, refinement)
├── dataset.py                    # Dataset loading and preprocessing
├── losses.py                     # Loss functions (L1, SSIM, perceptual, diversity)
├── trainer.py                    # Training loops and optimization
├── eval.py                       # Evaluation metrics and testing
├── eval_on_image.py             # Single image inference
├── main.py                       # Main entry point for all operations
├── utils.py                      # Utility functions
├── ablation_config.py           # Ablation study configurations
├── ablation_manager.py          # Ablation experiment orchestration
├── ablation_models.py           # Model variants for ablation
├── get_hardware_metrics.py      # Hardware performance profiling
├── train_all_experiments.sh     # Batch training script (Linux/Mac)
├── train_all_experiments.bat    # Batch training script (Windows)
├── run_ablations.sh             # Ablation study script (Linux/Mac)
├── run_ablations.bat            # Ablation study script (Windows)
└── README.md                     # This file
```

### Key Files

- **`model.py`**: Contains `DGMHRN` (base model) and `ProgressiveRestorationNetwork` (full two-stage model)
- **`trainer.py`**: Implements `train_model()`, `train_refinement()`, and `train_progressive_restoration()`
- **`losses.py`**: Defines all loss functions including diversity loss and degradation consistency loss
- **`ablation_manager.py`**: Orchestrates systematic ablation studies with automatic configuration

---

## Hardware Requirements

### Minimum Requirements
- GPU: NVIDIA GPU with 8GB VRAM (e.g., RTX 2070)
- RAM: 16GB system memory
- Storage: 50GB free space

### Recommended Setup
- GPU: NVIDIA GPU with 16GB+ VRAM (e.g., RTX 3090, A100)
- RAM: 32GB system memory
- Storage: 100GB SSD

### Performance Profiling

Get hardware utilization metrics:

```bash
python get_hardware_metrics.py \
    --model_path ./checkpoints/progressive/best_model.pth \
    --input_size 256 \
    --batch_size 1
```

---

## Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{venkatraman2026tide,
  title={TIDE: Two-Stage Inverse Degradation Estimation with Guided Prior Disentanglement for Underwater Image Restoration},
  author={Venkatraman, Shravan and Madavan, Rakesh Raj and Venkatesh, Pavan Kumar Sathya and Kavitha, Muthu Subash},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2609--2619},
  year={2026}
}
```
