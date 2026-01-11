# SegFormer Urban Scene Segmentation

[![Paper](https://img.shields.io/badge/SPIE-ICMVA%202025-blue)](https://doi.org/10.1117/12.3078755)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/nvidia/segformer-b3-finetuned-cityscapes-1024-1024)

> **Urban Scene Segmentation and Cross-Dataset Transfer Learning using SegFormer**  
> Published at SPIE International Conference on Machine Vision Applications (ICMVA) 2025

Transformer-based semantic segmentation for autonomous driving with cross-dataset transfer learning. We evaluate SegFormer architectural variants (B3, B4, B5) on CamVid and demonstrate effective knowledge transfer from CamVid to KITTI, achieving **61.1% faster convergence** and up to **30.75% class-specific improvement**.

<p align="center">
  <img src="results/methodology_draft.png" alt="Methodology" width="700"/>
</p>

---

## 📊 Key Results

### Architecture Scaling on CamVid

| Model | Params | mIoU | Inference Time | GPU Memory |
|-------|--------|------|----------------|------------|
| SegFormer-B3 | 47.1M | 77.9% | 25.3ms | 4.2 GB |
| SegFormer-B4 | 64.1M | 78.5% | 28.5ms | 5.1 GB |
| SegFormer-B5 | 84.7M | **82.4%** | 32.8ms | 6.3 GB |

### Transfer Learning: CamVid → KITTI

| Metric | From Scratch | Transfer Learning | Improvement |
|--------|--------------|-------------------|-------------|
| Mean IoU | 52.08% | 53.42% | **+2.57%** |
| Epochs to Converge | 18 | 7 | **-61.1%** |

### Class-Specific Improvements

| Class | From Scratch | Transfer | Gain |
|-------|--------------|----------|------|
| Wall | 49.53% | 64.76% | **+30.75%** |
| Sidewalk | 48.00% | 52.41% | +9.18% |
| Bus | 54.21% | 58.24% | +7.44% |

<p align="center">
  <img src="results/class_iou_comparison.png" alt="Per-class IoU Comparison" width="600"/>
</p>

---

## 🏗️ Method

### Architecture
- **Encoder**: Hierarchical transformer with efficient self-attention (reduction ratio for O(N²/R²) complexity)
- **Decoder**: Lightweight MLP for multi-scale feature fusion
- **No positional encoding**: Mix-FFN with 3×3 depthwise conv provides implicit spatial information

### Loss Function
```
L_total = L_CE + 0.4 × L_IoU + 0.8 × L_boundary
```
- **L_CE**: Class-weighted cross-entropy (3× weight for rare classes)
- **L_IoU**: Soft IoU loss for direct metric optimization
- **L_boundary**: Multi-scale boundary-aware loss

### Transfer Learning Strategy
1. Train SegFormer on CamVid (701 images, 32 classes)
2. Transfer encoder weights to KITTI
3. Reinitialize decoder for 19-class taxonomy
4. Fine-tune with reduced learning rate

---

<!-- 
## 📁 Repository Structure

```
segformer-urban-segmentation/
├── notebooks/
│   ├── SegFormer_B3_CamVid.py      # Base training (source domain)
│   ├── SegFormer_B4_CamVid.py      # B4 variant experiments
│   ├── SegFormer_B5_CamVid.py      # B5 variant experiments
│   ├── SegFormer_B3_KITTI.py       # Transfer learning to KITTI
│   └── SegFormer_B3_IDD.py         # Transfer learning to IDD
├── papers/
│   └── SPIE_ICMVA_2025.pdf
├── results/
├── requirements.txt
└── README.md
```
-->

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Tanmay-Hatkar/segformer-urban-scene-segmentation.git
cd segformer-urban-scene-segmentation
pip install -r requirements.txt
```

### Training on CamVid

```python
# In notebooks/SegFormer_B3_CamVid.py
# Update paths:
BASE_DIR = "/path/to/your/data"
ROOT_DIR = os.path.join(BASE_DIR, "CamVid")

# Run training
python SegFormer_B3_CamVid.py
```

### Transfer Learning to KITTI

```python
# In notebooks/SegFormer_B3_KITTI.py
# Update checkpoint path:
CAMVID_CHECKPOINT = "path/to/camvid/best_model.pth"

# Run experiments
run_transfer()   # Transfer learning from CamVid
run_baseline()   # Train from scratch (for comparison)
run_comparison() # Generate comparison metrics
```

---

## ⚙️ Configuration

Key hyperparameters in each script:

```python
# Model
MODEL_TYPE = "b3"              # b3, b4, or b5

# Training
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
BATCH_SIZE = 1
ACCUMULATION_STEPS = 16        # Effective batch size = 16
NUM_EPOCHS = 30
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01

# Transfer Learning
FREEZE_ENCODER = False         # Set True to freeze encoder weights
```

---

## 📚 Datasets

| Dataset | Location | Images | Classes | Role |
|---------|----------|--------|---------|------|
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) | UK | 701 | 32 | Source domain |
| [KITTI](http://www.cvlibs.net/datasets/kitti/) | Germany | ~200 | 19 | Target domain |

### Class Mapping Strategy

- **Direct mappings**: road→road, building→building, sky→sky
- **Semantic mappings**: tree→vegetation, pedestrian→person
- **Novel classes**: KITTI-specific categories (terrain, rider) initialized randomly

---

## 📈 Comparison with State-of-the-Art

| Model | Architecture | mIoU (%) | FPS | Params |
|-------|--------------|----------|-----|--------|
| DAFormer | Transformer | 61.2* | 28.3 | - |
| Mask2Former | Transformer | 56.7 | 24.8 | 158M |
| PIDNet | CNN-Hybrid | 80.1 | 153.7 | - |
| **SegFormer-B5 (Ours)** | Transformer | **82.4** | 30.5 | 84.7M |
| **SegFormer-B3 (Ours)** | Transformer | 53.42 | 39.5 | 47.1M |

*DAFormer uses synthetic pre-training data (Cityscapes→KITTI domain adaptation)

**Our advantage**: 61.1% faster convergence, 30.75% improvement on challenging classes, no synthetic data required.

---

## 🔬 Qualitative Results

<p align="center">
  <img src="results/qualitative_comparison.png" alt="Qualitative Results" width="800"/>
</p>

*Top Row: Left to right - Original image followed by baseline prediction
Bottom Row: Left to right - Transfer learning prediction followed by improvement visualization (brighter = better)*

---

## 📖 Citation

```bibtex
@inproceedings{hatkar2025segformer,
  title={Urban Scene Segmentation and Cross-Dataset Transfer Learning using SegFormer},
  author={Hatkar, Tanmay Sunil and Ahmed, Saad B.},
  booktitle={Proceedings of SPIE International Conference on Machine Vision Applications (ICMVA)},
  year={2025},
  doi={10.1117/12.3078755}
}
```

---

## 👤 Author

**Tanmay Hatkar**  
M.Sc. Computer Science, Lakehead University  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/tanmay-hatkar-82180a190/)
[![Email](https://img.shields.io/badge/Email-Contact-red)](mailto:thatkar@lakeheadu.ca)

## 🙏 Acknowledgments

- **Computational Resources**: Digital Research Alliance of Canada
- **Pre-trained Models**: NVIDIA SegFormer via HuggingFace
- **Supervisor**: Dr. Saad B. Ahmed, Lakehead University

---

## 📄 License

This project is for academic and research purposes. Please cite our paper if you use this code.
