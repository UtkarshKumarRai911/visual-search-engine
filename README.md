# 🔍 Visual Product Search Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/FAISS-ANN_Search-0467DF?style=for-the-badge&logo=meta&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

<p align="center">
  <b>Upload any product image → get visually similar results in under 150ms</b><br/>
  Fine-tuned ViT-B/16 · ArcFace Metric Learning · FAISS HNSW Index · 120,053 product catalog
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Recall@10-87.17%25-22C55E?style=flat-square"/>
  <img src="https://img.shields.io/badge/FAISS_Latency-~0.2ms-2563EB?style=flat-square"/>
  <img src="https://img.shields.io/badge/Speedup-11×_vs_brute--force-8B5CF6?style=flat-square"/>
  <img src="https://img.shields.io/badge/End--to--End-~90--150ms_warm-F59E0B?style=flat-square"/>
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Results](#results)
- [System Architecture](#system-architecture)
- [Project Phases](#project-phases)
  - [Phase 1 — EDA](#phase-1--exploratory-data-analysis)
  - [Phase 2 — ResNet50 Baseline](#phase-2--resnet50-baseline)
  - [Phase 3 — ViT + ArcFace Fine-tuning](#phase-3--vit--arcface-fine-tuning)
  - [Phase 4 — FAISS Integration](#phase-4--faiss-integration)
  - [Phase 5 — Deployment](#phase-5--deployment)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Technical Details](#technical-details)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project builds a **visual similarity search engine** for e-commerce products from the ground up — starting with dataset exploration, establishing a ResNet50 baseline, fine-tuning a Vision Transformer with ArcFace metric learning, scaling retrieval to 120K products with FAISS, and shipping it as a web application.

The system answers: *"Given a product photo, what are the most visually similar items in the catalog?"* — in real time.

**Colloquium Guided Project** under the supervision of **Dr. Neelesh Yadav**, Department of EEE, ABV-IIITM Gwalior.

---

## Results

### Recall Comparison — ViT ArcFace vs ResNet50 Baseline

| Model | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|
| ResNet50 (frozen, cosine) | 53.79% | 65.48% | 69.88% |
| **ViT-B/16 + ArcFace** | **73.23%** | **83.96%** | **87.17%** |
| Delta | **+19.44 pp** | **+18.48 pp** | **+17.29 pp** |

### FAISS Index Benchmark — 120,053 vectors, 500 queries

| Index Type | Recall@10 | Search Latency | Speedup vs NumPy |
|---|---|---|---|
| NumPy Brute-Force | — | 2.451 ms | 1× (baseline) |
| FAISS IndexFlatIP (exact) | 84.50% | 0.373 ms | 6.6× |
| FAISS IndexIVFFlat (nprobe=10) | 81.94% | 0.287 ms | 8.5× |
| **FAISS HNSW (efSearch=64)** | **~83.8%** | **~0.22 ms** | **~11×** |

### End-to-End Latency (warm, GPU)

| Step | Latency |
|---|---|
| Image preprocessing | ~2 ms |
| ViT-B/16 forward pass (RTX 3050) | ~70–130 ms |
| FAISS HNSW search (120K vectors) | ~0.2 ms |
| Result rendering | ~10 ms |
| **Total** | **~90–150 ms** |

---

## System Architecture

```
User uploads product image
           │
           ▼
┌─────────────────────────────┐
│   Streamlit Web Application  │
│   (drag-and-drop UI)         │
└────────────┬────────────────┘
             │  PIL Image
             ▼
┌─────────────────────────────┐
│   Image Preprocessing        │
│   Resize(256) → CenterCrop   │
│   (224) → Normalize          │
└────────────┬────────────────┘
             │  Tensor [1, 3, 224, 224]
             ▼
┌─────────────────────────────┐
│   ViT-B/16 Backbone          │
│   (6 blocks fine-tuned)      │
│   CLS token → 768-d          │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Projection Head            │
│   Linear(768→768) → GELU     │
│   → Linear(768→512) → L2-norm│
└────────────┬────────────────┘
             │  512-d unit embedding
             ▼
┌─────────────────────────────┐
│   FAISS HNSW Index           │
│   120,053 product vectors    │
│   index.search(query, K)     │
└────────────┬────────────────┘
             │  Top-K {rank, path, label, similarity}
             ▼
     Result grid displayed
```

---

## Project Phases

### Phase 1 — Exploratory Data Analysis

**Notebook:** `notebooks/01_EDA.ipynb`

The Stanford Online Products (SOP) dataset was explored to understand its structure, class distribution, and image characteristics before any modelling decisions were made.

**Dataset statistics confirmed from EDA:**

| Property | Value |
|---|---|
| Total images | 120,053 |
| Total product classes | 22,634 |
| Product super-categories | 12 |
| Train images | 59,551 |
| Train classes | 11,318 |
| Test images | 60,502 |
| Test classes | 11,316 |
| Avg images per class | ~5.3 |
| Image format | JPEG |
| Avg image resolution | ~400×400px |

**12 super-categories and characteristics:**

| Category | ~Images | ~Classes | Characteristics |
|---|---|---|---|
| Bicycle | 9,600 | 294 | High structural variation, angle diversity |
| Cabinet | 9,800 | 300 | Rectangular forms, colour/texture variety |
| Chair | 9,900 | 304 | High shape diversity, indoor settings |
| Coffee Maker | 8,500 | 264 | Small appliances, brand logos visible |
| Fan | 8,000 | 250 | Circular shapes, metallic textures |
| Kettle | 8,100 | 255 | Reflective surfaces, handle features |
| Lamp | 9,200 | 286 | Varied shapes, background clutter |
| Mug | 9,600 | 294 | Print/texture heavy, small object |
| Sofa | 8,600 | 267 | Large objects, fabric texture variety |
| Stapler | 7,600 | 242 | Small, metallic, minimal background |
| Table | 9,200 | 284 | Flat surfaces, leg variation |
| Toaster | 8,700 | 270 | Metallic, small appliance |

**Critical EDA finding — Zero-Shot Retrieval benchmark:**

Train and test classes are **completely disjoint**. Train classes are IDs 1–11,318; test classes are IDs 11,319–22,634. This means:
- The model **cannot memorize** specific products during training
- The model **must learn generalizable visual similarity**
- This simulates real-world deployment where new products are added daily
- Standard classification accuracy is **not a valid metric** — Recall@K is the correct evaluation protocol

**Preprocessing pipeline identified from EDA:**

| Step | Operation | Reason |
|---|---|---|
| 1 | Resize to 256×256 | Standardise scale |
| 2 | RandomCrop(224) + RandomHorizontalFlip + ColorJitter | Augmentation (train only) |
| 3 | CenterCrop(224) | Inference (test only) |
| 4 | Normalize (ImageNet mean/std) | Match pretrained backbone distribution |

**Why metric learning (not classification):** With only 5–6 images per product class, softmax classification will overfit and fail. The model must learn a similarity space where same-product images cluster together — this is why ArcFace loss was chosen in Phase 3.

---

### Phase 2 — ResNet50 Baseline

**Notebook:** `notebooks/02_Baseline_ResNet50.ipynb`

A frozen ImageNet-pretrained ResNet50 (25.6M parameters) was used as the feature extractor. The final FC layer was removed, producing 2048-d embeddings that were L2-normalised and compared via cosine similarity — no fine-tuning of any weights.

**Pipeline:**
```
RGB Image (224×224)
    → ResNet50 backbone (49 conv layers + skip connections)
    → Global Average Pooling → (2048,)
    → L2 Normalize
    → Brute-force cosine similarity against all 60,502 test embeddings
```

**Embedding extraction:**

| Split | Shape | Approx. Size | Time |
|---|---|---|---|
| Train | (59,551 × 2048) | ~460 MB | ~16 min |
| Test | (60,502 × 2048) | ~467 MB | ~17 min |

**Results (within test set, self-match excluded):**

| Metric | Score |
|---|---|
| Recall@1 | **53.79%** |
| Recall@5 | **65.48%** |
| Recall@10 | **69.88%** |

**Key observations from notebook output:**
- R@1 of 53.79% — impressive for a zero-shot frozen model; ImageNet features transfer well to product images
- The 16 pp gap between R@1 and R@10 signals poor ranking quality — correct product is retrieved but not ranked first
- Same super-category retrieval (staplers retrieve staplers) works well; **fine-grained instance-level matching fails** — this is the core problem ArcFace training in Phase 3 solves
- Products with strong shape features (bicycle, chair) outperform texture-heavy ones (mug fabric, sofa)
- Metallic/reflective surfaces (kettle, fan, stapler) show weaker retrieval due to lighting-induced embedding instability

**Why this baseline justifies ViT + ArcFace:**

| Limitation | Phase 3 Solution |
|---|---|
| No task-specific training | ArcFace fine-tuning on product similarity |
| CNN sees local patches only | ViT attends globally across all 196 patches |
| Fixed 2048-d embedding space | Learnable 512-d projection head optimised for cosine similarity |
| Brute-force O(N) retrieval | FAISS ANN index in Phase 4 |

**Phase 3 targets set by this baseline:**

| Metric | Baseline | Target | Required gain |
|---|---|---|---|
| Recall@1 | 53.79% | > 65% | +11 pp |
| Recall@5 | 65.48% | > 78% | +13 pp |
| Recall@10 | 69.88% | > 84% | +14 pp |

---

### Phase 3 — ViT + ArcFace Fine-tuning

**Notebook:** `notebooks/03_ViT_Finetuning_v2.ipynb`

The core modelling phase. A pretrained `google/vit-base-patch16-224-in21k` was fine-tuned end-to-end using ArcFace metric learning loss.

**Model architecture:**
```
ViT-B/16 backbone     86.4M parameters total
  └─ Frozen: layers 0–5 (first 6 transformer blocks)
  └─ Trainable: layers 6–11 + LayerNorm (42.5M, 49.2%)
Projection head       768 → 768 (GELU) → 512 → L2-normalize
```

**Why ArcFace over Triplet Loss:**
- ArcFace adds a fixed angular margin `m=0.5` (28.6°) in the cosine space, enforcing tighter intra-class clustering
- Classification-based formulation (softmax over 11,318 training classes) gives stable, dense gradients every batch vs sparse triplet mining
- Scale `s=64` ensures gradients don't vanish on the unit hypersphere
- Empirically outperforms triplet loss on SOP in literature by 3–5 pp Recall@10

**Training configuration:**

| Hyperparameter | Value |
|---|---|
| Backbone LR | 2e-5 |
| Projection head LR | 1e-4 |
| ArcFace head LR | 1e-4 |
| Batch size | 64 |
| Epochs | 15 |
| Scheduler | Cosine decay with 2-epoch linear warmup |
| Optimizer | AdamW (weight decay 1e-4) |
| Mixed precision | fp16 (torch.amp) |
| GPU | NVIDIA RTX 3050 Laptop (4.3 GB VRAM) |

**Training curve (Recall@10 on 4K val subset):**

| Epoch | Train Loss | Recall@10 |
|---|---|---|
| 1 | 41.57 | 48.1% |
| 2 | 37.73 | 64.8% |
| 4 | 29.11 | 79.2% |
| 6 | 21.95 | 80.6% |
| 10 | 14.25 | **84.1%** ← best checkpoint |
| 15 | 11.82 | 81.7% |

**Final results on full 60,502-image test set:**

| Metric | Score | vs Baseline |
|---|---|---|
| Recall@1 | **73.23%** | +19.44 pp |
| Recall@5 | **83.96%** | +18.48 pp |
| Recall@10 | **87.17%** | +17.29 pp |

---

### Phase 4 — FAISS Integration

**Notebook:** `notebooks/04_FAISS_Integration.ipynb`

The Phase 3 brute-force cosine similarity evaluation (which built a 60K×60K similarity matrix) was replaced with a production-grade FAISS ANN index over the **full 120,053-image catalog**.

**Why the full catalog:** In a real search engine, every listed product is searchable — not just a test partition. Using the full dataset makes the retrieval problem harder and more realistic.

**Why inner product metric:** All embeddings are L2-normalised. For unit vectors, cosine similarity = inner product. `METRIC_INNER_PRODUCT` avoids an unnecessary square root and aligns with ArcFace's angular training objective.

**Index types built:**

| Index | Description | Build time |
|---|---|---|
| `IndexFlatIP` | Exact brute-force, ground truth | Instant |
| `IndexIVFFlat` | 200 Voronoi cells, nprobe=10 | ~2 min |
| `IndexHNSWFlat` | M=32, efConstruction=200, efSearch=64 | ~10 min |

**nprobe sweep (IVF index):**

| nprobe | Recall@10 | Latency |
|---|---|---|
| 1 | ~75.2% | 0.09 ms |
| 5 | ~79.8% | 0.18 ms |
| 10 | 81.94% | 0.287 ms |
| 20 | ~83.1% | 0.41 ms |
| 50 | ~83.9% | 0.89 ms |

**HNSW was selected as the default** for deployment — best single-query latency on CPU, no training required, and better accuracy/speed trade-off than IVF for interactive workloads.

**Artifact sizes:**

| File | Size |
|---|---|
| `catalog_embeddings.npy` (120,053 × 512 float32) | ~234 MB |
| `sop_flat.index` | ~240 MB |
| `sop_ivf.index` | ~240 MB |
| `sop_hnsw.index` | ~340 MB |

---

### Phase 5 — Deployment

**Notebook:** `notebooks/05_Deployment.ipynb` · **App:** `app.py` · **HF Space:** `hf_space/`

The complete pipeline was wrapped in a Streamlit web application and packaged for HuggingFace Spaces deployment.

**Application features:**
- Drag-and-drop image upload (JPG, PNG, WEBP)
- Random catalog sample button for live demos
- Top-K result grid (1–10, configurable)
- Per-result class-match indicator (green ✓ / red ✗)
- Similarity score display per result
- Live index type switching (HNSW / IVF / Flat) without model reload
- Search latency display in milliseconds
- Catalog statistics sidebar

**HuggingFace Spaces adaptation:**
- Absolute Windows paths replaced with `os.path.dirname(__file__)` relative paths
- `DEVICE` hardcoded to `cpu` (HF free tier)
- `torch.amp.autocast` disabled on CPU
- Graceful checkpoint-missing error with actionable upload instructions

---

## Repository Structure

```
visual-search-engine/
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Dataset exploration & statistics
│   ├── 02_Baseline_ResNet50.ipynb    # ResNet50 frozen feature baseline
│   ├── 03_ViT_Finetuning_v2.ipynb   # ViT-B/16 + ArcFace training
│   ├── 04_FAISS_Integration.ipynb   # ANN index build & benchmark
│   └── 05_Deployment.ipynb          # Pipeline wiring & smoke test
│
├── hf_space/
│   ├── app.py                        # HuggingFace Spaces entry point
│   ├── requirements.txt              # Pinned dependencies
│   └── README.md                     # HF Space card
│
├── app.py                            # Local Streamlit application
├── requirements.txt                  # Project dependencies
└── README.md                         # This file
```

> **Note on large files:** Model checkpoint (`vit_sop_arcface_best.pt`, ~340 MB) and FAISS indexes are not committed to this repo due to size. See [Quick Start](#quick-start) for instructions to reproduce them or download pretrained artifacts.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/UtkarshKumarRai911/visual-search-engine.git
cd visual-search-engine
pip install -r requirements.txt
```

### 2. Download the Stanford Online Products dataset

```bash
# Dataset available from: https://cvgl.stanford.edu/projects/lifted_struct/
# Place extracted folder at: data/Stanford_Online_Products/
```

### 3. Run the notebooks in order

```
01_EDA.ipynb               ← understand the dataset
02_Baseline_ResNet50.ipynb ← establish baseline
03_ViT_Finetuning_v2.ipynb ← train ViT + ArcFace (~3 hrs on RTX 3050)
04_FAISS_Integration.ipynb ← build FAISS indexes
05_Deployment.ipynb        ← verify end-to-end pipeline
```

### 4. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Dataset

**Stanford Online Products (SOP)**

| Property | Value |
|---|---|
| Total images | 120,053 |
| Total classes | 22,634 |
| Train images | 59,551 |
| Train classes | 11,318 |
| Test images | 60,502 |
| Test classes | 11,316 |
| Train/test overlap | None (disjoint classes) |
| Avg images/class | ~5.3 |

12 super-categories: bicycle · cabinet · chair · coffee maker · fan · kettle · lamp · mug · sofa · stapler · table · toaster

Source: [Song et al., CVPR 2016](https://cvgl.stanford.edu/projects/lifted_struct/)

---

## Technical Details

### Why Vision Transformer over CNN?

ViT-B/16 captures **global context** via self-attention across all 196 patches simultaneously. For product retrieval, this means the model attends to the overall shape and composition of an object — not just local texture features that ResNet's convolutional filters specialize in. This global attention is particularly effective for distinguishing fine-grained product variants (e.g., different chair designs) where the overall silhouette matters more than local patterns.

### Why ArcFace over Triplet Loss?

Triplet loss requires careful mining of hard negative pairs to avoid collapsed gradients. ArcFace formulates metric learning as a classification problem with an angular margin constraint — every training sample contributes a gradient every batch, training is stable, and the angular margin directly optimizes the cosine similarity space used at inference. On SOP, ArcFace consistently outperforms triplet loss by 3–5 Recall@10 percentage points in the literature.

### Why HNSW over IVF for deployment?

HNSW (Hierarchical Navigable Small World) builds a multi-layer proximity graph over the embedding space. For **single-query interactive latency** on CPU — the dominant access pattern in a web application — HNSW consistently outperforms IVF because it doesn't require choosing `nprobe` (a tuning parameter that trades accuracy for speed) and has better cache locality on single queries. IVF has advantages for batch throughput workloads.

---

## Requirements

```
torch==2.2.2
torchvision==0.17.2
transformers==4.40.2
faiss-cpu==1.8.0
streamlit==1.35.0
Pillow==10.3.0
numpy==1.26.4
```

---

## Acknowledgements

- **Dr. Neelesh Yadav** — project guide, ABV-IIITM Gwalior
- [Stanford Online Products dataset](https://cvgl.stanford.edu/projects/lifted_struct/) — Song et al., CVPR 2016
- [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) — Dosovitskiy et al.
- [ArcFace](https://arxiv.org/abs/1801.07698) — Deng et al., CVPR 2019
- [FAISS](https://github.com/facebookresearch/faiss) — Johnson et al., Facebook AI Research

---

<p align="center">
  Built by <a href="https://github.com/UtkarshKumarRai911">Utkarsh Kumar Rai</a> · ABV-IIITM Gwalior · 2026
</p>
