# RoadGuardian AI
### Video-Based Road Accident Detection and Intelligent Emergency Incident Management

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Capstone](https://img.shields.io/badge/Project-B.Tech%20Capstone-informational.svg)]()

---

## 1. Project Overview

**RoadGuardian AI** is an end-to-end, software-based artificial intelligence system developed to detect vehicular road traffic accidents from recorded video feeds and coordinate simulated emergency incident response workflows.

Built as an academic capstone project for Computer Science and Engineering, RoadGuardian AI leverages a deep spatio-temporal neural network architecture combining **ResNet-18** (for spatial representation learning) and **Gated Recurrent Units (GRU)** (for temporal motion sequence modeling). Auxiliary computer vision modules utilizing **YOLOv8** and **ByteTrack** provide vehicle localization and trajectory tracking context. 

The application couples this deep learning core with a high-performance **FastAPI** backend, an **SQLite** persistence layer, an interactive **React + Vite** operator dashboard, and an automated **simulated emergency response dispatcher**.

---

## 2. Problem Statement

Road traffic accidents remain one of the leading causes of global fatalities and injuries. In conventional municipal traffic surveillance, human operators are tasked with monitoring dozens of CCTV video feeds simultaneously, inevitably leading to operator fatigue, overlooked collisions, and delayed emergency response.

The critical period immediately following an accident—the *"Golden Hour"*—dictates casualty survival rates. A software-driven, automated detection system that accurately recognizes collision dynamics from video feeds and immediately synthesizes incident telemetry can significantly reduce detection latency and improve emergency response preparedness.

---

## 3. Project Objectives

1. **Automated Spatio-Temporal Detection:** Analyze continuous video sequences to detect traffic accidents using a hybrid ResNet-18 + GRU architecture.
2. **Auxiliary Scene Context:** Employ YOLOv8 object detection and ByteTrack multi-object tracking to localize involved vehicles and observe trajectory changes.
3. **Automated Incident Logging:** Generate structured incident records containing precise timestamps, calibrated collision probabilities, and automatic evidence snapshot captures.
4. **Heuristic Severity Triage:** Categorize collision severity into standardized levels (Low, Medium, High) using observable visual dynamics and tracking metrics.
5. **Simulated Emergency Dispatch:** Model modern Computer-Aided Dispatch (CAD) workflows via a simulated dispatch engine without connecting to real emergency services.
6. **Unified Operator Dashboard:** Deliver a responsive web interface for video uploads, playback with detection probability timelines, incident review, and simulated dispatch telemetry.

---

## 4. Proposed Architecture

The system operates across six decoupled layers:

```
[ Uploaded Video File ]
          │
          ▼
┌───────────────────────────────────────┐
│ 1. Video Ingestion & Decoding (OpenCV)│
│    • Frame Extraction (224x224)       │
│    • Temporal Sliding Windows (T=16)  │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 2. Core Deep Learning Engine          │
│    • ResNet-18: Spatial Feature Map   │
│    • GRU: Temporal Dynamics Analysis  │
│    • Sigmoid: Collision Probability p │
└──────────────────┬────────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
┌──────────────┐       ┌────────────────────────┐
│ Auxiliary CV │       │ 3. Incident Engine     │
│ • YOLOv8     │       │ • Threshold Gating     │
│ • ByteTrack  │       │ • Severity Heuristics  │
│ (Context)    │       │ • Snapshot Extractor   │
└──────┬───────┘       └───────────┬────────────┘
       │                           │
       └───────────┬───────────────┘
                   ▼
┌───────────────────────────────────────┐
│ 4. Backend & Database (FastAPI/SQLite)│
│    • Video Job Management             │
│    • Incident & Evidence Snapshots    │
│    • Simulated Dispatch Engine        │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 5. Operator Web Dashboard (React+Vite)│
│    • Synchronized Video Player        │
│    • Probability Timeline Chart       │
│    • Simulated Dispatch Unit Tracker  │
└───────────────────────────────────────┘
```

For complete architectural specifications, see [docs/system_architecture.md](docs/system_architecture.md).

---

## 5. Technology Stack

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Deep Learning** | PyTorch, Torchvision | ResNet-18 feature extraction & GRU sequence model |
| **Computer Vision** | OpenCV, PIL | Frame extraction, image preprocessing, snapshot creation |
| **Auxiliary CV** | Ultralytics YOLOv8, Supervision | Vehicle detection & ByteTrack multi-object tracking |
| **Data & Metrics** | NumPy, Pandas, Scikit-learn | Data wrangling, ROC-AUC, F1-score, confusion matrix |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2 | High-throughput asynchronous REST API |
| **Database** | SQLite, SQLAlchemy 2.0 | Incident records, video metadata, simulation audit trail |
| **Frontend** | React 18, Vite | Interactive traffic operator dashboard |
| **Testing & Quality** | Pytest, Black, Flake8 | Automated unit/integration testing & code standards |

---

## 6. Dataset: Car Crash Dataset (CCD)

RoadGuardian AI utilizes the **Car Crash Dataset (CCD)**, an established academic traffic benchmark published at the **ACM Multimedia Conference 2020** ([Cogito2012/CarCrashDataset](https://github.com/Cogito2012/CarCrashDataset)).

### Key Dataset Characteristics:
- **Sample Distribution:** 1,500 real vehicular collision videos and 3,000 normal driving traffic videos (4,500 total clips).
- **Temporal Alignment:** Each video is an MP4 recording containing exactly **50 frames at 10 FPS** ($\approx 5.0\text{ seconds}$), providing continuous temporal sequence continuity.
- **Spatio-Temporal Ground Truth:** Frame-level annotations denote the exact moment of impact within each accident sequence.
- **Environmental Diversity:** Comprehensive metadata tags for lighting (Day/Night), weather (Normal, Snowy, Rainy), and vehicle involvement (Ego-car vs. Bystander).
- **Architecture Fit:** The 50-frame temporal progression is structurally optimal for sliding-window sequence slicing ($T=16$) into our **ResNet-18 + GRU** network.

For exhaustive evaluation rationale, alternative comparisons, and class imbalance strategies, refer to [docs/dataset_decision.md](docs/dataset_decision.md).

---

## 7. Project Modules & Directory Layout

```
RoadGuardian-AI/
│
├── dataset/                    # Dataset storage & manifests
│   ├── raw/                    # Original traffic video clips (gitignored)
│   ├── processed/              # Extracted frame sequences / features (gitignored)
│   └── metadata/               # Manifest files, train/val/test split indices
│
├── notebooks/                  # Interactive experimentation & research
│   ├── dataset_analysis/       # Video duration, resolution, & balance EDA
│   ├── baseline_model/         # Initial benchmark classifier notebooks
│   └── model_training/         # ResNet18 + GRU training & hyperparameter tuning
│
├── ml/                         # Core Machine Learning codebase
│   ├── preprocessing/          # Frame extraction, windowing, transformations
│   ├── baseline/               # Benchmark baseline model implementation
│   ├── resnet_gru/             # Spatial-temporal ResNet18 + GRU architecture
│   ├── inference/              # Prediction pipeline, temporal smoothing filter
│   └── evaluation/             # Metrics calculation (ROC-AUC, F1, PR curves)
│
├── computer_vision/            # Supporting Vision Modules (Decoupled)
│   ├── detection/              # YOLOv8 vehicle detection wrapper
│   └── tracking/               # ByteTrack multi-object tracking & trajectory
│
├── backend/                    # Asynchronous REST Backend
│   └── app/
│       ├── api/                # API routers (videos, incidents, simulation)
│       ├── services/           # Business logic & pipeline execution
│       ├── database/           # SQLite session & database configuration
│       ├── models/             # SQLAlchemy ORM models
│       └── schemas/            # Pydantic v2 data validation models
│
├── frontend/                   # React + Vite Single Page Application
│
├── docs/                       # Technical architecture & project documentation
│   ├── project_scope.md        # Explicit inclusions, exclusions & simulation scope
│   ├── development_plan.md     # 9-phase development strategy & milestones
│   ├── system_architecture.md  # Deep technical architecture & equations
│   ├── dataset_selection_criteria.md # 14-dimension evaluation rubric
│   └── dataset_decision.md     # Formal Car Crash Dataset (CCD) decision
│
├── tests/                      # Automated test suite
│
├── scripts/                    # Utility scripts (directory setup, data prep)
│
├── README.md                   # Project landing page & documentation
├── .gitignore                  # Comprehensive ignore rules
└── requirements.txt            # Dependency manifest grouped by purpose
```

---

## 8. Phased Development Roadmap

- [x] **Phase 1: Foundation, Architecture & Repository Setup** (Completed)
- [ ] **Phase 2: Dataset Acquisition, Audit & Verification** (Status: Dataset Selected - Acquisition In Progress)
- [ ] **Phase 3: Video Preprocessing & Baseline Model Development**
- [ ] **Phase 4: ResNet18 + GRU Spatio-Temporal Model**
- [ ] **Phase 5: Computer Vision Auxiliary Modules (YOLO + ByteTrack)**
- [ ] **Phase 6: FastAPI Backend & SQLite Incident Management**
- [ ] **Phase 7: React + Vite Operator Dashboard**
- [ ] **Phase 8: End-to-End System Integration & Testing**
- [ ] **Phase 9: Evaluation, Benchmarking & Academic Documentation**

Refer to [docs/development_plan.md](docs/development_plan.md) for full phase descriptions and validation gates.

---

## 9. Installation & Setup (Placeholder)

### 9.1 Prerequisites
- Python 3.10 or higher
- Node.js 18+ and npm (for frontend dashboard)
- Git

### 9.2 Environment Configuration (Planned)

```bash
# 1. Clone the repository
git clone https://github.com/<username>/RoadGuardian-AI.git
cd RoadGuardian-AI

# 2. Create and activate a Python virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify environment setup
pytest tests/
```

---

## 10. Academic & Ethical Scope Disclaimers

> [!IMPORTANT]
> - **Software-Only Prototype:** This system is built strictly as a software-only demonstration for academic evaluation and portfolio presentation.
> - **Simulated Emergency Response:** No connection is established with real-world 911, 112, or 108 emergency telephony services, police networks, or medical facilities. All dispatch actions and ETAs are **simulated**.
> - **Heuristic Severity Scoring:** The severity estimation is a prototype heuristic derived from visual collision dynamics and tracked vehicle counts. It does **not** constitute medical injury diagnosis or physical trauma assessment.
> - **Offline Video Ingestion:** The architecture is designed for uploaded video clips and makes no claims of unconstrained real-time municipal streaming surveillance.

For detailed scope specifications, consult [docs/project_scope.md](docs/project_scope.md).

---

## 11. Future Work

- **Edge Deployment Optimization:** Quantization and TensorRT / ONNX Runtime export for edge devices (e.g., NVIDIA Jetson).
- **Multimodal Sensor Fusion:** Incorporating acoustic audio signals (e.g., screeching tires, metal impact audio) to enhance detection reliability in low-visibility or occluded environments.
- **Explainable AI (XAI):** Integrating Grad-CAM heatmaps over ResNet-18 spatial feature maps to visually highlight regions triggering accident classifications.
- **Municipal CAD Integration:** Developing standardized Webhook adapters conforming to APCO / NENA standards for authorized smart city platforms.

---

## 12. Author & Academic Credits

- **Project:** B.Tech Final-Year Capstone Project
- **Discipline:** Computer Science & Engineering
- **Institution:** Department of Computer Science & Engineering
