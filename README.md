# ColdFusionX: Explainable Multi-Modal CTR Prediction

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Abstract:** An advanced, explainable multi-modal Click-Through Rate (CTR) prediction framework designed to solve the zero-shot / cold-start recommendation problem. ColdFusionX uses Modality-Attentive Meta-Embeddings and Temporal Drift Adaptations to maintain high predictive accuracy (AUC) across dynamic user-item interactions, operating efficiently under strict industrial latency constraints ($O(1)$ sub-10ms inference).

---

## Table of Contents
1. [System Architecture](#-system-architecture)
2. [Key Features](#-key-features)
3. [Reproducibility Guide (Quick Start)](#-reproducibility-guide-quick-start)
4. [Project Structure](#-project-structure)
5. [Testing & Security](#-testing--security)
6. [Academic Citation](#-academic-citation)

---

## System Architecture

ColdFusionX bypasses the limitations of traditional collaborative filtering by actively routing between collaborative IDs and multi-modal proxy embeddings (Vision and Text). 

### Core Modules:
* **Feature Store Caching:** Pre-computed embedding extraction to satisfy sub-100ms real-time SLAs.
* **Semantic Latent Space Alignment:** Forcing meta-generators to map multi-modal features into established collaborative spaces.
* **Temporal FiLM:** Modulating visual/textual features to capture cyclical micro-drift (e.g., weekend vs. weekday traffic shifts).

---

## Key Features & Dashboards

The repository includes a highly optimized Streamlit frontend (Lighthouse Performance: 91, Accessibility: 100) for interactive evaluation.

1. **Campaign Optimizer (`campaign_optimiser.py`):** Simulate ad performance, inject temporal context shifts, and forecast ROI using the $O(1)$ inference engine.
2. **Intrinsic Model X-Ray (`model_xray.py`):** Inspect internal tensor mechanics, modality routing weights (Attention), and mathematical nearest-neighbors in the latent space.
3. **Global Benchmarks (`performance.py`):** Reproduce the ablation studies, AUC drop penalties, and temporal drift line charts.

---

## Reproducibility Guide (Quick Start)

Follow these steps to replicate the local evaluation environment and run the interactive dashboards.

### 1. Prerequisites
Ensure you have Python 3.12+ installed. It is highly recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Clone the repository
git clone [https://github.com/your-username/ColdFusionX.git](https://github.com/your-username/ColdFusionX.git)
cd ColdFusionX

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Artifact Download (Pre-computed Weights)
To evaluate the zero-shot capabilities without retraining the entire multi-modal feature extractors, you must download the `release_demo` artifacts (~711MB).

1. Download the `artifacts.zip` file from [HuggingFace link (to be updated)].
2. Extract the contents directly into the `ColdFusionX/artifacts/` directory.
3. Ensure the path `artifacts/release_demo/` contains the `.pt` weights and `metrics/` CSV files.

### 4. Launch the Evaluation Suite
```bash
streamlit run ui/campaign_optimiser.py
```
The application will launch locally at `http://localhost:8501`.

---

## Project Structure

```text
ColdFusionX/
├── artifacts/                  # Model weights, pre-computed embeddings, metric logs
├── coldfusionx/                # Core PyTorch backend module
│   ├── data/                   # Feature store and data loaders
│   ├── models/                 # Neural network architectures (Meta-embeddings, FiLM)
│   └── inference/              # Inference manager (sub-10ms execution logic)
├── ui/                         # Streamlit frontend application
│   ├── utils/                  # CSS themes and layout configurations
│   ├── campaign_optimiser.py   # Main simulation dashboard
│   ├── model_xray.py           # Latent space and attention visualizer
│   └── pages/                  # Additional benchmark dashboards
├── tests/                      # Unit and integration tests
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Testing & Security

This framework has been audited for enterprise-grade deployment and Non-Functional Requirements (NFRs):

* **Performance Validation:** Chrome V8 engine profiling confirms a total event latency of 6ms (5ms scripting) during live multi-modal inference.
* **SAST (Static Application Security Testing):** Audited via `Bandit`, confirming zero High/Medium/Low vulnerabilities. Insecure deserialization risks (CWE-502) mitigated via strict `weights_only=True` PyTorch tensor loading.
* **SCA (Software Composition Analysis):** Automated supply chain vulnerability tracking integrated via GitHub Dependabot.

---

## Academic Citation

If you use ColdFusionX in your research, please cite our forthcoming papers:

**WCCCT 2026 Submission:**
```bibtex
@inproceedings{coldfusionx_wccct2026,
  title={ColdFusionX: Multi-Modal Fusion via Aligned Latent Manifolds and Stochastic Masking for Zero-Shot CTR},
  author={[Sharm Fernando, Nethmi Wijesinghe]},
  booktitle={World Congress on Computing and Communication Technologies (WCCCT)},
  year={2026}
}
```

**ICCRAIDS 2026 Submission:**
```bibtex
@article{coldfusionx_iccraids2026,
  title={A Focused Review of Multimodal CTR Prediction for Cold-Start Recommendation},
  author={[Sharm Fernando, Nethmi Wijesinghe]},
  journal={International Conference on Current Research in Artificial Intelligence and Data Science (ICCRAIDS)},
  year={2026}
}
```

---
*Developed as part of the Final Year Project at the Informatics Institute of Technology (IIT).*
```
