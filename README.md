# 🧠 Multimodal E-Commerce Price Optimization System  
### 📦 Smart Product Pricing Challenge — ML Challenge 2025  

---

## 📘 Overview  

This repository presents a **high-performance multimodal machine learning system** developed for the **Smart Product Pricing Challenge 2025**.  
The goal is to predict **optimal product prices** for e-commerce items by leveraging three key information sources:

- 📝 **Textual Metadata** — Product descriptions and catalog content  
- 🖼️ **Product Images** — Visual quality, color, and aesthetic cues  
- 📊 **Structured Data** — Brand information and Item Pack Quantity (IPQ)  

The system integrates **Transfer Learning**, **Feature Fusion**, and **Gradient Boosted Regression (LightGBM)** to minimize the **Symmetric Mean Absolute Percentage Error (SMAPE)** and achieve competitive leaderboard accuracy.

---

## 🧩 Methodology and Architecture  

### 🔹 Hybrid Multimodal Regression Pipeline  

| Modality | Model / Feature | Role in Pipeline |
|-----------|------------------|------------------|
| **Text** | `DistilBERT` | Extracts 768-dimensional semantic embeddings from product catalog content |
| **Image** | `ResNet-50` | Generates 2048-dimensional visual embeddings that capture product aesthetics and visual quality |
| **Structured** | `IPQ_Value`, `Brand Encoding` | Extracted using regex and one-hot encoding for normalization |
| **Prediction** | `LightGBM Regressor` | Learns on fused multimodal feature vectors (~2800 features) to predict log-transformed price per unit |

---

### ⚙️ Key Design Highlights  

- 🔸 **Transfer learning** with frozen CNN and Transformer backbones  
- 🔸 **Memory-efficient batch feature generation** for 150K+ product dataset  
- 🔸 **Feature Fusion Layer** combining textual, visual, and structured vectors  
- 🔸 **Optimized for SMAPE**, ensuring balanced accuracy across low and high price ranges  
- 🔸 **Stable and deterministic runs** through fixed seeds and controlled threading  

---

## ⚙️ Environment Setup  

### 🧾 Requirements  

- **Python 3.11** (Recommended for macOS / M-Series compatibility)  
- **Virtual Environment (venv)** for clean dependency management  

### 🔧 Installation  

bash
# Create and activate a virtual environment
python3 -m venv venv_311_stable
source venv_311_stable/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn lightgbm torch torchvision transformers pillow tqdm requests
