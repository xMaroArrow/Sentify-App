# 🌐 Sentify-App: Social Media Sentiment Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/Backend-PyTorch-red.svg)](https://pytorch.org/)
[![UI](https://img.shields.io/badge/UI-CustomTkinter-2B6CB0.svg)](https://github.com/TomSchimansky/CustomTkinter)

Sentify-App is an **AI-powered desktop application** for **multilingual sentiment analysis** across social media platforms.  
It combines **real-time tweet collection**, **HuggingFace transformers**, **deep learning model training**, and **publication-ready visualizations** — all inside a clean **CustomTkinter** graphical interface.

---

## 🚀 Key Capabilities

- 💬 **Multi-source input** — Analyze free text, tweet URLs, hashtags, or Reddit threads  
- ⚡ **Real-time monitoring** — Stream tweets and visualize sentiment trends live  
- 🌍 **Multilingual support** — Pretrained XLM-RoBERTa models for 100+ languages  
- 🧠 **Custom training** — Train and fine-tune LSTM, CNN, or Transformer models  
- 📊 **Rich visualizations** — Confusion matrices, ROC & PR curves, F1/accuracy charts  
- 🎨 **Modern GUI** — Responsive design with dark/light themes  

---

## 🧩 Application Pages

| Page | Purpose |
|------|----------|
| 🏠 **Home** | Overview and project dashboard |
| 💬 **Page 1** | Analyze text, tweets, hashtags, or accounts |
| 📈 **Page 2** | Real-time multilingual sentiment monitoring |
| 🔎 **Page 3** | Aspect-based sentiment and topic insights |
| 🧪 **Page 5** | Model training, fine-tuning, and evaluation |
| ⚖️ **Page 6** | Model comparison and visualization |

---

## 🧱 System Architecture

```
[ Twitter/Reddit API ] → [ Data Processor ]
           ↓
     [ Sentiment Analyzers ]
 (RoBERTa / XLM-R / BERTweet)
           ↓
     [ Model Trainer ]
  (LSTM, CNN, Transformers)
           ↓
     [ Visualizer ]
 (Metrics, Curves, Tables)
           ↓
     [ CustomTkinter GUI ]
```

---

## ⚙️ Installation Guide

### Prerequisites
- Python **3.9+**
- OS with **Tkinter GUI support** (Windows/macOS/Linux)
- Optional: CUDA-enabled GPU for PyTorch acceleration

### 1️⃣ Clone and Setup
```bash
git clone https://github.com/YourUsername/Sentify-App.git
cd Sentify-App
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### 2️⃣ Run the App
```bash
python main.py
```

---

## ⚡ GPU Acceleration (Optional)
If you have an NVIDIA GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Verify GPU use:
```python
import torch
print(torch.cuda.is_available())
```

---

## 📡 API Integrations

### 🐦 Twitter (X) Real-Time Streaming
Edit your **Bearer Token** in `addons/collector.py` or set as an environment variable:
```bash
set BEARER_TOKEN=your_token_here
```

### 👽 Reddit Thread Analysis
Set credentials:
```bash
export REDDIT_CLIENT_ID="your_id"
export REDDIT_CLIENT_SECRET="your_secret"
export REDDIT_USER_AGENT="SentifyApp/1.0"
```

---

## 🧮 Model Training & Fine-Tuning

1. Load a dataset (CSV with text + sentiment columns)
2. Configure preprocessing (remove URLs, mentions, punctuation, etc.)
3. Split dataset (Train/Val/Test)
4. Choose architecture:
   - **LSTM** (sequence-based)
   - **CNN** (spatial text features)
   - **Transformer** (contextual embeddings)
5. Adjust hyperparameters (epochs, learning rate, batch size)
6. Train, evaluate, and visualize performance  
   *(Confusion Matrix, ROC, PR Curves, Metrics Table)*

---

## 📊 Visualization Examples

| Metric | Example |
|:-------|:--------|
| Confusion Matrix | ![Confusion Matrix](docs/images/confusion_matrix.png) |
| ROC Curve | ![ROC Curve](docs/images/roc_curve.png) |
| Precision-Recall | ![PR Curve](docs/images/pr_curve.png) |

---

## 🧰 Tech Stack

- 🖥 **GUI:** CustomTkinter  
- 🤗 **NLP:** Hugging Face Transformers  
- 🔥 **Deep Learning:** PyTorch  
- 🧪 **ML Tools:** scikit-learn, pandas, numpy  
- 📈 **Visualization:** matplotlib, seaborn  
- 🐦 **Data:** Tweepy API, Nitter scraper (fallback)

---

## 📂 Project Structure

```
Sentify-App/
├── main.py                     # Application entry point
├── pages/                      # GUI pages (analysis, training, comparison)
├── addons/                     # Analyzers, collectors, multilingual models
├── models/                     # Model architectures and checkpoints
├── utils/                      # Preprocessing, visualization, helpers
├── data/                       # Datasets
├── outputs/                    # Generated CSVs, plots, and results
└── requirements.txt
```

---

## 🧑‍💻 Contributing

Pull requests are welcome!  
Please follow [PEP 8](https://peps.python.org/pep-0008/) and keep commits clean.

1. Fork the repo  
2. Create your feature branch  
3. Commit with descriptive messages  
4. Submit a pull request 🎉  

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🎓 Academic Use

If you use **Sentify-App** for research or a thesis, please cite it as:

> Sentify-App: An Interactive Platform for Social Media Sentiment Analysis and Model Fine-Tuning (2025)

---

⭐ **If you find this project useful, consider giving it a star!**
