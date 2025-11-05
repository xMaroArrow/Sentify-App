Sentify: Social Media Sentiment Analysis App

Sentify is a desktop application for analyzing sentiment in social media content with a modern CustomTkinter UI, real-time collection, multilingual models, and built-in tools to train and evaluate your own models.

Key capabilities:
- Multi-source analysis: free-text, tweets, hashtags, and Reddit threads
- Real-time tweet collection and live trend charts
- Multilingual transformers (RoBERTa/XLM-R) with GPU support
- Model training and evaluation (LSTM/CNN/Transformers) with rich visuals
- Clean, responsive UI with dark/light modes

**Features**
- Input options:
  - Analyze a single tweet URL or any free text
  - Monitor hashtags in real time (Twitter/X)
  - Fetch and analyze Reddit submissions and comments
- Visualizations:
  - Sentiment distribution, trends over time, confusion matrices
  - Precision–Recall and ROC curves for model comparison
- Model workflow:
  - Preprocess, split datasets, and train LSTM/CNN/Transformer models
  - Evaluate, compare, and save models with metadata
- Practical extras:
  - CSV export, logs, and publication-ready plots

**Project Structure**
```
Sentify-App/
  main.py                  # App entrypoint (CustomTkinter UI)
  pages/                   # UI pages (analysis, training, settings, etc.)
  addons/                  # Collectors and analyzers (Twitter, Reddit, HF models)
  models/                  # Training code and saved models
  utils/                   # Config, preprocessing, visualization helpers
  data/                    # Sample/input datasets (CSV)
  outputs/                 # Collected tweets and exports (created at runtime)
```

**Quickstart**
- Prerequisites: Python 3.9+ recommended; Windows/macOS/Linux with Tk available

- Create and activate a virtual environment, then install requirements:
  - Windows (PowerShell)
    - `python -m venv .venv`
    - `.\.venv\Scripts\Activate.ps1`
    - `pip install --upgrade pip`
    - `pip install -r requirements.txt`
  - macOS/Linux
    - `python3 -m venv .venv`
    - `source .venv/bin/activate`
    - `python -m pip install --upgrade pip`
    - `pip install -r requirements.txt`

- Run the app:
  - `python main.py`

**GPU Note (PyTorch)**
- The default `requirements.txt` installs a CPU build of PyTorch.
- For NVIDIA GPUs, install a CUDA-enabled build first: https://pytorch.org/get-started/locally/
  - Example: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

**Optional Integrations**
- Twitter/X (real-time collection)
  - Uses Tweepy (API v2 recent search). Provide a Bearer Token.
  - Simplest: edit the `bearer_token` in `addons/collector.py` with your token (do not commit secrets).

- Reddit (thread analysis)
  - Set environment variables:
    - `REDDIT_CLIENT_ID`
    - `REDDIT_CLIENT_SECRET`
    - `REDDIT_USER_AGENT` (e.g., `Sentify-App/1.0`)
  - The app will also fall back to values in `app_config.json` if present.

**Datasets and Training**
- The Training page supports CSV datasets with a text column and a label column.
- Workflow:
  - Load CSV (optionally sample large files)
  - Configure preprocessing (URLs/mentions/punctuation removal, case, etc.)
  - Split into train/validation/test
  - Train LSTM/CNN or evaluate local Transformer checkpoints
  - View metrics/curves and save models with metadata

**Troubleshooting**
- HF model downloads: the first run may download model weights; ensure network access and disk space.
- XLM-R models require `sentencepiece` (included in requirements).
- On some Linux distros, install Tk (e.g., `sudo apt-get install python3-tk`).
- If matplotlib figures don’t render, ensure a GUI backend is available on your OS.

**Contributing**
- Use clear, documented code (PEP 8) and small PRs
- Avoid committing API keys or large datasets
- Propose improvements via issues and pull requests

**License**
- Add your license file (e.g., MIT) at the repository root.
