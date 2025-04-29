# 🔍 SentiMate: Social Media Sentiment Analysis Application

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![CustomTkinter](https://img.shields.io/badge/CustomTkinter-5.1.2-orange)
![HuggingFace](https://img.shields.io/badge/Transformers-4.28.1-red)

A powerful desktop application for real-time sentiment analysis of social media content with advanced visualization capabilities.

![Application Screenshot](https://via.placeholder.com/800x400?text=SentiMate+Application+Screenshot)

## 📋 Features

- **Multi-Source Analysis**: Analyze sentiment from different inputs:
  - Individual tweets via URL
  - Custom text input
  - Real-time hashtag monitoring
  - Clipboard content analysis

- **Real-Time Monitoring**: Track sentiment for specific hashtags on Twitter/X with automatic updates every minute

- **Clipboard Sentiment Analyzer**: Instantly analyze selected text with Ctrl+C hotkey for quick sentiment evaluation

- **Advanced Visualizations**:
  - Interactive pie charts showing sentiment distribution
  - Time-series graphs for tracking sentiment trends
  - Real-time visual feedback with color-coded indicators

- **Data Management**:
  - Export analysis results to CSV and images
  - Load historical data from previous analyses
  - Automatic logging of sentiment results

- **Customizable UI**: Modern dark-themed interface with responsive layout using CustomTkinter

## 🏗️ Architecture

SentiMate is built with a modular architecture that separates the UI components from the analysis logic:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     UI Layer    │     │  Service Layer  │     │    Data Layer   │
│  (CustomTkinter)│────▶│  (Analyzers &   │────▶│  (Collectors &  │
│                 │     │   Processing)   │     │    Storage)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

- **Pages**: The application is organized into multiple pages with specific functionalities:
  - `home_page.py`: Welcome screen
  - `page1.py`: Single item analysis (Tweet/Text)
  - `page2.py`: Real-time opinion polling with hashtag monitoring
  - `page3.py`: Clipboard sentiment analyzer

- **Addons**: Backend modules that provide specialized functionality:
  - `sentiment_analyzer.py`: Core singleton for sentiment analysis using HuggingFace models
  - `collector.py`/`bypass_collector.py`: Twitter data collection modules
  - `analyzer.py`: Processes collected tweets for sentiment analysis
  - `model_service.py`: Provides access to the sentiment analysis model
  - `db_service.py`: Database interactions for storing results
  - `aspect_analyzer.py`: Aspect-based sentiment analysis

## 🔧 Technology Stack

- **UI Framework**: 
  - [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - Modern UI toolkit based on Tkinter

- **Machine Learning**:
  - [HuggingFace Transformers](https://huggingface.co/transformers/) - State-of-the-art NLP models
  - [PyTorch](https://pytorch.org/) - Deep learning framework for model inference

- **Data Collection**:
  - [Tweepy](https://www.tweepy.org/) - Twitter API client for Python
  - Beautiful Soup - For web scraping fallback option

- **Data Visualization**:
  - [Matplotlib](https://matplotlib.org/) - Comprehensive plotting library
  - [FigureCanvasTkAgg](https://matplotlib.org/stable/api/figure_canvas_api.html) - Matplotlib integration with Tkinter

- **Data Processing**:
  - [Pandas](https://pandas.pydata.org/) - Data analysis and manipulation
  - [NumPy](https://numpy.org/) - Numerical computing

- **Additional Libraries**:
  - spaCy - For aspect-based sentiment analysis
  - Keyboard - For hotkey monitoring

## 📥 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/sentimate.git
cd sentimate
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
```bash
venv\Scripts\activate
```

- On macOS/Linux:
```bash
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

#### Core Dependencies

```
customtkinter>=5.1.2
torch>=1.9.0
transformers>=4.28.1
tweepy>=4.10.0
matplotlib>=3.5.0
pandas>=1.3.0
numpy>=1.20.0
pyperclip>=1.8.2
keyboard>=0.13.5
beautifulsoup4>=4.9.3
spacy>=3.2.0
scikit-learn>=0.24.2
```

### Step 4: Download SpaCy model (for aspect-based analysis)

```bash
python -m spacy download en_core_web_sm
```

### Step 5: Configure Twitter API (Optional)

To use the Twitter data collection features, you need to set up Twitter API credentials:

1. Create a Twitter Developer account at [developer.twitter.com](https://developer.twitter.com)
2. Create a new project and app to get your API keys
3. Update the bearer token in `addons/collector.py`

## 🚀 Usage

### Running the Application

```bash
python main.py
```

### Main Pages and Features

#### 1. Single Item Analysis (Tweet/Text)

![Tweet Analysis](https://via.placeholder.com/400x200?text=Tweet+Analysis+Screenshot)

- Select input type from the dropdown (Tweet, Text, Hashtag, Account)
- Enter the content (URL or text)
- Click Submit to analyze sentiment
- View results in the pie chart visualization

#### 2. Real-Time Opinion Polling

![Opinion Polling](https://via.placeholder.com/400x200?text=Opinion+Polling+Screenshot)

- Enter a hashtag to monitor
- Click "Start Monitoring" to begin collecting and analyzing tweets
- Watch sentiment trends update in real-time
- Export results with "Export Results" button

#### 3. Clipboard Sentiment Analyzer

![Clipboard Analyzer](https://via.placeholder.com/400x200?text=Clipboard+Analyzer+Screenshot)

- Copy any text with Ctrl+C to automatically analyze its sentiment
- View results in the pie chart
- See detailed breakdown of positive, neutral, and negative scores
- Results are automatically logged for future reference

## 📂 File Structure

```
sentimate/
│
├── main.py               # Application entry point
│
├── pages/                # UI pages
│   ├── home_page.py      # Welcome page
│   ├── page1.py          # Single item analysis
│   ├── page2.py          # Real-time opinion polling
│   ├── page3.py          # Clipboard analyzer
│   └── settings_page.py  # Application settings
│
├── addons/               # Backend modules
│   ├── sentiment_analyzer.py  # Core sentiment analysis
│   ├── collector.py      # Twitter API data collection
│   ├── bypass_collector.py    # Alternative scraping collector
│   ├── analyzer.py       # Tweet analysis
│   ├── model_service.py  # ML model management
│   ├── db_service.py     # Database operations
│   ├── aspect_analyzer.py     # Aspect-based analysis
│   └── evaluation.py     # Model evaluation tools
│
├── outputs/              # Data storage directory
│   └── *.csv             # Exported analysis results
│
├── logs/                 # Application logs
│
└── requirements.txt      # Project dependencies
```

## 🤝 Contributing

Contributions are welcome! Here's how you can contribute to SentiMate:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines for Python code
- Write docstrings for new functions and classes
- Add appropriate comments for complex logic
- Write tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [CardiffNLP](https://github.com/cardiffnlp) for the pre-trained sentiment analysis model
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) for the modern UI components
- [HuggingFace](https://huggingface.co/) for the Transformers library

---

Made with ❤️ by [Ammar]
