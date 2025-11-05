import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Optional
from utils.config_manager import ConfigManager
from models.model_trainer import LSTMSentimentModel, CNNSentimentModel
try:
    # AdvancedRNNModel is defined later in model_trainer; import guardedly
    from models.model_trainer import AdvancedRNNModel
except Exception:
    AdvancedRNNModel = None
import pickle

class SentimentAnalyzer:
    """
    Singleton class for sentiment analysis using HuggingFace models.
    Ensures model is loaded only once and shared across the application.
    """
    _instance: Optional['SentimentAnalyzer'] = None
    _model = None
    _tokenizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Load model and tokenizer once for the entire application based on config."""
        cfg = ConfigManager()
        source = (cfg.get("model_source", "huggingface") or "huggingface").lower()
        try:
            if source == "local":
                local_path = cfg.get("local_model_path", "") or ""
                local_type = (cfg.get("local_model_type", "transformer") or "transformer").lower()
                if local_path and os.path.isdir(local_path):
                    if local_type == "pytorch":
                        self._load_local_pytorch_model(local_path)
                    else:
                        print(f"Loading local transformer model from: {local_path}")
                        self._tokenizer = AutoTokenizer.from_pretrained(local_path)
                        self._model = AutoModelForSequenceClassification.from_pretrained(local_path)
                else:
                    # Fallback to HF if local path invalid
                    print("Local model path invalid or not set; falling back to Hugging Face model")
                    model_id = cfg.get("model", "cardiffnlp/twitter-roberta-base-sentiment")
                    print(f"Loading HF model: {model_id}")
                    self._tokenizer = AutoTokenizer.from_pretrained(model_id)
                    self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
            else:
                model_id = cfg.get("model", "cardiffnlp/twitter-roberta-base-sentiment")
                print(f"Loading HF model: {model_id}")
                self._tokenizer = AutoTokenizer.from_pretrained(model_id)
                self._model = AutoModelForSequenceClassification.from_pretrained(model_id)

            self._model.eval()
            print("Sentiment analysis model loaded successfully")
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            # Provide a fallback if model loading fails
            self._model = None
            self._tokenizer = None

    def _load_local_pytorch_model(self, model_dir: str) -> None:
        """Load a locally trained PyTorch model with required artifacts.

        Expects files in model_dir:
          - model.pt (weights)
          - hyperparams.json (contains model_type, num_classes, etc.)
          - vectorizer.pkl (CountVectorizer fitted on training data)
          - label_encoder.pkl (LabelEncoder fitted on training labels)
        """
        import json
        import torch
        hp_path = os.path.join(model_dir, "hyperparams.json")
        sd_path = os.path.join(model_dir, "model.pt")
        vec_path = os.path.join(model_dir, "vectorizer.pkl")
        le_path = os.path.join(model_dir, "label_encoder.pkl")

        if not (os.path.exists(hp_path) and os.path.exists(sd_path)):
            raise RuntimeError("Missing hyperparams.json or model.pt in local PyTorch model directory")
        if not (os.path.exists(vec_path) and os.path.exists(le_path)):
            raise RuntimeError("Missing vectorizer.pkl or label_encoder.pkl; cannot run inference for local PyTorch model")

        with open(hp_path, "r", encoding="utf-8") as f:
            hp = json.load(f)

        with open(vec_path, "rb") as f:
            self._vectorizer = pickle.load(f)
        with open(le_path, "rb") as f:
            self._label_encoder = pickle.load(f)

        model_type = (hp.get("model_type") or "advanced_rnn").lower()
        num_classes = int(hp.get("num_classes", 3))
        vocab_size = len(getattr(self._vectorizer, "vocabulary_", {})) + 1
        embedding_dim = int(hp.get("embedding_dim", 300))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type in ("lstm", "basic_lstm"):
            hidden_size = int(hp.get("hidden_size", 256))
            num_layers = int(hp.get("num_layers", 2))
            bidirectional = bool(hp.get("bidirectional", True))
            dropout = float(hp.get("dropout", 0.5))
            model = LSTMSentimentModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif model_type in ("cnn", "basic_cnn"):
            num_filters = int(hp.get("num_filters", 100))
            filter_sizes = hp.get("filter_sizes", [3,4,5])
            dropout = float(hp.get("dropout", 0.5))
            model = CNNSentimentModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_filters=num_filters,
                filter_sizes=filter_sizes,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            # advanced_rnn (GRU/LSTM with attention) supported if available
            if AdvancedRNNModel is None:
                raise RuntimeError("AdvancedRNNModel not available; cannot load this PyTorch model type")
            rnn_type = hp.get("rnn_type", "LSTM")
            hidden_size = int(hp.get("hidden_size", 256))
            num_layers = int(hp.get("num_layers", 2))
            dropout = float(hp.get("dropout", 0.5))
            embedding_dropout = float(hp.get("embedding_dropout", 0.2))
            bidirectional = bool(hp.get("bidirectional", True))
            use_attention = bool(hp.get("use_attention", False))
            model = AdvancedRNNModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
                embedding_dropout=embedding_dropout,
                bidirectional=bidirectional,
                rnn_type=rnn_type,
                use_attention=use_attention,
            )

        # Load state dict
        state = torch.load(sd_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        self._pt_model = model.to(device)
        self._pt_device = device

        # Ensure tokenizer is None to signal we are in PyTorch local mode
        self._tokenizer = None
        self._model = model
    
    def is_initialized(self) -> bool:
        """Check if the model was properly initialized (transformer or local PyTorch)."""
        if self._model is None:
            return False
        # Transformer path has tokenizer; PyTorch local path has _pt_model
        return self._tokenizer is not None or hasattr(self, "_pt_model")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of the provided text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with sentiment labels as keys and confidence scores (0-100) as values
        """
        if not self.is_initialized():
            return {"Error": 100.0}
            
        try:
            # Ensure we're not tracking gradients for inference
            with torch.no_grad():
                if self._tokenizer is None and hasattr(self, "_pt_model"):
                    # Local PyTorch model path: use vectorizer -> indices -> forward
                    words = (text or "").split()
                    vocab = getattr(self._vectorizer, "vocabulary_", {})
                    indices = [vocab.get(w, 0) + 1 for w in words]
                    max_len = 128
                    if len(indices) > max_len:
                        indices = indices[:max_len]
                    else:
                        indices = indices + [0] * (max_len - len(indices))
                    x = torch.tensor([indices], dtype=torch.long, device=self._pt_device)
                    outputs = self._pt_model(x)
                else:
                    inputs = self._tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True, 
                        padding=True, 
                        max_length=512
                    )
                    outputs = self._model(**inputs)
                
            # Convert logits to probabilities
            logits = outputs if not hasattr(outputs, "logits") else outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            
            # Map to sentiment labels
            # Map class indices to labels
            if hasattr(self, "_label_encoder") and self._label_encoder is not None:
                classes = list(getattr(self._label_encoder, "classes_", ["negative","neutral","positive"]))
            else:
                classes = ["negative","neutral","positive"]
            # Normalize to title case keys
            mapping = {"negative":"Negative","neutral":"Neutral","positive":"Positive"}
            sentiment = {}
            for i in range(min(len(probs), len(classes))):
                key = mapping.get(str(classes[i]).lower(), str(classes[i]))
                sentiment[key] = float(probs[i]) * 100
            return sentiment
            
        except Exception as e:
            print(f"Error during sentiment analysis: {e}")
            return {"Error": 100.0}
    
    def get_ordered_sentiment_values(self, sentiment_dict: Dict[str, float]) -> list:
        """
        Return sentiment values in a consistent order for visualization.
        Order: [Neutral, Negative, Positive]
        
        Args:
            sentiment_dict: Dictionary with sentiment scores
            
        Returns:
            List of sentiment values in the specified order
        """
        if "Error" in sentiment_dict:
            return [33.3, 33.3, 33.3]  # Equal distribution for error case
            
        sentiments = ["Neutral", "Negative", "Positive"]
        return [sentiment_dict.get(s, 0) for s in sentiments]

    @classmethod
    def reload(cls) -> None:
        """Reload the singleton model using current configuration."""
        try:
            if cls._instance is None:
                # Creating an instance will initialize
                cls()
                return
            # Clear existing tokenizer/model to free memory
            try:
                cls._instance._model = None
                cls._instance._tokenizer = None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception:
                pass
            # Reinitialize
            cls._instance._initialize()
            print("SentimentAnalyzer reloaded from config")
        except Exception as e:
            print(f"Failed to reload SentimentAnalyzer: {e}")
