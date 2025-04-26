# sentiment_analyzer.py - New file to add to your project

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Optional

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
        """Load model and tokenizer once for the entire application."""
        try:
            print("Loading sentiment analysis model...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                "cardiffnlp/twitter-roberta-base-sentiment"
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                "cardiffnlp/twitter-roberta-base-sentiment"
            )
            self._model.eval()
            print("Sentiment analysis model loaded successfully")
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            # Provide a fallback if model loading fails
            self._model = None
            self._tokenizer = None
    
    def is_initialized(self) -> bool:
        """Check if the model was properly initialized."""
        return self._model is not None and self._tokenizer is not None
    
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
                inputs = self._tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                outputs = self._model(**inputs)
                
            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            
            # Map to sentiment labels
            labels = ["Negative", "Neutral", "Positive"]
            sentiment = {label: float(probs[i]) * 100 for i, label in enumerate(labels)}
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