# model_service.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance
    
    def _init_model(self):
        print("Loading sentiment model (this happens only once)...")
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model.eval()
        
    def analyze_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        labels = ["Negative", "Neutral", "Positive"]
        sentiment = {label: float(probs[i]) * 100 for i, label in enumerate(labels)}
        return sentiment