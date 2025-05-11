# addons/xlm_roberta_analyzer.py
import threading

class MultilingualSentimentAnalyzer:
    """
    Provides multilingual sentiment analysis using XLM-RoBERTa.
    This class loads the model once and serves all sentiment analysis requests.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.initialized = False
        # Default to English if language detection fails
        self.default_language = "en"
        
        # Initialize in a separate thread to avoid blocking the UI
        threading.Thread(target=self._initialize_model, daemon=True).start()
        
    def _initialize_model(self):
        """Initialize the XLM-RoBERTa model and tokenizer."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            print("Loading XLM-RoBERTa multilingual sentiment model...")
            
            # Load tokenizer and model
            model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            print(f"XLM-RoBERTa model loaded successfully (device: {self.device})")
            self.initialized = True
        
        except Exception as e:
            print(f"Error initializing XLM-RoBERTa model: {e}")
            
            # Fallback to another model if first one fails
            try:
                print("Attempting to load fallback model...")
                model_name = "finiteautomata/bertweet-base-sentiment-analysis"
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.model.eval()
                
                print("Fallback model loaded successfully")
                self.initialized = True
            except Exception as fallback_error:
                print(f"Error loading fallback model: {fallback_error}")
            
    def detect_language(self, text):
        """
        Detect the language of the input text.
        Uses langdetect if available, otherwise returns the default language.
        """
        try:
            from langdetect import detect
            return detect(text)
        except:
            try:
                # Try using langid as fallback
                import langid
                lang, _ = langid.classify(text)
                return lang
            except:
                # Return default language if both fail
                return self.default_language
                
    def preprocess_text(self, text):
        """Preprocess text before sentiment analysis."""
        # Perform basic preprocessing
        import re
        
        # 1. Normalize whitespace
        text = ' '.join(text.split())
        
        # 2. Remove URLs 
        text = re.sub(r'https?://\S+', '', text)
        
        # 3. Remove user mentions for privacy
        text = re.sub(r'@\w+', '@user', text)
        
        # 4. Replace multiple punctuation with single punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        return text
                
    def analyze_sentiment(self, text, return_confidence=False):
        """Analyze sentiment of text and return the sentiment label and confidence scores."""
        if not self.initialized:
            if return_confidence:
                return "Neutral", {"Positive": 33.33, "Neutral": 33.33, "Negative": 33.33}
            return "Neutral"
            
        try:
            import torch
            
            # Preprocess text
            text = self.preprocess_text(text)
            
            # Detect language (for debugging/logging)
            lang = self.detect_language(text)
            
            # Tokenize and convert to model inputs
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            
            # Move inputs to same device as model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Get number of labels and map accordingly
            num_labels = probs.shape[1]
            
            if num_labels == 3:
                # Standard 3-label sentiment model
                id_to_label = {0: "Negative", 1: "Neutral", 2: "Positive"}
            elif num_labels == 2:
                # Binary sentiment model
                id_to_label = {0: "Negative", 1: "Positive"}
            else:
                # Generic mapping
                id_to_label = {i: f"Label_{i}" for i in range(num_labels)}
                
            # Get sentiment label
            sentiment_id = torch.argmax(probs, dim=1).item()
            sentiment_label = id_to_label.get(sentiment_id, "Neutral")
            
            # Calculate confidence scores
            confidence = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
            
            # Fill in values based on model output
            for i, label in id_to_label.items():
                if i < len(probs[0]):
                    mapped_label = label
                    if mapped_label in confidence:
                        confidence[mapped_label] = float(probs[0, i]) * 100
                    elif label == "Label_0":
                        confidence["Negative"] = float(probs[0, i]) * 100
                    elif label == "Label_1":
                        if num_labels == 2:
                            confidence["Positive"] = float(probs[0, i]) * 100
                        else:
                            confidence["Neutral"] = float(probs[0, i]) * 100
                    elif label == "Label_2":
                        confidence["Positive"] = float(probs[0, i]) * 100
            
            # Handle 2-class models - estimate neutral as lack of strong confidence
            if num_labels == 2 and "Neutral" not in id_to_label.values():
                # If neither class has strong confidence, consider it neutral
                max_conf = max(confidence["Positive"], confidence["Negative"])
                if max_conf < 70:  # Threshold for "confident" prediction
                    sentiment_label = "Neutral"
                    # Adjust confidences
                    diff = 100 - (confidence["Positive"] + confidence["Negative"])
                    confidence["Neutral"] = max(0, diff)
                    confidence["Positive"] *= 0.7  # Reduce positive/negative confidences
                    confidence["Negative"] *= 0.7
                    
            if return_confidence:
                return sentiment_label, confidence
            return sentiment_label
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            if return_confidence:
                return "Neutral", {"Positive": 33.33, "Neutral": 33.33, "Negative": 33.33}
            return "Neutral"  # Return Neutral as fallback