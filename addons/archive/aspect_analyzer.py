# aspect_analyzer.py
import spacy
from transformers import pipeline

class AspectBasedSentiment:
    def __init__(self):
        # Load spaCy for entity and dependency parsing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load sentiment pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
    
    def extract_aspects_with_sentiment(self, text):
        """Extract aspects and their associated sentiment."""
        doc = self.nlp(text)
        aspects = {}
        
        # Extract named entities as potential aspects
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG", "PERSON", "GPE"]:
                # Get the sentence containing this entity
                for sent in doc.sents:
                    if ent.start >= sent.start and ent.end <= sent.end:
                        sentiment = self.sentiment_pipeline(sent.text)[0]
                        aspects[ent.text] = {
                            "type": ent.label_,
                            "sentiment": sentiment["label"],
                            "score": sentiment["score"],
                            "context": sent.text
                        }
        
        # Extract noun phrases that might be aspects
        for chunk in doc.noun_chunks:
            # Skip if already captured as named entity
            if any(chunk.text in aspect for aspect in aspects.keys()):
                continue
                
            # Find sentiment words (adjectives) connected to this noun
            for token in chunk.root.children:
                if token.pos_ == "ADJ":
                    for sent in doc.sents:
                        if chunk.start >= sent.start and chunk.end <= sent.end:
                            sentiment = self.sentiment_pipeline(sent.text)[0]
                            aspects[chunk.text] = {
                                "type": "FEATURE",
                                "sentiment": sentiment["label"],
                                "score": sentiment["score"],
                                "context": sent.text,
                                "modifier": token.text
                            }
        
        return aspects