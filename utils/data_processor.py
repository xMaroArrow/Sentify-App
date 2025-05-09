"""
Data processor for sentiment analysis models.

This module handles:
- Loading datasets from various sources
- Preprocessing text data
- Splitting data into train/test sets
- Converting data to model-specific formats
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

class DataProcessor:
    """Handles data loading, preprocessing, and preparation for model training."""
    
    def __init__(self, cache_dir: Optional[str] = "data_cache"):
        """
        Initialize the data processor.
        
        Args:
            cache_dir: Directory to cache processed datasets
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize state variables
        self.raw_data = None
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.label_encoder = LabelEncoder()
        self.vectorizer = None
        self.tokenizer = None
        self.class_names = None
        self.vocab_size = None
        self.max_seq_length = 128
        
        # Set default preprocessing options
        self.preprocessing_options = {
            "remove_urls": True,
            "remove_mentions": True,
            "remove_hashtags": False,
            "lowercase": True,
            "remove_punctuation": True,
            "remove_numbers": False
        }
    
    def load_csv(self, file_path: str, text_col: str = None, 
                label_col: str = None, sample_size: int = None) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            text_col: Column name containing text data
            label_col: Column name containing labels
            sample_size: Optionally load only a sample of the data
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Try different encodings for potentially problematic files
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
            df = None
            
            # Check pandas version for the right parameter name
            import pandas as pd
            pd_version = pd.__version__
            if int(pd_version.split('.')[0]) >= 1 and int(pd_version.split('.')[1]) >= 3:
                # For pandas >= 1.3.0, use on_bad_lines
                bad_lines_param = {'on_bad_lines': 'skip'}
            else:
                # For older pandas versions, use error_bad_lines
                bad_lines_param = {'error_bad_lines': False}
            
            for encoding in encodings_to_try:
                try:
                    print(f"Trying to load with {encoding} encoding...")
                    if sample_size:
                        df = pd.read_csv(file_path, nrows=sample_size, encoding=encoding, **bad_lines_param)
                    else:
                        df = pd.read_csv(file_path, encoding=encoding, **bad_lines_param)
                    print(f"Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    print(f"Failed to load with {encoding} encoding")
                    continue
                except Exception as e:
                    print(f"Error with {encoding} encoding: {str(e)}")
                    continue
            
            if df is None:
                # If all encodings fail, try with engine='python' which can be more forgiving
                print("Trying with python engine...")
                if sample_size:
                    df = pd.read_csv(file_path, nrows=sample_size, engine='python')
                else:
                    df = pd.read_csv(file_path, engine='python')
                
            print(f"Successfully loaded {len(df)} rows")
            print(f"Columns in the dataset: {df.columns.tolist()}")
            
            # Special handling for the Twitter Sentiment Dataset with clean_text and category columns
            if 'clean_text' in df.columns and 'category' in df.columns:
                print("Detected Twitter Sentiment Dataset format with clean_text and category columns")
                text_col = 'clean_text'
                label_col = 'category'
            else:
                # Auto-detect text and label columns if not specified or identified
                if text_col is None:
                    text_candidates = ["clean_text", "tweet", "text", "Text", "content", "message", "review"]
                    for col in text_candidates:
                        if col in df.columns:
                            text_col = col
                            break
                    else:
                        raise ValueError(f"Could not find text column. Available columns: {df.columns.tolist()}")
                
                if label_col is None:
                    label_candidates = ["category", "label", "sentiment", "Sentiment", "class", "Class", "target"]
                    for col in label_candidates:
                        if col in df.columns:
                            label_col = col
                            break
                    else:
                        raise ValueError(f"Could not find label column. Available columns: {df.columns.tolist()}")
            
            print(f"Using '{text_col}' for text and '{label_col}' for sentiment")
            
            # Look at sample values to determine the encoding format
            print(f"Sample values in the label column: {df[label_col].value_counts().to_dict()}")
            
            # Check if label column contains numeric values
            if pd.api.types.is_numeric_dtype(df[label_col]):
                # Check if values are in the -1, 0, 1 range
                if set(df[label_col].unique()).issubset({-1, 0, 1}):
                    print("Detected sentiment labels: -1 (negative), 0 (neutral), 1 (positive)")
                    sentiment_map = {
                        -1: "negative",
                        0: "neutral", 
                        1: "positive"
                    }
                    df[label_col] = df[label_col].map(sentiment_map)
            
            # Standardize column names
            df = df.rename(columns={text_col: "text", label_col: "sentiment"})
            
            # Clean and normalize the data
            print("Preprocessing data...")
            df = self._preprocess_dataframe(df)
            print(f"Preprocessing complete. {len(df)} rows remaining after cleaning.")
            
            # Store the data
            self.raw_data = df
            
            # Encode labels
            self._encode_labels()
            
            return df
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise RuntimeError(f"Error loading CSV file: {str(e)}")
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize the DataFrame."""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Drop rows with missing text or sentiment
        df = df.dropna(subset=["text", "sentiment"])
        
        # Normalize sentiment labels
        df["sentiment"] = df["sentiment"].astype(str).str.lower()
        
        # Map various sentiment representations to standard values
        sentiment_map = {
            # Positive variations
            "positive": "positive",
            "pos": "positive",
            "1": "positive",
            "p": "positive",
            "good": "positive",
            "4": "positive",
            "5": "positive",
            
            # Neutral variations
            "neutral": "neutral",
            "neu": "neutral",
            "0": "neutral",
            "n": "neutral",
            "ok": "neutral",
            "3": "neutral",
            
            # Negative variations
            "negative": "negative",
            "neg": "negative",
            "-1": "negative",
            "bad": "negative",
            "1": "negative",
            "2": "negative"
        }
        
        df["sentiment"] = df["sentiment"].map(sentiment_map).fillna(df["sentiment"])
        
        # If we still have more than 3 classes, try to simplify
        if df["sentiment"].nunique() > 3:
            numeric_cols = pd.to_numeric(df["sentiment"], errors="coerce")
            if not numeric_cols.isna().all():
                # Seems like numeric ratings, convert to sentiment
                df["sentiment"] = numeric_cols.apply(
                    lambda x: "positive" if x > 3 else ("neutral" if x == 3 else "negative")
                )
        
        # Preprocess text
        df["text"] = df["text"].apply(self._preprocess_text)
        
        return df
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess a single text string based on preprocessing options."""
        if not isinstance(text, str):
            return ""
        
        # Apply preprocessing based on options
        if self.preprocessing_options.get("remove_urls", True):
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        if self.preprocessing_options.get("remove_mentions", True):
            text = re.sub(r'@\w+', '', text)
        
        if self.preprocessing_options.get("remove_hashtags", False):
            text = re.sub(r'#\w+', '', text)
        
        if self.preprocessing_options.get("lowercase", True):
            text = text.lower()
        
        if self.preprocessing_options.get("remove_punctuation", True):
            text = re.sub(r'[^\w\s]', '', text)
        
        if self.preprocessing_options.get("remove_numbers", False):
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _encode_labels(self):
        """Encode sentiment labels to numeric values."""
        if self.raw_data is None:
            return
        
        # Fit label encoder
        self.label_encoder.fit(self.raw_data["sentiment"])
        self.class_names = self.label_encoder.classes_.tolist()
        
        
    def set_preprocessing_options(self, options: Dict[str, bool]):
        """
        Set preprocessing options.
        
        Args:
            options: Dictionary of preprocessing options
        """
        self.preprocessing_options.update(options)
    
    def prepare_train_test_split(self, train_size: float = 0.8, 
                                    use_validation: bool = True,
                                    random_state: int = 42,
                                    stratify: bool = True):
            """
            Split the data into train, test, and optionally validation sets.
            
            Args:
                train_size: Proportion of data to use for training
                use_validation: Whether to create a validation set from training data
                random_state: Random seed for reproducibility
                stratify: Whether to maintain label distribution in splits
            """
            if self.raw_data is None:
                raise ValueError("No data loaded. Call load_csv() first.")
            
            # Get data and labels
            texts = self.raw_data["text"].values
            sentiments = self.raw_data["sentiment"].values
            
            # Encode labels if needed
            if not hasattr(self, 'label_encoder') or self.label_encoder is None:
                self._encode_labels()
            
            # Convert labels to numeric
            labels = self.label_encoder.transform(sentiments)
            
            # Stratify if requested and possible
            stratify_param = labels if stratify else None
            
            # Split into train and test
            if use_validation:
                # First split into train+val and test
                train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
                    texts, labels, train_size=train_size, random_state=random_state, 
                    stratify=stratify_param
                )
                
                # Then split train into train and validation
                # Use same proportion for validation as for test
                val_proportion = 1 - train_size
                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    train_val_texts, train_val_labels, 
                    test_size=val_proportion,
                    random_state=random_state,
                    stratify=train_val_labels if stratify else None
                )
                
                # Store the splits
                self.train_data = {
                    "texts": train_texts,
                    "labels": train_labels
                }
                
                self.val_data = {
                    "texts": val_texts,
                    "labels": val_labels
                }
                
                self.test_data = {
                    "texts": test_texts,
                    "labels": test_labels
                }
            else:
                # Simple train/test split
                train_texts, test_texts, train_labels, test_labels = train_test_split(
                    texts, labels, train_size=train_size, random_state=random_state, 
                    stratify=stratify_param
                )
                
                # Store the splits
                self.train_data = {
                    "texts": train_texts,
                    "labels": train_labels
                }
                
                self.val_data = None
                
                self.test_data = {
                    "texts": test_texts,
                    "labels": test_labels
                }
    
    def prepare_tfidf_features(self, max_features: int = 5000):
        """
        Create TF-IDF features for traditional ML models.
        
        Args:
            max_features: Maximum number of features to extract
        
        Returns:
            Dictionary with train, validation, and test features
        """
        if self.train_data is None:
            raise ValueError("Data not split. Call prepare_train_test_split() first.")
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
        # Fit on training data only
        train_features = self.vectorizer.fit_transform(self.train_data["texts"])
        
        # Transform test data
        test_features = self.vectorizer.transform(self.test_data["texts"])
        
        # Transform validation data if available
        val_features = None
        if self.val_data is not None:
            val_features = self.vectorizer.transform(self.val_data["texts"])
        
        self.vocab_size = len(self.vectorizer.vocabulary_)
        
        return {
            "train": train_features,
            "test": test_features,
            "val": val_features
        }
    
    def prepare_pytorch_datasets(self, batch_size: int = 32, 
                            tokenizer_name: str = None,
                            tokenizer = None,  # Add this parameter
                            max_length: int = 128):
        """
        Create PyTorch DataLoaders for deep learning models.
        
        Args:
            batch_size: Batch size for training
            tokenizer_name: Name of the tokenizer to use (for transformer models)
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with train, validation, and test dataloaders
        """
        if self.train_data is None:
            raise ValueError("Data not split. Call prepare_train_test_split() first.")
        
        # If tokenizer or tokenizer_name specified, use transformers tokenizer
        if tokenizer or tokenizer_name:
            if tokenizer is None:
                # Load tokenizer by name if not provided
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            self.tokenizer = tokenizer
            self.max_seq_length = max_length
            
            # Tokenize train data
            train_encodings = tokenizer(
                list(self.train_data["texts"]),
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            
            train_dataset = TensorDataset(
                train_encodings["input_ids"],
                train_encodings["attention_mask"],
                torch.tensor(self.train_data["labels"])
            )
            
            # Tokenize test data
            test_encodings = tokenizer(
                list(self.test_data["texts"]),
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            
            test_dataset = TensorDataset(
                test_encodings["input_ids"],
                test_encodings["attention_mask"],
                torch.tensor(self.test_data["labels"])
            )
            
            # Tokenize validation data if available
            val_dataset = None
            if self.val_data is not None:
                val_encodings = tokenizer(
                    list(self.val_data["texts"]),
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt"
                )
                
                val_dataset = TensorDataset(
                    val_encodings["input_ids"],
                    val_encodings["attention_mask"],
                    torch.tensor(self.val_data["labels"])
                )
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            val_loader = None
            if val_dataset is not None:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False
                )
            
            return {
                "train": train_loader,
                "test": test_loader,
                "val": val_loader
            }
        else:
            return self._prepare_basic_pytorch_datasets(batch_size=batch_size)
    
    def _prepare_transformer_datasets(self, tokenizer_name: str, 
                                      max_length: int, batch_size: int):
        """Prepare datasets for transformer models."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_length
        
        # Tokenize train data
        train_encodings = self.tokenizer(
            list(self.train_data["texts"]),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        train_dataset = TensorDataset(
            train_encodings["input_ids"],
            train_encodings["attention_mask"],
            torch.tensor(self.train_data["labels"])
        )
        
        # Tokenize test data
        test_encodings = self.tokenizer(
            list(self.test_data["texts"]),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        test_dataset = TensorDataset(
            test_encodings["input_ids"],
            test_encodings["attention_mask"],
            torch.tensor(self.test_data["labels"])
        )
        
        # Tokenize validation data if available
        val_dataset = None
        if self.val_data is not None:
            val_encodings = self.tokenizer(
                list(self.val_data["texts"]),
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            
            val_dataset = TensorDataset(
                val_encodings["input_ids"],
                val_encodings["attention_mask"],
                torch.tensor(self.val_data["labels"])
            )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        return {
            "train": train_loader,
            "test": test_loader,
            "val": val_loader
        }
    
    def _prepare_basic_pytorch_datasets(self, batch_size: int):
        """Prepare basic datasets for custom models (LSTM, CNN)."""
        # Create vocabulary and word-to-index mapping
        if self.vectorizer is None:
            # Use CountVectorizer to build vocabulary
            self.vectorizer = CountVectorizer(max_features=10000)
            self.vectorizer.fit(self.train_data["texts"])
            self.vocab_size = len(self.vectorizer.vocabulary_) + 1  # +1 for padding
        
        # Convert texts to indices
        word_to_idx = self.vectorizer.vocabulary_
        
        # Function to convert text to sequence of indices
        def text_to_indices(text, max_len=128):
            words = text.split()
            indices = [word_to_idx.get(word, 0) + 1 for word in words]  # +1 so 0 can be padding
            if len(indices) > max_len:
                indices = indices[:max_len]
            else:
                indices = indices + [0] * (max_len - len(indices))
            return indices
        
        # Convert train texts to sequences
        train_sequences = np.array([
            text_to_indices(text, self.max_seq_length) for text in self.train_data["texts"]
        ])
        
        # Convert test texts to sequences
        test_sequences = np.array([
            text_to_indices(text, self.max_seq_length) for text in self.test_data["texts"]
        ])
        
        # Create train dataset
        train_dataset = TensorDataset(
            torch.tensor(train_sequences),
            torch.tensor(self.train_data["labels"])
        )
        
        # Create test dataset
        test_dataset = TensorDataset(
            torch.tensor(test_sequences),
            torch.tensor(self.test_data["labels"])
        )
        
        # Create val dataset if available
        val_dataset = None
        val_loader = None
        if self.val_data is not None:
            val_sequences = np.array([
                text_to_indices(text, self.max_seq_length) for text in self.val_data["texts"]
            ])
            
            val_dataset = TensorDataset(
                torch.tensor(val_sequences),
                torch.tensor(self.val_data["labels"])
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return {
            "train": train_loader,
            "test": test_loader,
            "val": val_loader
        }
    
    
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced datasets."""
        if self.train_data is None:
            raise ValueError("Data not split. Call prepare_train_test_split() first.")
        
        # Count labels
        labels = self.train_data["labels"]
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Calculate weights (inverse of frequency)
        weights = 1.0 / counts
        
        # Normalize weights
        weights = weights / np.sum(weights) * len(unique_labels)
        
        # Return as dictionary and torch tensor
        weight_dict = {label: weight for label, weight in zip(unique_labels, weights)}
        weight_tensor = torch.tensor(weights, dtype=torch.float)
        
        return weight_dict, weight_tensor
    
    def get_data_summary(self):
        """Generate a summary of the loaded data."""
        if self.raw_data is None:
            return "No data loaded."
        
        df = self.raw_data
        
        # Basic statistics
        total_samples = len(df)
        label_counts = df["sentiment"].value_counts()
        unique_labels = df["sentiment"].nunique()
        
        # Text statistics
        avg_length = df["text"].str.len().mean()
        max_length = df["text"].str.len().max()
        
        # Format summary string
        summary = (
            f"Dataset Summary:\n"
            f"Total samples: {total_samples}\n"
            f"Labels: {unique_labels} unique classes: {', '.join(label_counts.index.tolist())}\n"
            f"Class distribution:\n"
        )
        
        for label, count in label_counts.items():
            percentage = count / total_samples * 100
            summary += f"  - {label}: {count} ({percentage:.1f}%)\n"
        
        summary += (
            f"\nText statistics:\n"
            f"Average length: {avg_length:.1f} characters\n"
            f"Maximum length: {max_length} characters\n"
        )
        
        return summary
    
    def get_split_summary(self):
        """Generate a summary of the data splits."""
        if self.train_data is None:
            return "Data not split yet."
        
        # Training data info
        train_samples = len(self.train_data["labels"])
        train_label_counts = np.bincount(self.train_data["labels"])
        
        # Test data info
        test_samples = len(self.test_data["labels"])
        test_label_counts = np.bincount(self.test_data["labels"])
        
        # Format summary string
        summary = (
            f"Data Split Summary:\n"
            f"Training set: {train_samples} samples\n"
            f"Test set: {test_samples} samples\n"
        )
        
        # Add validation info if available
        if self.val_data is not None:
            val_samples = len(self.val_data["labels"])
            val_label_counts = np.bincount(self.val_data["labels"])
            summary += f"Validation set: {val_samples} samples\n"
        
        # Add class distribution
        summary += "\nClass distribution:\n"
        
        for i, class_name in enumerate(self.class_names):
            if i < len(train_label_counts):
                train_count = train_label_counts[i]
                train_pct = train_count / train_samples * 100
                
                test_count = test_label_counts[i] if i < len(test_label_counts) else 0
                test_pct = test_count / test_samples * 100
                
                summary += f"  - {class_name}:\n"
                summary += f"      Train: {train_count} ({train_pct:.1f}%)\n"
                summary += f"      Test: {test_count} ({test_pct:.1f}%)\n"
                
                if self.val_data is not None:
                    val_count = val_label_counts[i] if i < len(val_label_counts) else 0
                    val_pct = val_count / val_samples * 100
                    summary += f"      Validation: {val_count} ({val_pct:.1f}%)\n"
        
        return summary