"""
Model trainer for sentiment analysis.

This module handles the training, evaluation, and persistence of
different sentiment analysis model architectures with optimized GPU support.
"""

import os
import time
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

# Import our custom data processor
from utils.data_processor import DataProcessor

# Define model architectures
class LSTMSentimentModel(nn.Module):
    """LSTM-based model for sentiment analysis."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_layers: int, num_classes: int, dropout: float = 0.5,
                 bidirectional: bool = True):
        """
        Initialize LSTM model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMSentimentModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True, 
            bidirectional=bidirectional
        )
        
        # If bidirectional, double the hidden size for the fully connected layer
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.fc = nn.Linear(fc_input_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Embed the input
        embedded = self.embedding(x)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Use the final time step output
        out = lstm_out[:, -1, :]
        
        # Apply dropout and pass through fully connected layer
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

class CNNSentimentModel(nn.Module):
    """CNN-based model for sentiment analysis."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_filters: int,
                 filter_sizes: List[int], num_classes: int, dropout: float = 0.5):
        """
        Initialize CNN model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            num_filters: Number of filters per filter size
            filter_sizes: List of filter sizes
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(CNNSentimentModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Create multiple convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) 
            for k in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Embed the input
        embedded = self.embedding(x)
        
        # Add channel dimension for conv2d
        embedded = embedded.unsqueeze(1)
        
        # Apply convolutions and max-over-time pooling
        conv_results = []
        for conv in self.convs:
            # Conv output shape: (batch_size, num_filters, seq_len - filter_size + 1, 1)
            conv_out = conv(embedded)
            # Remove last dimension
            conv_out = conv_out.squeeze(3)
            # Apply ReLU
            conv_out = torch.relu(conv_out)
            # Max pooling
            pooled = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_results.append(pooled)
        
        # Concatenate results from all convolutions
        out = torch.cat(conv_results, dim=1)
        
        # Apply dropout and pass through fully connected layer
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

class ModelTrainer:
    """Trains and evaluates sentiment analysis models with optimized GPU support."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize model trainer with optimized GPU configuration.
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize callbacks dictionary
        self.callbacks = {}
        
        # Set up GPU with optimized configuration
        self._setup_gpu()
        
        # Print model directory
        print(f"Models will be saved to: {os.path.abspath(model_dir)}")
    
    def _setup_gpu(self):
        """Configure GPU for optimal performance."""
        if torch.cuda.is_available():
            # Get GPU information
            gpu_count = torch.cuda.device_count()
            print(f"GPU Support: {gpu_count} device(s) detected")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_capability = torch.cuda.get_device_capability(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # GB
                print(f"  GPU {i}: {gpu_name} (Compute capability: {gpu_capability[0]}.{gpu_capability[1]}, Memory: {total_memory:.2f} GB)")
            
            # Set device to first available GPU
            self.device = torch.device("cuda:0")
            
            # Print CUDA version
            cuda_version = torch.version.cuda
            print(f"CUDA Version: {cuda_version}")
            
            # Enable cuDNN auto-tuner to find the best algorithm
            torch.backends.cudnn.benchmark = True
            print("cuDNN benchmark enabled for optimal performance")
            
            # Check if we can use mixed precision training (tensor cores)
            self.use_amp = hasattr(torch.cuda, 'amp')
            if self.use_amp:
                print("Mixed precision training (AMP) available and enabled")
            else:
                print("Mixed precision training not available")
                
            # Clear any existing cache
            torch.cuda.empty_cache()
            print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
            
        else:
            print("No GPU detected, using CPU for training (this will be slow)")
            self.device = torch.device("cpu")
            self.use_amp = False
    
    def set_callbacks(self, callbacks: Dict[str, Callable]):
        """
        Set callback functions for training events.
        
        Args:
            callbacks: Dictionary mapping event names to callback functions
        """
        self.callbacks = callbacks
    
    def _call_callback(self, event: str, *args, **kwargs):
        """
        Call a callback function if it exists.
        
        Args:
            event: Name of the event
            *args, **kwargs: Arguments to pass to the callback
        """
        if event in self.callbacks and callable(self.callbacks[event]):
            self.callbacks[event](*args, **kwargs)
            
    def scan_models(self) -> Dict[str, Dict]:
        """
        Scan the models directory and load metadata for existing models.
        
        Returns:
            Dictionary mapping model names to their metadata
        """
        models_info = {}
        
        # Check if models directory exists
        if not os.path.exists(self.model_dir):
            return models_info
        
        # Scan for model directories
        for item in os.listdir(self.model_dir):
            item_path = os.path.join(self.model_dir, item)
            
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, "metadata.json")
                
                if os.path.exists(metadata_path):
                    try:
                        # Load metadata
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        
                        # Load evaluation results if available
                        eval_results_path = os.path.join(item_path, "evaluation_results.json")
                        eval_results = None
                        
                        if os.path.exists(eval_results_path):
                            with open(eval_results_path, "r") as f:
                                eval_results = json.load(f)
                        
                        # Load hyperparameters if available
                        hyperparams_path = os.path.join(item_path, "hyperparams.json")
                        hyperparams = None
                        
                        if os.path.exists(hyperparams_path):
                            with open(hyperparams_path, "r") as f:
                                hyperparams = json.load(f)
                        
                        # Load training history if available
                        history_path = os.path.join(item_path, "history.json")
                        history = None
                        
                        if os.path.exists(history_path):
                            with open(history_path, "r") as f:
                                history = json.load(f)
                        
                        # Store model info
                        models_info[item] = {
                            "metadata": metadata,
                            "evaluation": eval_results,
                            "hyperparams": hyperparams,
                            "history": history
                        }
                        
                    except Exception as e:
                        print(f"Error loading metadata for {item}: {e}")
        
        return models_info
            
            
    def load_and_evaluate_local_model(self, data_processor: DataProcessor, model_name: str,
                                model_dir: str, batch_size: int = 32) -> Dict:
        """
        Load a model from a local directory and evaluate it on the test set.
        
        Args:
            data_processor: Data processor with test data
            model_name: Name to save the model results under (for comparison purposes)
            model_dir: Path to the local model directory
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n===== Loading Local Model for Evaluation =====")
        print(f"Model directory: {model_dir}")
        
        # Try to determine model type from the directory
        metadata_path = os.path.join(model_dir, "metadata.json")
        config_path = os.path.join(model_dir, "config.json")
        pytorch_model_path = os.path.join(model_dir, "model.pt")
        transformer_model_path = os.path.join(model_dir, "pytorch_model.bin")
        
        # Check if it's a transformer model
        is_transformer = os.path.exists(transformer_model_path) or os.path.exists(config_path)
        
        # Prepare test data
        print("Preparing test data...")
        if is_transformer:
            # For transformer models, try to use the model's tokenizer
            try:
                dataloaders = data_processor.prepare_pytorch_datasets(
                    batch_size=batch_size,
                    tokenizer_name=model_dir,  # Use the model directory for tokenizer
                    max_length=128
                )
                test_loader = dataloaders["test"]
                print(f"Using transformer tokenizer from {model_dir}")
            except Exception as e:
                print(f"Error loading tokenizer: {e}")
                print("Falling back to standard PyTorch dataloader")
                dataloaders = data_processor.prepare_pytorch_datasets(batch_size=batch_size)
                test_loader = dataloaders["test"]
        else:
            # For PyTorch models
            dataloaders = data_processor.prepare_pytorch_datasets(batch_size=batch_size)
            test_loader = dataloaders["test"]
        
        # Get number of classes
        num_classes = len(data_processor.class_names)
        print(f"Test dataset prepared with {num_classes} classes")
        
        # Load the model based on type
        try:
            if is_transformer:
                # Load transformer model
                print("Loading transformer model...")
                from transformers import AutoModelForSequenceClassification, AutoConfig
                
                # Load configuration first to check number of labels
                if os.path.exists(config_path):
                    config = AutoConfig.from_pretrained(model_dir)
                    original_num_labels = getattr(config, "num_labels", -1)
                    
                    # Check if we need to adjust the number of labels
                    if original_num_labels != num_classes:
                        print(f"Adjusting model output from {original_num_labels} to {num_classes} classes")
                        config.num_labels = num_classes
                        model = AutoModelForSequenceClassification.from_pretrained(
                            model_dir,
                            config=config,
                            ignore_mismatched_sizes=True
                        )
                    else:
                        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                else:
                    # If no config found, try loading with num_labels and ignore_mismatched_sizes
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_dir,
                        num_labels=num_classes,
                        ignore_mismatched_sizes=True
                    )
                    
                model_type = "transformer"
            else:
                # Check metadata for PyTorch model type
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    model_type = metadata.get("model_type", "unknown")
                    hyperparams_path = os.path.join(model_dir, "hyperparams.json")
                    
                    if os.path.exists(hyperparams_path):
                        with open(hyperparams_path, "r") as f:
                            hyperparams = json.load(f)
                    else:
                        hyperparams = {}
                    
                    # Load the appropriate model architecture
                    if model_type == "lstm" or "lstm" in model_name.lower():
                        print("Loading LSTM model...")
                        model = self._load_lstm_model(model_dir, data_processor, hyperparams)
                    elif model_type == "cnn" or "cnn" in model_name.lower():
                        print("Loading CNN model...")
                        model = self._load_cnn_model(model_dir, data_processor, hyperparams)
                    elif model_type == "advanced_rnn":
                        print("Loading Advanced RNN model...")
                        model = self._load_advanced_rnn_model(model_dir, data_processor, hyperparams)
                    elif model_type == "advanced_cnn":
                        print("Loading Advanced CNN model...")
                        model = self._load_advanced_cnn_model(model_dir, data_processor, hyperparams)
                    else:
                        raise ValueError(f"Unknown model type: {model_type}")
                else:
                    # Try to infer from file structure
                    if os.path.exists(pytorch_model_path):
                        # Basic fallback - assume it's an LSTM if we can't determine
                        print("Model type not found in metadata, assuming LSTM...")
                        model = self._load_lstm_model(model_dir, data_processor)
                        model_type = "lstm"
                    else:
                        raise ValueError("Could not determine model type and no model.pt found")
            
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load local model: {str(e)}")
        
        # Move model to device
        model.to(self.device)
        print(f"Model moved to {self.device}")
        
        # Print model summary (parameter count)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")
        
        # Create a directory for the model to store evaluation results
        results_dir = os.path.join(self.model_dir, model_name)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save model metadata
        metadata = {
            "model_type": model_type,
            "original_path": model_dir,
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "device": str(self.device),
            "is_local_model": True
        }
        
        with open(os.path.join(results_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Evaluate model
        print("Running evaluation...")
        eval_start_time = time.time()
        results = self._evaluate_model_on_test_data(
            model=model,
            test_loader=test_loader,
            model_type=("transformer" if model_type == "transformer" else "pytorch"),
            class_names=data_processor.class_names
        )
        eval_time = time.time() - eval_start_time
        
        # Add metadata to results
        results["model_name"] = model_name
        results["model_type"] = model_type
        results["original_path"] = model_dir
        results["evaluation_time"] = eval_time
        results["is_local_model"] = True
        
        # Save results
        results_path = os.path.join(results_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for k, v in results.items():
                if isinstance(v, np.ndarray):
                    serializable_results[k] = v.tolist()
                elif isinstance(v, dict):
                    serializable_results[k] = {
                        sk: sv.tolist() if isinstance(sv, np.ndarray) else sv
                        for sk, sv in v.items()
                    }
                else:
                    serializable_results[k] = v
            
            json.dump(serializable_results, f, indent=2)
        
        # Print summary
        print(f"\nEvaluation Summary for local model {os.path.basename(model_dir)}:")
        print(f"Model stored as: {model_name}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print("Class-wise metrics:")
        for i, class_name in enumerate(results["classes"]):
            print(f"  {class_name}: Precision={results['precision'][i]:.4f}, Recall={results['recall'][i]:.4f}, F1={results['f1'][i]:.4f}")
        
        print(f"Evaluation completed in {eval_time:.2f}s")
        print(f"Results saved to {results_path}")
        
        return results
                
                
    def _load_advanced_rnn_model(self, model_path: str, data_processor: DataProcessor, 
                            hyperparams: Optional[Dict] = None) -> nn.Module:
        """
        Load a saved advanced RNN model with hyperparameters.
        
        Args:
            model_path: Path to the saved model
            data_processor: Data processor with model parameters
            hyperparams: Optional hyperparameters dictionary
            
        Returns:
            Loaded RNN model
        """
        # Use provided hyperparams or try to load from file
        if hyperparams is None:
            hyperparams_path = os.path.join(model_path, "hyperparams.json")
            if os.path.exists(hyperparams_path):
                with open(hyperparams_path, "r") as f:
                    hyperparams = json.load(f)
            else:
                hyperparams = {}
        
        # Set default hyperparameters if not found
        num_layers = hyperparams.get("num_layers", 2)
        hidden_size = hyperparams.get("hidden_size", 256)
        embedding_dim = hyperparams.get("embedding_dim", 300)
        dropout = hyperparams.get("dropout", 0.5)
        embedding_dropout = hyperparams.get("embedding_dropout", 0.2)
        bidirectional = hyperparams.get("bidirectional", True)
        rnn_type = hyperparams.get("rnn_type", "LSTM")
        use_attention = hyperparams.get("use_attention", False)
        
        # Create model
        from .model_trainer import AdvancedRNNModel  # Import locally to avoid circular imports
        
        model = AdvancedRNNModel(
            vocab_size=data_processor.vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=len(data_processor.class_names),
            dropout=dropout,
            embedding_dropout=embedding_dropout,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            use_attention=use_attention
        )
        
        # Load weights
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), 
                                        map_location=self.device))
        
        return model

    def _load_advanced_cnn_model(self, model_path: str, data_processor: DataProcessor,
                            hyperparams: Optional[Dict] = None) -> nn.Module:
        """
        Load a saved advanced CNN model with hyperparameters.
        
        Args:
            model_path: Path to the saved model
            data_processor: Data processor with model parameters
            hyperparams: Optional hyperparameters dictionary
            
        Returns:
            Loaded CNN model
        """
        # Use provided hyperparams or try to load from file
        if hyperparams is None:
            hyperparams_path = os.path.join(model_path, "hyperparams.json")
            if os.path.exists(hyperparams_path):
                with open(hyperparams_path, "r") as f:
                    hyperparams = json.load(f)
            else:
                hyperparams = {}
        
        # Set default hyperparameters if not found
        filter_sizes = hyperparams.get("filter_sizes", [3, 4, 5])
        num_filters = hyperparams.get("num_filters", 100)
        embedding_dim = hyperparams.get("embedding_dim", 300)
        dropout = hyperparams.get("dropout", 0.5)
        embedding_dropout = hyperparams.get("embedding_dropout", 0.2)
        activation = hyperparams.get("activation", "relu")
        batch_norm = hyperparams.get("batch_norm", False)
        pool_type = hyperparams.get("pool_type", "max")
        
        # Create model
        from .model_trainer import AdvancedCNNModel  # Import locally to avoid circular imports
        
        model = AdvancedCNNModel(
            vocab_size=data_processor.vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            num_classes=len(data_processor.class_names),
            dropout=dropout,
            embedding_dropout=embedding_dropout,
            activation=activation,
            batch_norm=batch_norm,
            pool_type=pool_type
        )
        
        # Load weights
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), 
                                        map_location=self.device))
        
        return model
                
                
                
    def load_and_evaluate_pretrained(self, data_processor: DataProcessor, model_name: str,
                                pretrained_model: str = "cardiffnlp/twitter-roberta-base-sentiment",
                                max_seq_length: int = 128, batch_size: int = 32) -> Dict:
        """
        Load a pre-trained model from Hugging Face and evaluate it on the test set without training.
        
        Args:
            data_processor: Data processor with test data
            model_name: Name to save the model results under (for comparison purposes)
            pretrained_model: Hugging Face model ID to load
            max_seq_length: Maximum sequence length for tokenization
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n===== Loading Pre-trained Model for Evaluation =====")
        print(f"Model: {pretrained_model}")
        print(f"This will evaluate the model on the test set without any fine-tuning.")
        
        # Prepare test data with the model's tokenizer directly from Hugging Face
        print(f"Preparing test data with tokenizer from Hugging Face: {pretrained_model}")
        
        try:
            # Import the required classes from transformers
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Load the tokenizer directly from Hugging Face
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            print(f"Tokenizer loaded successfully from Hugging Face")
            
            # Prepare datasets using the loaded tokenizer
            dataloaders = data_processor.prepare_pytorch_datasets(
                batch_size=batch_size,
                tokenizer=tokenizer,  # Pass the tokenizer object directly
                max_length=max_seq_length
            )
            test_loader = dataloaders["test"]
            print(f"Test dataset prepared with tokenizer")
            
            # Get number of classes
            num_classes = len(data_processor.class_names)
            print(f"Number of classes in test data: {num_classes}")
            
            # First try loading with default configuration
            print(f"Loading model from Hugging Face: {pretrained_model}")
            try:
                # Load the model configuration
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(pretrained_model)
                original_labels = getattr(config, "num_labels", -1)
                print(f"Original model has {original_labels} labels, we need {num_classes}")
                
                # Modify config for our number of classes
                config.num_labels = num_classes
                
                # Load the model with our updated config
                model = AutoModelForSequenceClassification.from_pretrained(
                    pretrained_model,
                    config=config,
                    ignore_mismatched_sizes=True
                )
                print(f"Model loaded successfully from Hugging Face")
            except Exception as e:
                print(f"Error loading model with config modification: {e}")
                
                # Try alternative loading method
                try:
                    print("Attempting alternative loading method...")
                    model = AutoModelForSequenceClassification.from_pretrained(
                        pretrained_model,
                        num_labels=num_classes,
                        ignore_mismatched_sizes=True
                    )
                    print("Alternative loading successful")
                except Exception as e2:
                    print(f"Alternative loading failed: {e2}")
                    raise RuntimeError(f"Could not load model: {str(e)}\nAlso tried: {str(e2)}")
            
            # Move model to device
            model.to(self.device)
            print(f"Model moved to {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load pre-trained model: {str(e)}")
        
        # Create a directory for the model to store evaluation results
        model_path = os.path.join(self.model_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Save model metadata
        metadata = {
            "model_type": "transformer",
            "pretrained_model": pretrained_model,
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "device": str(self.device),
            "is_pretrained_evaluation": True,
            "num_classes": num_classes
        }
        
        with open(os.path.join(model_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save hyperparameters
        hyperparams = {
            "model_type": "transformer",
            "pretrained_model": pretrained_model,
            "max_seq_length": max_seq_length,
            "num_classes": num_classes,
            "is_pretrained_evaluation": True
        }
        
        self._save_hyperparameters(model_name, hyperparams)
        
        # Evaluate model
        print("Running evaluation...")
        eval_start_time = time.time()
        results = self._evaluate_model_on_test_data(
            model=model,
            test_loader=test_loader,
            model_type="transformer",
            class_names=data_processor.class_names
        )
        eval_time = time.time() - eval_start_time
        
        # Add metadata to results
        results["model_name"] = model_name
        results["model_type"] = "transformer"
        results["pretrained_model"] = pretrained_model
        results["evaluation_time"] = eval_time
        results["is_pretrained_evaluation"] = True
        
        # Save results
        results_path = os.path.join(model_path, "evaluation_results.json")
        with open(results_path, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for k, v in results.items():
                if isinstance(v, np.ndarray):
                    serializable_results[k] = v.tolist()
                elif isinstance(v, dict):
                    serializable_results[k] = {
                        sk: sv.tolist() if isinstance(sv, np.ndarray) else sv
                        for sk, sv in v.items()
                    }
                else:
                    serializable_results[k] = v
            
            json.dump(serializable_results, f, indent=2)
        
        # Print summary
        print(f"\nEvaluation Summary for {pretrained_model}:")
        print(f"Model stored as: {model_name}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print("Class-wise metrics:")
        for i, class_name in enumerate(results["classes"]):
            print(f"  {class_name}: Precision={results['precision'][i]:.4f}, Recall={results['recall'][i]:.4f}, F1={results['f1'][i]:.4f}")
        
        print(f"Evaluation completed in {eval_time:.2f}s")
        print(f"Results saved to {results_path}")
        
        return results
    
    def compare_models(self, model_names: List[str], metrics: List[str] = None, 
                    include_class_metrics: bool = True) -> Dict:
        """
        Compare multiple models based on their evaluation results.
        
        Args:
            model_names: List of model names to compare
            metrics: List of metrics to compare (default: accuracy, precision, recall, f1)
            include_class_metrics: Whether to include class-specific metrics
            
        Returns:
            Dictionary with comparison results
        """
        if not metrics:
            metrics = ["accuracy", "precision", "recall", "f1", "inference_time"]
        
        print(f"\n===== Model Comparison =====")
        print(f"Comparing models: {', '.join(model_names)}")
        
        # Collect results for each model
        comparison = {}
        model_results = {}
        
        for model_name in model_names:
            model_path = os.path.join(self.model_dir, model_name)
            
            # Check if model exists
            if not os.path.exists(model_path):
                print(f"Warning: Model '{model_name}' not found, skipping")
                continue
            
            # Load metadata
            metadata_path = os.path.join(model_path, "metadata.json")
            if not os.path.exists(metadata_path):
                print(f"Warning: Metadata for '{model_name}' not found, skipping")
                continue
                
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Check if model has evaluation results
            results_path = os.path.join(model_path, "evaluation_results.json")
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    evaluation = json.load(f)
                    
                # Store results
                model_type = metadata.get("model_type", "unknown")
                is_pretrained = metadata.get("is_pretrained_evaluation", False)
                
                # Format model name for display
                display_name = model_name
                if is_pretrained:
                    pretrained_model = metadata.get("pretrained_model", "unknown")
                    display_name = f"{model_name} ({pretrained_model})"
                    
                model_results[model_name] = {
                    "display_name": display_name,
                    "model_type": model_type,
                    "is_pretrained": is_pretrained,
                    "evaluation": evaluation
                }
            else:
                print(f"Warning: Evaluation results for '{model_name}' not found, skipping")
        
        if not model_results:
            print("No valid models found for comparison")
            return {}
        
        # Compare metrics
        print("\nComparison Results:")
        
        # Overall metrics table
        print("\nOverall Metrics:")
        headers = ["Model"] + [m.capitalize() for m in metrics if m != "precision" and m != "recall" and m != "f1"]
        
        # Format as a table
        rows = []
        for model_name, data in model_results.items():
            eval_data = data["evaluation"]
            row = [data["display_name"]]
            
            for metric in metrics:
                if metric == "precision" or metric == "recall" or metric == "f1":
                    # These are per-class metrics, handle separately
                    continue
                elif metric == "inference_time":
                    # Handle nested inference time dictionary
                    if "inference_time" in eval_data:
                        samples_per_sec = eval_data["inference_time"].get("samples_per_second", 0)
                        row.append(f"{samples_per_sec:.2f} samples/s")
                    else:
                        row.append("N/A")
                else:
                    # Regular metric
                    value = eval_data.get(metric, "N/A")
                    if isinstance(value, (int, float)):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
            
            rows.append(row)
        
        # Print table
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]
        
        # Print header
        header_row = "".join(str(headers[i]).ljust(col_widths[i]) for i in range(len(headers)))
        print(header_row)
        print("-" * sum(col_widths))
        
        # Print data rows
        for row in rows:
            data_row = "".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))
            print(data_row)
        
        # Class-specific metrics if requested
        if include_class_metrics:
            # Get class names from the first model
            first_model = next(iter(model_results.values()))
            class_names = first_model["evaluation"].get("classes", [])
            
            if class_names:
                print("\nClass-wise Metrics:")
                
                for class_idx, class_name in enumerate(class_names):
                    print(f"\nClass: {class_name}")
                    
                    # Headers for class metrics
                    class_headers = ["Model", "Precision", "Recall", "F1"]
                    class_rows = []
                    
                    for model_name, data in model_results.items():
                        eval_data = data["evaluation"]
                        row = [data["display_name"]]
                        
                        # Add precision, recall, F1
                        for metric in ["precision", "recall", "f1"]:
                            if metric in eval_data and str(class_idx) in eval_data[metric]:
                                value = eval_data[metric][str(class_idx)]
                                row.append(f"{value:.4f}")
                            else:
                                row.append("N/A")
                        
                        class_rows.append(row)
                    
                    # Print class metrics table
                    col_widths = [max(len(str(row[i])) for row in [class_headers] + class_rows) + 2 
                                for i in range(len(class_headers))]
                    
                    # Print header
                    header_row = "".join(str(class_headers[i]).ljust(col_widths[i]) for i in range(len(class_headers)))
                    print(header_row)
                    print("-" * sum(col_widths))
                    
                    # Print data rows
                    for row in class_rows:
                        data_row = "".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))
                        print(data_row)
        
        # Create comparison dictionary
        comparison["models"] = model_results
        comparison["metrics"] = metrics
        comparison["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        return comparison
    
    
    def train_lstm(self, data_processor: DataProcessor, model_name: str,
                num_layers: int = 2, hidden_size: int = 256, 
                embedding_dim: int = 300, dropout: float = 0.5,
                bidirectional: bool = True, rnn_type: str = "LSTM",
                batch_size: int = 32, lr: float = 0.001, 
                weight_decay: float = 0.0001, epochs: int = 5, 
                optimizer: str = "Adam", scheduler: str = "none",
                embedding_dropout: float = 0.2, use_attention: bool = False,
                clip_grad: bool = True, max_grad_norm: float = 1.0):
        """
        Train an LSTM/GRU model with advanced hyperparameters and GPU optimizations.
        
        Args:
            data_processor: Data processor with prepared data
            model_name: Name for the saved model
            num_layers: Number of RNN layers
            hidden_size: Size of RNN hidden state
            embedding_dim: Dimension of word embeddings
            dropout: Dropout probability for RNN layers
            bidirectional: Whether to use bidirectional RNN
            rnn_type: Type of RNN ("LSTM" or "GRU")
            batch_size: Batch size for training
            lr: Learning rate
            weight_decay: L2 regularization factor
            epochs: Number of training epochs
            optimizer: Optimizer type ("Adam", "AdamW", "SGD", or "RMSprop")
            scheduler: Learning rate scheduler ("none", "step", "cosine", "plateau")
            embedding_dropout: Dropout probability for embedding layer
            use_attention: Whether to use attention mechanism
            clip_grad: Whether to clip gradients during training
            max_grad_norm: Maximum norm for gradient clipping
        """
        # Prepare datasets
        dataloaders = data_processor.prepare_pytorch_datasets(batch_size=batch_size)
        train_loader = dataloaders["train"]
        # Use validation loader if available; otherwise fall back to test for validation curves
        val_loader = dataloaders["val"] if dataloaders.get("val") is not None else dataloaders.get("test")
        
        # Get vocabulary size and number of classes
        vocab_size = data_processor.vocab_size
        num_classes = len(data_processor.class_names)
        
        # Print model configuration
        print(f"\n===== {rnn_type} Model Configuration =====")
        print(f"Vocabulary Size: {vocab_size}")
        print(f"Number of Classes: {num_classes}")
        print(f"Embedding Dimension: {embedding_dim}")
        print(f"Embedding Dropout: {embedding_dropout}")
        print(f"Hidden Size: {hidden_size}")
        print(f"Number of Layers: {num_layers}")
        print(f"RNN Type: {rnn_type}")
        print(f"Bidirectional: {bidirectional}")
        print(f"Attention Mechanism: {use_attention}")
        print(f"Dropout Rate: {dropout}")
        print(f"Batch Size: {batch_size}")
        print(f"Learning Rate: {lr}")
        print(f"Weight Decay: {weight_decay}")
        print(f"Gradient Clipping: {clip_grad} (max norm: {max_grad_norm})")
        print(f"Epochs: {epochs}")
        print(f"Optimizer: {optimizer}")
        print(f"Scheduler: {scheduler}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"=====================================\n")
        
        # Create model with enhanced options
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
            use_attention=use_attention
        )
        
        # Move model to device
        model.to(self.device)
        
        # Print model summary (parameter count)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")
        
        # Create optimizer
        if optimizer == "Adam":
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "AdamW":
            opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "SGD":
            opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer == "RMSprop":
            opt = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Create learning rate scheduler
        if scheduler == "step":
            lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        elif scheduler == "cosine":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        elif scheduler == "plateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, verbose=True)
        else:
            lr_scheduler = None
        
        # Create loss function
        # Use class weights to handle imbalance
        try:
            _, class_weights = data_processor.get_class_weights()
            class_weights = class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        except Exception:
            criterion = nn.CrossEntropyLoss()
        
        # Save hyperparameters
        hyperparams = {
            "model_type": "advanced_rnn",
            "rnn_type": rnn_type,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "embedding_dim": embedding_dim,
            "dropout": dropout,
            "embedding_dropout": embedding_dropout,
            "bidirectional": bidirectional,
            "use_attention": use_attention,
            "vocab_size": vocab_size,
            "num_classes": num_classes,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "clip_grad": clip_grad,
            "max_grad_norm": max_grad_norm,
            "epochs": epochs
        }
        
        self._save_hyperparameters(model_name, hyperparams)
        
        # Train the model with advanced options
        self._train_advanced_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=opt,
            epochs=epochs,
            model_name=model_name,
            scheduler=lr_scheduler,
            clip_grad=clip_grad,
            max_grad_norm=max_grad_norm
        )
    
    def train_cnn(self, data_processor: DataProcessor, model_name: str,
                filter_sizes: List[int] = [3, 4, 5], num_filters: int = 100,
                embedding_dim: int = 300, dropout: float = 0.5,
                embedding_dropout: float = 0.2, activation: str = "relu",
                batch_norm: bool = False, pool_type: str = "max",
                batch_size: int = 32, lr: float = 0.001, 
                weight_decay: float = 0.0001, epochs: int = 5, 
                optimizer: str = "Adam", scheduler: str = "none",
                clip_grad: bool = False, max_grad_norm: float = 1.0):
        """
        Train a CNN model with advanced hyperparameters and GPU optimizations.
        
        Args:
            data_processor: Data processor with prepared data
            model_name: Name for the saved model
            filter_sizes: List of filter sizes
            num_filters: Number of filters per filter size
            embedding_dim: Dimension of word embeddings
            dropout: Dropout probability for CNN layers
            embedding_dropout: Dropout probability for embedding layer
            activation: Activation function ("relu", "leaky_relu", "tanh", "elu")
            batch_norm: Whether to use batch normalization
            pool_type: Pooling type ("max", "avg", "adaptive")
            batch_size: Batch size for training
            lr: Learning rate
            weight_decay: L2 regularization factor
            epochs: Number of training epochs
            optimizer: Optimizer type ("Adam", "AdamW", "SGD", or "RMSprop")
            scheduler: Learning rate scheduler ("none", "step", "cosine", "plateau")
            clip_grad: Whether to clip gradients during training
            max_grad_norm: Maximum norm for gradient clipping
        """
        # Prepare datasets
        dataloaders = data_processor.prepare_pytorch_datasets(batch_size=batch_size)
        train_loader = dataloaders["train"]
        # Use validation loader if available; otherwise fall back to test for validation curves
        val_loader = dataloaders["val"] if dataloaders.get("val") is not None else dataloaders.get("test")
        
        # Get vocabulary size and number of classes
        vocab_size = data_processor.vocab_size
        num_classes = len(data_processor.class_names)
        
        # Print model configuration
        print(f"\n===== CNN Model Configuration =====")
        print(f"Vocabulary Size: {vocab_size}")
        print(f"Number of Classes: {num_classes}")
        print(f"Embedding Dimension: {embedding_dim}")
        print(f"Embedding Dropout: {embedding_dropout}")
        print(f"Filter Sizes: {filter_sizes}")
        print(f"Number of Filters: {num_filters}")
        print(f"Activation Function: {activation}")
        print(f"Batch Normalization: {batch_norm}")
        print(f"Pooling Type: {pool_type}")
        print(f"Dropout Rate: {dropout}")
        print(f"Batch Size: {batch_size}")
        print(f"Learning Rate: {lr}")
        print(f"Weight Decay: {weight_decay}")
        print(f"Gradient Clipping: {clip_grad} (max norm: {max_grad_norm})")
        print(f"Epochs: {epochs}")
        print(f"Optimizer: {optimizer}")
        print(f"Scheduler: {scheduler}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"===================================\n")
        
        # Create model with enhanced options
        model = AdvancedCNNModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            num_classes=num_classes,
            dropout=dropout,
            embedding_dropout=embedding_dropout,
            activation=activation,
            batch_norm=batch_norm,
            pool_type=pool_type
        )
        
        # Move model to device
        model.to(self.device)
        
        # Print model summary (parameter count)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")
        
        # Create optimizer
        if optimizer == "Adam":
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "AdamW":
            opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "SGD":
            opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer == "RMSprop":
            opt = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Create learning rate scheduler
        if scheduler == "step":
            lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        elif scheduler == "cosine":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        elif scheduler == "plateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, verbose=True)
        else:
            lr_scheduler = None
        
        # Create loss function
        # Use class weights to handle imbalance
        try:
            _, class_weights = data_processor.get_class_weights()
            class_weights = class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        except Exception:
            criterion = nn.CrossEntropyLoss()
        
        # Save hyperparameters
        hyperparams = {
            "model_type": "advanced_cnn",
            "filter_sizes": filter_sizes,
            "num_filters": num_filters,
            "embedding_dim": embedding_dim,
            "dropout": dropout,
            "embedding_dropout": embedding_dropout,
            "activation": activation,
            "batch_norm": batch_norm,
            "pool_type": pool_type,
            "vocab_size": vocab_size,
            "num_classes": num_classes,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "clip_grad": clip_grad,
            "max_grad_norm": max_grad_norm,
            "epochs": epochs
        }
        
        self._save_hyperparameters(model_name, hyperparams)
        
        # Train the model with advanced options
        self._train_advanced_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=opt,
            epochs=epochs,
            model_name=model_name,
            scheduler=lr_scheduler,
            clip_grad=clip_grad,
            max_grad_norm=max_grad_norm
        )

    def _train_advanced_model(self, model: nn.Module, train_loader: DataLoader,
                            val_loader: Optional[DataLoader], criterion: nn.Module,
                            optimizer: optim.Optimizer, epochs: int, model_name: str,
                            scheduler: Optional[Any] = None, clip_grad: bool = False,
                            max_grad_norm: float = 1.0):
        """
        Advanced training loop for PyTorch models with GPU optimization, LR scheduling and gradient clipping.
        
        Args:
            model: PyTorch model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            epochs: Number of training epochs
            model_name: Name for the saved model
            scheduler: Learning rate scheduler (optional)
            clip_grad: Whether to clip gradients
            max_grad_norm: Maximum norm for gradient clipping
        """
        # Initialize metrics tracking
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'lr': []
        }
        
        best_val_acc = 0.0
        total_batches = len(train_loader)
        start_time = time.time()
        
        # Set up AMP (Automatic Mixed Precision) for faster training if available
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Notify training start
        self._call_callback('on_training_start', epochs, total_batches)
        
        # Log initial GPU stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        
        # Track learning rates
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Track epoch time
            epoch_start = time.time()
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Move inputs and labels to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                if self.use_amp:
                    # Forward pass with automatic mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    # Backward and optimize with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping if enabled
                    if clip_grad:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backward and optimize
                    loss.backward()
                    
                    # Gradient clipping if enabled
                    if clip_grad:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                
                # Track metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Notify batch end
                self._call_callback('on_batch_end', batch_idx, total_batches, epoch, epochs)
                
                # Report progress every 50 batches
                if batch_idx % 50 == 0:
                    batch_loss = loss.item()
                    batch_acc = (predicted == labels).sum().item() / labels.size(0)
                    curr_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{total_batches} | Loss: {batch_loss:.4f} | Acc: {batch_acc:.4f} | LR: {curr_lr:.6f}")
            
            # Calculate training metrics
            train_loss = running_loss / total_batches
            train_acc = correct / total
            
            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            
            if val_loader:
                val_loss, val_acc = self._validate_model(model, val_loader, criterion)
                
                # Save model if it's the best so far
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_model(model, model_name, 'pytorch')
                    print(f"New best validation accuracy: {val_acc:.4f} - Saved model to {model_name}")
            else:
                # Save model at each epoch if no validation data
                self._save_model(model, model_name, 'pytorch')
            
            # Step the scheduler if provided
            if scheduler is not None:
                # Check if it's a ReduceLROnPlateau scheduler which requires a validation metric
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = val_acc if val_loader else train_acc
                    scheduler.step(metric)
                else:
                    scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Update history
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            
            if val_loader:
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
            
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            
            # Report GPU memory usage
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1e6  # MB
                peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
                print(f"GPU memory: current={current_memory:.2f} MB, peak={peak_memory:.2f} MB")
            
            # Notify epoch end
            metrics = {
                'loss': train_loss,
                'accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'epoch_time': epoch_time,
                'learning_rate': current_lr
            }
            self._call_callback('on_epoch_end', epoch, epochs, metrics)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}", end="")
            if val_loader:
                print(f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}", end="")
            print(f" | LR: {current_lr:.6f}")
        
        # Training complete
        training_time = time.time() - start_time
        
        # Save history
        self._save_training_history(model_name, history)
        
        # Report final GPU memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
            print(f"Peak GPU memory usage during training: {peak_memory:.2f} MB")
            torch.cuda.empty_cache()
        
        # Notify training end
        final_metrics = {
            'loss': history['loss'][-1],
            'accuracy': history['accuracy'][-1],
            'val_loss': history['val_loss'][-1] if history['val_loss'] else 0.0,
            'val_accuracy': history['val_accuracy'][-1] if history['val_accuracy'] else 0.0,
            'training_time': training_time,
            'learning_rate': history['lr'][-1]
        }
        self._call_callback('on_training_end', model_name, final_metrics, training_time)
        
        # Print final summary
        print(f"\nTraining completed in {training_time:.2f}s")
        print(f"Final training accuracy: {final_metrics['accuracy']:.4f}")
        if val_loader:
            print(f"Final validation accuracy: {final_metrics['val_accuracy']:.4f}")
        print(f"Model saved to {os.path.join(self.model_dir, model_name)}")

    def train_transformer(self, data_processor: DataProcessor, model_name: str,
                        pretrained_model: str = "cardiffnlp/twitter-roberta-base-sentiment",
                        finetune: bool = True, freeze_layers: int = 0,
                        dropout: float = 0.1, attention_dropout: float = 0.1,
                        classifier_dropout: float = 0.1, hidden_dropout: float = 0.1,
                        batch_size: int = 16, lr: float = 0.00005, 
                        weight_decay: float = 0.01, warmup_steps: int = 500,
                        epochs: int = 3, optimizer: str = "AdamW",
                        scheduler: str = "linear", max_seq_length: int = 128,
                        gradient_accumulation_steps: int = 1, fp16_training: bool = True,
                        clip_grad: bool = True, max_grad_norm: float = 1.0):
        """
        Train a transformer model with comprehensive hyperparameters and GPU optimizations.
        
        Args:
            data_processor: Data processor with prepared data
            model_name: Name for the saved model
            pretrained_model: Name of pretrained model to use
            finetune: Whether to finetune the model or freeze parts of it
            freeze_layers: Number of layers to freeze from the bottom
            dropout: General dropout probability
            attention_dropout: Dropout for attention layers
            classifier_dropout: Dropout for classification head
            hidden_dropout: Dropout for hidden layers
            batch_size: Batch size for training
            lr: Learning rate
            weight_decay: L2 regularization factor
            warmup_steps: Number of warmup steps for learning rate scheduler
            epochs: Number of training epochs
            optimizer: Optimizer type ("AdamW", "Adam", "Adafactor")
            scheduler: Learning rate scheduler ("linear", "cosine", "constant", "polynomial")
            max_seq_length: Maximum sequence length for tokenization
            gradient_accumulation_steps: Number of steps to accumulate gradients
            fp16_training: Whether to use mixed precision training (fp16)
            clip_grad: Whether to clip gradients during training
            max_grad_norm: Maximum norm for gradient clipping
        """
        # Prepare datasets with appropriate tokenizer
        dataloaders = data_processor.prepare_pytorch_datasets(
            batch_size=batch_size,
            tokenizer_name=pretrained_model,
            max_length=max_seq_length
        )
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        
        # Get number of classes
        num_classes = len(data_processor.class_names)
        
        # Use mixed precision if requested and available
        use_fp16 = fp16_training and self.use_amp
        
        # Print model configuration
        print(f"\n===== Transformer Model Configuration =====")
        print(f"Pre-trained Model: {pretrained_model}")
        print(f"Number of Classes: {num_classes}")
        print(f"Fine-tuning Strategy: {'Full fine-tuning' if finetune else 'Feature extraction'}")
        print(f"Frozen Layers: {freeze_layers}")
        print(f"Dropout Rates:")
        print(f"  General: {dropout}")
        print(f"  Attention: {attention_dropout}")
        print(f"  Classifier: {classifier_dropout}")
        print(f"  Hidden: {hidden_dropout}")
        print(f"Max Sequence Length: {max_seq_length}")
        print(f"Training Parameters:")
        print(f"  Batch Size: {batch_size}")
        print(f"  Gradient Accumulation Steps: {gradient_accumulation_steps}")
        print(f"  Effective Batch Size: {batch_size * gradient_accumulation_steps}")
        print(f"  Learning Rate: {lr}")
        print(f"  Weight Decay: {weight_decay}")
        print(f"  Warmup Steps: {warmup_steps}")
        print(f"  Mixed Precision (FP16): {use_fp16}")
        print(f"  Gradient Clipping: {clip_grad} (max norm: {max_grad_norm})")
        print(f"  Epochs: {epochs}")
        print(f"  Optimizer: {optimizer}")
        print(f"  Scheduler: {scheduler}")
        print(f"  Device: {self.device}")
        print(f"============================================\n")
        
        # Load pretrained model with configuration options
        config_kwargs = {
            "hidden_dropout_prob": hidden_dropout,
            "attention_probs_dropout_prob": attention_dropout,
            "classifier_dropout": classifier_dropout
        }
        
        try:
            print(f"Loading pre-trained model: {pretrained_model}")
            from transformers import AutoConfig
            
            # Load configuration first
            config = AutoConfig.from_pretrained(pretrained_model, **config_kwargs)
            config.num_labels = num_classes
            
            # Load model with updated configuration
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model,
                config=config
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load pre-trained model: {str(e)}")
        
        # Handle layer freezing if specified
        if not finetune or freeze_layers > 0:
            print(f"{'Freezing encoder entirely' if not finetune else f'Freezing first {freeze_layers} layers'}")
            
            # Freeze specific components based on model type
            if hasattr(model, "base_model"):
                if not finetune:
                    # Freeze the entire encoder
                    for param in model.base_model.parameters():
                        param.requires_grad = False
                elif freeze_layers > 0:
                    # Freeze only specific layers (BERT-like models)
                    if hasattr(model.base_model, "encoder") and hasattr(model.base_model.encoder, "layer"):
                        # For BERT, RoBERTa, etc.
                        for i, layer in enumerate(model.base_model.encoder.layer):
                            if i < freeze_layers:
                                for param in layer.parameters():
                                    param.requires_grad = False
                    # Handle other architectures
                    elif hasattr(model.base_model, "layers"):
                        # For some other models
                        for i, layer in enumerate(model.base_model.layers):
                            if i < freeze_layers:
                                for param in layer.parameters():
                                    param.requires_grad = False
        
        # Move model to device
        model.to(self.device)
        
        # Print model summary (parameter count)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,} - {trainable_params/total_params:.1%})")
        
        # Create optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        
        if optimizer == "AdamW":
            opt = optim.AdamW(optimizer_grouped_parameters, lr=lr)
        elif optimizer == "Adam":
            opt = optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif optimizer == "Adafactor":
            from transformers import Adafactor
            opt = Adafactor(optimizer_grouped_parameters, lr=lr, relative_step=False, scale_parameter=False)
        else:
            opt = optim.AdamW(optimizer_grouped_parameters, lr=lr)
        
        # Create learning rate scheduler
        if scheduler == "linear":
            total_steps = len(train_loader) * epochs // gradient_accumulation_steps
            from transformers import get_linear_schedule_with_warmup
            lr_scheduler = get_linear_schedule_with_warmup(
                opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        elif scheduler == "cosine":
            total_steps = len(train_loader) * epochs // gradient_accumulation_steps
            from transformers import get_cosine_schedule_with_warmup
            lr_scheduler = get_cosine_schedule_with_warmup(
                opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        elif scheduler == "polynomial":
            total_steps = len(train_loader) * epochs // gradient_accumulation_steps
            from transformers import get_polynomial_decay_schedule_with_warmup
            lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        elif scheduler == "constant":
            from transformers import get_constant_schedule_with_warmup
            lr_scheduler = get_constant_schedule_with_warmup(
                opt, num_warmup_steps=warmup_steps
            )
        else:
            lr_scheduler = None
        
        # Create loss function
        criterion = nn.CrossEntropyLoss()
        
        # Save hyperparameters
        hyperparams = {
            "model_type": "transformer",
            "pretrained_model": pretrained_model,
            "finetune": finetune,
            "freeze_layers": freeze_layers,
            "dropout": dropout,
            "attention_dropout": attention_dropout,
            "classifier_dropout": classifier_dropout,
            "hidden_dropout": hidden_dropout,
            "num_classes": num_classes,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "max_seq_length": max_seq_length,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "fp16_training": use_fp16,
            "clip_grad": clip_grad,
            "max_grad_norm": max_grad_norm,
            "epochs": epochs
        }
        
        self._save_hyperparameters(model_name, hyperparams)
        
        # Train the model with advanced options
        self._train_advanced_transformer_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=opt,
            epochs=epochs,
            model_name=model_name,
            scheduler=lr_scheduler,
            clip_grad=clip_grad,
            max_grad_norm=max_grad_norm,
            use_fp16=use_fp16,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
    def _train_advanced_transformer_model(self, model: nn.Module, train_loader: DataLoader,
                                        val_loader: Optional[DataLoader], criterion: nn.Module,
                                        optimizer: optim.Optimizer, epochs: int, model_name: str,
                                        scheduler: Optional[Any] = None, clip_grad: bool = True,
                                        max_grad_norm: float = 1.0, use_fp16: bool = True,
                                        gradient_accumulation_steps: int = 1):
        """
        Advanced training loop for transformer models with gradient accumulation and GPU optimizations.
        
        Args:
            model: Transformer model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            epochs: Number of training epochs
            model_name: Name for the saved model
            scheduler: Learning rate scheduler (optional)
            clip_grad: Whether to clip gradients
            max_grad_norm: Maximum norm for gradient clipping
            use_fp16: Whether to use mixed precision training
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        # Initialize metrics tracking
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'lr': []
        }
        
        best_val_acc = 0.0
        total_batches = len(train_loader)
        start_time = time.time()
        
        # Set up AMP (Automatic Mixed Precision) for faster training if available
        scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
        
        # Notify training start
        self._call_callback('on_training_start', epochs, total_batches)
        
        # Log initial GPU stats and training setup
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        
        print(f"Training with {gradient_accumulation_steps} gradient accumulation steps")
        print(f"Effective batch size: {train_loader.batch_size * gradient_accumulation_steps}")
        
        # Track learning rates
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Track epoch time
            epoch_start = time.time()
            
            # Reset gradients at start of epoch for clarity
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                # Unpack batch data
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                # Determine if this is an optimization step
                is_optimization_step = (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx + 1 == len(train_loader)
                
                # Scale the loss by the number of accumulation steps for consistent gradient scale
                loss_scale = 1.0 / gradient_accumulation_steps
                
                if use_fp16:
                    # Forward pass with automatic mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids, attention_mask=attention_mask)
                        loss = criterion(outputs.logits, labels) * loss_scale
                    
                    # Backward with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Only optimize at the specified interval
                    if is_optimization_step:
                        # Unscale gradients for clipping
                        if clip_grad:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        
                        # Optimizer step with scaler
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # Standard forward pass
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels) * loss_scale
                    
                    # Backward
                    loss.backward()
                    
                    # Only optimize at the specified interval
                    if is_optimization_step:
                        # Gradient clipping if enabled
                        if clip_grad:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        
                        # Optimizer step
                        optimizer.step()
                        optimizer.zero_grad()
                
                # Track metrics (using the full, unscaled loss)
                running_loss += (loss.item() / loss_scale)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Step the scheduler if it's an iteration-based scheduler
                if scheduler is not None and is_optimization_step and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if hasattr(scheduler, "step_every_batch") and scheduler.step_every_batch:
                        scheduler.step()
                
                # Notify batch end
                self._call_callback('on_batch_end', batch_idx, total_batches, epoch, epochs)
                
                # Report progress periodically
                if batch_idx % 10 == 0 or is_optimization_step:
                    batch_loss = loss.item() / loss_scale
                    batch_acc = (predicted == labels).sum().item() / labels.size(0)
                    curr_lr = optimizer.param_groups[0]['lr']
                    steps_info = f"(Opt step: {(batch_idx + 1) // gradient_accumulation_steps})" if gradient_accumulation_steps > 1 else ""
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{total_batches} {steps_info} | Loss: {batch_loss:.4f} | Acc: {batch_acc:.4f} | LR: {curr_lr:.6f}")
            
            # Calculate training metrics
            train_loss = running_loss / total_batches
            train_acc = correct / total
            
            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            
            if val_loader:
                val_loss, val_acc = self._validate_transformer_model(model, val_loader, criterion, use_fp16)
                
                # Save model if it's the best so far
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_model(model, model_name, 'transformer')
                    print(f"New best validation accuracy: {val_acc:.4f} - Saved model to {model_name}")
            else:
                # Save model at each epoch if no validation data
                self._save_model(model, model_name, 'transformer')
            
            # Step the learning rate scheduler if it's epoch-based or requires validation metric
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau requires a validation metric
                    metric = val_acc if val_loader else train_acc
                    scheduler.step(metric)
                elif not hasattr(scheduler, "step_every_batch") or not scheduler.step_every_batch:
                    # Only step if it's an epoch-based scheduler
                    scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Update history
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            
            if val_loader:
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
            
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            
            # Report GPU memory usage
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1e6  # MB
                peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
                print(f"GPU memory: current={current_memory:.2f} MB, peak={peak_memory:.2f} MB")
            
            # Notify epoch end
            metrics = {
                'loss': train_loss,
                'accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'epoch_time': epoch_time,
                'learning_rate': current_lr
            }
            self._call_callback('on_epoch_end', epoch, epochs, metrics)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}", end="")
            if val_loader:
                print(f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}", end="")
            print(f" | LR: {current_lr:.6f}")
        
        # Training complete
        training_time = time.time() - start_time
        
        # Save history
        self._save_training_history(model_name, history)
        
        # Report final GPU memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
            print(f"Peak GPU memory usage during training: {peak_memory:.2f} MB")
            torch.cuda.empty_cache()
        
        # Notify training end
        final_metrics = {
            'loss': history['loss'][-1],
            'accuracy': history['accuracy'][-1],
            'val_loss': history['val_loss'][-1] if history['val_loss'] else 0.0,
            'val_accuracy': history['val_accuracy'][-1] if history['val_accuracy'] else 0.0,
            'training_time': training_time,
            'learning_rate': history['lr'][-1]
        }
        self._call_callback('on_training_end', model_name, final_metrics, training_time)
        
        # Print final summary
        print(f"\nTraining completed in {training_time:.2f}s")
        print(f"Final training accuracy: {final_metrics['accuracy']:.4f}")
        if val_loader:
            print(f"Final validation accuracy: {final_metrics['val_accuracy']:.4f}")
        print(f"Model saved to {os.path.join(self.model_dir, model_name)}")
    
    def _train_pytorch_model(self, model: nn.Module, train_loader: DataLoader,
                            val_loader: Optional[DataLoader], criterion: nn.Module,
                            optimizer: optim.Optimizer, epochs: int, model_name: str):
        """
        Generic training loop for PyTorch models with full GPU optimization.
        
        Args:
            model: PyTorch model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            epochs: Number of training epochs
            model_name: Name for the saved model
        """
        # Initialize metrics tracking
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_acc = 0.0
        total_batches = len(train_loader)
        start_time = time.time()
        
        # Set up AMP (Automatic Mixed Precision) for faster training if available
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Notify training start
        self._call_callback('on_training_start', epochs, total_batches)
        
        # Log initial GPU stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Track epoch time
            epoch_start = time.time()
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Move inputs and labels to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                if self.use_amp:
                    # Forward pass with automatic mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    # Backward and optimize with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backward and optimize
                    loss.backward()
                    optimizer.step()
                
                # Track metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Notify batch end
                self._call_callback('on_batch_end', batch_idx, total_batches, epoch, epochs)
                
                # Report progress every 50 batches
                if batch_idx % 50 == 0:
                    batch_loss = loss.item()
                    batch_acc = (predicted == labels).sum().item() / labels.size(0)
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{total_batches} | Loss: {batch_loss:.4f} | Acc: {batch_acc:.4f}")
            
            # Calculate training metrics
            train_loss = running_loss / total_batches
            train_acc = correct / total
            
            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            
            if val_loader:
                val_loss, val_acc = self._validate_model(model, val_loader, criterion)
                
                # Save model if it's the best so far
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_model(model, model_name, 'pytorch')
                    print(f"New best validation accuracy: {val_acc:.4f} - Saved model to {model_name}")
            else:
                # Save model at each epoch if no validation data
                self._save_model(model, model_name, 'pytorch')
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Update history
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            
            if val_loader:
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
            
            # Report GPU memory usage
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1e6  # MB
                peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
                print(f"GPU memory: current={current_memory:.2f} MB, peak={peak_memory:.2f} MB")
            
            # Notify epoch end
            metrics = {
                'loss': train_loss,
                'accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'epoch_time': epoch_time
            }
            self._call_callback('on_epoch_end', epoch, epochs, metrics)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}", end="")
            if val_loader:
                print(f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            else:
                print("")
        
        # Training complete
        training_time = time.time() - start_time
        
        # Save history
        self._save_training_history(model_name, history)
        
        # Report final GPU memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
            print(f"Peak GPU memory usage during training: {peak_memory:.2f} MB")
            torch.cuda.empty_cache()
        
        # Notify training end
        final_metrics = {
            'loss': history['loss'][-1],
            'accuracy': history['accuracy'][-1],
            'val_loss': history['val_loss'][-1] if history['val_loss'] else 0.0,
            'val_accuracy': history['val_accuracy'][-1] if history['val_accuracy'] else 0.0,
            'training_time': training_time
        }
        self._call_callback('on_training_end', model_name, final_metrics, training_time)
        
        # Print final summary
        print(f"\nTraining completed in {training_time:.2f}s")
        print(f"Final training accuracy: {final_metrics['accuracy']:.4f}")
        if val_loader:
            print(f"Final validation accuracy: {final_metrics['val_accuracy']:.4f}")
        print(f"Model saved to {os.path.join(self.model_dir, model_name)}")
    
    def _train_transformer_model(self, model: nn.Module, train_loader: DataLoader,
                                val_loader: Optional[DataLoader], criterion: nn.Module,
                                optimizer: optim.Optimizer, epochs: int, model_name: str):
        """
        Optimized training loop for transformer models with GPU acceleration.
        
        Args:
            model: Transformer model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            epochs: Number of training epochs
            model_name: Name for the saved model
        """
        # Initialize metrics tracking
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_acc = 0.0
        total_batches = len(train_loader)
        start_time = time.time()
        
        # Set up AMP (Automatic Mixed Precision) for faster training if available
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Notify training start
        self._call_callback('on_training_start', epochs, total_batches)
        
        # Log initial GPU stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Track epoch time
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                # Unpack batch data
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                if self.use_amp:
                    # Forward pass with automatic mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids, attention_mask=attention_mask)
                        loss = criterion(outputs.logits, labels)
                    
                    # Backward and optimize with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard forward pass
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                    
                    # Backward and optimize
                    loss.backward()
                    optimizer.step()
                
                # Track metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Notify batch end
                self._call_callback('on_batch_end', batch_idx, total_batches, epoch, epochs)
                
# Report progress every 20 batches (transformers tend to have fewer, larger batches)
                if batch_idx % 20 == 0:
                    batch_loss = loss.item()
                    batch_acc = (predicted == labels).sum().item() / labels.size(0)
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{total_batches} | Loss: {batch_loss:.4f} | Acc: {batch_acc:.4f}")
            
            # Calculate training metrics
            train_loss = running_loss / total_batches
            train_acc = correct / total
            
            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            
            if val_loader:
                val_loss, val_acc = self._validate_transformer_model(model, val_loader, criterion)
                
                # Save model if it's the best so far
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_model(model, model_name, 'transformer')
                    print(f"New best validation accuracy: {val_acc:.4f} - Saved model to {model_name}")
            else:
                # Save model at each epoch if no validation data
                self._save_model(model, model_name, 'transformer')
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Update history
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            
            if val_loader:
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
            
            # Report GPU memory usage
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1e6  # MB
                peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
                print(f"GPU memory: current={current_memory:.2f} MB, peak={peak_memory:.2f} MB")
            
            # Notify epoch end
            metrics = {
                'loss': train_loss,
                'accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'epoch_time': epoch_time
            }
            self._call_callback('on_epoch_end', epoch, epochs, metrics)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}", end="")
            if val_loader:
                print(f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            else:
                print("")
        
        # Training complete
        training_time = time.time() - start_time
        
        # Save history
        self._save_training_history(model_name, history)
        
        # Report final GPU memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
            print(f"Peak GPU memory usage during training: {peak_memory:.2f} MB")
            torch.cuda.empty_cache()
        
        # Notify training end
        final_metrics = {
            'loss': history['loss'][-1],
            'accuracy': history['accuracy'][-1],
            'val_loss': history['val_loss'][-1] if history['val_loss'] else 0.0,
            'val_accuracy': history['val_accuracy'][-1] if history['val_accuracy'] else 0.0,
            'training_time': training_time
        }
        self._call_callback('on_training_end', model_name, final_metrics, training_time)
        
        # Print final summary
        print(f"\nTraining completed in {training_time:.2f}s")
        print(f"Final training accuracy: {final_metrics['accuracy']:.4f}")
        if val_loader:
            print(f"Final validation accuracy: {final_metrics['val_accuracy']:.4f}")
        print(f"Model saved to {os.path.join(self.model_dir, model_name)}")
    
    def _validate_model(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate model performance with GPU optimization.
        
        Args:
            model: PyTorch model
            val_loader: DataLoader for validation data
            criterion: Loss function
            
        Returns:
            Tuple of (validation loss, validation accuracy)
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Use mixed precision if available
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def _validate_transformer_model(self, model: nn.Module, val_loader: DataLoader, 
                                   criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate transformer model performance with GPU optimization.
        
        Args:
            model: Transformer model
            val_loader: DataLoader for validation data
            criterion: Loss function
            
        Returns:
            Tuple of (validation loss, validation accuracy)
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                # Use mixed precision if available
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids, attention_mask=attention_mask)
                        loss = criterion(outputs.logits, labels)
                else:
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def _save_model(self, model: nn.Module, model_name: str, model_type: str):
        """
        Save model to disk with appropriate format based on model type.
        
        Args:
            model: PyTorch model
            model_name: Name for the saved model
            model_type: Type of model ('pytorch' or 'transformer')
        """
        # Create model directory if it doesn't exist
        model_path = os.path.join(self.model_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Save model state dictionary or the entire model depending on type
        if model_type == 'pytorch':
            torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
            print(f"Saved PyTorch model state to {os.path.join(model_path, 'model.pt')}")
        else:
            model.save_pretrained(model_path)
            print(f"Saved transformer model to {model_path}")
        
        # Save model metadata
        metadata = {
            "model_type": model_type,
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "device": str(self.device)
        }
        
        with open(os.path.join(model_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _save_hyperparameters(self, model_name: str, hyperparams: Dict):
        """
        Save model hyperparameters to disk.
        
        Args:
            model_name: Name of the model
            hyperparams: Dictionary of hyperparameters
        """
        model_path = os.path.join(self.model_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        with open(os.path.join(model_path, "hyperparams.json"), "w") as f:
            json.dump(hyperparams, f, indent=2)
    
    def _save_training_history(self, model_name: str, history: Dict):
        """
        Save training history to disk.
        
        Args:
            model_name: Name of the model
            history: Dictionary with training history
        """
        model_path = os.path.join(self.model_dir, model_name)
        
        with open(os.path.join(model_path, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available trained models.
        
        Returns:
            List of model names
        """
        models = []
        
        for item in os.listdir(self.model_dir):
            item_path = os.path.join(self.model_dir, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, "metadata.json")
                if os.path.exists(metadata_path):
                    models.append(item)
        
        return models
    
    def evaluate_model(self, model_name: str, data_processor: DataProcessor) -> Dict:
        """
        Evaluate a trained model on test data with detailed metrics.
        
        Args:
            model_name: Name of the model to evaluate
            data_processor: Data processor with test data
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        print(f"\n===== Evaluating Model: {model_name} =====")
        model_path = os.path.join(self.model_dir, model_name)
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model '{model_name}' not found")
        
        # Load model metadata
        with open(os.path.join(model_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        model_type = metadata["model_type"]
        print(f"Model type: {model_type}")
        
        # Load training history if available
        history = None
        history_path = os.path.join(model_path, "history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
            print("Loaded training history")
        
        # Load hyperparameters if available
        hyperparams = None
        hyperparams_path = os.path.join(model_path, "hyperparams.json")
        if os.path.exists(hyperparams_path):
            with open(hyperparams_path, "r") as f:
                hyperparams = json.load(f)
            print("Loaded hyperparameters")
        
        # Load model
        print(f"Loading model from {model_path}...")
        if model_type == "transformer":
            # For transformer models, try local dir first; if missing weights, fall back to HF ID
            is_pretrained_eval = metadata.get("is_pretrained_evaluation", False)
            pretrained_id = metadata.get("pretrained_model")
            num_classes = len(data_processor.class_names)
            try:
                if not is_pretrained_eval:
                    # Attempt to load from local directory
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    print("Loaded transformer model from local directory")
                else:
                    raise RuntimeError("Pretrained-eval entry has no local weights; loading from Hugging Face")
            except Exception as e:
                if not pretrained_id:
                    raise
                # Load from Hugging Face using recorded model ID with label count override
                print(f"Falling back to Hugging Face model: {pretrained_id} ({e})")
                try:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(pretrained_id)
                    _orig = getattr(config, "num_labels", -1)
                    print(f"Original HF num_labels={_orig}, overriding to {num_classes}")
                    config.num_labels = num_classes
                    model = AutoModelForSequenceClassification.from_pretrained(
                        pretrained_id,
                        config=config,
                        ignore_mismatched_sizes=True
                    )
                except Exception as e2:
                    # Final fallback
                    model = AutoModelForSequenceClassification.from_pretrained(
                        pretrained_id,
                        num_labels=num_classes,
                        ignore_mismatched_sizes=True
                    )
                    print(f"Loaded HF model with direct num_labels override ({e2})")
        else:
            # PyTorch family (LSTM/CNN/advanced variants)
            arch = None
            if hyperparams and "model_type" in hyperparams:
                arch = hyperparams["model_type"]
            else:
                # If metadata stored advanced type, prefer it
                if model_type in ("lstm", "cnn", "advanced_rnn", "advanced_cnn"):
                    arch = model_type
                else:
                    if "lstm" in model_name.lower():
                        arch = "lstm"
                    elif "cnn" in model_name.lower():
                        arch = "cnn"

            # If still unknown, try to infer from checkpoint keys
            if arch is None:
                sd_path = os.path.join(model_path, "model.pt")
                if os.path.exists(sd_path):
                    try:
                        state_dict = torch.load(sd_path, map_location=self.device)
                        if any(k.startswith("convs.") for k in state_dict.keys()):
                            arch = "advanced_cnn"
                        elif any(k.startswith("rnn.") or k.startswith("lstm.") for k in state_dict.keys()):
                            arch = "advanced_rnn"
                    except Exception:
                        pass

            def _load_by_arch(a: str):
                if a == "lstm":
                    return self._load_lstm_model(model_path, data_processor, hyperparams)
                if a == "cnn":
                    return self._load_cnn_model(model_path, data_processor, hyperparams)
                if a == "advanced_rnn":
                    return self._load_advanced_rnn_model(model_path, data_processor, hyperparams)
                if a == "advanced_cnn":
                    return self._load_advanced_cnn_model(model_path, data_processor, hyperparams)
                raise ValueError(f"Unknown PyTorch model type: {a}")

            try:
                model = _load_by_arch(arch)
            except (RuntimeError, ValueError) as e:
                # Try alternative arch if mismatch between saved weights and chosen arch
                sd_path = os.path.join(model_path, "model.pt")
                alt = None
                if os.path.exists(sd_path):
                    try:
                        state_dict = torch.load(sd_path, map_location=self.device)
                        if any(k.startswith("convs.") for k in state_dict.keys()):
                            alt = "advanced_cnn"
                        elif any(k.startswith("rnn.") or k.startswith("lstm.") for k in state_dict.keys()):
                            alt = "advanced_rnn"
                    except Exception:
                        pass
                if alt and alt != arch:
                    print(f"Architecture mismatch detected, retrying load as {alt}")
                    model = _load_by_arch(alt)
                else:
                    raise
        
        # Move model to device
        model.to(self.device)
        print(f"Model loaded and moved to {self.device}")
        
        # Prepare test data
        print("Preparing test data...")
        if model_type == "transformer":
            # Use the correct tokenizer source: local dir if present, otherwise HF ID
            tokenizer_name = metadata.get("pretrained_model") if metadata.get("is_pretrained_evaluation", False) else os.path.join(model_path)
            dataloaders = data_processor.prepare_pytorch_datasets(
                batch_size=32,
                tokenizer_name=tokenizer_name,
                max_length=128
            )
        else:
            dataloaders = data_processor.prepare_pytorch_datasets(batch_size=32)
        
        test_loader = dataloaders["test"]
        print(f"Test data prepared, {len(test_loader)} batches")
        
        # Evaluate model
        print("Running evaluation...")
        eval_start_time = time.time()
        results = self._evaluate_model_on_test_data(
            model=model,
            test_loader=test_loader,
            model_type=model_type,
            class_names=data_processor.class_names
        )
        eval_time = time.time() - eval_start_time
        print(f"Evaluation completed in {eval_time:.2f}s")
        
        # Add metadata to results
        results["model_name"] = model_name
        results["model_type"] = model_type
        results["history"] = history
        results["hyperparams"] = hyperparams
        results["evaluation_time"] = eval_time
        
        # Print summary
        print(f"\nEvaluation Summary for {model_name}:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print("Class-wise metrics:")
        for i, class_name in enumerate(results["classes"]):
            print(f"  {class_name}: Precision={results['precision'][i]:.4f}, Recall={results['recall'][i]:.4f}, F1={results['f1'][i]:.4f}")
        
        return results
    
    def _load_lstm_model(self, model_path: str, data_processor: DataProcessor, 
                        hyperparams: Optional[Dict] = None) -> nn.Module:
        """
        Load a saved LSTM model with hyperparameters.
        
        Args:
            model_path: Path to the saved model
            data_processor: Data processor with model parameters
            hyperparams: Optional hyperparameters dictionary
            
        Returns:
            Loaded LSTM model
        """
        # Use provided hyperparams or try to load from file
        if hyperparams is None:
            hyperparams_path = os.path.join(model_path, "hyperparams.json")
            if os.path.exists(hyperparams_path):
                with open(hyperparams_path, "r") as f:
                    hyperparams = json.load(f)
            else:
                hyperparams = {}
        
        # Set default hyperparameters if not found
        num_layers = hyperparams.get("num_layers", 2)
        hidden_size = hyperparams.get("hidden_size", 256)
        embedding_dim = hyperparams.get("embedding_dim", 300)
        dropout = hyperparams.get("dropout", 0.5)
        bidirectional = hyperparams.get("bidirectional", True)
        
        # Create model
        model = LSTMSentimentModel(
            vocab_size=data_processor.vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=len(data_processor.class_names),
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Load weights
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), 
                                        map_location=self.device))
        
        return model
    
    def _load_cnn_model(self, model_path: str, data_processor: DataProcessor,
                       hyperparams: Optional[Dict] = None) -> nn.Module:
        """
        Load a saved CNN model with hyperparameters.
        
        Args:
            model_path: Path to the saved model
            data_processor: Data processor with model parameters
            hyperparams: Optional hyperparameters dictionary
            
        Returns:
            Loaded CNN model
        """
        # Use provided hyperparams or try to load from file
        if hyperparams is None:
            hyperparams_path = os.path.join(model_path, "hyperparams.json")
            if os.path.exists(hyperparams_path):
                with open(hyperparams_path, "r") as f:
                    hyperparams = json.load(f)
            else:
                hyperparams = {}
        
        # Set default hyperparameters if not found
        filter_sizes = hyperparams.get("filter_sizes", [3, 4, 5])
        num_filters = hyperparams.get("num_filters", 100)
        embedding_dim = hyperparams.get("embedding_dim", 300)
        dropout = hyperparams.get("dropout", 0.5)
        
        # Create model
        model = CNNSentimentModel(
            vocab_size=data_processor.vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            num_classes=len(data_processor.class_names),
            dropout=dropout
        )
        
        # Load weights
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), 
                                        map_location=self.device))
        
        return model
    
    def _evaluate_model_on_test_data(self, model: nn.Module, test_loader: DataLoader,
                                   model_type: str, class_names: List[str]) -> Dict:
        """
        Evaluate model on test data with comprehensive metrics.
        
        Args:
            model: PyTorch model
            test_loader: DataLoader for test data
            model_type: Type of model ('pytorch' or 'transformer')
            class_names: List of class names
            
        Returns:
            Dictionary with detailed evaluation results
        """
        model.eval()
        
        # Collect predictions and true labels
        all_preds = []
        all_labels = []
        all_probs = []  # Store probabilities for ROC and PR curves
        inference_times = []  # Track inference time per batch
        
        with torch.no_grad():
            for batch in test_loader:
                # Track inference time
                start_time = time.time()
                
                if model_type == "pytorch":
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Use mixed precision if available
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                    else:
                        outputs = model(inputs)
                else:
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    labels = batch[2].to(self.device)
                    
                    # Use mixed precision if available
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(input_ids, attention_mask=attention_mask)
                            outputs = outputs.logits
                    else:
                        outputs = model(input_ids, attention_mask=attention_mask)
                        outputs = outputs.logits
                
                # Record inference time
                inference_times.append(time.time() - start_time)
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Get classification report
        class_names_list = list(class_names)
        report = classification_report(all_labels, all_preds, 
                                      target_names=class_names_list, 
                                      output_dict=True)
        
        # Calculate accuracy
        accuracy = np.mean(all_preds == all_labels)
        
        # Extract precision, recall, and F1 for each class
        precision = {}
        recall = {}
        f1 = {}
        
        for i, class_name in enumerate(class_names_list):
            if class_name in report:
                precision[i] = report[class_name]['precision']
                recall[i] = report[class_name]['recall']
                f1[i] = report[class_name]['f1-score']
        
        # Calculate ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        # Convert to one-hot encoding for multiclass ROC
        y_true_onehot = np.eye(len(class_names))[all_labels]
        
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Calculate PR curve
        pr_precision = {}
        pr_recall = {}
        average_precision = {}
        
        for i in range(len(class_names)):
            pr_precision[i], pr_recall[i], _ = precision_recall_curve(
                y_true_onehot[:, i], 
                all_probs[:, i]
            )
            average_precision[i] = np.mean(pr_precision[i])
        
        # Calculate inference speed metrics
        avg_inference_time = np.mean(inference_times)
        total_samples = len(all_labels)
        samples_per_second = total_samples / sum(inference_times)
        
        # Return comprehensive results
        return {
            "confusion_matrix": cm,
            "classification_report": report,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
            "precision_curve": pr_precision,
            "recall_curve": pr_recall,
            "average_precision": average_precision,
            "classes": class_names_list,
            "y_true": all_labels.tolist(),
            "y_pred": all_preds.tolist(),
            "inference_time": {
                "average_batch_time": avg_inference_time,
                "samples_per_second": samples_per_second,
                "total_inference_time": sum(inference_times)
            }
        }
        
        
class AdvancedRNNModel(nn.Module):
    """Advanced RNN-based model for sentiment analysis with attention option."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_layers: int, num_classes: int, dropout: float = 0.5,
                 embedding_dropout: float = 0.2, bidirectional: bool = True,
                 rnn_type: str = "LSTM", use_attention: bool = False):
        """
        Initialize advanced RNN model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_size: Size of RNN hidden state
            num_layers: Number of RNN layers
            num_classes: Number of output classes
            dropout: Dropout probability for RNN layers
            embedding_dropout: Dropout probability for embedding layer
            bidirectional: Whether to use bidirectional RNN
            rnn_type: Type of RNN ("LSTM" or "GRU")
            use_attention: Whether to use attention mechanism
        """
        super(AdvancedRNNModel, self).__init__()
        
        # Embedding layer with dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        
        # RNN layer (LSTM or GRU)
        if rnn_type == "GRU":
            self.rnn = nn.GRU(
                embedding_dim, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True, 
                bidirectional=bidirectional
            )
        else:  # Default to LSTM
            self.rnn = nn.LSTM(
                embedding_dim, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True, 
                bidirectional=bidirectional
            )
        
        # If bidirectional, double the hidden size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention layer
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(rnn_output_size, rnn_output_size),
                nn.Tanh(),
                nn.Linear(rnn_output_size, 1, bias=False)
            )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_output_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass with optional attention mechanism.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Embed the input
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        # Pass through RNN
        outputs, _ = self.rnn(embedded)
        # outputs shape: (batch_size, sequence_length, hidden_size * bidirectional)
        
        # Apply attention if enabled
        if self.use_attention:
            # Calculate attention weights
            attention_weights = self.attention(outputs)  # (batch_size, seq_len, 1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            
            # Apply attention to get context vector
            context = torch.bmm(outputs.transpose(1, 2), attention_weights)  # (batch_size, hidden_size, 1)
            context = context.squeeze(2)  # (batch_size, hidden_size)
        else:
            # Use the final time step output
            context = outputs[:, -1, :]
        
        # Apply dropout and pass through fully connected layer
        context = self.dropout(context)
        output = self.fc(context)
        
        return output

class AdvancedCNNModel(nn.Module):
    """Advanced CNN-based model for sentiment analysis with various options."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_filters: int,
                 filter_sizes: List[int], num_classes: int, dropout: float = 0.5,
                 embedding_dropout: float = 0.2, activation: str = "relu",
                 batch_norm: bool = False, pool_type: str = "max"):
        """
        Initialize advanced CNN model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            num_filters: Number of filters per filter size
            filter_sizes: List of filter sizes
            num_classes: Number of output classes
            dropout: Dropout probability for CNN layers
            embedding_dropout: Dropout probability for embedding layer
            activation: Activation function ("relu", "leaky_relu", "tanh", "elu")
            batch_norm: Whether to use batch normalization
            pool_type: Pooling type ("max", "avg", "adaptive")
        """
        super(AdvancedCNNModel, self).__init__()
        
        # Embedding layer with dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        
        # Create convolutional layers with different kernel sizes
        self.convs = nn.ModuleList()
        for filter_size in filter_sizes:
            conv = nn.Conv2d(1, num_filters, (filter_size, embedding_dim))
            
            # Add batch normalization if requested
            if batch_norm:
                bn = nn.BatchNorm2d(num_filters)
                self.convs.append(nn.Sequential(conv, bn))
            else:
                self.convs.append(conv)
        
        # Set activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
            
        # Save pooling type for forward pass
        self.pool_type = pool_type
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        """
        Forward pass with configurable activation and pooling.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Embed the input
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        # Add channel dimension for conv2d
        embedded = embedded.unsqueeze(1)
        
        # Apply convolutions with activation and pooling
        conv_results = []
        for conv in self.convs:
            # Conv output shape: (batch_size, num_filters, seq_len - filter_size + 1, 1)
            conv_out = conv(embedded)
            # Remove last dimension
            conv_out = conv_out.squeeze(3)
            # Apply activation function
            conv_out = self.activation(conv_out)
            
            # Apply pooling based on selected type
            if self.pool_type == "max":
                # Max pooling over time
                pooled = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            elif self.pool_type == "avg":
                # Average pooling over time
                pooled = torch.avg_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            elif self.pool_type == "adaptive":
                # Adaptive max pooling to fixed size
                pooled = torch.nn.functional.adaptive_max_pool1d(conv_out, 1).squeeze(2)
            else:
                # Default to max pooling
                pooled = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                
            conv_results.append(pooled)
        
        # Concatenate results from all convolutions
        out = torch.cat(conv_results, dim=1)
        
        # Apply dropout and pass through fully connected layer
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
            
            
    
