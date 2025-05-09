"""
Model trainer for sentiment analysis.

This module handles the training, evaluation, and persistence of
different sentiment analysis model architectures.
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
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

# Import our custom data processor
from utils.data_processor import DataProcessor

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
    """Trains and evaluates sentiment analysis models."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize model trainer.
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize callbacks dictionary
        self.callbacks = {}
        
        # Initialize device (use GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    def train_lstm(self, data_processor: DataProcessor, model_name: str,
                  num_layers: int = 2, hidden_size: int = 256, 
                  embedding_dim: int = 300, dropout: float = 0.5,
                  bidirectional: bool = True, batch_size: int = 32,
                  lr: float = 0.001, epochs: int = 5, optimizer: str = "Adam"):
        """
        Train an LSTM model.
        
        Args:
            data_processor: Data processor with prepared data
            model_name: Name for the saved model
            num_layers: Number of LSTM layers
            hidden_size: Size of LSTM hidden state
            embedding_dim: Dimension of word embeddings
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            batch_size: Batch size for training
            lr: Learning rate
            epochs: Number of training epochs
            optimizer: Optimizer type ("Adam", "AdamW", or "SGD")
        """
        # Prepare datasets
        dataloaders = data_processor.prepare_pytorch_datasets(batch_size=batch_size)
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        
        # Get vocabulary size and number of classes
        vocab_size = data_processor.vocab_size
        num_classes = len(data_processor.class_names)
        
        # Create model
        model = LSTMSentimentModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Move model to device
        model.to(self.device)
        
        # Create optimizer
        if optimizer == "Adam":
            opt = optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "AdamW":
            opt = optim.AdamW(model.parameters(), lr=lr)
        elif optimizer == "SGD":
            opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            opt = optim.Adam(model.parameters(), lr=lr)
        
        # Create loss function
        criterion = nn.CrossEntropyLoss()
        
        # Train the model
        self._train_pytorch_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=opt,
            epochs=epochs,
            model_name=model_name
        )
    
    def train_cnn(self, data_processor: DataProcessor, model_name: str,
                 filter_sizes: List[int] = [3, 4, 5], num_filters: int = 100,
                 embedding_dim: int = 300, dropout: float = 0.5,
                 batch_size: int = 32, lr: float = 0.001, 
                 epochs: int = 5, optimizer: str = "Adam"):
        """
        Train a CNN model.
        
        Args:
            data_processor: Data processor with prepared data
            model_name: Name for the saved model
            filter_sizes: List of filter sizes
            num_filters: Number of filters per filter size
            embedding_dim: Dimension of word embeddings
            dropout: Dropout probability
            batch_size: Batch size for training
            lr: Learning rate
            epochs: Number of training epochs
            optimizer: Optimizer type ("Adam", "AdamW", or "SGD")
        """
        # Prepare datasets
        dataloaders = data_processor.prepare_pytorch_datasets(batch_size=batch_size)
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        
        # Get vocabulary size and number of classes
        vocab_size = data_processor.vocab_size
        num_classes = len(data_processor.class_names)
        
        # Create model
        model = CNNSentimentModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Move model to device
        model.to(self.device)
        
        # Create optimizer
        if optimizer == "Adam":
            opt = optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "AdamW":
            opt = optim.AdamW(model.parameters(), lr=lr)
        elif optimizer == "SGD":
            opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            opt = optim.Adam(model.parameters(), lr=lr)
        
        # Create loss function
        criterion = nn.CrossEntropyLoss()
        
        # Train the model
        self._train_pytorch_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=opt,
            epochs=epochs,
            model_name=model_name
        )
    
    def train_transformer(self, data_processor: DataProcessor, model_name: str,
                         pretrained_model: str = "cardiffnlp/twitter-roberta-base-sentiment",
                         finetune: bool = True, batch_size: int = 16,
                         lr: float = 0.00005, epochs: int = 3,
                         optimizer: str = "AdamW"):
        """
        Train a transformer model.
        
        Args:
            data_processor: Data processor with prepared data
            model_name: Name for the saved model
            pretrained_model: Name of pretrained model to use
            finetune: Whether to finetune the model or freeze the encoder
            batch_size: Batch size for training
            lr: Learning rate
            epochs: Number of training epochs
            optimizer: Optimizer type (only used for non-transformers Trainer)
        """
        # Prepare datasets with appropriate tokenizer
        dataloaders = data_processor.prepare_pytorch_datasets(
            batch_size=batch_size,
            tokenizer_name=pretrained_model,
            max_length=128
        )
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        
        # Get number of classes
        num_classes = len(data_processor.class_names)
        
        # Load pretrained model
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes
        )
        
        # Freeze encoder if not finetuning
        if not finetune:
            for param in model.base_model.parameters():
                param.requires_grad = False
        
        # Move model to device
        model.to(self.device)
        
        # Create optimizer
        if optimizer == "Adam":
            opt = optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "AdamW":
            opt = optim.AdamW(model.parameters(), lr=lr)
        elif optimizer == "SGD":
            opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            opt = optim.AdamW(model.parameters(), lr=lr)
        
        # Create loss function
        criterion = nn.CrossEntropyLoss()
        
        # Train the model
        self._train_transformer_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=opt,
            epochs=epochs,
            model_name=model_name
        )
    
    def _train_pytorch_model(self, model: nn.Module, train_loader: DataLoader,
                            val_loader: Optional[DataLoader], criterion: nn.Module,
                            optimizer: optim.Optimizer, epochs: int, model_name: str):
        """
        Generic training loop for PyTorch models.
        
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
        
        # Notify training start
        self._call_callback('on_training_start', epochs, total_batches)
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Move inputs and labels to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Notify batch end
                self._call_callback('on_batch_end', batch_idx, total_batches, epoch, epochs)
            
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
            else:
                # Save model at each epoch if no validation data
                self._save_model(model, model_name, 'pytorch')
            
            # Update history
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            
            if val_loader:
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
            
            # Notify epoch end
            metrics = {
                'loss': train_loss,
                'accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }
            self._call_callback('on_epoch_end', epoch, epochs, metrics)
        
        # Training complete
        training_time = time.time() - start_time
        
        # Save history
        self._save_training_history(model_name, history)
        
        # Notify training end
        final_metrics = {
            'loss': history['loss'][-1],
            'accuracy': history['accuracy'][-1],
            'val_loss': history['val_loss'][-1] if history['val_loss'] else 0.0,
            'val_accuracy': history['val_accuracy'][-1] if history['val_accuracy'] else 0.0
        }
        self._call_callback('on_training_end', model_name, final_metrics, training_time)
    
    def _train_transformer_model(self, model: nn.Module, train_loader: DataLoader,
                                val_loader: Optional[DataLoader], criterion: nn.Module,
                                optimizer: optim.Optimizer, epochs: int, model_name: str):
        """
        Training loop for transformer models using PyTorch.
        
        Args:
            model: Transformer model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            epochs: Number of training epochs
            model_name: Name for the saved model
        """
        # This method uses the same approach as _train_pytorch_model, but with
        # specific handling for transformer outputs
        
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
        
        # Notify training start
        self._call_callback('on_training_start', epochs, total_batches)
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Unpack batch data (format depends on whether using Hugging Face datasets)
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Notify batch end
                self._call_callback('on_batch_end', batch_idx, total_batches, epoch, epochs)
            
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
            else:
                # Save model at each epoch if no validation data
                self._save_model(model, model_name, 'transformer')
            
            # Update history
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            
            if val_loader:
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
            
            # Notify epoch end
            metrics = {
                'loss': train_loss,
                'accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }
            self._call_callback('on_epoch_end', epoch, epochs, metrics)
        
        # Training complete
        training_time = time.time() - start_time
        
        # Save history
        self._save_training_history(model_name, history)
        
        # Notify training end
        final_metrics = {
            'loss': history['loss'][-1],
            'accuracy': history['accuracy'][-1],
            'val_loss': history['val_loss'][-1] if history['val_loss'] else 0.0,
            'val_accuracy': history['val_accuracy'][-1] if history['val_accuracy'] else 0.0
        }
        self._call_callback('on_training_end', model_name, final_metrics, training_time)
    
    def _validate_model(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate model performance.
        
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
        Validate transformer model performance.
        
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
        Save model to disk.
        
        Args:
            model: PyTorch model
            model_name: Name for the saved model
            model_type: Type of model ('pytorch' or 'transformer')
        """
        # Create model directory if it doesn't exist
        model_path = os.path.join(self.model_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Save model state dictionary
        if model_type == 'pytorch':
            torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
        else:
            model.save_pretrained(model_path)
        
        # Save model metadata
        metadata = {
            "model_type": model_type,
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name
        }
        
        with open(os.path.join(model_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    
    def _save_training_history(self, model_name: str, history: Dict):
        """
        Save training history to disk.
        
        Args:
            model_name: Name of the model
            history: Dictionary with training history
        """
        model_path = os.path.join(self.model_dir, model_name)
        
        with open(os.path.join(model_path, "history.json"), "w") as f:
            json.dump(history, f)
    
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
        Evaluate a trained model on test data.
        
        Args:
            model_name: Name of the model to evaluate
            data_processor: Data processor with test data
            
        Returns:
            Dictionary with evaluation results
        """
        model_path = os.path.join(self.model_dir, model_name)
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model '{model_name}' not found")
        
        # Load model metadata
        with open(os.path.join(model_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        model_type = metadata["model_type"]
        
        # Load training history if available
        history = None
        history_path = os.path.join(model_path, "history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
        
# Load model
            if model_type == "pytorch":
                # For PyTorch models, we need to re-create the model architecture
                # Here we assume we can infer the architecture from model_name
                if "lstm" in model_name.lower():
                    model = self._load_lstm_model(model_path, data_processor)
                elif "cnn" in model_name.lower():
                    model = self._load_cnn_model(model_path, data_processor)
                else:
                    raise ValueError(f"Unknown PyTorch model type for {model_name}")
            else:
                # For transformer models, we can load directly with from_pretrained
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Move model to device
        model.to(self.device)
        
        # Prepare test data
        if model_type == "pytorch":
            dataloaders = data_processor.prepare_pytorch_datasets(batch_size=32)
        else:
            dataloaders = data_processor.prepare_pytorch_datasets(
                batch_size=32,
                tokenizer_name=os.path.join(model_path),
                max_length=128
            )
        
        test_loader = dataloaders["test"]
        
        # Evaluate model
        results = self._evaluate_model_on_test_data(
            model=model,
            test_loader=test_loader,
            model_type=model_type,
            class_names=data_processor.class_names
        )
        
        # Add history to results
        results["history"] = history
        
        return results
    
    def _load_lstm_model(self, model_path: str, data_processor: DataProcessor) -> nn.Module:
        """
        Load a saved LSTM model.
        
        Args:
            model_path: Path to the saved model
            data_processor: Data processor with model parameters
            
        Returns:
            Loaded LSTM model
        """
        # Try to load hyperparameters if available
        params = {}
        params_path = os.path.join(model_path, "hyperparams.json")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                params = json.load(f)
        
        # Set default hyperparameters if not found
        num_layers = params.get("num_layers", 2)
        hidden_size = params.get("hidden_size", 256)
        embedding_dim = params.get("embedding_dim", 300)
        dropout = params.get("dropout", 0.5)
        bidirectional = params.get("bidirectional", True)
        
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
    
    def _load_cnn_model(self, model_path: str, data_processor: DataProcessor) -> nn.Module:
        """
        Load a saved CNN model.
        
        Args:
            model_path: Path to the saved model
            data_processor: Data processor with model parameters
            
        Returns:
            Loaded CNN model
        """
        # Try to load hyperparameters if available
        params = {}
        params_path = os.path.join(model_path, "hyperparams.json")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                params = json.load(f)
        
        # Set default hyperparameters if not found
        filter_sizes = params.get("filter_sizes", [3, 4, 5])
        num_filters = params.get("num_filters", 100)
        embedding_dim = params.get("embedding_dim", 300)
        dropout = params.get("dropout", 0.5)
        
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
        Evaluate model on test data.
        
        Args:
            model: PyTorch model
            test_loader: DataLoader for test data
            model_type: Type of model ('pytorch' or 'transformer')
            class_names: List of class names
            
        Returns:
            Dictionary with evaluation results
        """
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                if model_type == "pytorch":
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                else:
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    labels = batch[2].to(self.device)
                    outputs = model(input_ids, attention_mask=attention_mask)
                    outputs = outputs.logits
                
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Get classification report
        class_names_list = list(class_names)
        report = classification_report(all_labels, all_preds, 
                                      target_names=class_names_list, 
                                      output_dict=True)
        
        # Calculate accuracy
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Calculate precision, recall, and F1 for each class
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
        
        # Convert predictions to one-hot encoding for ROC
        n_classes = len(class_names)
        y_true_onehot = np.eye(n_classes)[all_labels]
        y_pred_onehot = np.eye(n_classes)[all_preds]
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_onehot[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Calculate PR curve
        pr_precision = {}
        pr_recall = {}
        average_precision = {}
        
        for i in range(n_classes):
            pr_precision[i], pr_recall[i], _ = precision_recall_curve(
                y_true_onehot[:, i], 
                y_pred_onehot[:, i]
            )
            average_precision[i] = np.mean(pr_precision[i])
        
        # Return results
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
            "classes": class_names_list
        }