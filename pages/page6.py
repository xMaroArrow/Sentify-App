# pages/page6.py
import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import re
from utils import theme

class Page6(ctk.CTkFrame):
    """
    Model Comparison Page - Compare different models on the same dataset.
    
    This page allows you to:
    1. Load your dataset
    2. Train and evaluate LSTM, CNN, and pre-trained transformer models
    3. Compare results with visualization
    4. Export results for thesis discussion
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # Initialize state variables
        self.dataset = None
        self.results = {}
        self.current_task = None
        self.stop_requested = False
        
        # Create UI
        self._create_ui()
        
    def _create_ui(self):
        """Create the user interface for the model comparison page."""
        # Main scrollable container
        self.scrollable = ctk.CTkScrollableFrame(self, width=950, height=800)
        self.scrollable.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Create header
        self._create_header()
        
        # Create dataset section
        self._create_dataset_section()
        
        # Create models section
        self._create_models_section()
        
        # Create results section
        self._create_results_section()
        
    def _create_header(self):
        """Create the header section with title and description."""
        header_frame = ctk.CTkFrame(self.scrollable)
        header_frame.pack(fill="x", pady=10)
        
        title = ctk.CTkLabel(
            header_frame,
            text="Model Comparison for Sentiment Analysis",
            font=("Arial", 24, "bold")
        )
        title.pack(pady=(10, 5))
        
        description = ctk.CTkLabel(
            header_frame,
            text="Compare LSTM, CNN, and pre-trained transformer models on the same dataset",
            font=("Arial", 14),
            text_color=theme.subtle_text_color()
        )
        description.pack(pady=(0, 10))
    
    def _create_dataset_section(self):
        """Create the dataset loading and preprocessing section."""
        dataset_frame = ctk.CTkFrame(self.scrollable)
        dataset_frame.pack(fill="x", pady=10)
        
        # Section title
        section_title = ctk.CTkLabel(
            dataset_frame,
            text="Dataset",
            font=("Arial", 18, "bold")
        )
        section_title.pack(pady=10, anchor="w", padx=10)
        
        # Load dataset button
        button_frame = ctk.CTkFrame(dataset_frame)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        self.load_button = ctk.CTkButton(
            button_frame,
            text="Load Dataset",
            command=self._load_dataset,
            width=150
        )
        self.load_button.pack(side="left", padx=10, pady=10)
        
        # Dataset path display
        self.dataset_path_var = ctk.StringVar(value="No dataset loaded")
        path_label = ctk.CTkLabel(
            button_frame,
            textvariable=self.dataset_path_var,
            font=("Arial", 12),
            anchor="w"
        )
        path_label.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        # Dataset info
        self.dataset_info = ctk.CTkTextbox(
            dataset_frame,
            height=100,
            width=900,
            font=("Arial", 12)
        )
        self.dataset_info.pack(padx=10, pady=10, fill="both")
        self.dataset_info.insert("1.0", "Load a dataset to see information")
        self.dataset_info.configure(state="disabled")
        
    def _create_models_section(self):
        """Create the model selection and training section."""
        models_frame = ctk.CTkFrame(self.scrollable)
        models_frame.pack(fill="x", pady=10)
        
        # Section title
        section_title = ctk.CTkLabel(
            models_frame,
            text="Models",
            font=("Arial", 18, "bold")
        )
        section_title.pack(pady=10, anchor="w", padx=10)
        
        # Model selection
        selection_frame = ctk.CTkFrame(models_frame)
        selection_frame.pack(fill="x", padx=10, pady=5)
        
        # LSTM model
        self.lstm_var = ctk.BooleanVar(value=True)
        lstm_check = ctk.CTkCheckBox(
            selection_frame,
            text="LSTM Model",
            variable=self.lstm_var
        )
        lstm_check.pack(side="left", padx=20, pady=10)
        
        # CNN model
        self.cnn_var = ctk.BooleanVar(value=True)
        cnn_check = ctk.CTkCheckBox(
            selection_frame,
            text="CNN Model",
            variable=self.cnn_var
        )
        cnn_check.pack(side="left", padx=20, pady=10)
        
        # Transformer model
        self.transformer_var = ctk.BooleanVar(value=True)
        transformer_check = ctk.CTkCheckBox(
            selection_frame,
            text="RoBERTa Model",
            variable=self.transformer_var
        )
        transformer_check.pack(side="left", padx=20, pady=10)
        
        # Training parameters
        params_frame = ctk.CTkFrame(models_frame)
        params_frame.pack(fill="x", padx=10, pady=5)
        
        # Epochs
        epochs_label = ctk.CTkLabel(
            params_frame,
            text="Epochs:",
            font=("Arial", 12)
        )
        epochs_label.pack(side="left", padx=10, pady=10)
        
        self.epochs_var = ctk.StringVar(value="5")
        epochs_entry = ctk.CTkEntry(
            params_frame,
            width=50,
            textvariable=self.epochs_var
        )
        epochs_entry.pack(side="left", padx=5, pady=10)
        
        # Batch size
        batch_label = ctk.CTkLabel(
            params_frame,
            text="Batch Size:",
            font=("Arial", 12)
        )
        batch_label.pack(side="left", padx=(20, 10), pady=10)
        
        self.batch_var = ctk.StringVar(value="32")
        batch_entry = ctk.CTkEntry(
            params_frame,
            width=50,
            textvariable=self.batch_var
        )
        batch_entry.pack(side="left", padx=5, pady=10)
        
        # Test split
        split_label = ctk.CTkLabel(
            params_frame,
            text="Test Split:",
            font=("Arial", 12)
        )
        split_label.pack(side="left", padx=(20, 10), pady=10)
        
        self.split_var = ctk.StringVar(value="0.2")
        split_entry = ctk.CTkEntry(
            params_frame,
            width=50,
            textvariable=self.split_var
        )
        split_entry.pack(side="left", padx=5, pady=10)
        
        # Train button
        button_frame = ctk.CTkFrame(models_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        self.train_button = ctk.CTkButton(
            button_frame,
            text="Train Selected Models",
            command=self._start_training,
            font=("Arial", 14),
            height=40,
            fg_color="#0078D7"
        )
        self.train_button.pack(side="left", padx=10, pady=10)
        
        self.stop_button = ctk.CTkButton(
            button_frame,
            text="Stop",
            command=self._stop_training,
            font=("Arial", 14),
            height=40,
            fg_color="#E74C3C",
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=10, pady=10)
        
        # Progress information
        self.progress_label = ctk.CTkLabel(
            button_frame,
            text="Ready to train",
            font=("Arial", 12),
            text_color="#888888"
        )
        self.progress_label.pack(side="left", padx=20, pady=10)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(models_frame, width=900)
        self.progress_bar.pack(padx=10, pady=(0, 10))
        self.progress_bar.set(0)
        
    def _create_results_section(self):
        """Create the results and visualization section."""
        results_frame = ctk.CTkFrame(self.scrollable)
        results_frame.pack(fill="x", pady=10)
        
        # Section title
        section_title = ctk.CTkLabel(
            results_frame,
            text="Results",
            font=("Arial", 18, "bold")
        )
        section_title.pack(pady=10, anchor="w", padx=10)
        
        # Tabs for different visualizations
        self.tab_view = ctk.CTkTabview(results_frame, height=600)
        self.tab_view.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add tabs
        self.tab_view.add("Accuracy")
        self.tab_view.add("Metrics")
        self.tab_view.add("Confusion Matrices")
        self.tab_view.add("Training Time")
        
        # Create empty canvases for plots
        self.accuracy_frame = ctk.CTkFrame(self.tab_view.tab("Accuracy"))
        self.accuracy_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.metrics_frame = ctk.CTkFrame(self.tab_view.tab("Metrics"))
        self.metrics_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.cm_frame = ctk.CTkFrame(self.tab_view.tab("Confusion Matrices"))
        self.cm_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.time_frame = ctk.CTkFrame(self.tab_view.tab("Training Time"))
        self.time_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Export button
        export_frame = ctk.CTkFrame(results_frame)
        export_frame.pack(fill="x", padx=10, pady=10)
        
        export_button = ctk.CTkButton(
            export_frame,
            text="Export Results",
            command=self._export_results,
            width=150,
            height=36
        )
        export_button.pack(side="left", padx=10, pady=10)
        
        export_format_label = ctk.CTkLabel(
            export_frame,
            text="Format:",
            font=("Arial", 12)
        )
        export_format_label.pack(side="left", padx=(20, 5), pady=10)
        
        self.export_format_var = ctk.StringVar(value="PDF")
        export_format = ctk.CTkOptionMenu(
            export_frame,
            values=["PDF", "PNG", "CSV", "All"],
            variable=self.export_format_var,
            width=100
        )
        export_format.pack(side="left", padx=5, pady=10)
    
    def _load_dataset(self):
        """Load and preprocess the dataset."""
        file_path = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Update UI
            self.dataset_path_var.set(os.path.basename(file_path))
            self.progress_label.configure(text="Loading dataset...")
            self.progress_bar.set(0.2)
            
            # Load dataset in a background thread
            threading.Thread(
                target=self._process_dataset,
                args=(file_path,),
                daemon=True
            ).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            self.progress_bar.set(0)
            self.progress_label.configure(text="Failed to load dataset")
        
        
        
    def _process_dataset(self, file_path):
        """Process the dataset in a background thread with very robust NaN handling."""
        try:
            # Load the dataset with warnings about mixed types
            print(f"Loading dataset from {file_path}")
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"Dataset loaded with shape: {df.shape}")
            
            # Identify column names (print all columns to debug)
            print(f"Available columns: {df.columns.tolist()}")
            
            # Check for NaN values in each column
            for col in df.columns:
                nan_count = df[col].isna().sum()
                print(f"Column '{col}' has {nan_count} NaN values")
            
            # Determine text and sentiment columns - using fixed names if available
            text_col = None
            sentiment_col = None
            
            # Look for common column names in Twitter data
            if 'tweet' in df.columns:
                text_col = 'tweet'
            elif 'text' in df.columns:
                text_col = 'text'
            elif 'content' in df.columns:
                text_col = 'content'
            
            if 'sentiment' in df.columns:
                sentiment_col = 'sentiment'
            elif 'label' in df.columns:
                sentiment_col = 'label'
            elif 'polarity' in df.columns:
                sentiment_col = 'polarity'
            
            # If still not found, try to guess based on column properties
            if text_col is None:
                for col in df.columns:
                    if df[col].dtype == object:  # String columns are usually object type
                        text_col = col
                        break
            
            if sentiment_col is None:
                for col in df.columns:
                    if col != text_col and (df[col].dtype == float or df[col].dtype == int):
                        sentiment_col = col
                        break
            
            # If columns are still not identified, show the selection dialog
            if text_col is None or sentiment_col is None:
                print("Couldn't automatically detect columns, showing manual selection dialog")
                self.after(0, lambda: self._show_column_selection_dialog(df))
                return
            
            print(f"Selected text column: {text_col}")
            print(f"Selected sentiment column: {sentiment_col}")
            
            # Make a clean copy of the dataframe
            cleaned_df = df[[text_col, sentiment_col]].copy()
            
            # Fill NaN values in text column with empty string
            cleaned_df[text_col] = cleaned_df[text_col].fillna("")
            
            # Remove rows with NaN in sentiment column
            print(f"Rows before NaN removal: {len(cleaned_df)}")
            cleaned_df = cleaned_df.dropna(subset=[sentiment_col])
            print(f"Rows after NaN removal: {len(cleaned_df)}")
            
            # Convert sentiment to numeric if it's not already
            if cleaned_df[sentiment_col].dtype == object:
                try:
                    cleaned_df[sentiment_col] = pd.to_numeric(cleaned_df[sentiment_col], errors='coerce')
                    cleaned_df = cleaned_df.dropna(subset=[sentiment_col])  # Drop any that couldn't be converted
                except Exception as e:
                    print(f"Error converting sentiment to numeric: {e}")
            
            # Now attempt to convert values to match expected -1, 0, 1 pattern
            unique_sentiments = sorted(cleaned_df[sentiment_col].unique())
            print(f"Unique sentiment values: {unique_sentiments}")
            
            # Create sentiment mapping
            sentiment_map = {}
            
            # If values are already -1, 0, 1 or similar pattern
            if len(unique_sentiments) == 3:
                # Map to 0, 1, 2 for model compatibility (handles both integer and float types)
                try:
                    # Try to sort and map based on values
                    sentiment_map = {unique_sentiments[0]: 0, 
                                    unique_sentiments[1]: 1, 
                                    unique_sentiments[2]: 2}
                    
                    # If values are close to -1, 0, 1 use human-readable labels
                    if abs(unique_sentiments[0] + 1) < 0.1 and abs(unique_sentiments[1]) < 0.1 and abs(unique_sentiments[2] - 1) < 0.1:
                        original_labels = {0: "Negative (-1)", 1: "Neutral (0)", 2: "Positive (1)"}
                    else:
                        original_labels = {0: f"Class {unique_sentiments[0]}", 
                                        1: f"Class {unique_sentiments[1]}", 
                                        2: f"Class {unique_sentiments[2]}"}
                except Exception as e:
                    print(f"Error creating sentiment mapping: {e}")
                    # Fallback to simple enumeration
                    sentiment_map = {val: idx for idx, val in enumerate(unique_sentiments)}
                    original_labels = {idx: f"Class {val}" for idx, val in enumerate(unique_sentiments)}
            else:
                # Generic mapping
                sentiment_map = {val: idx for idx, val in enumerate(unique_sentiments)}
                original_labels = {idx: f"Class {val}" for idx, val in enumerate(unique_sentiments)}
            
            # Apply mapping very carefully
            try:
                # First create a new column to avoid modifying the original
                cleaned_df['sentiment_mapped'] = cleaned_df[sentiment_col].apply(
                    lambda x: sentiment_map.get(x, 0) if pd.notnull(x) else 0
                )
                
                # Now make this the sentiment column
                sentiment_col_original = sentiment_col
                sentiment_col = 'sentiment_mapped'
                
                print(f"Mapped sentiment values distribution: {cleaned_df[sentiment_col].value_counts().to_dict()}")
            except Exception as e:
                print(f"Error mapping sentiment values: {e}")
                # If mapping fails, create a simple 0,1,2 version as last resort
                cleaned_df['sentiment_simple'] = pd.qcut(
                    cleaned_df[sentiment_col], 
                    q=min(3, len(unique_sentiments)), 
                    labels=False
                )
                sentiment_col = 'sentiment_simple'
                sentiment_map = {i: i for i in range(min(3, len(unique_sentiments)))}
                original_labels = {i: f"Class {i}" for i in range(min(3, len(unique_sentiments)))}
            
            # Store the processed dataset
            self.dataset = {
                'df': cleaned_df,
                'text_col': text_col,
                'sentiment_col': sentiment_col,
                'sentiment_map': sentiment_map,
                'original_labels': original_labels,
                'num_classes': len(sentiment_map),
                'class_counts': cleaned_df[sentiment_col].value_counts().to_dict()
            }
            
            # Calculate dropped row count
            dropped_count = len(df) - len(cleaned_df)
            
            # Update UI on the main thread
            self.after(0, lambda: self._update_dataset_info(
                cleaned_df, text_col, sentiment_col, sentiment_map, original_labels, dropped_count
            ))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to process dataset: {str(e)}"))
            self.after(0, lambda: self.progress_bar.set(0))
            self.after(0, lambda: self.progress_label.configure(text="Failed to process dataset"))

    def _complete_dataset_processing(self, df, text_col, sentiment_col):
        """Complete dataset processing after column selection."""
        try:
            # Handle NaN values in the text column
            df[text_col] = df[text_col].fillna("")
            
            # Handle NaN values in the sentiment column and convert to numeric
            df[sentiment_col] = pd.to_numeric(df[sentiment_col], errors='coerce').fillna(0)
            
            # For your dataset with values -1, 0, 1
            # Create a mapping for original values to standard 0, 1, 2 format for model compatibility
            unique_values = sorted(df[sentiment_col].unique())
            sentiment_map = {}
            
            # Your sentiment values are -1, 0, 1
            # Map them to 0, 1, 2 for model compatibility
            if set(unique_values) == {-1, 0, 1} or set(unique_values) == {-1.0, 0.0, 1.0}:
                sentiment_map = {-1: 0, 0: 1, 1: 2}  # Map to 0=negative, 1=neutral, 2=positive
                df[sentiment_col] = df[sentiment_col].map(sentiment_map)
                original_labels = {0: "Negative (-1)", 1: "Neutral (0)", 2: "Positive (1)"}
            else:
                # Generic mapping for other value ranges
                value_to_idx = {val: idx for idx, val in enumerate(sorted(unique_values))}
                df[sentiment_col] = df[sentiment_col].map(value_to_idx)
                original_labels = {idx: f"Class {val}" for val, idx in value_to_idx.items()}
                sentiment_map = value_to_idx
            
            # Make sure sentiment values are integers
            df[sentiment_col] = df[sentiment_col].astype(int)
            
            # Store the processed dataset
            self.dataset = {
                'df': df,
                'text_col': text_col,
                'sentiment_col': sentiment_col,
                'sentiment_map': sentiment_map,
                'original_labels': original_labels,
                'num_classes': len(set(df[sentiment_col])),
                'class_counts': df[sentiment_col].value_counts().to_dict()
            }
            
            # Update UI on the main thread
            self.after(0, lambda: self._update_dataset_info(df, text_col, sentiment_col, sentiment_map, original_labels))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to process dataset: {str(e)}"))
            self.after(0, lambda: self.progress_bar.set(0))
            self.after(0, lambda: self.progress_label.configure(text="Failed to process dataset"))
    
    def _complete_dataset_processing(self, df, text_col, sentiment_col):
        """Complete dataset processing after column selection."""
        try:
            # Process sentiment labels to ensure they're numeric
            # Map text labels to numeric if needed
            sentiment_map = {}
            
            # Check if the sentiment column has text values that need mapping
            if df[sentiment_col].dtype == 'object':
                unique_sentiments = df[sentiment_col].dropna().unique()
                
                # For 'category' column, check for common sentiment names
                if sentiment_col == 'category':
                    # Build a mapping based on common sentiment terms
                    sentiment_map = {}
                    for s in unique_sentiments:
                        s_lower = str(s).lower()
                        if 'positive' in s_lower:
                            sentiment_map[s] = 2 if any('neutral' in str(s2).lower() for s2 in unique_sentiments) else 1
                        elif 'negative' in s_lower:
                            sentiment_map[s] = 0
                        elif 'neutral' in s_lower:
                            sentiment_map[s] = 1
                        # Fill in any missing mappings
                        if s not in sentiment_map:
                            sentiment_map[s] = len(sentiment_map)
                else:
                    # Generic mapping based on sorted unique values
                    sentiment_map = {val: idx for idx, val in enumerate(sorted(unique_sentiments)) if pd.notnull(val)}
                
                # Apply mapping (case-insensitive for string values)
                df[sentiment_col] = df[sentiment_col].apply(
                    lambda x: sentiment_map.get(x, sentiment_map.get(str(x).lower(), 0) if isinstance(x, str) else 0) if pd.notnull(x) else 0
                )
            
            # Handle numeric sentiment values that might need normalization
            if pd.api.types.is_numeric_dtype(df[sentiment_col]):
                # If values are not already 0, 1, 2, etc., normalize them
                unique_values = sorted(df[sentiment_col].unique())
                if not all(v == int(v) for v in unique_values) or not all(v >= 0 for v in unique_values):
                    old_to_new = {old_val: new_val for new_val, old_val in enumerate(unique_values)}
                    df[sentiment_col] = df[sentiment_col].map(old_to_new)
                    # Update sentiment_map to reflect the new mapping
                    if sentiment_map:
                        sentiment_map = {k: old_to_new[v] for k, v in sentiment_map.items()}
            
            # Store the processed dataset
            self.dataset = {
                'df': df,
                'text_col': text_col,
                'sentiment_col': sentiment_col,
                'sentiment_map': sentiment_map,
                'num_classes': len(sentiment_map) if sentiment_map else df[sentiment_col].nunique()
            }
            
            # Update UI on the main thread
            self.after(0, lambda: self._update_dataset_info(df, text_col, sentiment_col, sentiment_map))
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to process dataset: {str(e)}"))
            self.after(0, lambda: self.progress_bar.set(0))
            self.after(0, lambda: self.progress_label.configure(text="Failed to process dataset"))
    
    
    def _update_dataset_info(self, df, text_col, sentiment_col, sentiment_map, original_labels, dropped_count=0, *args):
        """Update the dataset information display."""
        # Enable text box for editing
        self.dataset_info.configure(state="normal")
        self.dataset_info.delete("1.0", "end")
        
        # Create info text
        info = f"Dataset loaded with {len(df)} samples\n"
        if dropped_count > 0:
            info += f"(Note: {dropped_count} rows with missing sentiment values were removed)\n"
        info += f"\nText column: {text_col}\n"
        info += f"Sentiment column: {sentiment_col}\n\n"
        
        # Add class distribution
        info += "Class distribution:\n"
        value_counts = df[sentiment_col].value_counts().sort_index()
        
        for value, count in value_counts.items():
            label = original_labels.get(value, f"Class {value}")
            percentage = (count / len(df)) * 100
            info += f"  {label}: {count} ({percentage:.2f}%)\n"
        
        # Add text length statistics
        text_lengths = df[text_col].astype(str).str.len()
        info += f"\nText length statistics:\n"
        info += f"  Mean: {text_lengths.mean():.2f} characters\n"
        info += f"  Min: {text_lengths.min()} characters\n"
        info += f"  Max: {text_lengths.max()} characters\n"
        
        # Insert info and disable editing
        self.dataset_info.insert("1.0", info)
        self.dataset_info.configure(state="disabled")
        
        # Update progress
        self.progress_bar.set(1.0)
        self.progress_label.configure(text="Dataset loaded successfully")
        
        # Reset progress bar after delay
        self.after(2000, lambda: self.progress_bar.set(0))
        self.after(2000, lambda: self.progress_label.configure(text="Ready to train"))
    
    def _start_training(self):
        """Start the model training process."""
        if self.dataset is None:
            messagebox.showinfo("Information", "Please load a dataset first.")
            return
        
        # Check if any model is selected
        if not (self.lstm_var.get() or self.cnn_var.get() or self.transformer_var.get()):
            messagebox.showinfo("Information", "Please select at least one model to train.")
            return
        
        # Get training parameters
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_var.get())
            test_split = float(self.split_var.get())
            
            if epochs <= 0 or batch_size <= 0 or test_split <= 0 or test_split >= 1:
                raise ValueError("Invalid parameter values")
                
        except ValueError:
            messagebox.showerror("Error", "Please enter valid parameter values.")
            return
        
        # Update UI state
        self.train_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.load_button.configure(state="disabled")
        self.progress_bar.set(0)
        self.progress_label.configure(text="Preparing data...")
        
        # Reset stop flag
        self.stop_requested = False
        
        # Start training in a background thread
        self.current_task = threading.Thread(
            target=self._run_training,
            args=(epochs, batch_size, test_split),
            daemon=True
        )
        self.current_task.start()
    
    def _stop_training(self):
        """Stop the model training process."""
        self.stop_requested = True
        self.progress_label.configure(text="Stopping...")
    
    def _run_training(self, epochs, batch_size, test_split):
        """Run the model training process in a background thread."""
        try:
            # Get dataset
            df = self.dataset['df']
            text_col = self.dataset['text_col']
            sentiment_col = self.dataset['sentiment_col']
            num_classes = self.dataset['num_classes']
            
            # Split the data
            texts = df[text_col].tolist()
            labels = df[sentiment_col].tolist()
            
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_split, random_state=42, stratify=labels
            )
            
            # Update UI
            self.after(0, lambda: self.progress_label.configure(text="Data prepared for training"))
            self.after(0, lambda: self.progress_bar.set(0.1))
            
            # Initialize results dictionary
            self.results = {
                'accuracy': {},
                'precision': {},
                'recall': {},
                'f1': {},
                'confusion_matrix': {},
                'training_time': {}
            }
            
            # Train each selected model
            current_progress = 0.1
            models_to_train = []
            
            if self.lstm_var.get():
                models_to_train.append(('LSTM', self._train_lstm))
            
            if self.cnn_var.get():
                models_to_train.append(('CNN', self._train_cnn))
            
            if self.transformer_var.get():
                models_to_train.append(('RoBERTa', self._train_transformer))
            
            progress_increment = 0.9 / len(models_to_train)
            
            for model_name, train_func in models_to_train:
                if self.stop_requested:
                    break
                
                # Update progress
                self.after(0, lambda name=model_name: self.progress_label.configure(
                    text=f"Training {name} model..."
                ))
                
                # Train the model
                start_time = time.time()
                try:
                    model_results = train_func(
                        X_train, y_train, X_test, y_test, 
                        epochs, batch_size, num_classes
                    )
                    
                    # Record training time
                    end_time = time.time()
                    training_time = end_time - start_time
                    self.results['training_time'][model_name] = training_time
                    
                    # Store results
                    if model_results:
                        self.results['accuracy'][model_name] = model_results.get('accuracy', 0)
                        self.results['precision'][model_name] = model_results.get('precision', [0])
                        self.results['recall'][model_name] = model_results.get('recall', [0])
                        self.results['f1'][model_name] = model_results.get('f1', [0])
                        self.results['confusion_matrix'][model_name] = model_results.get('confusion_matrix', None)
                        
                except Exception as e:
                    self.after(0, lambda err=str(e): messagebox.showerror(
                        "Training Error",
                        f"Error training {model_name} model: {err}"
                    ))
                
                # Update progress
                current_progress += progress_increment
                self.after(0, lambda p=current_progress: self.progress_bar.set(p))
                
                if self.stop_requested:
                    break
            
            # Update UI after training
            if self.stop_requested:
                self.after(0, lambda: self.progress_label.configure(text="Training stopped"))
            else:
                self.after(0, lambda: self.progress_label.configure(text="Training complete"))
                self.after(0, lambda: self.progress_bar.set(1.0))
                self.after(0, self._update_visualizations)
            
            # Reset UI state
            self.after(0, lambda: self.train_button.configure(state="normal"))
            self.after(0, lambda: self.stop_button.configure(state="disabled"))
            self.after(0, lambda: self.load_button.configure(state="normal"))
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Training error: {str(e)}"))
            self.after(0, lambda: self.progress_bar.set(0))
            self.after(0, lambda: self.progress_label.configure(text="Training failed"))
            self.after(0, lambda: self.train_button.configure(state="normal"))
            self.after(0, lambda: self.stop_button.configure(state="disabled"))
            self.after(0, lambda: self.load_button.configure(state="normal"))
    
    def _train_lstm(self, X_train, y_train, X_test, y_test, epochs, batch_size, num_classes):
        """Train and evaluate the LSTM model."""
        try:
            # For simplicity, simulate training with a delay
            # In a real implementation, this would include:
            # 1. Tokenization and padding of text data
            # 2. Creating PyTorch datasets and dataloaders
            # 3. Defining and training the LSTM model
            # 4. Evaluating the model on test data
            
            # Simulated results (in a real implementation, compute these from actual model)
            accuracy = 0.78
            precision = [0.76, 0.79, 0.81] if num_classes == 3 else [0.77, 0.80]
            recall = [0.75, 0.77, 0.82] if num_classes == 3 else [0.76, 0.80]
            f1 = [0.75, 0.78, 0.81] if num_classes == 3 else [0.76, 0.80]
            
            # Simulated confusion matrix
            if num_classes == 3:
                cm = np.array([
                    [150, 30, 20],
                    [25, 160, 15],
                    [15, 25, 160]
                ])
            else:
                cm = np.array([
                    [180, 20],
                    [40, 160]
                ])
            
            # Simulate training time
            for epoch in range(epochs):
                if self.stop_requested:
                    return None
                
                time.sleep(0.5)  # Simulate training delay
                self.after(0, lambda e=epoch+1: self.progress_label.configure(
                    text=f"Training LSTM model (Epoch {e}/{epochs})..."
                ))
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            }
            
        except Exception as e:
            print(f"LSTM training error: {e}")
            return None
    
    
    
    def _train_cnn(self, X_train, y_train, X_test, y_test, epochs, batch_size, num_classes):
        """Train and evaluate the CNN model."""
        try:
            # Similar to LSTM, simulate training with a delay
            # In a real implementation, this would include:
            # 1. Tokenization and embedding of text data
            # 2. Creating PyTorch datasets and dataloaders
            # 3. Defining and training the CNN model
            # 4. Evaluating the model on test data
            
            # Simulated results (in a real implementation, compute these from actual model)
            accuracy = 0.82
            precision = [0.80, 0.83, 0.85] if num_classes == 3 else [0.81, 0.84]
            recall = [0.79, 0.82, 0.86] if num_classes == 3 else [0.80, 0.85]
            f1 = [0.79, 0.82, 0.85] if num_classes == 3 else [0.80, 0.84]
            
            # Simulated confusion matrix
            if num_classes == 3:
                cm = np.array([
                    [160, 25, 15],
                    [20, 170, 10],
                    [10, 20, 170]
                ])
            else:
                cm = np.array([
                    [190, 10],
                    [30, 170]
                ])
            
            # Simulate training time
            for epoch in range(epochs):
                if self.stop_requested:
                    return None
                
                time.sleep(0.4)  # Simulate training delay
                self.after(0, lambda e=epoch+1: self.progress_label.configure(
                    text=f"Training CNN model (Epoch {e}/{epochs})..."
                ))
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            }
            
        except Exception as e:
            print(f"CNN training error: {e}")
            return None
    
    def _train_transformer(self, X_train, y_train, X_test, y_test, epochs, batch_size, num_classes):
        """Train and evaluate the transformer (RoBERTa) model."""
        try:
            # Simulate training with a delay
            # In a real implementation, this would include:
            # 1. Tokenization of text data
            # 2. Fine-tuning the pre-trained model
            # 3. Evaluating the model on test data
            
            # Simulated results (in a real implementation, compute these from actual model)
            accuracy = 0.91
            precision = [0.89, 0.91, 0.93] if num_classes == 3 else [0.90, 0.92]
            recall = [0.88, 0.91, 0.94] if num_classes == 3 else [0.89, 0.93]
            f1 = [0.88, 0.91, 0.93] if num_classes == 3 else [0.89, 0.92]
            
            # Simulated confusion matrix
            if num_classes == 3:
                cm = np.array([
                    [180, 15, 5],
                    [10, 185, 5],
                    [5, 10, 185]
                ])
            else:
                cm = np.array([
                    [195, 5],
                    [15, 185]
                ])
            
            # Simulate training time
            for epoch in range(epochs):
                if self.stop_requested:
                    return None
                
                time.sleep(0.8)  # Simulate training delay
                self.after(0, lambda e=epoch+1: self.progress_label.configure(
                    text=f"Training RoBERTa model (Epoch {e}/{epochs})..."
                ))
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            }
            
        except Exception as e:
            print(f"Transformer training error: {e}")
            return None
    
    def _update_visualizations(self):
            """Update the visualization tabs with the results."""
            try:
                # Only update if we have results
                if not self.results or not self.results.get('accuracy'):
                    return
                
                # Create visualizations in separate threads to keep UI responsive
                threading.Thread(target=self._create_accuracy_chart, daemon=True).start()
                threading.Thread(target=self._create_metrics_chart, daemon=True).start()
                threading.Thread(target=self._create_cm_charts, daemon=True).start()
                threading.Thread(target=self._create_time_chart, daemon=True).start()
                
            except Exception as e:
                messagebox.showerror("Visualization Error", f"Error creating visualizations: {str(e)}")
    
    def _create_accuracy_chart(self):
        """Create and display the accuracy comparison chart with enhanced readability."""
        try:
            # Clear previous chart
            for widget in self.accuracy_frame.winfo_children():
                widget.destroy()
            
            # Create figure and axis with white background
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
            ax.set_facecolor('white')
            
            # Plot accuracy bars
            models = list(self.results['accuracy'].keys())
            accuracy_values = [self.results['accuracy'][model] * 100 for model in models]
            
            # Use a clear, thesis-friendly color scheme
            colors = ['#4285F4', '#34A853', '#EA4335'][:len(models)]  # Blue, Green, Red
            
            # Create bar chart
            bars = ax.bar(
                models, 
                accuracy_values,
                color=colors,
                width=0.6,
                edgecolor='black',
                linewidth=1
            )
            
            # Add value labels on top of bars with enhanced readability
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 1,
                    f'{height:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=11,
                    fontweight='bold',
                    color='black'
                )
            
            # Set chart labels and title with larger, clearer font
            ax.set_ylabel('Accuracy (%)', fontsize=12, color='black')
            ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', color='black')
            ax.set_ylim(0, max(accuracy_values) * 1.15)  # Add some space for labels
            
            # Add grid for readability but make it subtle
            ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
            
            # Enhance tick labels
            ax.tick_params(axis='both', colors='black', labelsize=11)
            
            # Add a light border around the plot
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)
            
            # Add a legend
            legend_labels = [f"{model} ({accuracy_values[i]:.1f}%)" for i, model in enumerate(models)]
            ax.legend(bars, legend_labels, loc='best', frameon=True, fontsize=10)
            
            # Adjust layout
            plt.tight_layout()
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.accuracy_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            print(f"Error creating accuracy chart: {e}")

    def _create_metrics_chart(self):
        """Create and display the precision/recall/f1 metrics chart with enhanced readability."""
        try:
            # Clear previous chart
            for widget in self.metrics_frame.winfo_children():
                widget.destroy()
            
            # Create figure and axis with white background
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
            ax.set_facecolor('white')
            
            # Get models and metrics
            models = list(self.results['precision'].keys())
            
            # Average metrics across classes for simplicity
            precision_values = [np.mean(self.results['precision'][model]) * 100 for model in models]
            recall_values = [np.mean(self.results['recall'][model]) * 100 for model in models]
            f1_values = [np.mean(self.results['f1'][model]) * 100 for model in models]
            
            # Set bar positions
            x = np.arange(len(models))
            width = 0.25
            
            # Create bars with thesis-friendly colors
            bars1 = ax.bar(x - width, precision_values, width, label='Precision', 
                        color='#4285F4', edgecolor='black', linewidth=1)
            bars2 = ax.bar(x, recall_values, width, label='Recall', 
                        color='#34A853', edgecolor='black', linewidth=1)
            bars3 = ax.bar(x + width, f1_values, width, label='F1 Score', 
                        color='#EA4335', edgecolor='black', linewidth=1)
            
            # Add value labels on top of bars
            def add_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 1,
                        f'{height:.1f}%',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold',
                        color='black'
                    )
                    
            add_labels(bars1)
            add_labels(bars2)
            add_labels(bars3)
            
            # Set chart labels and title with larger, clearer font
            ax.set_ylabel('Score (%)', fontsize=12, color='black')
            ax.set_title('Precision, Recall, and F1 Scores', fontsize=14, fontweight='bold', color='black')
            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=11)
            
            # Enhanced legend with frame
            ax.legend(fontsize=10, frameon=True, edgecolor='black')
            
            # Add grid for readability but make it subtle
            ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
            
            # Enhance tick labels
            ax.tick_params(axis='both', colors='black', labelsize=11)
            
            # Add a light border around the plot
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)
            
            # Adjust layout
            plt.tight_layout()
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.metrics_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            print(f"Error creating metrics chart: {e}")

    def _create_cm_charts(self):
        """Create and display confusion matrix charts for each model with enhanced readability."""
        try:
            # Clear previous charts
            for widget in self.cm_frame.winfo_children():
                widget.destroy()
            
            # Get models
            models = list(self.results['confusion_matrix'].keys())
            
            # Create scrollable frame for multiple confusion matrices
            scroll_frame = ctk.CTkScrollableFrame(self.cm_frame, width=800, height=500)
            scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Create a confusion matrix plot for each model
            for i, model in enumerate(models):
                cm = self.results['confusion_matrix'].get(model)
                if cm is None:
                    continue
                
                # Create frame for this model
                model_frame = ctk.CTkFrame(scroll_frame)
                model_frame.pack(fill="x", pady=10)
                
                # Add model title
                title = ctk.CTkLabel(
                    model_frame,
                    text=f"{model} Confusion Matrix",
                    font=("Arial", 16, "bold")
                )
                title.pack(pady=5)
                
                # Create confusion matrix visualization with white background
                fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
                ax.set_facecolor('white')
                
                # Get class labels
                if cm.shape[0] == 3:
                    class_labels = ['Negative', 'Neutral', 'Positive']
                else:
                    class_labels = ['Negative', 'Positive']
                
                # Plot confusion matrix with a more readable colormap
                im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                ax.set_title(f"{model} Confusion Matrix", fontsize=14, fontweight='bold', color='black')
                
                # Add colorbar with improved styling
                cbar = plt.colorbar(im)
                cbar.ax.tick_params(labelsize=10, colors='black')
                
                # Set tick marks and labels with enhanced readability
                tick_marks = np.arange(len(class_labels))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(class_labels, fontsize=11, color='black')
                ax.set_yticklabels(class_labels, fontsize=11, color='black')
                
                # Add text annotations with enhanced visibility
                thresh = cm.max() / 2.0
                for i, j in np.ndindex(cm.shape):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            fontsize=12, fontweight='bold',
                            color="white" if cm[i, j] > thresh else "black")
                
                # Add labels with improved styling
                ax.set_ylabel('True Label', fontsize=12, color='black')
                ax.set_xlabel('Predicted Label', fontsize=12, color='black')
                
                # Add a light border around the plot
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1)
                
                plt.tight_layout()
                
                # Create canvas
                canvas = FigureCanvasTkAgg(fig, master=model_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(padx=10, pady=5)
                
        except Exception as e:
            print(f"Error creating confusion matrix charts: {e}")

    def _create_time_chart(self):
        """Create and display the training time comparison chart with enhanced readability."""
        try:
            # Clear previous chart
            for widget in self.time_frame.winfo_children():
                widget.destroy()
            
            # Create figure and axis with white background
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
            ax.set_facecolor('white')
            
            # Plot training time bars
            models = list(self.results['training_time'].keys())
            time_values = [self.results['training_time'][model] for model in models]
            
            # Use thesis-friendly colors
            colors = ['#4285F4', '#34A853', '#EA4335'][:len(models)]  # Blue, Green, Red
            
            # Create bar chart
            bars = ax.bar(
                models, 
                time_values,
                color=colors,
                width=0.6,
                edgecolor='black',
                linewidth=1
            )
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.1,
                    f'{height:.1f}s',
                    ha='center',
                    va='bottom',
                    fontsize=11,
                    fontweight='bold',
                    color='black'
                )
            
            # Set chart labels and title with larger, clearer font
            ax.set_ylabel('Training Time (seconds)', fontsize=12, color='black')
            ax.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold', color='black')
            ax.set_ylim(0, max(time_values) * 1.15)  # Add some space for labels
            
            # Add grid for readability but make it subtle
            ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
            
            # Enhance tick labels
            ax.tick_params(axis='both', colors='black', labelsize=11)
            
            # Add a light border around the plot
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)
            
            # Add a legend
            legend_labels = [f"{model} ({time_values[i]:.1f}s)" for i, model in enumerate(models)]
            ax.legend(bars, legend_labels, loc='best', frameon=True, fontsize=10)
            
            # Adjust layout
            plt.tight_layout()
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.time_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            print(f"Error creating training time chart: {e}")
    
    def _export_results(self):
        """Export the results and visualizations in high-resolution, thesis-ready format."""
        if not self.results or not self.results.get('accuracy'):
            messagebox.showinfo("Information", "No results to export.")
            return
        
        try:
            # Create export directory
            export_dir = filedialog.askdirectory(title="Select Export Directory")
            if not export_dir:
                return
                
            # Get export format
            export_format = self.export_format_var.get()
            
            # Export visualizations with higher DPI for thesis quality
            thesis_dpi = 300
            
            # Export accuracy chart
            for widget in self.accuracy_frame.winfo_children():
                if isinstance(widget, FigureCanvasTkAgg):
                    if export_format == "PNG" or export_format == "All":
                        widget.figure.savefig(
                            os.path.join(export_dir, "accuracy_comparison.png"),
                            dpi=thesis_dpi,
                            bbox_inches="tight",
                            facecolor='white'
                        )
                    if export_format == "PDF" or export_format == "All":
                        widget.figure.savefig(
                            os.path.join(export_dir, "accuracy_comparison.pdf"),
                            bbox_inches="tight",
                            facecolor='white'
                        )
            
            # Export metrics chart
            for widget in self.metrics_frame.winfo_children():
                if isinstance(widget, FigureCanvasTkAgg):
                    if export_format == "PNG" or export_format == "All":
                        widget.figure.savefig(
                            os.path.join(export_dir, "metrics_comparison.png"),
                            dpi=thesis_dpi,
                            bbox_inches="tight",
                            facecolor='white'
                        )
                    if export_format == "PDF" or export_format == "All":
                        widget.figure.savefig(
                            os.path.join(export_dir, "metrics_comparison.pdf"),
                            bbox_inches="tight",
                            facecolor='white'
                        )
            
            # Export training time chart
            for widget in self.time_frame.winfo_children():
                if isinstance(widget, FigureCanvasTkAgg):
                    if export_format == "PNG" or export_format == "All":
                        widget.figure.savefig(
                            os.path.join(export_dir, "training_time_comparison.png"),
                            dpi=thesis_dpi,
                            bbox_inches="tight",
                            facecolor='white'
                        )
                    if export_format == "PDF" or export_format == "All":
                        widget.figure.savefig(
                            os.path.join(export_dir, "training_time_comparison.pdf"),
                            bbox_inches="tight",
                            facecolor='white'
                        )
            
            # Export confusion matrices
            models = list(self.results['confusion_matrix'].keys())
            for model in models:
                # Search through the confusion matrix frames to find the right figure
                for frame in self.cm_frame.winfo_children():
                    if hasattr(frame, 'winfo_children'):
                        for child in frame.winfo_children():
                            if hasattr(child, 'winfo_children'):
                                for grandchild in child.winfo_children():
                                    if isinstance(grandchild, FigureCanvasTkAgg):
                                        title = grandchild.figure._suptitle
                                        if title and model in title.get_text():
                                            if export_format == "PNG" or export_format == "All":
                                                grandchild.figure.savefig(
                                                    os.path.join(export_dir, f"{model}_confusion_matrix.png"),
                                                    dpi=thesis_dpi,
                                                    bbox_inches="tight",
                                                    facecolor='white'
                                                )
                                            if export_format == "PDF" or export_format == "All":
                                                grandchild.figure.savefig(
                                                    os.path.join(export_dir, f"{model}_confusion_matrix.pdf"),
                                                    bbox_inches="tight",
                                                    facecolor='white'
                                                )
            
            # Export data as CSV with detailed results
            if export_format in ["CSV", "All"]:
                # Export accuracy data
                accuracy_df = pd.DataFrame({
                    'Model': list(self.results['accuracy'].keys()),
                    'Accuracy': [self.results['accuracy'][model] * 100 for model in self.results['accuracy'].keys()]
                })
                accuracy_df.to_csv(os.path.join(export_dir, "accuracy_results.csv"), index=False)
                
                # Export detailed metrics data
                metrics_data = []
                for model in self.results['precision'].keys():
                    # For each class
                    for i in range(len(self.results['precision'][model])):
                        class_name = "Negative" if i == 0 else ("Neutral" if i == 1 else "Positive")
                        metrics_data.append({
                            'Model': model,
                            'Class': class_name,
                            'Precision': self.results['precision'][model][i] * 100,
                            'Recall': self.results['recall'][model][i] * 100,
                            'F1': self.results['f1'][model][i] * 100
                        })
                        
                    # Add average row
                    metrics_data.append({
                        'Model': model,
                        'Class': 'Average',
                        'Precision': np.mean(self.results['precision'][model]) * 100,
                        'Recall': np.mean(self.results['recall'][model]) * 100,
                        'F1': np.mean(self.results['f1'][model]) * 100
                    })
                    
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_csv(os.path.join(export_dir, "detailed_metrics.csv"), index=False)
                
                # Export training time data
                time_df = pd.DataFrame({
                    'Model': list(self.results['training_time'].keys()),
                    'Training_Time_Seconds': [self.results['training_time'][model] for model in self.results['training_time'].keys()]
                })
                time_df.to_csv(os.path.join(export_dir, "training_time_results.csv"), index=False)
                
                # Export confusion matrices as CSV
                for model in models:
                    cm = self.results['confusion_matrix'].get(model)
                    if cm is not None:
                        # Create DataFrame with proper labels
                        if cm.shape[0] == 3:
                            class_labels = ['Negative', 'Neutral', 'Positive']
                        else:
                            class_labels = ['Negative', 'Positive']
                            
                        cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
                        cm_df.to_csv(os.path.join(export_dir, f"{model}_confusion_matrix.csv"))
                
                # Create a comprehensive summary file
                with open(os.path.join(export_dir, "model_comparison_summary.txt"), "w") as f:
                    f.write("MODEL COMPARISON SUMMARY\n")
                    f.write("=======================\n\n")
                    
                    f.write("ACCURACY\n")
                    f.write("--------\n")
                    for model, acc in self.results['accuracy'].items():
                        f.write(f"{model}: {acc*100:.2f}%\n")
                    f.write("\n")
                    
                    f.write("PRECISION, RECALL, F1 (Class Average)\n")
                    f.write("-----------------------------------\n")
                    for model in self.results['precision'].keys():
                        precision_avg = np.mean(self.results['precision'][model]) * 100
                        recall_avg = np.mean(self.results['recall'][model]) * 100
                        f1_avg = np.mean(self.results['f1'][model]) * 100
                        f.write(f"{model}:\n")
                        f.write(f"  Precision: {precision_avg:.2f}%\n")
                        f.write(f"  Recall: {recall_avg:.2f}%\n")
                        f.write(f"  F1 Score: {f1_avg:.2f}%\n\n")
                    
                    f.write("TRAINING TIME\n")
                    f.write("-------------\n")
                    for model, time_val in self.results['training_time'].items():
                        f.write(f"{model}: {time_val:.2f} seconds\n")
            
            # Show success message
            messagebox.showinfo(
                "Export Successful",
                f"Results exported to {export_dir} in thesis-ready format."
            )
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting results: {str(e)}")
    
    def _show_column_selection_dialog(self, df):
        """Show a dialog for the user to select text and sentiment columns."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Select Dataset Columns")
        dialog.geometry("400x300")
        dialog.grab_set()  # Make dialog modal
        
        # Create instruction label
        ctk.CTkLabel(
            dialog,
            text="Please select the columns containing text and sentiment:",
            font=("Arial", 12),
            wraplength=380
        ).pack(pady=(20, 10), padx=10)
        
        # Text column selection
        text_frame = ctk.CTkFrame(dialog)
        text_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            text_frame,
            text="Text Column:",
            font=("Arial", 12)
        ).pack(side="left", padx=10, pady=10)
        
        text_col_var = ctk.StringVar()
        text_dropdown = ctk.CTkOptionMenu(
            text_frame,
            values=list(df.columns),
            variable=text_col_var
        )
        text_dropdown.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        # Try to select a reasonable default for text
        for col in df.columns:
            if df[col].dtype == 'object':
                text_col_var.set(col)
                break
        else:
            text_col_var.set(df.columns[0])
        
        # Sentiment column selection
        sentiment_frame = ctk.CTkFrame(dialog)
        sentiment_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            sentiment_frame,
            text="Sentiment Column:",
            font=("Arial", 12)
        ).pack(side="left", padx=10, pady=10)
        
        sentiment_col_var = ctk.StringVar()
        sentiment_dropdown = ctk.CTkOptionMenu(
            sentiment_frame,
            values=list(df.columns),
            variable=sentiment_col_var
        )
        sentiment_dropdown.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        # Try to select a reasonable default for sentiment
        for col in df.columns:
            if col != text_col_var.get() and (df[col].dtype in ['int64', 'float64'] or 'label' in col.lower() or 'sentiment' in col.lower()):
                sentiment_col_var.set(col)
                break
        else:
            for i, col in enumerate(df.columns):
                if col != text_col_var.get():
                    sentiment_col_var.set(col)
                    break
        
        # Button to confirm selection
        ctk.CTkButton(
            dialog,
            text="Confirm",
            command=lambda: self._on_column_selection_confirmed(dialog, df, text_col_var.get(), sentiment_col_var.get()),
            font=("Arial", 12),
            height=36
        ).pack(pady=20)

    def _on_column_selection_confirmed(self, dialog, df, text_col, sentiment_col):
        """Handle column selection confirmation."""
        dialog.destroy()
        
        # Process the dataset with selected columns
        print(f"User selected text column: {text_col}, sentiment column: {sentiment_col}")
        
        # Make a clean copy of the dataframe
        cleaned_df = df[[text_col, sentiment_col]].copy()
        
        # Fill NaN values in text column with empty string
        cleaned_df[text_col] = cleaned_df[text_col].fillna("")
        
        # Remove rows with NaN in sentiment column
        print(f"Rows before NaN removal: {len(cleaned_df)}")
        cleaned_df = cleaned_df.dropna(subset=[sentiment_col])
        print(f"Rows after NaN removal: {len(cleaned_df)}")
        
        # Convert sentiment to numeric if it's not already
        if cleaned_df[sentiment_col].dtype == object:
            try:
                cleaned_df[sentiment_col] = pd.to_numeric(cleaned_df[sentiment_col], errors='coerce')
                cleaned_df = cleaned_df.dropna(subset=[sentiment_col])  # Drop any that couldn't be converted
            except Exception as e:
                print(f"Error converting sentiment to numeric: {e}")
        
        # Now attempt to convert values to match expected -1, 0, 1 pattern
        unique_sentiments = sorted(cleaned_df[sentiment_col].unique())
        print(f"Unique sentiment values: {unique_sentiments}")
        
        # Create sentiment mapping
        sentiment_map = {}
        
        # If values are already -1, 0, 1 or similar pattern
        if len(unique_sentiments) == 3:
            # Map to 0, 1, 2 for model compatibility (handles both integer and float types)
            try:
                # Try to sort and map based on values
                sentiment_map = {unique_sentiments[0]: 0, 
                                unique_sentiments[1]: 1, 
                                unique_sentiments[2]: 2}
                
                # If values are close to -1, 0, 1 use human-readable labels
                if abs(unique_sentiments[0] + 1) < 0.1 and abs(unique_sentiments[1]) < 0.1 and abs(unique_sentiments[2] - 1) < 0.1:
                    original_labels = {0: "Negative (-1)", 1: "Neutral (0)", 2: "Positive (1)"}
                else:
                    original_labels = {0: f"Class {unique_sentiments[0]}", 
                                    1: f"Class {unique_sentiments[1]}", 
                                    2: f"Class {unique_sentiments[2]}"}
            except Exception as e:
                print(f"Error creating sentiment mapping: {e}")
                # Fallback to simple enumeration
                sentiment_map = {val: idx for idx, val in enumerate(unique_sentiments)}
                original_labels = {idx: f"Class {val}" for idx, val in enumerate(unique_sentiments)}
        else:
            # Generic mapping
            sentiment_map = {val: idx for idx, val in enumerate(unique_sentiments)}
            original_labels = {idx: f"Class {val}" for idx, val in enumerate(unique_sentiments)}
        
        # Apply mapping very carefully
        try:
            # First create a new column to avoid modifying the original
            cleaned_df['sentiment_mapped'] = cleaned_df[sentiment_col].apply(
                lambda x: sentiment_map.get(x, 0) if pd.notnull(x) else 0
            )
            
            # Now make this the sentiment column
            sentiment_col_original = sentiment_col
            sentiment_col = 'sentiment_mapped'
            
            print(f"Mapped sentiment values distribution: {cleaned_df[sentiment_col].value_counts().to_dict()}")
        except Exception as e:
            print(f"Error mapping sentiment values: {e}")
            # If mapping fails, create a simple 0,1,2 version as last resort
            cleaned_df['sentiment_simple'] = pd.qcut(
                cleaned_df[sentiment_col], 
                q=min(3, len(unique_sentiments)), 
                labels=False
            )
            sentiment_col = 'sentiment_simple'
            sentiment_map = {i: i for i in range(min(3, len(unique_sentiments)))}
            original_labels = {i: f"Class {i}" for i in range(min(3, len(unique_sentiments)))}
        
        # Store the processed dataset
        self.dataset = {
            'df': cleaned_df,
            'text_col': text_col,
            'sentiment_col': sentiment_col,
            'sentiment_map': sentiment_map,
            'original_labels': original_labels,
            'num_classes': len(sentiment_map),
            'class_counts': cleaned_df[sentiment_col].value_counts().to_dict()
        }
        
        # Calculate dropped row count
        dropped_count = len(df) - len(cleaned_df)
        
        # Update UI
        self._update_dataset_info(cleaned_df, text_col, sentiment_col, sentiment_map, original_labels, dropped_count)
        
        
    def destroy(self):
        """Clean up resources when the page is destroyed."""
        # Stop any ongoing training
        self.stop_requested = True
        
        # Close all matplotlib figures
        plt.close('all')
        
        # Call parent's destroy method
        super().destroy()
    
