# pages/page5.py
import customtkinter as ctk
import threading
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox
import torch
from datetime import datetime

# Import custom modules
from utils.data_processor import DataProcessor
from utils.visualization import ModelComparisonVisualizer
from models.model_trainer import ModelTrainer

class Page5(ctk.CTkFrame):
    """
    Model Training and Evaluation Page.
    
    This page provides comprehensive tools for:
    1. Loading and preprocessing datasets
    2. Splitting data into train/test sets
    3. Training various model architectures
    4. Evaluating and comparing model performance
    5. Visualizing results for publication
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # Initialize state
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        self.visualizer = ModelComparisonVisualizer()
        
        # Initialize UI components dictionary (to keep track of dynamic elements)
        self.ui_components = {}
        
        # Tracks run models and their metrics
        self.model_results = {}
        
        # Track active threads
        self.active_threads = []
        
        # Create the main UI
        self._create_ui()
        
        # Scan for existing models at startup
        self._update_model_lists()
        
    def _create_ui(self):
        """Create the main user interface components."""
        # Create a master scrollable frame
        self.master_scroll = ctk.CTkScrollableFrame(self, width=950, height=800)
        self.master_scroll.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Create sections
        self._create_header_section()
        self._create_data_section()
        self._create_model_section()
        self._create_training_section()
        self._create_evaluation_section()
        self._create_comparison_section()
        
    def _create_header_section(self):
        """Create the header section with title and description."""
        header_frame = ctk.CTkFrame(self.master_scroll)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="Model Training & Evaluation",
            font=("Arial", 24, "bold")
        )
        title_label.pack(pady=(10, 5))
        
        # Description
        desc_label = ctk.CTkLabel(
            header_frame,
            text="Train, evaluate, and compare sentiment analysis models",
            font=("Arial", 14),
            text_color="#888888"
        )
        desc_label.pack(pady=(0, 10))
        
    def _create_data_section(self):
        """Create the data loading and preprocessing section."""
        data_frame = ctk.CTkFrame(self.master_scroll)
        data_frame.pack(fill="x", padx=5, pady=5)
        
        # Section title
        section_label = ctk.CTkLabel(
            data_frame,
            text="1. Data Preparation",
            font=("Arial", 18, "bold")
        )
        section_label.pack(pady=(10, 5), padx=10, anchor="w")
        
        # Data loading controls
        load_frame = ctk.CTkFrame(data_frame)
        load_frame.pack(fill="x", padx=10, pady=5)
        
        load_label = ctk.CTkLabel(
            load_frame,
            text="Dataset:",
            font=("Arial", 14)
        )
        load_label.pack(side="left", padx=(10, 5), pady=10)
        
        self.data_path_var = ctk.StringVar(value="No file selected")
        data_path_label = ctk.CTkLabel(
            load_frame,
            textvariable=self.data_path_var,
            font=("Arial", 12),
            width=400,
            anchor="w"
        )
        data_path_label.pack(side="left", padx=5, pady=10, fill="x", expand=True)
        
        load_button = ctk.CTkButton(
            load_frame,
            text="Load Dataset",
            command=self._load_dataset,
            width=150
        )
        load_button.pack(side="right", padx=10, pady=10)
        
        # Add sample option for large datasets
        self.use_sample_var = ctk.BooleanVar(value=True)
        sample_check = ctk.CTkCheckBox(
            load_frame,
            text="Use sample (10,000 rows)",
            variable=self.use_sample_var
        )
        sample_check.pack(side="left", padx=10, pady=10)
        
        load_button = ctk.CTkButton(
            load_frame,
            text="Load Dataset",
            command=self._load_dataset,
            width=150
        )
        load_button.pack(side="right", padx=10, pady=10)
        
        # Data splitting controls
        split_frame = ctk.CTkFrame(data_frame)
        split_frame.pack(fill="x", padx=10, pady=5)
        
        split_label = ctk.CTkLabel(
            split_frame,
            text="Train/Test Split:",
            font=("Arial", 14)
        )
        split_label.pack(side="left", padx=(10, 5), pady=10)
        
        self.split_ratio_var = ctk.StringVar(value="80/20")
        split_options = ["50/50", "60/40", "70/30", "80/20", "90/10"]
        split_menu = ctk.CTkOptionMenu(
            split_frame,
            values=split_options,
            variable=self.split_ratio_var,
            width=100
        )
        split_menu.pack(side="left", padx=10, pady=10)
        
        self.use_validation_var = ctk.BooleanVar(value=True)
        validation_check = ctk.CTkCheckBox(
            split_frame,
            text="Use validation set",
            variable=self.use_validation_var
        )
        validation_check.pack(side="left", padx=10, pady=10)
        
        split_button = ctk.CTkButton(
            split_frame,
            text="Prepare Data Split",
            command=self._prepare_data_split,
            width=150
        )
        split_button.pack(side="right", padx=10, pady=10)
        
        # Preprocessing options
        preprocess_frame = ctk.CTkFrame(data_frame)
        preprocess_frame.pack(fill="x", padx=10, pady=5)
        
        preprocess_label = ctk.CTkLabel(
            preprocess_frame,
            text="Preprocessing Options:",
            font=("Arial", 14)
        )
        preprocess_label.pack(side="left", padx=(10, 5), pady=10)
        
        # Create checkboxes for preprocessing options
        self.preprocess_vars = {}
        options = [
            ("remove_urls", "Remove URLs"),
            ("remove_mentions", "Remove @mentions"),
            ("lowercase", "Convert to lowercase"),
            ("remove_punctuation", "Remove punctuation"),
            ("remove_hashtags", "Remove hashtags"),
            ("remove_numbers", "Remove numbers")
        ]
        
        for i, (option_key, option_text) in enumerate(options):
            self.preprocess_vars[option_key] = ctk.BooleanVar(value=True)
            checkbox = ctk.CTkCheckBox(
                preprocess_frame,
                text=option_text,
                variable=self.preprocess_vars[option_key]
            )
            checkbox.pack(side="left", padx=10, pady=10)
        
        # Data summary
        self.data_summary_frame = ctk.CTkFrame(data_frame)
        self.data_summary_frame.pack(fill="x", padx=10, pady=5)
        
        self.data_summary_label = ctk.CTkLabel(
            self.data_summary_frame,
            text="Load a dataset to see summary statistics",
            font=("Arial", 12),
            justify="left",
            wraplength=900
        )
        self.data_summary_label.pack(pady=10, padx=10, fill="x")
    
    def _create_model_section(self):
        """Create the model selection and configuration section."""
        model_frame = ctk.CTkFrame(self.master_scroll)
        model_frame.pack(fill="x", padx=5, pady=5)
        
        # Section title
        section_label = ctk.CTkLabel(
            model_frame,
            text="2. Model Selection",
            font=("Arial", 18, "bold")
        )
        section_label.pack(pady=(10, 5), padx=10, anchor="w")
        
        
        local_model_frame = ctk.CTkFrame(model_frame)
        local_model_frame.pack(fill="x", padx=10, pady=5)
        
        local_model_label = ctk.CTkLabel(
            local_model_frame,
            text="Load Local Model:",
            font=("Arial", 14, "bold")
        )
        local_model_label.pack(side="left", padx=10, pady=10)
        
        self.local_model_path_var = ctk.StringVar(value="No model selected")
        local_model_path_label = ctk.CTkLabel(
            local_model_frame,
            textvariable=self.local_model_path_var,
            font=("Arial", 12),
            width=300,
            anchor="w"
        )
        local_model_path_label.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        browse_model_button = ctk.CTkButton(
            local_model_frame,
            text="Browse...",
            command=self._browse_local_model,
            width=100
        )
        browse_model_button.pack(side="left", padx=10, pady=10)
        
        load_model_button = ctk.CTkButton(
            local_model_frame,
            text="Load & Evaluate Model",
            command=self._load_and_evaluate_local_model,
            width=200
        )
        load_model_button.pack(side="right", padx=10, pady=10)
        
        pretrained_frame = ctk.CTkFrame(model_frame)
        pretrained_frame.pack(fill="x", padx=10, pady=5)
        
        pretrained_label = ctk.CTkLabel(
            pretrained_frame,
            text="Evaluate Pre-trained Model:",
            font=("Arial", 14, "bold")
        )
        pretrained_label.pack(side="left", padx=10, pady=10)
        
        # Dropdown for pretrained model selection
        self.pretrained_model_var = ctk.StringVar(value="cardiffnlp/twitter-roberta-base-sentiment")
        pretrained_options = [
            "cardiffnlp/twitter-roberta-base-sentiment",
            "distilbert-base-uncased-finetuned-sst-2-english",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "finiteautomata/bertweet-base-sentiment-analysis"
        ]
        
        pretrained_dropdown = ctk.CTkOptionMenu(
            pretrained_frame,
            values=pretrained_options,
            variable=self.pretrained_model_var,
            width=350
        )
        pretrained_dropdown.pack(side="left", padx=10, pady=10)
        
        # Button to evaluate the pretrained model without training
        pretrained_eval_button = ctk.CTkButton(
            pretrained_frame,
            text="Evaluate Without Training",
            command=self._evaluate_pretrained,
            width=200
        )
        pretrained_eval_button.pack(side="right", padx=10, pady=10)
        
        
        # Model type selection
        model_type_frame = ctk.CTkFrame(model_frame)
        model_type_frame.pack(fill="x", padx=10, pady=5)
        
        # Title and description
        model_header = ctk.CTkLabel(
            model_type_frame,
            text="Select model architecture:",
            font=("Arial", 14, "bold")
        )
        model_header.pack(pady=5, padx=10, anchor="w")
        
        # Create tabview for different model types
        self.model_tabview = ctk.CTkTabview(model_frame)
        self.model_tabview.pack(fill="x", padx=10, pady=10)
        
        # Add tabs for different model types
        self.model_tabview.add("Pre-trained")
        self.model_tabview.add("LSTM")
        self.model_tabview.add("CNN")
        self.model_tabview.add("Custom")
        
        # Pre-trained models tab
        pretrained_tab = self.model_tabview.tab("Pre-trained")
        
        self.pretrained_var = ctk.StringVar(value="cardiffnlp/twitter-roberta-base-sentiment")
        pretrained_options = [
            "cardiffnlp/twitter-roberta-base-sentiment",
            "distilbert-base-uncased-finetuned-sst-2-english",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "finiteautomata/bertweet-base-sentiment-analysis"
        ]
        
        pretrained_label = ctk.CTkLabel(
            pretrained_tab,
            text="Pre-trained Model:",
            font=("Arial", 14)
        )
        pretrained_label.pack(pady=5, padx=10, anchor="w")
        
        pretrained_menu = ctk.CTkOptionMenu(
            pretrained_tab,
            values=pretrained_options,
            variable=self.pretrained_var,
            width=400
        )
        pretrained_menu.pack(pady=5, padx=10, anchor="w")
        
        self.finetune_var = ctk.BooleanVar(value=True)
        finetune_check = ctk.CTkCheckBox(
            pretrained_tab,
            text="Fine-tune model",
            variable=self.finetune_var
        )
        finetune_check.pack(pady=5, padx=10, anchor="w")
        
        # LSTM models tab
        lstm_tab = self.model_tabview.tab("LSTM")
        
        # LSTM configuration
        lstm_config_frame = ctk.CTkFrame(lstm_tab)
        lstm_config_frame.pack(fill="x", padx=10, pady=5)
        
        # Create a grid of LSTM parameters
        params = [
            ("LSTM Layers:", "lstm_layers", ["1", "2", "3"], "2"),
            ("Hidden Size:", "lstm_hidden_size", ["128", "256", "384", "512"], "256"),
            ("Embedding Dim:", "lstm_embedding_dim", ["100", "200", "300"], "300"),
            ("Dropout Rate:", "lstm_dropout", ["0.1", "0.2", "0.3", "0.5"], "0.5"),
            ("RNN Type:", "lstm_rnn_type", ["LSTM", "GRU"], "LSTM"),
            ("Embedding Dropout:", "lstm_embedding_dropout", ["0.0", "0.1", "0.2", "0.3", "0.5"], "0.2")
        ]
        
        self.lstm_params = {}
        
        for i, (label_text, param_name, options, default) in enumerate(params):
            label = ctk.CTkLabel(
                lstm_config_frame,
                text=label_text,
                font=("Arial", 14)
            )
            label.grid(row=i, column=0, padx=10, pady=5, sticky="w")
            
            self.lstm_params[param_name] = ctk.StringVar(value=default)
            menu = ctk.CTkOptionMenu(
                lstm_config_frame,
                values=options,
                variable=self.lstm_params[param_name],
                width=100
            )
            menu.grid(row=i, column=1, padx=10, pady=5, sticky="w")
        
        # Add bidirectional and attention options
        self.lstm_bidirectional = ctk.BooleanVar(value=True)
        bidirectional_check = ctk.CTkCheckBox(
            lstm_config_frame,
            text="Bidirectional",
            variable=self.lstm_bidirectional
        )
        bidirectional_check.grid(row=len(params), column=0, padx=10, pady=5, sticky="w")

        self.lstm_use_attention = ctk.BooleanVar(value=True)
        attention_check = ctk.CTkCheckBox(
            lstm_config_frame,
            text="Use Attention",
            variable=self.lstm_use_attention
        )
        attention_check.grid(row=len(params), column=1, padx=10, pady=5, sticky="w")
        
        # CNN models tab
        cnn_tab = self.model_tabview.tab("CNN")
        
        # CNN configuration
        cnn_config_frame = ctk.CTkFrame(cnn_tab)
        cnn_config_frame.pack(fill="x", padx=10, pady=5)
        
        # Create a grid of CNN parameters
        cnn_params = [
            ("Filter Sizes:", "cnn_filter_sizes", ["[3,4,5]", "[2,3,4]", "[1,2,3,4,5]"], "[3,4,5]"),
            ("Num Filters:", "cnn_num_filters", ["100", "200", "300"], "200"),
            ("Embedding Dim:", "cnn_embedding_dim", ["100", "200", "300"], "300"),
            ("Dropout Rate:", "cnn_dropout", ["0.1", "0.2", "0.3", "0.5"], "0.5"),
            ("Embedding Dropout:", "cnn_embedding_dropout", ["0.0", "0.1", "0.2", "0.3", "0.5"], "0.2"),
            ("Activation:", "cnn_activation", ["relu", "leaky_relu", "tanh", "elu"], "relu"),
            ("Pooling:", "cnn_pool_type", ["max", "avg", "adaptive"], "max")
        ]
        
        self.cnn_params = {}
        
        for i, (label_text, param_name, options, default) in enumerate(cnn_params):
            label = ctk.CTkLabel(
                cnn_config_frame,
                text=label_text,
                font=("Arial", 14)
            )
            label.grid(row=i, column=0, padx=10, pady=5, sticky="w")
            
            self.cnn_params[param_name] = ctk.StringVar(value=default)
            menu = ctk.CTkOptionMenu(
                cnn_config_frame,
                values=options,
                variable=self.cnn_params[param_name],
                width=100
            )
            menu.grid(row=i, column=1, padx=10, pady=5, sticky="w")
        
        # Add BatchNorm option (checkbox below the grid)
        self.cnn_batch_norm = ctk.BooleanVar(value=True)
        batchnorm_check = ctk.CTkCheckBox(
            cnn_config_frame,
            text="Batch Normalization",
            variable=self.cnn_batch_norm
        )
        batchnorm_check.grid(row=len(cnn_params), column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Custom models tab
        custom_tab = self.model_tabview.tab("Custom")
        
        custom_label = ctk.CTkLabel(
            custom_tab,
            text="Coming soon: Custom model architecture builder",
            font=("Arial", 14)
        )
        custom_label.pack(pady=20, padx=20)
    
    def _browse_local_model(self):
        """Browse for a local model directory."""
        model_dir = filedialog.askdirectory(
            title="Select Model Directory"
        )
        
        if model_dir:
            self.local_model_path_var.set(model_dir)

    def _load_and_evaluate_local_model(self):
        """Load and evaluate a model from a local directory."""
        model_dir = self.local_model_path_var.get()
        
        if model_dir == "No model selected" or not os.path.exists(model_dir):
            messagebox.showinfo("Information", "Please select a valid model directory.")
            return
        
        if self.data_processor.test_data is None:
            messagebox.showinfo("Information", "Please prepare data split first.")
            return
        
        # Generate a unique name for the model
        model_name = f"local_model_{os.path.basename(model_dir)}_{int(time.time())}"
        
        # Update UI
        self.progress_text.configure(text=f"Loading model from: {model_dir}")
        self.progress_bar.set(0.2)
        
        # Start evaluation in a separate thread
        thread = threading.Thread(
            target=self._evaluate_local_model_thread,
            args=(model_dir, model_name),
            daemon=True
        )
        thread.start()
        self.active_threads.append(thread)

    def _evaluate_local_model_thread(self, model_dir, model_name):
        """Run local model evaluation in a background thread."""
        try:
            # Update progress
            self.after(0, lambda: self.progress_bar.set(0.4))
            self.after(0, lambda: self.progress_text.configure(text="Loading local model..."))
            
            # Evaluate the model
            results = self.model_trainer.load_and_evaluate_local_model(
                data_processor=self.data_processor,
                model_name=model_name,
                model_dir=model_dir
            )
            
            # Update progress
            self.after(0, lambda: self.progress_bar.set(1.0))
            
            # Update model lists to include the new evaluation
            self.after(0, self._update_model_lists)
            
            # Show success message
            self.after(0, lambda: self.progress_text.configure(
                text=f"Evaluation complete. Accuracy: {results['accuracy']:.4f}"
            ))
            
            # Update evaluation visualization if this is the currently selected model
            self.after(0, lambda: self.eval_model_var.set(model_name))
            self.after(0, lambda: self._update_visualization())
            
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda msg=error_msg: self._log_message(f"Error during evaluation: {msg}"))
            self.after(0, lambda msg=error_msg: self.progress_text.configure(text=f"Evaluation failed: {msg}"))
            self.after(0, lambda msg=error_msg: messagebox.showerror("Evaluation Error", msg))
    
    
    def _create_training_section(self):
        """Create the model training section."""
        training_frame = ctk.CTkFrame(self.master_scroll)
        training_frame.pack(fill="x", padx=5, pady=5)
        
        # Section title
        section_label = ctk.CTkLabel(
            training_frame,
            text="3. Model Training",
            font=("Arial", 18, "bold")
        )
        section_label.pack(pady=(10, 5), padx=10, anchor="w")
        
        # Training parameters
        params_frame = ctk.CTkFrame(training_frame)
        params_frame.pack(fill="x", padx=10, pady=5)
        
        # Create a grid for training parameters
        train_params = [
            ("Batch Size:", "batch_size", ["16", "32", "64"], "32"),
            ("Learning Rate:", "learning_rate", ["0.001", "0.0005", "0.0001"], "0.0005"),
            ("Epochs:", "epochs", ["3", "5", "10", "20"], "5"),
            ("Optimizer:", "optimizer", ["Adam", "AdamW", "SGD", "RMSprop"], "AdamW"),
            ("Weight Decay:", "weight_decay", ["0.0", "0.0001", "0.0005", "0.001"], "0.0001"),
            ("Scheduler:", "scheduler", ["none", "step", "cosine", "plateau"], "plateau"),
            ("Clip Gradients:", "clip_grad", ["True", "False"], "True"),
            ("Max Grad Norm:", "max_grad_norm", ["0.5", "1.0", "2.0", "5.0"], "1.0")
        ]
        
        self.train_params = {}
        # Device selection and indicator
        device_row = ctk.CTkFrame(training_frame)
        device_row.pack(fill="x", padx=10, pady=(0, 5))
        ctk.CTkLabel(device_row, text="Compute Device:", font=("Arial", 13)).pack(side="left", padx=6)
        self.device_pref_var = ctk.StringVar(value="Auto")
        ctk.CTkOptionMenu(device_row, values=["Auto", "GPU", "CPU"], variable=self.device_pref_var,
                          command=self._on_device_change, width=100).pack(side="left", padx=6)
        self.device_info_label = ctk.CTkLabel(device_row, text=f"Device: {self.model_trainer.current_device_info()}")
        self.device_info_label.pack(side="left", padx=10)
        # Manual overrides (optional)
        manual_frame = ctk.CTkFrame(training_frame)
        manual_frame.pack(fill="x", padx=10, pady=(0, 5))
        manual_title = ctk.CTkLabel(manual_frame, text="Manual Overrides (optional)", font=("Arial", 13, "bold"))
        manual_title.grid(row=0, column=0, columnspan=6, padx=10, pady=(8, 2), sticky="w")

        self.manual_batch_size = ctk.StringVar(value="")
        self.manual_learning_rate = ctk.StringVar(value="")
        self.manual_epochs = ctk.StringVar(value="")

        ctk.CTkLabel(manual_frame, text="Batch Size:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(manual_frame, textvariable=self.manual_batch_size, width=100, placeholder_text="e.g. 48").grid(row=1, column=1, padx=5, pady=5)
        ctk.CTkLabel(manual_frame, text="Learning Rate:").grid(row=1, column=2, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(manual_frame, textvariable=self.manual_learning_rate, width=120, placeholder_text="e.g. 0.0003").grid(row=1, column=3, padx=5, pady=5)
        ctk.CTkLabel(manual_frame, text="Epochs:").grid(row=1, column=4, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(manual_frame, textvariable=self.manual_epochs, width=100, placeholder_text="e.g. 12").grid(row=1, column=5, padx=5, pady=5)
        
        for i, (label_text, param_name, options, default) in enumerate(train_params):
            label = ctk.CTkLabel(
                params_frame,
                text=label_text,
                font=("Arial", 14)
            )
            label.grid(row=i // 2, column=i % 2 * 2, padx=10, pady=5, sticky="w")
            
            self.train_params[param_name] = ctk.StringVar(value=default)
            menu = ctk.CTkOptionMenu(
                params_frame,
                values=options,
                variable=self.train_params[param_name],
                width=100
            )
            menu.grid(row=i // 2, column=i % 2 * 2 + 1, padx=10, pady=5, sticky="w")
        
        # Model name entry
        name_frame = ctk.CTkFrame(training_frame)
        name_frame.pack(fill="x", padx=10, pady=5)
        
        name_label = ctk.CTkLabel(
            name_frame,
            text="Model Name:",
            font=("Arial", 14)
        )
        name_label.pack(side="left", padx=10, pady=5)
        
        self.model_name_var = ctk.StringVar(value=f"model_{int(time.time())}")
        name_entry = ctk.CTkEntry(
            name_frame,
            textvariable=self.model_name_var,
            width=300
        )
        name_entry.pack(side="left", padx=10, pady=5)
        
        # Training button
        train_button = ctk.CTkButton(
            training_frame,
            text="Train Model",
            command=self._train_model,
            width=150,
            height=40,
            font=("Arial", 14, "bold"),
            fg_color="#0078D7",
            hover_color="#005A9E"
        )
        train_button.pack(padx=10, pady=10)
        
        # Progress section
        progress_frame = ctk.CTkFrame(training_frame)
        progress_frame.pack(fill="x", padx=10, pady=5)
        
        progress_label = ctk.CTkLabel(
            progress_frame,
            text="Training Progress:",
            font=("Arial", 14)
        )
        progress_label.pack(pady=5, padx=10, anchor="w")
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=700)
        self.progress_bar.pack(pady=5, padx=10)
        self.progress_bar.set(0)
        
        self.progress_text = ctk.CTkLabel(
            progress_frame,
            text="Waiting to start training...",
            font=("Arial", 12)
        )
        self.progress_text.pack(pady=5, padx=10)
        
        # Training log
        log_frame = ctk.CTkFrame(training_frame)
        log_frame.pack(fill="x", padx=10, pady=5)
        
        log_label = ctk.CTkLabel(
            log_frame,
            text="Training Log:",
            font=("Arial", 14)
        )
        log_label.pack(pady=5, padx=10, anchor="w")
        
        self.log_text = ctk.CTkTextbox(
            log_frame,
            height=100,
            width=700,
            font=("Courier", 12)
        )
        self.log_text.pack(pady=5, padx=10, fill="x")
        self.log_text.configure(state="disabled")
    
    def _create_evaluation_section(self):
        """Create the model evaluation section."""
        eval_frame = ctk.CTkFrame(self.master_scroll)
        eval_frame.pack(fill="x", padx=5, pady=5)
        
        # Section title
        section_label = ctk.CTkLabel(
            eval_frame,
            text="4. Model Evaluation",
            font=("Arial", 18, "bold")
        )
        section_label.pack(pady=(10, 5), padx=10, anchor="w")
        
        # Model selection for evaluation
        model_select_frame = ctk.CTkFrame(eval_frame)
        model_select_frame.pack(fill="x", padx=10, pady=5)
        
        # First row: Model selection dropdown and refresh button
        select_row = ctk.CTkFrame(model_select_frame)
        select_row.pack(fill="x", pady=5)
        
        select_label = ctk.CTkLabel(
            select_row,
            text="Select Model:",
            font=("Arial", 14)
        )
        select_label.pack(side="left", padx=10, pady=5)
        
        self.eval_model_var = ctk.StringVar(value="No models available")
        self.eval_model_menu = ctk.CTkOptionMenu(
            select_row,
            values=["No models available"],
            variable=self.eval_model_var,
            width=300,
            command=self._model_selected
        )
        self.eval_model_menu.pack(side="left", padx=10, pady=5)
        
        # Add refresh button
        refresh_button = ctk.CTkButton(
            select_row,
            text="Refresh Models",
            command=self._update_model_lists,
            width=120
        )
        refresh_button.pack(side="left", padx=10, pady=5)
        
        # Second row: Action buttons
        action_row = ctk.CTkFrame(model_select_frame)
        action_row.pack(fill="x", pady=5)
        
        # Model info button (left side)
        info_button = ctk.CTkButton(
            action_row,
            text="Model Info",
            command=lambda: self._display_model_info(self.eval_model_var.get()),
            width=150
        )
        info_button.pack(side="left", padx=10, pady=5)
        
        # Only one evaluate button (right side)
        eval_button = ctk.CTkButton(
            action_row,
            text="Evaluate Model",
            command=self._evaluate_model,
            width=150
        )
        eval_button.pack(side="right", padx=10, pady=5)
        
        # Visualization options
        viz_frame = ctk.CTkFrame(eval_frame)
        viz_frame.pack(fill="x", padx=10, pady=5)
        
        viz_label = ctk.CTkLabel(
            viz_frame,
            text="Visualization:",
            font=("Arial", 14)
        )
        viz_label.pack(side="left", padx=10, pady=5)
        
        self.viz_type_var = ctk.StringVar(value="Confusion Matrix")
        viz_options = [
            "Confusion Matrix",
            "Precision-Recall Curve",
            "ROC Curve",
            "Metrics Table",
            "Loss Curve",
            "Word Cloud"
        ]
        viz_menu = ctk.CTkOptionMenu(
            viz_frame,
            values=viz_options,
            variable=self.viz_type_var,
            width=200,
            command=self._update_visualization
        )
        viz_menu.pack(side="left", padx=10, pady=5)
        
        self.publication_ready_var = ctk.BooleanVar(value=True)
        publication_check = ctk.CTkCheckBox(
            viz_frame,
            text="Publication ready (white background)",
            variable=self.publication_ready_var,
            command=self._update_visualization
        )
        publication_check.pack(side="left", padx=20, pady=5)
        
        export_button = ctk.CTkButton(
            viz_frame,
            text="Export Visualization",
            command=self._export_visualization,
            width=150
        )
        export_button.pack(side="right", padx=10, pady=5)
        
        # Results display area
        results_frame = ctk.CTkFrame(eval_frame)
        results_frame.pack(fill="x", padx=10, pady=5)
        
        # Container for visualization
        self.viz_container = ctk.CTkFrame(results_frame)
        self.viz_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Placeholder text
        self.viz_placeholder = ctk.CTkLabel(
            self.viz_container,
            text="Evaluate a model to see results",
            font=("Arial", 14)
        )
        self.viz_placeholder.pack(expand=True, pady=50)
        
        # Placeholder for visualization canvas
        self.viz_canvas = None
        
    def _model_selected(self, model_name):
        """Handle model selection from dropdown."""
        # If no data is loaded yet, show model info
        if self.data_processor.test_data is None:
            self._display_model_info(model_name)
            
            
    def _display_model_info(self, model_name: str):
        """
        Display information about a model without evaluating it.
        This is useful when data hasn't been loaded yet.
        """
        # Get model information from the model trainer
        models_info = self.model_trainer.scan_models()
        
        if model_name not in models_info:
            messagebox.showinfo("Information", f"No information available for model {model_name}.")
            return
        
        # Get the model info
        model_info = models_info[model_name]
        metadata = model_info.get("metadata", {})
        hyperparams = model_info.get("hyperparams", {})
        evaluation = model_info.get("evaluation", {})
        
        # Create a popup window
        info_window = ctk.CTkToplevel(self)
        info_window.title(f"Model Information: {model_name}")
        info_window.geometry("700x500")
        info_window.grab_set()  # Make window modal
        
        # Create scrollable frame
        info_frame = ctk.CTkScrollableFrame(info_window, width=680, height=480)
        info_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Title
        title_label = ctk.CTkLabel(
            info_frame,
            text=f"Model: {model_name}",
            font=("Arial", 20, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        # Model type and save time
        model_type = metadata.get("model_type", "Unknown")
        save_time = metadata.get("save_time", "Unknown")
        
        type_label = ctk.CTkLabel(
            info_frame,
            text=f"Type: {model_type} | Saved: {save_time}",
            font=("Arial", 14)
        )
        type_label.pack(pady=(0, 20))
        
        # Sections
        
        # Hyperparameters section
        if hyperparams:
            hyper_frame = ctk.CTkFrame(info_frame)
            hyper_frame.pack(fill="x", pady=10)
            
            hyper_label = ctk.CTkLabel(
                hyper_frame,
                text="Hyperparameters",
                font=("Arial", 16, "bold")
            )
            hyper_label.pack(pady=5, padx=10, anchor="w")
            
            # Format hyperparameters as text
            hyper_text = ""
            for key, value in hyperparams.items():
                # Format arrays more nicely
                if isinstance(value, list):
                    value = str(value)
                hyper_text += f"{key}: {value}\n"
            
            hyper_textbox = ctk.CTkTextbox(
                hyper_frame,
                width=650,
                height=150,
                font=("Courier", 12)
            )
            hyper_textbox.pack(pady=5, padx=10, fill="x")
            hyper_textbox.insert("1.0", hyper_text)
            hyper_textbox.configure(state="disabled")
        
        # Evaluation section
        if evaluation:
            eval_frame = ctk.CTkFrame(info_frame)
            eval_frame.pack(fill="x", pady=10)
            
            eval_label = ctk.CTkLabel(
                eval_frame,
                text="Evaluation Results",
                font=("Arial", 16, "bold")
            )
            eval_label.pack(pady=5, padx=10, anchor="w")
            
            # Format key metrics
            accuracy = evaluation.get("accuracy", "N/A")
            if isinstance(accuracy, (int, float)):
                accuracy = f"{accuracy:.4f}"
            
            metrics_text = f"Accuracy: {accuracy}\n\nClass-wise metrics:\n"
            
            # Add class-wise metrics
            classes = evaluation.get("classes", [])
            precision = evaluation.get("precision", {})
            recall = evaluation.get("recall", {})
            f1 = evaluation.get("f1", {})
            
            for i, class_name in enumerate(classes):
                i_str = str(i)
                if i_str in precision and i_str in recall and i_str in f1:
                    metrics_text += f"  {class_name}:\n"
                    metrics_text += f"    Precision: {precision[i_str]:.4f}\n"
                    metrics_text += f"    Recall: {recall[i_str]:.4f}\n"
                    metrics_text += f"    F1: {f1[i_str]:.4f}\n"
            
            eval_textbox = ctk.CTkTextbox(
                eval_frame,
                width=650,
                height=200,
                font=("Courier", 12)
            )
            eval_textbox.pack(pady=5, padx=10, fill="x")
            eval_textbox.insert("1.0", metrics_text)
            eval_textbox.configure(state="disabled")
        
        # Close button only - removed the evaluate button
        button_frame = ctk.CTkFrame(info_frame)
        button_frame.pack(fill="x", pady=10)
        
        close_button = ctk.CTkButton(
            button_frame,
            text="Close",
            command=info_window.destroy,
            width=100
        )
        close_button.pack(side="right", padx=10, pady=10)
        
    def _evaluate_pretrained(self):
        """Evaluate a pre-trained model without training."""
        if self.data_processor.test_data is None:
            messagebox.showinfo("Information", "Please prepare data split first.")
            return
        
        # Get the selected pre-trained model
        pretrained_model = self.pretrained_model_var.get()
        
        # Create a model name
        model_name = f"pretrained_{pretrained_model.split('/')[-1]}_{int(time.time())}"
        
        # Update UI
        self.progress_text.configure(text=f"Evaluating pre-trained model: {pretrained_model}")
        self.progress_bar.set(0.2)
        
        # Start evaluation in a separate thread
        thread = threading.Thread(
            target=self._evaluate_pretrained_thread,
            args=(pretrained_model, model_name),
            daemon=True
        )
        thread.start()
        self.active_threads.append(thread)

    def _evaluate_pretrained_thread(self, pretrained_model, model_name):
        """Run pre-trained model evaluation in a background thread."""
        try:
            # Update progress
            self.after(0, lambda: self.progress_bar.set(0.4))
            self.after(0, lambda: self.progress_text.configure(text="Loading pre-trained model..."))
            
            # Evaluate the model
            results = self.model_trainer.load_and_evaluate_pretrained(
                data_processor=self.data_processor,
                model_name=model_name,
                pretrained_model=pretrained_model
            )
            
            # Update progress
            self.after(0, lambda: self.progress_bar.set(1.0))
            
            # Update model lists to include the new evaluation
            self.after(0, self._update_model_lists)
            
            # Show success message
            self.after(0, lambda: self.progress_text.configure(
                text=f"Evaluation complete. Accuracy: {results['accuracy']:.4f}"
            ))
            
            # Update evaluation visualization if this is the currently selected model
            self.after(0, lambda: self.eval_model_var.set(model_name))
            self.after(0, lambda: self._update_visualization())
        
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda msg=error_msg: self._log_message(f"Error during evaluation: {msg}"))
            self.after(0, lambda msg=error_msg: self.progress_text.configure(text=f"Evaluation failed: {msg}"))
            self.after(0, lambda msg=error_msg: messagebox.showerror("Evaluation Error", msg))
    
    def _create_comparison_section(self):
        """Create the model comparison section."""
        compare_frame = ctk.CTkFrame(self.master_scroll)
        compare_frame.pack(fill="x", padx=5, pady=5)
        
        # Section title
        section_label = ctk.CTkLabel(
            compare_frame,
            text="5. Model Comparison",
            font=("Arial", 18, "bold")
        )
        section_label.pack(pady=(10, 5), padx=10, anchor="w")
        
        # Add a "Compare All Models" button
        compare_all_frame = ctk.CTkFrame(compare_frame)
        compare_all_frame.pack(fill="x", padx=10, pady=5)
        
        compare_all_button = ctk.CTkButton(
            compare_all_frame,
            text="Show Model Comparison Table",
            command=self._show_model_comparison,
            width=250,
            height=40,
            font=("Arial", 14)
        )
        compare_all_button.pack(side="left", padx=10, pady=10)
        
        # Model selection for comparison
        model_select_frame = ctk.CTkFrame(compare_frame)
        model_select_frame.pack(fill="x", padx=10, pady=5)
        
        select_label = ctk.CTkLabel(
            model_select_frame,
            text="Select Models to Compare:",
            font=("Arial", 14)
        )
        select_label.pack(pady=5, padx=10, anchor="w")
        
        # Checkboxes will be added dynamically once models are trained
        # Use a scrollable frame to ensure all models are visible
        self.model_checkboxes_frame = ctk.CTkScrollableFrame(model_select_frame, height=140)
        self.model_checkboxes_frame.pack(fill="x", padx=10, pady=5)
        
        self.model_checkbox_vars = {}
        
        # Initial placeholder
        self.model_compare_placeholder = ctk.CTkLabel(
            self.model_checkboxes_frame,
            text="Train models to enable comparison",
            font=("Arial", 12),
            text_color="#888888"
        )
        self.model_compare_placeholder.pack(pady=10)
        
        # Comparison type selection
        compare_type_frame = ctk.CTkFrame(compare_frame)
        compare_type_frame.pack(fill="x", padx=10, pady=5)
        
        type_label = ctk.CTkLabel(
            compare_type_frame,
            text="Comparison Type:",
            font=("Arial", 14)
        )
        type_label.pack(side="left", padx=10, pady=5)
        
        self.compare_type_var = ctk.StringVar(value="Metrics Bar Chart")
        compare_options = [
            "Metrics Bar Chart", 
            "Confusion Matrices", 
            "ROC Curves", 
            "Precision-Recall Curves",
            "Training History",
            "Loss Curves"
        ]
        compare_type_menu = ctk.CTkOptionMenu(
            compare_type_frame,
            values=compare_options,
            variable=self.compare_type_var,
            width=200
        )
        compare_type_menu.pack(side="left", padx=10, pady=5)
        
        compare_button = ctk.CTkButton(
            compare_type_frame,
            text="Generate Comparison",
            command=self._generate_comparison,
            width=150
        )
        compare_button.pack(side="right", padx=10, pady=5)
        
        # Container for comparison visualization
        self.compare_container = ctk.CTkFrame(compare_frame)
        self.compare_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Placeholder text
        self.compare_placeholder = ctk.CTkLabel(
            self.compare_container,
            text="Select models and comparison type, then click 'Generate Comparison'",
            font=("Arial", 14)
        )
        self.compare_placeholder.pack(expand=True, pady=50)
        
        # Placeholder for comparison canvas
        self.compare_canvas = None
        
    def _show_model_comparison(self):
        """Show a comparison of all trained and evaluated models."""
        # Get available models
        models = self.model_trainer.get_available_models()
        
        if not models:
            messagebox.showinfo("Information", "No models available for comparison.")
            return
        
        # Open a new window for the comparison
        comparison_window = ctk.CTkToplevel(self)
        comparison_window.title("Model Comparison")
        comparison_window.geometry("900x600")
        comparison_window.grab_set()  # Make window modal
        
        # Create scrollable frame for comparison content
        comparison_frame = ctk.CTkScrollableFrame(comparison_window, width=880, height=580)
        comparison_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Add a title
        title_label = ctk.CTkLabel(
            comparison_frame,
            text="Model Comparison",
            font=("Arial", 24, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Use our comparison function to get the data
        comparison = self.model_trainer.compare_models(models)
        
        if not comparison:
            no_data_label = ctk.CTkLabel(
                comparison_frame,
                text="No valid models found for comparison",
                font=("Arial", 14)
            )
            no_data_label.pack(pady=20)
            return
        
        # Create text widget to display the comparison
        comparison_text = ctk.CTkTextbox(
            comparison_frame,
            width=860,
            height=500,
            font=("Courier", 12)
        )
        comparison_text.pack(pady=10, fill="both", expand=True)
        
        # Format comparison as text
        text_output = "===== MODEL COMPARISON =====\n\n"
        text_output += "Overall Metrics:\n"
        text_output += "-" * 80 + "\n"
        
        # Get headers
        metrics = ["accuracy", "precision", "recall", "f1", "inference_time"]
        headers = ["Model"] + [m.capitalize() for m in metrics if m != "precision" and m != "recall" and m != "f1"]
        
        # Format headers
        header_row = " ".join(h.ljust(15) for h in headers)
        text_output += header_row + "\n"
        text_output += "-" * 80 + "\n"
        
        # Format model data
        for model_name, data in comparison.get("models", {}).items():
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
                        row.append(f"{samples_per_sec:.2f}/s")
                    else:
                        row.append("N/A")
                else:
                    # Regular metric
                    value = eval_data.get(metric, "N/A")
                    if isinstance(value, (int, float)):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
            
            # Format as a row
            data_row = " ".join(str(r).ljust(15) for r in row)
            text_output += data_row + "\n"
        
        # Add class-specific metrics
        first_model = next(iter(comparison.get("models", {}).values()))
        class_names = first_model["evaluation"].get("classes", [])
        
        if class_names:
            text_output += "\n\nClass-wise Metrics:\n"
            
            for class_idx, class_name in enumerate(class_names):
                text_output += f"\nClass: {class_name}\n"
                text_output += "-" * 80 + "\n"
                
                # Headers for class metrics
                class_headers = ["Model", "Precision", "Recall", "F1"]
                header_row = " ".join(h.ljust(15) for h in class_headers)
                text_output += header_row + "\n"
                text_output += "-" * 80 + "\n"
                
                for model_name, data in comparison.get("models", {}).items():
                    eval_data = data["evaluation"]
                    row = [data["display_name"]]
                    
                    # Add precision, recall, F1
                    for metric in ["precision", "recall", "f1"]:
                        if metric in eval_data and str(class_idx) in eval_data[metric]:
                            value = eval_data[metric][str(class_idx)]
                            row.append(f"{value:.4f}")
                        else:
                            row.append("N/A")
                    
                    # Format as a row
                    data_row = " ".join(str(r).ljust(15) for r in row)
                    text_output += data_row + "\n"
        
        # Insert the formatted text into the text widget
        comparison_text.insert("1.0", text_output)
        comparison_text.configure(state="disabled")
        
        # Add export button
        export_button = ctk.CTkButton(
            comparison_frame,
            text="Export Comparison",
            command=lambda: self._export_comparison(text_output),
            width=150
        )
        export_button.pack(pady=10)

    def _export_comparison(self, comparison_text):
        """Export the model comparison to a file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Export Model Comparison"
        )
        
        if file_path:
            with open(file_path, "w") as f:
                f.write(comparison_text)
            messagebox.showinfo("Success", f"Comparison exported to {file_path}")
    
    def _load_dataset(self):
        """Load and process a dataset from a file."""
        file_path = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Update UI
            self.data_path_var.set(os.path.basename(file_path))
            
            # Load data in a background thread
            thread = threading.Thread(
                target=self._load_data_thread,
                args=(file_path,),
                daemon=True
            )
            thread.start()
            self.active_threads.append(thread)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            
    def _load_data_thread(self, file_path):
        """Load and process data in a background thread."""
        try:
            # Check if we should use a sample
            sample_size = 10000 if self.use_sample_var.get() else None
            
            # Update preprocessing options based on UI
            preprocessing_options = {
                key: var.get() for key, var in self.preprocess_vars.items()
            }
            
            # Pass options to data processor - Fix the typo here
            self.data_processor.set_preprocessing_options(preprocessing_options)  # Note the 's' at the end
            
            # Load the dataset with sample option
            df = self.data_processor.load_csv(file_path, sample_size=sample_size)
            
            # Update UI with data summary
            summary = self.data_processor.get_data_summary()
            self.after(0, lambda s=summary: self._update_data_summary(s))
            
        except Exception as e:
            # Capture the error in a variable
            error_message = str(e)
            # Show error on main thread with captured message
            self.after(0, lambda msg=error_message: messagebox.showerror("Error", f"Failed to process data: {msg}"))
    
    def _prepare_data_split(self):
        """Prepare train/test/validation split of the dataset."""
        if self.data_processor.raw_data is None:
            messagebox.showinfo("Information", "Please load a dataset first.")
            return
            
        try:
            # Parse split ratio
            split_ratio = self.split_ratio_var.get()
            train_size = float(split_ratio.split('/')[0]) / 100.0
            
            # Check if validation set is needed
            use_val = self.use_validation_var.get()
            
            # Update preprocessing options based on UI
            preprocessing_options = {
                key: var.get() for key, var in self.preprocess_vars.items()
            }
            
            # Pass options to data processor
            self.data_processor.set_preprocessing_options(preprocessing_options)
            
            # Create a thread for data preparation
            thread = threading.Thread(
                target=self._prepare_data_thread,
                args=(train_size, use_val),
                daemon=True
            )
            thread.start()
            self.active_threads.append(thread)
            
            # Update UI
            self.data_summary_label.configure(text="Preparing data split...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare data split: {str(e)}")
    
    def _prepare_data_thread(self, train_size, use_val):
        """Prepare data split in a background thread."""
        try:
            # Split the data
            self.data_processor.prepare_train_test_split(
                train_size=train_size,
                use_validation=use_val
            )
            
            # Update UI with the new data info
            summary = self.data_processor.get_split_summary()
            self.after(0, lambda s=summary: self._update_data_summary(s))
            
        except Exception as e:
            # Capture the error in a variable
            error_message = str(e)
            # Show error on main thread with captured message
            self.after(0, lambda msg=error_message: messagebox.showerror("Error", f"Failed to split data: {msg}"))
    
    def _update_data_summary(self, summary_text):
        """Update the data summary display with provided text."""
        self.data_summary_label.configure(text=summary_text)
    
    def _train_model(self):
        """Train the selected model with specified parameters."""
        if self.data_processor.train_data is None:
            messagebox.showinfo("Information", "Please prepare data split first.")
            return
        
        # Get model type from selected tab
        model_type = self.model_tabview.get()
        
        # Get model name
        model_name = self.model_name_var.get()
        if not model_name:
            model_name = f"model_{int(time.time())}"
            self.model_name_var.set(model_name)
        
        # Get training parameters
        training_params = {
            key: var.get() for key, var in self.train_params.items()
        }
        
        # Convert string parameters to appropriate types
        training_params['batch_size'] = int(training_params['batch_size'])
        training_params['epochs'] = int(training_params['epochs'])
        training_params['learning_rate'] = float(training_params['learning_rate'])
        training_params['weight_decay'] = float(training_params.get('weight_decay', 0.0))
        # Boolean for clip_grad
        cg = training_params.get('clip_grad', 'True')
        training_params['clip_grad'] = True if str(cg).lower() == 'true' else False
        training_params['max_grad_norm'] = float(training_params.get('max_grad_norm', 1.0))
        # Apply manual overrides if provided and valid
        try:
            if self.manual_batch_size.get().strip():
                mb = int(float(self.manual_batch_size.get().strip()))
                if mb > 0:
                    training_params['batch_size'] = mb
        except Exception:
            pass
        try:
            if self.manual_learning_rate.get().strip():
                mlr = float(self.manual_learning_rate.get().strip())
                if mlr > 0:
                    training_params['learning_rate'] = mlr
        except Exception:
            pass
        try:
            if self.manual_epochs.get().strip():
                me = int(float(self.manual_epochs.get().strip()))
                if me > 0:
                    training_params['epochs'] = me
        except Exception:
            pass
        
        # Get model-specific parameters
        model_params = {}
        
        if model_type == "Pre-trained":
            model_params['model_name'] = self.pretrained_var.get()
            model_params['finetune'] = self.finetune_var.get()
        
        elif model_type == "LSTM":
            model_params = {
                key: var.get() for key, var in self.lstm_params.items()
            }
            # Convert to appropriate types
            model_params['lstm_layers'] = int(model_params['lstm_layers'])
            model_params['lstm_hidden_size'] = int(model_params['lstm_hidden_size'])
            model_params['lstm_embedding_dim'] = int(model_params['lstm_embedding_dim'])
            model_params['lstm_dropout'] = float(model_params['lstm_dropout'])
            model_params['bidirectional'] = self.lstm_bidirectional.get()
            model_params['lstm_rnn_type'] = model_params.get('lstm_rnn_type', 'LSTM')
            model_params['lstm_embedding_dropout'] = float(model_params.get('lstm_embedding_dropout', '0.2'))
            model_params['lstm_use_attention'] = self.lstm_use_attention.get()
            
        elif model_type == "CNN":
            model_params = {
                key: var.get() for key, var in self.cnn_params.items()
            }
            # Convert to appropriate types
            model_params['cnn_embedding_dim'] = int(model_params['cnn_embedding_dim'])
            model_params['cnn_dropout'] = float(model_params['cnn_dropout'])
            model_params['cnn_num_filters'] = int(model_params['cnn_num_filters'])
            # Parse filter sizes string to actual list
            model_params['cnn_filter_sizes'] = eval(model_params['cnn_filter_sizes'])
            model_params['cnn_embedding_dropout'] = float(model_params.get('cnn_embedding_dropout', '0.2'))
            model_params['cnn_activation'] = model_params.get('cnn_activation', 'relu')
            model_params['cnn_pool_type'] = model_params.get('cnn_pool_type', 'max')
            model_params['cnn_batch_norm'] = self.cnn_batch_norm.get()
        
        # Reset progress bar and log
        self.progress_bar.set(0)
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", f"Starting training for {model_type} model: {model_name}\n")
        self.log_text.configure(state="disabled")
        self.progress_text.configure(text="Initializing training...")
        
        # Start training in a separate thread
        thread = threading.Thread(
            target=self._train_model_thread,
            args=(model_type, model_name, model_params, training_params),
            daemon=True
        )
        thread.start()
        self.active_threads.append(thread)
    
    def _train_model_thread(self, model_type, model_name, model_params, training_params):
        """Train model in a background thread."""
        try:
            # Configure the model trainer with callbacks
            self.model_trainer.set_callbacks({
                'on_epoch_end': self._on_epoch_end,
                'on_training_start': self._on_training_start,
                'on_training_end': self._on_training_end,
                'on_batch_end': self._on_batch_end
            })
            
            # Train the model based on its type
            if model_type == "Pre-trained":
                self.model_trainer.train_transformer(
                    data_processor=self.data_processor,
                    model_name=model_name,
                    pretrained_model=model_params['model_name'],
                    finetune=model_params['finetune'],
                    batch_size=training_params['batch_size'],
                    lr=training_params['learning_rate'],
                    epochs=training_params['epochs'],
                    optimizer=training_params['optimizer']
                )
            
            elif model_type == "LSTM":
                self.model_trainer.train_lstm(
                    data_processor=self.data_processor,
                    model_name=model_name,
                    num_layers=model_params['lstm_layers'],
                    hidden_size=model_params['lstm_hidden_size'],
                    embedding_dim=model_params['lstm_embedding_dim'],
                    dropout=model_params['lstm_dropout'],
                    bidirectional=model_params['bidirectional'],
                    rnn_type=model_params['lstm_rnn_type'],
                    batch_size=training_params['batch_size'],
                    lr=training_params['learning_rate'],
                    weight_decay=training_params['weight_decay'],
                    epochs=training_params['epochs'],
                    optimizer=training_params['optimizer'],
                    scheduler=training_params.get('scheduler', 'none'),
                    embedding_dropout=model_params['lstm_embedding_dropout'],
                    use_attention=model_params['lstm_use_attention'],
                    clip_grad=training_params['clip_grad'],
                    max_grad_norm=training_params['max_grad_norm']
                )
            
            elif model_type == "CNN":
                self.model_trainer.train_cnn(
                    data_processor=self.data_processor,
                    model_name=model_name,
                    filter_sizes=model_params['cnn_filter_sizes'],
                    num_filters=model_params['cnn_num_filters'],
                    embedding_dim=model_params['cnn_embedding_dim'],
                    dropout=model_params['cnn_dropout'],
                    embedding_dropout=model_params['cnn_embedding_dropout'],
                    activation=model_params['cnn_activation'],
                    batch_norm=model_params['cnn_batch_norm'],
                    pool_type=model_params['cnn_pool_type'],
                    batch_size=training_params['batch_size'],
                    lr=training_params['learning_rate'],
                    weight_decay=training_params['weight_decay'],
                    epochs=training_params['epochs'],
                    optimizer=training_params['optimizer'],
                    scheduler=training_params.get('scheduler', 'none'),
                    clip_grad=training_params['clip_grad'],
                    max_grad_norm=training_params['max_grad_norm']
                )
            
            # Update model list for evaluation
            self.after(0, self._update_model_lists)
            
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: self._log_message(f"Error during training: {error_msg}"))
            self.after(0, lambda: self.progress_text.configure(text=f"Training failed: {error_msg}"))
            self.after(0, lambda: messagebox.showerror("Training Error", error_msg))
    
    def _on_training_start(self, total_epochs, total_batches):
        """Callback for when training starts."""
        self.after(0, lambda: self.progress_text.configure(text=f"Training started. Total epochs: {total_epochs}"))
    
    def _on_epoch_end(self, epoch, total_epochs, metrics):
        """Callback for when an epoch ends."""
        # Update progress bar
        progress = (epoch + 1) / total_epochs
        self.after(0, lambda: self.progress_bar.set(progress))
        
        # Format metrics string (use scientific notation for very small values like LR)
        def _fmt(v):
            try:
                v = float(v)
                if abs(v) < 1e-3:
                    return f"{v:.2e}"
                return f"{v:.4f}"
            except Exception:
                return str(v)
        metrics_str = ", ".join([f"{k}: {_fmt(v)}" for k, v in metrics.items()])
        
        # Update progress text and log
        self.after(0, lambda: self.progress_text.configure(
            text=f"Epoch {epoch+1}/{total_epochs} completed. {metrics_str}"
        ))
        self.after(0, lambda: self._log_message(f"Epoch {epoch+1}/{total_epochs}: {metrics_str}"))
    
    def _on_batch_end(self, batch, total_batches, epoch, total_epochs):
        """Callback for when a batch ends."""
        # Update progress bar more frequently for better user feedback
        progress = (epoch + (batch + 1) / total_batches) / total_epochs
        self.after(0, lambda: self.progress_bar.set(progress))
        
        # Update progress text every 10 batches to avoid UI overload
        if batch % 10 == 0:
            self.after(0, lambda: self.progress_text.configure(
                text=f"Epoch {epoch+1}/{total_epochs}, Batch {batch+1}/{total_batches}"
            ))
    
    def _on_training_end(self, model_name, metrics, training_time):
        """Callback for when training ends."""
        # Format metrics for display
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        # Format training time
        time_str = f"{training_time:.2f}s" if training_time < 60 else f"{training_time/60:.2f}min"
        
        # Update UI
        self.after(0, lambda: self.progress_bar.set(1.0))
        self.after(0, lambda: self.progress_text.configure(
            text=f"Training completed. Time: {time_str}. {metrics_str}"
        ))
        self.after(0, lambda: self._log_message(
            f"Training completed for {model_name}. Time: {time_str}.\nFinal metrics: {metrics_str}"
        ))
        
        # Store model results for comparison
        self.model_results[model_name] = {
            'metrics': metrics,
            'training_time': training_time
        }
        
        # Update model lists
        self.after(0, self._update_model_lists)
    
    def _log_message(self, message):
        """Add a message to the training log."""
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
    
    def _update_model_lists(self):
        """Update the model selection dropdowns and checkboxes with available models."""
        # Get list of available models
        available_models = self.model_trainer.get_available_models()
        
        if not available_models:
            return
        
        # Update evaluation dropdown
        self.eval_model_menu.configure(values=available_models)
        
        # Only update the selected value if it's currently set to "No models available"
        if self.eval_model_var.get() == "No models available":
            self.eval_model_var.set(available_models[0])
        
        # Update comparison checkboxes
        self.model_compare_placeholder.pack_forget()
        
        # Clear existing checkboxes
        for widget in self.model_checkboxes_frame.winfo_children():
            if widget != self.model_compare_placeholder:
                widget.destroy()
        
        # Add new checkboxes
        self.model_checkbox_vars = {}
        for model_name in available_models:
            var = ctk.BooleanVar(value=False)
            self.model_checkbox_vars[model_name] = var
            
            checkbox = ctk.CTkCheckBox(
                self.model_checkboxes_frame,
                text=model_name,
                variable=var
            )
            checkbox.pack(side="top", anchor="w", padx=10, pady=3)
    
    def _evaluate_model(self):
        """Evaluate the selected model on the test set."""
        model_name = self.eval_model_var.get()
        
        if model_name == "No models available":
            messagebox.showinfo("Information", "No trained models available for evaluation.")
            return
        
        # Check if data is loaded and split
        if self.data_processor.test_data is None:
            # Ask if user wants to load and split data first
            answer = messagebox.askyesno(
                "Data Required", 
                "Test data is required for evaluation. Do you want to load and prepare data now?"
            )
            
            if answer:
                # Prompt to load data
                self._load_dataset()
                # We'll need to exit here and let the user prepare the data
                messagebox.showinfo(
                    "Prepare Data", 
                    "Please prepare data by setting the train/test split and clicking 'Prepare Data Split'."
                )
                return
            else:
                return
        
        # Continue with evaluation as before
        # Start evaluation in a separate thread
        thread = threading.Thread(
            target=self._evaluate_model_thread,
            args=(model_name,),
            daemon=True
        )
        thread.start()
        self.active_threads.append(thread)
        
        # Update UI
        self.viz_placeholder.configure(text=f"Evaluating {model_name}...")
    
    def _evaluate_model_thread(self, model_name):
        """Evaluate model in a background thread."""
        try:
            # Evaluate the model
            result = self.model_trainer.evaluate_model(
                model_name=model_name,
                data_processor=self.data_processor
            )
            
            # Store results for visualization (initialize if missing)
            if model_name not in self.model_results:
                self.model_results[model_name] = {}
            self.model_results[model_name]['evaluation'] = result
            
            # Update visualization
            self.after(0, lambda: self._update_visualization())
            
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Evaluation Error", error_msg))
            self.after(0, lambda: self.viz_placeholder.configure(text=f"Evaluation failed: {error_msg}"))
    
    def _update_visualization(self, *args):
        """Update visualization based on selected type."""
        model_name = self.eval_model_var.get()
        
        if model_name == "No models available" or model_name not in self.model_results:
            return
        
        if 'evaluation' not in self.model_results[model_name]:
            messagebox.showinfo("Information", f"Please evaluate {model_name} first.")
            return
        
        viz_type = self.viz_type_var.get()
        publication_ready = self.publication_ready_var.get()
        
        # Clear placeholder
        self.viz_placeholder.pack_forget()
        
        # Clear previous visualization
        if self.viz_canvas:
            self.viz_canvas.get_tk_widget().destroy()
            self.viz_canvas = None
        
        # Get evaluation result
        result = self.model_results[model_name]['evaluation']
        
        # Create figure with appropriate style
        if publication_ready:
            plt.style.use('default')
        else:
            plt.style.use('dark_background')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if viz_type == "Confusion Matrix":
            self.visualizer.plot_confusion_matrix(
                fig=fig, 
                ax=ax,
                confusion_matrix=result['confusion_matrix'],
                classes=result['classes'],
                title=f"Confusion Matrix - {model_name}",
                publication_ready=publication_ready
            )
        
        elif viz_type == "Precision-Recall Curve":
            self.visualizer.plot_precision_recall_curve(
                fig=fig,
                ax=ax,
                precision=result['precision_curve'],
                recall=result['recall_curve'],
                average_precision=result['average_precision'],
                classes=result['classes'],
                title=f"Precision-Recall Curve - {model_name}",
                publication_ready=publication_ready
            )
        
        elif viz_type == "ROC Curve":
            self.visualizer.plot_roc_curve(
                fig=fig,
                ax=ax,
                fpr=result['fpr'],
                tpr=result['tpr'],
                roc_auc=result['roc_auc'],
                classes=result['classes'],
                title=f"ROC Curve - {model_name}",
                publication_ready=publication_ready
            )
        
        elif viz_type == "Metrics Table":
            self.visualizer.plot_metrics_table(
                fig=fig,
                ax=ax,
                classification_report=result['classification_report'],
                accuracy=result['accuracy'],
                model_name=model_name,
                publication_ready=publication_ready
            )
        
        elif viz_type == "Loss Curve":
            history = result.get('history') or self.model_results.get(model_name, {}).get('history')
            if not history:
                plt.close(fig)
                self.viz_canvas = None
                messagebox.showinfo("Information", "No training history available for this model.")
                return
            self.visualizer.plot_loss_curves(
                fig=fig,
                ax=ax,
                history=history,
                title=f"Loss Curves - {model_name}",
                publication_ready=publication_ready
            )
        
        elif viz_type == "Word Cloud":
            texts = []
            if hasattr(self.data_processor, 'test_data') and self.data_processor.test_data is not None:
                texts = list(self.data_processor.test_data.get('texts', []))
            # Prefer misclassified texts if predictions are available
            y_true = result.get('y_true')
            y_pred = result.get('y_pred')
            if texts and isinstance(y_true, list) and isinstance(y_pred, list) and len(y_true) == len(texts) == len(y_pred):
                mis_texts = [t for t, yt, yp in zip(texts, y_true, y_pred) if yt != yp]
                if mis_texts:
                    texts = mis_texts
                    title = f"Word Cloud (Misclassified) - {model_name}"
                else:
                    title = f"Word Cloud (All Test Texts) - {model_name}"
            else:
                title = f"Word Cloud (All Test Texts) - {model_name}"
            self.visualizer.plot_word_cloud(
                fig=fig,
                ax=ax,
                texts=texts,
                title=title,
                publication_ready=publication_ready
            )
        
        # Create canvas with the new figure
        self.viz_canvas = FigureCanvasTkAgg(fig, master=self.viz_container)
        self.viz_canvas.draw()
        self.viz_canvas.get_tk_widget().pack(fill="both", expand=True)
        # Detach from pyplot to prevent figure accumulation warnings
        plt.close(fig)
    
    def _export_visualization(self):
        """Export current visualization to a file."""
        if not self.viz_canvas:
            messagebox.showinfo("Information", "No visualization to export.")
            return
        
        # Get file path
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("PDF", "*.pdf"),
                ("SVG", "*.svg"),
                ("EPS", "*.eps")
            ],
            title="Export Visualization"
        )
        
        if not file_path:
            return
        
        try:
            # Export the figure
            self.viz_canvas.figure.savefig(
                file_path,
                bbox_inches="tight",
                dpi=300
            )
            messagebox.showinfo("Success", f"Visualization exported to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
    
    def _generate_comparison(self):
        """Generate comparison visualization for selected models."""
        # Get selected models
        selected_models = [
            model_name for model_name, var in self.model_checkbox_vars.items()
            if var.get()
        ]
        
        if not selected_models:
            messagebox.showinfo("Information", "Please select at least one model to compare.")
            return
        
        if len(selected_models) < 2 and self.compare_type_var.get() != "Training History":
            messagebox.showinfo("Information", "Please select at least two models to compare.")
            return
        
        # Ensure evaluation results and histories are available on demand
        for model_name in selected_models:
            # Try to load evaluation if missing (from saved files)
            if model_name not in self.model_results:
                self.model_results[model_name] = {}
            if 'evaluation' not in self.model_results[model_name]:
                try:
                    eval_path = os.path.join(self.model_trainer.model_dir, model_name, "evaluation_results.json")
                    if os.path.exists(eval_path):
                        with open(eval_path, "r") as f:
                            self.model_results[model_name]['evaluation'] = json.load(f)
                    else:
                        messagebox.showinfo("Information", f"Please evaluate {model_name} first.")
                        return
                except Exception:
                    messagebox.showinfo("Information", f"Please evaluate {model_name} first.")
                    return
            # Try to load training history if missing
            if 'history' not in self.model_results[model_name]:
                try:
                    hist_path = os.path.join(self.model_trainer.model_dir, model_name, "history.json")
                    if os.path.exists(hist_path):
                        with open(hist_path, "r") as f:
                            self.model_results[model_name]['history'] = json.load(f)
                except Exception:
                    pass
        
        # Get comparison type
        compare_type = self.compare_type_var.get()
        
        # Remove placeholder
        self.compare_placeholder.pack_forget()
        
        # Clear previous visualization
        if self.compare_canvas:
            self.compare_canvas.get_tk_widget().destroy()
            self.compare_canvas = None
        
        # Set publication ready style
        plt.style.use('default')
        
        # For certain comparison types, we need a larger figure
        if compare_type in ["Confusion Matrices", "ROC Curves", "Precision-Recall Curves"]:
            # Calculate subplot grid size based on number of models
            n_models = len(selected_models)
            cols = min(3, n_models)
            rows = (n_models + cols - 1) // cols  # Ceiling division
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            
            # Flatten axes array for easy iteration if there are multiple subplots
            if n_models > 1:
                if isinstance(axes, np.ndarray):
                    axes = axes.flatten()
                else:
                    axes = [axes]  # Convert to list if only one axis
            else:
                axes = [axes]
            
            # Hide unused subplots
            for i in range(n_models, len(axes)):
                axes[i].axis('off')
            
            if compare_type == "Confusion Matrices":
                for i, model_name in enumerate(selected_models):
                    result = self.model_results[model_name]['evaluation']
                    self.visualizer.plot_confusion_matrix(
                        fig=None,
                        ax=axes[i],
                        confusion_matrix=result['confusion_matrix'],
                        classes=result['classes'],
                        title=model_name,
                        publication_ready=True
                    )
            
            elif compare_type == "ROC Curves":
                for i, model_name in enumerate(selected_models):
                    result = self.model_results[model_name]['evaluation']
                    self.visualizer.plot_roc_curve(
                        fig=None,
                        ax=axes[i],
                        fpr=result['fpr'],
                        tpr=result['tpr'],
                        roc_auc=result['roc_auc'],
                        classes=result['classes'],
                        title=model_name,
                        publication_ready=True
                    )
            
            elif compare_type == "Precision-Recall Curves":
                for i, model_name in enumerate(selected_models):
                    result = self.model_results[model_name]['evaluation']
                    self.visualizer.plot_precision_recall_curve(
                        fig=None,
                        ax=axes[i],
                        precision=result['precision_curve'],
                        recall=result['recall_curve'],
                        average_precision=result['average_precision'],
                        classes=result['classes'],
                        title=model_name,
                        publication_ready=True
                    )
            
            plt.tight_layout()
        
        elif compare_type == "Metrics Bar Chart":
            # Create a single figure for bar chart comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract metrics for comparison
            model_names = []
            accuracy = []
            precision = []
            recall = []
            f1 = []
            
            for model_name in selected_models:
                result = self.model_results[model_name]['evaluation']
                model_names.append(model_name)
                accuracy.append(float(result.get('accuracy', 0.0)))
                def _avg(d):
                    try:
                        if isinstance(d, dict):
                            vals = []
                            for v in d.values():
                                try:
                                    vals.append(float(v))
                                except Exception:
                                    pass
                            return float(np.mean(vals)) if vals else 0.0
                        return float(d)
                    except Exception:
                        return 0.0
                precision.append(_avg(result.get('precision', {})))
                recall.append(_avg(result.get('recall', {})))
                f1.append(_avg(result.get('f1', {})))
            
            self.visualizer.plot_metrics_comparison(
                fig=fig,
                ax=ax,
                model_names=model_names,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                title="Model Performance Comparison",
                publication_ready=True
            )
        
        elif compare_type == "Training History":
            # For training history, even a single model is fine
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if len(selected_models) == 1:
                # Single model history
                model_name = selected_models[0]
                if 'history' in self.model_results[model_name]:
                    history = self.model_results[model_name]['history']
                    self.visualizer.plot_training_history(
                        fig=fig,
                        ax=ax,
                        history=history,
                        title=f"Training History - {model_name}",
                        publication_ready=True
                    )
                else:
                    messagebox.showinfo("Information", f"No training history available for {model_name}.")
                    return
            else:
                # Multiple models history comparison
                histories = {}
                for model_name in selected_models:
                    if 'history' in self.model_results[model_name]:
                        histories[model_name] = self.model_results[model_name]['history']
                
                if not histories:
                    messagebox.showinfo("Information", "No training history available for selected models.")
                    return
                
                self.visualizer.plot_training_history_comparison(
                    fig=fig,
                    ax=ax,
                    histories=histories,
                    title="Training History Comparison",
                    publication_ready=True
                )
        
        elif compare_type == "Loss Curves":
            # Compare loss curves for selected models
            fig, ax = plt.subplots(figsize=(10, 6))
            histories = {}
            for model_name in selected_models:
                if 'history' in self.model_results[model_name]:
                    histories[model_name] = self.model_results[model_name]['history']
            if not histories:
                messagebox.showinfo("Information", "No training history available for selected models.")
                return
            self.visualizer.plot_loss_history_comparison(
                fig=fig,
                ax=ax,
                histories=histories,
                title="Loss Curves Comparison",
                publication_ready=True
            )
        
        # Create canvas with the new figure
        self.compare_canvas = FigureCanvasTkAgg(fig, master=self.compare_container)
        self.compare_canvas.draw()
        self.compare_canvas.get_tk_widget().pack(fill="both", expand=True)
        # Detach from pyplot to prevent figure accumulation warnings
        plt.close(fig)
    
    def destroy(self):
        """Clean up resources when the page is destroyed."""
        # Close any open matplotlib figures
        plt.close('all')
        
        # Cancel any active threads (though they're daemon so will terminate anyway)
        self.active_threads = []
        
        # Call parent's destroy method
        super().destroy()

    def _on_device_change(self, choice):
        """Handle compute device preference change (Auto/GPU/CPU)."""
        pref = (choice or 'Auto').lower()
        if pref == 'gpu':
            self.model_trainer.set_compute_device('gpu')
        elif pref == 'cpu':
            self.model_trainer.set_compute_device('cpu')
        else:
            self.model_trainer.set_compute_device('auto')
        # refresh indicator
        try:
            self.device_info_label.configure(text=f"Device: {self.model_trainer.current_device_info()}")
        except Exception:
            pass



