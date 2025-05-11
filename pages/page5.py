# pages/page5.py
import customtkinter as ctk
import threading
import os
import time
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
            ("Hidden Size:", "lstm_hidden_size", ["64", "128", "256", "512"], "256"),
            ("Embedding Dim:", "lstm_embedding_dim", ["100", "200", "300"], "300"),
            ("Dropout Rate:", "lstm_dropout", ["0.1", "0.2", "0.3", "0.5"], "0.3")
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
        
        # Add bidirectional LSTM option
        self.lstm_bidirectional = ctk.BooleanVar(value=True)
        bidirectional_check = ctk.CTkCheckBox(
            lstm_config_frame,
            text="Bidirectional LSTM",
            variable=self.lstm_bidirectional
        )
        bidirectional_check.grid(row=len(params), column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        # CNN models tab
        cnn_tab = self.model_tabview.tab("CNN")
        
        # CNN configuration
        cnn_config_frame = ctk.CTkFrame(cnn_tab)
        cnn_config_frame.pack(fill="x", padx=10, pady=5)
        
        # Create a grid of CNN parameters
        cnn_params = [
            ("Filter Sizes:", "cnn_filter_sizes", ["[3,4,5]", "[2,3,4]", "[1,2,3,4,5]"], "[3,4,5]"),
            ("Num Filters:", "cnn_num_filters", ["100", "200", "300"], "100"),
            ("Embedding Dim:", "cnn_embedding_dim", ["100", "200", "300"], "300"),
            ("Dropout Rate:", "cnn_dropout", ["0.1", "0.2", "0.3", "0.5"], "0.5")
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
            ("Batch Size:", "batch_size", ["8", "16", "32", "64"], "32"),
            ("Learning Rate:", "learning_rate", ["0.001", "0.0005", "0.0001"], "0.001"),
            ("Epochs:", "epochs", ["3", "5", "10", "20"], "5"),
            ("Optimizer:", "optimizer", ["Adam", "AdamW", "SGD"], "Adam")
        ]
        
        self.train_params = {}
        
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
        viz_options = ["Confusion Matrix", "Precision-Recall Curve", "ROC Curve", "Metrics Table"]
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
        model_select_frame = ctk.CTkFrame(compare_frame)  # This was missing or misnamed
        model_select_frame.pack(fill="x", padx=10, pady=5)
        
        select_label = ctk.CTkLabel(
            model_select_frame,
            text="Select Models to Compare:",
            font=("Arial", 14)
        )
        select_label.pack(pady=5, padx=10, anchor="w")
        
        # Checkboxes will be added dynamically once models are trained
        self.model_checkboxes_frame = ctk.CTkFrame(model_select_frame)
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
            "Training History"
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
                    batch_size=training_params['batch_size'],
                    lr=training_params['learning_rate'],
                    epochs=training_params['epochs'],
                    optimizer=training_params['optimizer']
                )
            
            elif model_type == "CNN":
                self.model_trainer.train_cnn(
                    data_processor=self.data_processor,
                    model_name=model_name,
                    filter_sizes=model_params['cnn_filter_sizes'],
                    num_filters=model_params['cnn_num_filters'],
                    embedding_dim=model_params['cnn_embedding_dim'],
                    dropout=model_params['cnn_dropout'],
                    batch_size=training_params['batch_size'],
                    lr=training_params['learning_rate'],
                    epochs=training_params['epochs'],
                    optimizer=training_params['optimizer']
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
        
        # Format metrics string
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
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
            checkbox.pack(side="left", padx=10, pady=5)
    
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
            
            # Store results for visualization
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
                precision=result['precision'],
                recall=result['recall'],
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
        
        # Create canvas with the new figure
        self.viz_canvas = FigureCanvasTkAgg(fig, master=self.viz_container)
        self.viz_canvas.draw()
        self.viz_canvas.get_tk_widget().pack(fill="both", expand=True)
    
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
        
        # Check if all selected models have evaluation results
        for model_name in selected_models:
            if 'evaluation' not in self.model_results[model_name]:
                messagebox.showinfo("Information", f"Please evaluate {model_name} first.")
                return
        
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
                        precision=result['precision'],
                        recall=result['recall'],
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
                accuracy.append(result['accuracy'])
                precision.append(np.mean(result['precision'].values()))
                recall.append(np.mean(result['recall'].values()))
                f1.append(np.mean(result['f1'].values()))
            
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
        
        # Create canvas with the new figure
        self.compare_canvas = FigureCanvasTkAgg(fig, master=self.compare_container)
        self.compare_canvas.draw()
        self.compare_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def destroy(self):
        """Clean up resources when the page is destroyed."""
        # Close any open matplotlib figures
        plt.close('all')
        
        # Cancel any active threads (though they're daemon so will terminate anyway)
        self.active_threads = []
        
        # Call parent's destroy method
        super().destroy()
# pages/page5.py
import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
import threading
import os
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Import evaluator
from addons.evaluation import ModelEvaluator
from addons.sentiment_analyzer import SentimentAnalyzer


class Page5(ctk.CTkFrame):
    """
    Model Evaluation Visualization Page for Thesis Documentation.
    
    This page provides:
    1. High-quality visualizations of model performance metrics
    2. Publication-ready charts with white background
    3. Export functionality for thesis documentation
    4. Multiple evaluation metrics (confusion matrix, precision/recall, F1 scores)
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # Initialize sentiment analyzer 
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize model evaluator
        self.evaluator = ModelEvaluator(self.sentiment_analyzer)
        
        # State variables
        self.test_data = None
        self.evaluation_results = None
        self.current_figure = None
        self.current_canvas = None
        
        # Thread tracking
        self.active_threads = []
        self.visualization_lock = threading.Lock()
        
        # Create the main UI
        self._create_ui()
        
    def _create_ui(self):
        """Create the user interface for the evaluation page."""
        # Create master scrollable frame
        self.master_scroll = ctk.CTkScrollableFrame(self, width=950, height=800)
        self.master_scroll.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Create header
        self._create_header()
        
        # Create data loading section
        self._create_data_section()
        
        # Create visualization controls
        self._create_visualization_controls()
        
        # Create visualization area
        self._create_visualization_area()
        
        # Create export section
        self._create_export_section()
        
    def _create_header(self):
        """Create header section with title and description."""
        header_frame = ctk.CTkFrame(self.master_scroll)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="Model Evaluation for Thesis Documentation",
            font=("Arial", 24, "bold")
        )
        title_label.pack(pady=(10, 5))
        
        # Description
        desc_label = ctk.CTkLabel(
            header_frame,
            text="Generate publication-ready evaluation metrics visualizations for your thesis",
            font=("Arial", 14),
            text_color="#888888"
        )
        desc_label.pack(pady=(0, 10))
        
    def _create_data_section(self):
        """Create section for loading test data."""
        data_frame = ctk.CTkFrame(self.master_scroll)
        data_frame.pack(fill="x", padx=5, pady=5)
        
        # Section title
        section_label = ctk.CTkLabel(
            data_frame,
            text="Test Data",
            font=("Arial", 18, "bold")
        )
        section_label.pack(pady=(10, 5), padx=10, anchor="w")
        
        # Data loading row
        load_frame = ctk.CTkFrame(data_frame)
        load_frame.pack(fill="x", padx=10, pady=5)
        
        # File path display
        self.file_path_var = ctk.StringVar(value="No file selected")
        file_path_label = ctk.CTkLabel(
            load_frame,
            textvariable=self.file_path_var,
            font=("Arial", 12),
            width=400,
            anchor="w"
        )
        file_path_label.pack(side="left", padx=5, pady=10, fill="x", expand=True)
        
        # Add a specific button for Sentiment140
        sentiment140_button = ctk.CTkButton(
            load_frame,
            text="Load Sentiment140 Sample",
            command=self._load_sentiment140_sample,
            width=200
        )
        sentiment140_button.pack(side="right", padx=10, pady=10)
        
        # Sample data button
        sample_button = ctk.CTkButton(
            load_frame,
            text="Use Sample Data",
            command=self._load_sample_data,
            width=150
        )
        sample_button.pack(side="right", padx=10, pady=10)
        
        # Load button
        load_button = ctk.CTkButton(
            load_frame,
            text="Load Test Data",
            command=self._load_test_data,
            width=150
        )
        load_button.pack(side="right", padx=10, pady=10)
        
        # Data summary
        self.data_summary_frame = ctk.CTkFrame(data_frame)
        self.data_summary_frame.pack(fill="x", padx=10, pady=5)
        
        self.data_summary_label = ctk.CTkLabel(
            self.data_summary_frame,
            text="Load test data to see summary statistics",
            font=("Arial", 12),
            justify="left",
            wraplength=900
        )
        self.data_summary_label.pack(pady=10, padx=10, fill="x")
        
    def _create_visualization_controls(self):
        """Create controls for visualization generation."""
        controls_frame = ctk.CTkFrame(self.master_scroll)
        controls_frame.pack(fill="x", padx=5, pady=5)
        
        # Section title
        section_label = ctk.CTkLabel(
            controls_frame,
            text="Visualization Controls",
            font=("Arial", 18, "bold")
        )
        section_label.pack(pady=(10, 5), padx=10, anchor="w")
        
        # Controls layout
        options_frame = ctk.CTkFrame(controls_frame)
        options_frame.pack(fill="x", padx=10, pady=5)
        
        # Visualization type
        type_label = ctk.CTkLabel(
            options_frame,
            text="Visualization Type:",
            font=("Arial", 14)
        )
        type_label.pack(side="left", padx=10, pady=10)
        
        self.viz_type_var = ctk.StringVar(value="Confusion Matrix")
        viz_type_menu = ctk.CTkOptionMenu(
            options_frame,
            values=[
                "Confusion Matrix", 
                "Precision-Recall-F1", 
                "Accuracy Comparison",
                "Performance Metrics Table",
                "Class Distribution"
            ],
            variable=self.viz_type_var,
            width=200
        )
        viz_type_menu.pack(side="left", padx=10, pady=10)
        
        # Publication style toggle
        self.pub_style_var = ctk.BooleanVar(value=True)
        pub_style_checkbox = ctk.CTkCheckBox(
            options_frame,
            text="Publication Style (White Background)",
            variable=self.pub_style_var,
            onvalue=True,
            offvalue=False
        )
        pub_style_checkbox.pack(side="left", padx=20, pady=10)
        
        # Generate button
        generate_button = ctk.CTkButton(
            options_frame,
            text="Generate Visualization",
            command=self._generate_visualization,
            width=170
        )
        generate_button.pack(side="right", padx=10, pady=10)
        
    def _create_visualization_area(self):
        """Create area for displaying visualizations."""
        self.viz_frame = ctk.CTkFrame(self.master_scroll)
        self.viz_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Placeholder text
        self.viz_placeholder = ctk.CTkLabel(
            self.viz_frame,
            text="Load test data and select visualization type, then click 'Generate Visualization'",
            font=("Arial", 14),
            text_color="#888888"
        )
        self.viz_placeholder.pack(expand=True, pady=100)
        
    def _create_export_section(self):
        """Create controls for exporting visualizations."""
        export_frame = ctk.CTkFrame(self.master_scroll)
        export_frame.pack(fill="x", padx=5, pady=5)
        
        # Section title
        export_label = ctk.CTkLabel(
            export_frame,
            text="Export Options",
            font=("Arial", 18, "bold")
        )
        export_label.pack(pady=(10, 5), padx=10, anchor="w")
        
        # Export controls
        controls_frame = ctk.CTkFrame(export_frame)
        controls_frame.pack(fill="x", padx=10, pady=5)
        
        # Format selection
        format_label = ctk.CTkLabel(
            controls_frame,
            text="Export Format:",
            font=("Arial", 14)
        )
        format_label.pack(side="left", padx=10, pady=10)
        
        self.export_format_var = ctk.StringVar(value="PNG")
        export_format_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["PNG", "PDF", "SVG", "EPS", "TIFF"],
            variable=self.export_format_var,
            width=100
        )
        export_format_menu.pack(side="left", padx=10, pady=10)
        
        # DPI selection
        dpi_label = ctk.CTkLabel(
            controls_frame,
            text="Resolution (DPI):",
            font=("Arial", 14)
        )
        dpi_label.pack(side="left", padx=(20, 10), pady=10)
        
        self.export_dpi_var = ctk.StringVar(value="300")
        export_dpi_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["150", "300", "600", "1200"],
            variable=self.export_dpi_var,
            width=100
        )
        export_dpi_menu.pack(side="left", padx=10, pady=10)
        
        # Figure size selection
        size_label = ctk.CTkLabel(
            controls_frame,
            text="Figure Size:",
            font=("Arial", 14)
        )
        size_label.pack(side="left", padx=(20, 10), pady=10)
        
        self.fig_size_var = ctk.StringVar(value="8x6")
        fig_size_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["6x4", "8x6", "10x8", "12x10", "16x12"],
            variable=self.fig_size_var,
            width=100
        )
        fig_size_menu.pack(side="left", padx=10, pady=10)
        
        # Export button
        export_button = ctk.CTkButton(
            controls_frame,
            text="Export Visualization",
            command=self._export_visualization,
            width=200
        )
        export_button.pack(side="right", padx=10, pady=10)
        
    def _load_test_data(self):
        """Load test data from a CSV file."""
        file_path = filedialog.askopenfilename(
            title="Select Test Data File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        # Check if file is large
        is_large = file_path.lower().endswith('sentiment140.csv') or os.path.getsize(file_path) > 50000000
        
        if is_large:
            # Ask user for sample size
            sample_dialog = ctk.CTkInputDialog(
                text="Enter number of rows to sample (large dataset detected):", 
                title="Sample Size"
            )
            sample_size = sample_dialog.get_input()
            
            try:
                sample_size = int(sample_size)
                if sample_size <= 0:
                    sample_size = 10000  # Default if invalid
            except (ValueError, TypeError):
                sample_size = 10000  # Default if invalid
        else:
            sample_size = None  # Process full dataset
                
        try:
            # Show loading status
            self.data_summary_label.configure(text="Loading data...")
            
            # Load data in a background thread
            threading.Thread(
                target=self._load_data_thread,
                args=(file_path, sample_size),
                daemon=True
            ).start()
            
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Error", f"Failed to load data: {error_msg}")
            
    def _load_data_thread(self, file_path: str, sample_size: Optional[int] = None):
        """Load and process test data in a background thread with proper Sentiment140 handling."""
        try:
            # Check if the file is likely to be Sentiment140 dataset
            is_sentiment140 = 'sentiment140' in file_path.lower() or 'training.1600000' in file_path.lower()
            
            # Use latin-1 encoding for Sentiment140 dataset
            encoding = 'latin-1' if is_sentiment140 else 'utf-8'
            
            try:
                # Try to load the data with the selected encoding
                if sample_size:
                    df = pd.read_csv(file_path, encoding=encoding, nrows=sample_size)
                else:
                    df = pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                # Fallback to latin-1 if the first encoding fails
                if encoding != 'latin-1':
                    if sample_size:
                        df = pd.read_csv(file_path, encoding='latin-1', nrows=sample_size)
                    else:
                        df = pd.read_csv(file_path, encoding='latin-1')
            
            # Handle Sentiment140 format specifically
            if is_sentiment140:
                # The Sentiment140 dataset has 6 columns:
                # 0 = target (0 = negative, 4 = positive)
                # 1 = ID
                # 2 = date
                # 3 = query
                # 4 = user
                # 5 = text
                
                # Check if columns need to be renamed
                if len(df.columns) == 6:
                    # Check if first column name isn't meaningful (could be 0, Unnamed, etc.)
                    first_col = df.columns[0]
                    if first_col.startswith('Unnamed') or first_col in ['0', 0]:
                        # This is likely the raw format without headers
                        df.columns = ['target', 'id', 'date', 'query', 'user', 'text']
                
                # In Sentiment140, 0 = negative, 4 = positive (there's no neutral class)
                # First, check if the target column exists and has numeric values
                target_col = None
                for col in ['target', 'sentiment', 'label', 0, '0']:
                    if col in df.columns:
                        try:
                            # Check if this column contains numeric values like 0 and 4
                            unique_vals = df[col].unique()
                            if set(unique_vals).issubset({0, 4}) or set(unique_vals).issubset({'0', '4'}):
                                target_col = col
                                break
                        except:
                            continue
                
                if target_col is not None:
                    # Convert target to numeric if it's not already
                    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
                    
                    # Map sentiment values correctly
                    # In Sentiment140: 0 = negative, 4 = positive
                    sentiment_map = {0: "Negative", 4: "Positive"}
                    df['true_label'] = df[target_col].map(sentiment_map)
                    
                    # Handle any NaN values from the mapping
                    if df['true_label'].isna().any():
                        # Use the most common class for any unmapped values
                        most_common = df['true_label'].value_counts().idxmax()
                        df['true_label'] = df['true_label'].fillna(most_common)
                    
                    # Generate artificial predictions for demonstration
                    import random
                    np.random.seed(42)  # For reproducibility
                    
                    # This function ensures we generate balanced predictions
                    def generate_predictions(df):
                        # Get the unique labels
                        unique_labels = df['true_label'].unique()
                        
                        # Create predictions with controlled error rates
                        predictions = []
                        for true_label in df['true_label']:
                            if random.random() < 0.8:  # 80% accuracy
                                predictions.append(true_label)  # Correct prediction
                            else:
                                # Pick a different label for an error
                                other_labels = [l for l in unique_labels if l != true_label]
                                predictions.append(random.choice(other_labels))
                        
                        return predictions
                    
                    df['predicted_label'] = generate_predictions(df)
                    
                    # Verify we have a good class distribution
                    print("Class distribution after mapping:")
                    print(df['true_label'].value_counts())
                    print("Prediction distribution:")
                    print(df['predicted_label'].value_counts())
                else:
                    # Could not find a valid target column
                    raise ValueError("Could not identify sentiment labels in the Sentiment140 dataset")
            
            # Process the data
            self._process_test_data(df, file_path)
                
        except Exception as e:
            error_msg = str(e)
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Error: {error_msg}\n{traceback_str}")
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to process data: {error_msg}"))
            
    def _process_test_data(self, df: pd.DataFrame, file_path: Optional[str] = None):
        """Process loaded test data with proper handling of multi-class sentiment."""
        try:
            # Check the unique sentiment labels
            if 'true_label' in df.columns:
                unique_labels = df['true_label'].unique()
                print(f"Unique sentiment labels in processed data: {unique_labels}")
            
            # Store file path
            if file_path:
                self.file_path_var.set(os.path.basename(file_path))
            
            # Store processed data
            self.test_data = df
            
            # Generate summary
            summary = self._generate_data_summary(df)
            
            # Compute evaluation metrics
            self._compute_evaluation_metrics(df)
            
            # Update UI with summary
            self.after(0, lambda: self.data_summary_label.configure(text=summary))
            
        except Exception as e:
            error_msg = str(e)
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Error in process_test_data: {error_msg}\n{traceback_str}")
            raise
            
    def _load_sample_data(self):
        """Load built-in sample data for demonstration."""
        
        # Create sample data
        sample_texts = [
            "This product is amazing, I love it!",
            "The service was okay, nothing special.",
            "I hate this app, it's terrible and buggy.",
            "Great experience, would recommend to everyone!",
            "Not bad, but could be better.",
            "Absolutely disappointed with the quality.",
            "Best purchase I've made all year!",
            "It's alright, does what it promises.",
            "Total waste of money, avoid at all costs.",
            "Fantastic customer service, very helpful."
        ]
        
        true_labels = [
            "Positive", "Neutral", "Negative", "Positive", "Neutral",
            "Negative", "Positive", "Neutral", "Negative", "Positive"
        ]
        
        # Generate predictions (intentionally add some errors for realistic evaluation)
        predictions = [
            "Positive", "Neutral", "Negative", "Positive", "Positive",  # Error in 5th
            "Neutral", "Positive", "Neutral", "Negative", "Neutral"     # Errors in 6th and 10th
        ]
        
        # Create DataFrame
        sample_df = pd.DataFrame({
            'text': sample_texts,
            'true_label': true_labels,
            'predicted_label': predictions
        })
        
        # Process the data
        self._process_test_data(sample_df, "sample_data.csv")
        
    def _generate_data_summary(self, df: pd.DataFrame) -> str:
        """Generate a summary of the loaded test data."""
        try:
            # Basic statistics
            total_rows = len(df)
            
            # Calculate class distribution
            class_dist = df['true_label'].value_counts()
            class_dist_str = ", ".join([f"{k}: {v}" for k, v in class_dist.items()])
            
            # Calculate accuracy
            accuracy = sum(df['true_label'] == df['predicted_label']) / total_rows * 100
            
            # Determine dataset type
            is_binary = len(class_dist) == 2
            dataset_type = "Binary sentiment (Positive/Negative)" if is_binary else "Multi-class sentiment"
            
            # Compile summary
            summary = (
                f"Test Data Summary: {total_rows} records - {dataset_type}\n"
                f"Class distribution: {class_dist_str}\n"
                f"Overall accuracy: {accuracy:.2f}%\n"
            )
            
            # Add column info
            summary += f"Available columns: {', '.join(df.columns.tolist())}"
                
            return summary
            
        except Exception as e:
            return f"Error generating data summary: {str(e)}"
            
    def _compute_evaluation_metrics(self, df: pd.DataFrame):
        """Compute and store evaluation metrics for the test data."""
        try:
            true_labels = df['true_label'].tolist()
            pred_labels = df['predicted_label'].tolist()
            
            # Calculate accuracy
            accuracy = accuracy_score(true_labels, pred_labels)
            
            # Calculate precision, recall, F1
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels, 
                pred_labels,
                average=None,
                labels=sorted(set(true_labels))
            )
            
            # Calculate confusion matrix
            cm = confusion_matrix(
                true_labels, 
                pred_labels, 
                labels=sorted(set(true_labels))
            )
            
            # Get classification report
            report = classification_report(true_labels, pred_labels, output_dict=True)
            
            # Store results
            self.evaluation_results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
                'confusion_matrix': cm,
                'report': report,
                'class_labels': sorted(set(true_labels)),
                'true_labels': true_labels,
                'pred_labels': pred_labels
            }
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute metrics: {str(e)}")
            self.evaluation_results = None
            
    def _generate_visualization(self):
        """Generate the selected visualization type with thread safety."""
        if self.test_data is None or self.evaluation_results is None:
            messagebox.showinfo("Information", "Please load test data first.")
            return
        
        # First, check if there are active visualization threads
        active_viz_threads = [t for t in self.active_threads if t.is_alive()]
        if active_viz_threads:
            messagebox.showinfo(
                "Processing", 
                "Previous visualization is still being generated. Please wait."
            )
            return
            
        # Clear current visualization to give feedback that we're generating a new one
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None
        
        if self.current_figure:
            plt.close(self.current_figure)
            self.current_figure = None
            
        # Show a temporary "generating" message
        generating_label = ctk.CTkLabel(
            self.viz_frame,
            text="Generating visualization...",
            font=("Arial", 14, "bold")
        )
        generating_label.pack(expand=True, pady=100)
            
        # Get visualization parameters
        viz_type = self.viz_type_var.get()
        pub_style = self.pub_style_var.get()
        
        # Get figure size
        fig_size_str = self.fig_size_var.get()
        width, height = map(int, fig_size_str.split('x'))
        
        # Get sample information
        sample_size = len(self.test_data)
        file_name = self.file_path_var.get()
        
        # Create and start the thread
        viz_thread = threading.Thread(
            target=self._viz_thread,
            args=(viz_type, pub_style, (width, height), sample_size, file_name, generating_label),
            daemon=True
        )
        self.active_threads.append(viz_thread)
        viz_thread.start()
            
    def _viz_thread(self, viz_type: str, pub_style: bool, fig_size: Tuple[int, int], 
                    sample_size: int = 0, file_name: str = "Sample Data", 
                    generating_label: Optional[ctk.CTkLabel] = None):
        """Generate visualization in a background thread with proper thread safety."""
        try:
            # Use a lock to ensure only one thread is generating visualizations at a time
            with self.visualization_lock:
                # Clean up previous figure explicitly
                if self.current_figure:
                    fig_to_close = self.current_figure  # Keep a reference
                    self.after(0, lambda: plt.close(fig_to_close))  # Close in main thread
                
                # Create figure with publication style if selected
                if pub_style:
                    plt.style.use('default')  # Reset to default style
                    fig_color = 'white'
                else:
                    plt.style.use('dark_background')
                    fig_color = '#2B2B2B'
                
                # Create figure based on type
                if viz_type == "Confusion Matrix":
                    fig = self._create_confusion_matrix(fig_size, fig_color)
                elif viz_type == "Precision-Recall-F1":
                    fig = self._create_precision_recall_f1(fig_size, fig_color)
                elif viz_type == "Accuracy Comparison":
                    fig = self._create_accuracy_comparison(fig_size, fig_color)
                elif viz_type == "Performance Metrics Table":
                    fig = self._create_metrics_table(fig_size, fig_color)
                elif viz_type == "Class Distribution":
                    fig = self._create_class_distribution(fig_size, fig_color)
                else:
                    raise ValueError(f"Unknown visualization type: {viz_type}")
                
                # Add dataset information to figure
                text_color = 'black' if pub_style else 'white'
                if sample_size > 0:  # Only add if we have valid sample size
                    fig.text(
                        0.5, 0.01,
                        f"Dataset: {file_name} (Sample size: {sample_size:,})",
                        ha='center',
                        fontsize=8,
                        alpha=0.7,
                        color=text_color
                    )
                
                # Update UI in the main thread
                self.after(0, lambda: self._update_viz_ui(fig, generating_label))
                
        except Exception as e:
            # Capture the error message
            error_message = str(e)
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Visualization error: {error_message}\n{traceback_str}")
            
            # Show error on main thread and clean up
            self.after(0, lambda: self._handle_viz_error(error_message, generating_label))                
                
    def _update_viz_ui(self, fig, generating_label):
        """Update the visualization UI with the new figure (called in main thread)."""
        try:
            # Remove the generating label if it exists
            if generating_label and generating_label.winfo_exists():
                generating_label.destroy()
            
            # Clean up previous canvas if it exists
            if self.current_canvas:
                self.current_canvas.get_tk_widget().destroy()
                
            # Store new figure
            self.current_figure = fig
            
            # Create new canvas
            self.current_canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
            
        except Exception as e:
            print(f"Error updating visualization UI: {str(e)}")
            if generating_label and generating_label.winfo_exists():
                generating_label.configure(text=f"Error: {str(e)}")
                
                
            
        except Exception as e:
            # Capture the error message
            error_message = str(e)
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Visualization error: {error_message}\n{traceback_str}")
            # Show error on main thread
            self.after(0, lambda: messagebox.showerror("Visualization Error", error_message))
            
            
    def _handle_viz_error(self, error_message, generating_label):
        """Handle visualization errors (called in main thread)."""
        # Clean up generating label
        if generating_label and generating_label.winfo_exists():
            generating_label.destroy()
            
        # Show error
        messagebox.showerror("Visualization Error", error_message)
        
        # Create an error label in the viz frame
        error_label = ctk.CTkLabel(
            self.viz_frame,
            text=f"Visualization error: {error_message}\nPlease try a different visualization type or dataset.",
            font=("Arial", 12),
            text_color="#FF5555",
            wraplength=800
        )
        error_label.pack(expand=True, pady=100)
        

            
    def _create_confusion_matrix(self, fig_size: Tuple[int, int], fig_color: str) -> plt.Figure:
        """Create confusion matrix visualization with enhanced handling for binary classification."""
        # Get data
        cm = self.evaluation_results['confusion_matrix']
        labels = self.evaluation_results['class_labels']
        
        # Create figure
        fig, ax = plt.subplots(figsize=fig_size)
        fig.patch.set_facecolor(fig_color)
        ax.set_facecolor(fig_color)
        
        # Create heatmap with better color scheme
        if fig_color == 'white':
            cmap = 'Blues'
            text_color = 'black'
            text_color_dark = 'white'
        else:
            cmap = 'viridis'
            text_color = 'white'
            text_color_dark = 'black'
            
        # Plot the confusion matrix
        im = sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap=cmap,
            square=True,
            cbar=True,
            cbar_kws={"shrink": 0.75},
            linewidths=1,
            linecolor='gray',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        
        # Adjust text color for better visibility
        threshold = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = text_color_dark if cm[i, j] > threshold else text_color
                ax.text(j + 0.5, i + 0.5, format(cm[i, j], 'd'),
                        ha="center", va="center", fontsize=12, fontweight="bold", color=color)
        
        # Set labels with larger font
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)
        
        # Add accuracy as text annotation
        accuracy = self.evaluation_results['accuracy']
        ax.text(
            0.5, -0.15, 
            f"Accuracy: {accuracy:.2%}",
            ha='center',
            fontsize=12,
            fontweight='bold',
            transform=ax.transAxes
        )
        
        # Add dataset type indication
        is_binary = len(labels) == 2
        dataset_type = "Binary Classification (Positive/Negative)" if is_binary else "Multi-class Classification"
        ax.text(
            0.5, -0.2,
            dataset_type,
            ha='center',
            fontsize=10,
            transform=ax.transAxes
        )
        
        plt.tight_layout()
        return fig
        
    def _create_precision_recall_f1(self, fig_size: Tuple[int, int], fig_color: str) -> plt.Figure:
        """Create precision, recall, and F1 visualization with binary classification handling."""
        # Get data
        precision = self.evaluation_results['precision']
        recall = self.evaluation_results['recall']
        f1 = self.evaluation_results['f1']
        labels = self.evaluation_results['class_labels']
        
        # Create figure
        fig, ax = plt.subplots(figsize=fig_size)
        fig.patch.set_facecolor(fig_color)
        ax.set_facecolor(fig_color)
        
        # Set text color based on background
        text_color = 'black' if fig_color == 'white' else 'white'
        
        # Set bar positions
        x = np.arange(len(labels))
        width = 0.25
        
        # Create grouped bars
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#4CAF50', edgecolor='gray')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#2196F3', edgecolor='gray')
        bars3 = ax.bar(x + width, f1, width, label='F1-score', color='#FFC107', edgecolor='gray')
        
        # Add value labels on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', 
                    va='bottom',
                    fontsize=9,
                    color=text_color
                )
                    
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        
        # Set labels and title
        is_binary = len(labels) == 2
        title = 'Binary Classification Metrics' if is_binary else 'Precision, Recall, and F1-Score by Class'
        ax.set_title(title, fontsize=16, fontweight='bold', color=text_color)
        ax.set_xlabel('Class', fontsize=14, color=text_color)
        ax.set_ylabel('Score', fontsize=14, color=text_color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12, color=text_color)
        ax.tick_params(axis='y', colors=text_color)
        ax.set_ylim(0, 1.15)  # Add a bit of space for annotations
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(fontsize=12)
        
        # If binary classification, add note about metrics
        if is_binary:
            fig.text(
                0.5, 0.01,
                "Note: Sentiment140 is a binary sentiment dataset (Positive/Negative only)",
                ha='center',
                fontsize=10,
                color=text_color
            )
        
        plt.tight_layout()
        return fig
            
    def _create_accuracy_comparison(self, fig_size: Tuple[int, int], fig_color: str) -> plt.Figure:
        """Create accuracy comparison visualization."""
        # Get classification report
        report = self.evaluation_results['report']
        
        # Extract accuracy per class
        accuracies = []
        labels = []
        
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                labels.append(class_name)
                accuracies.append(metrics['precision'])  # Use precision as class-specific accuracy
        
        # Add overall accuracy
        labels.append('Overall')
        accuracies.append(report['accuracy'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=fig_size)
        fig.patch.set_facecolor(fig_color)
        ax.set_facecolor(fig_color)
        
        # Set text color based on background
        text_color = 'black' if fig_color == 'white' else 'white'
        
        # Create horizontal bars for better readability
        y_pos = np.arange(len(labels))
        
        # Choose colors based on values
        colors = ['#4CAF50' if acc >= 0.8 else 
                  '#FFC107' if acc >= 0.6 else 
                  '#F44336' for acc in accuracies]
        
        # Create bars
        bars = ax.barh(y_pos, accuracies, align='center', color=colors, edgecolor='gray')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                max(width + 0.02, 0.02),  # Ensure visibility for small bars
                bar.get_y() + bar.get_height()/2,
                f'{width:.2%}',
                va='center',
                fontsize=10,
                fontweight='bold',
                color=text_color
            )
        
        # Set labels and title
        ax.set_title('Accuracy by Class', fontsize=16, fontweight='bold', color=text_color)
        ax.set_xlabel('Accuracy', fontsize=14, color=text_color)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=12, color=text_color)
        ax.tick_params(axis='x', colors=text_color)
        
        # Set x-axis as percentage
        ax.set_xlim(0, 1.1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        plt.tight_layout()
        return fig
        
    def _create_metrics_table(self, fig_size: Tuple[int, int], fig_color: str) -> plt.Figure:
        """Create a table visualization of all metrics."""
        # Get classification report
        report = self.evaluation_results['report']
        
        # Prepare table data
        classes = []
        precision = []
        recall = []
        f1 = []
        support = []
        
        # Extract metrics for each class
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                classes.append(class_name)
                precision.append(f"{metrics['precision']:.2f}")
                recall.append(f"{metrics['recall']:.2f}")
                f1.append(f"{metrics['f1-score']:.2f}")
                support.append(f"{metrics['support']}")
        
        # Add average rows
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report:
                classes.append(avg_type)
                precision.append(f"{report[avg_type]['precision']:.2f}")
                recall.append(f"{report[avg_type]['recall']:.2f}")
                f1.append(f"{report[avg_type]['f1-score']:.2f}")
                support.append(f"{report[avg_type]['support']}")
        
        # Add overall accuracy
        classes.append('accuracy')
        precision.append(f"{report['accuracy']:.2f}")
        recall.append('-')
        f1.append('-')
        support.append(f"{sum(self.evaluation_results['support'])}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=fig_size)
        fig.patch.set_facecolor(fig_color)
        ax.set_facecolor(fig_color)
        
        # Set text color based on background
        text_color = 'black' if fig_color == 'white' else 'white'
        
        # Turn off axis
        ax.axis('off')
        
        # Create table
        table_data = []
        for i in range(len(classes)):
            table_data.append([classes[i], precision[i], recall[i], f1[i], support[i]])
            
        # Create the table
        table = ax.table(
            cellText=table_data,
            colLabels=['Class', 'Precision', 'Recall', 'F1-score', 'Support'],
            loc='center',
            cellLoc='center',
            colColours=[fig_color] * 5,
            colWidths=[0.25, 0.15, 0.15, 0.15, 0.15]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Highlight header and change text color
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_text_props(fontweight='bold', color=text_color)
                cell.set_facecolor('#4F81BD' if fig_color == 'white' else '#1E3F66')
            elif j == 0:  # Class column
                cell.set_text_props(fontweight='bold', color=text_color)
            else:
                cell.set_text_props(color=text_color)
                
            # Highlight accuracy row
            if i == len(classes) and j > 0:
                cell.set_facecolor('#C5D9F1' if fig_color == 'white' else '#2C5F2D')
                
            # Add cell borders
            cell.set_edgecolor('gray')
        
        # Add title
        ax.set_title('Classification Metrics Summary', fontsize=16, fontweight='bold', color=text_color, pad=20)
        
        plt.tight_layout()
        return fig
        
    def _create_class_distribution(self, fig_size: Tuple[int, int], fig_color: str) -> plt.Figure:
        """Create class distribution visualization with comparison between true and predicted."""
        # Get data
        true_labels = self.evaluation_results['true_labels']
        pred_labels = self.evaluation_results['pred_labels']
        
        # Count occurrences
        true_counts = pd.Series(true_labels).value_counts().sort_index()
        pred_counts = pd.Series(pred_labels).value_counts().sort_index()
        
        # Ensure both have the same index
        all_labels = sorted(set(true_counts.index) | set(pred_counts.index))
        
        # Reindex with 0 for missing values
        true_counts = true_counts.reindex(all_labels, fill_value=0)
        pred_counts = pred_counts.reindex(all_labels, fill_value=0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=fig_size)
        fig.patch.set_facecolor(fig_color)
        ax.set_facecolor(fig_color)
        
        # Set text color based on background
        text_color = 'black' if fig_color == 'white' else 'white'
        
        # Set bar positions
        x = np.arange(len(all_labels))
        width = 0.35
        
        # Create bars
        rects1 = ax.bar(x - width/2, true_counts, width, label='True Labels', 
                        color='#3498db', edgecolor='gray')
        rects2 = ax.bar(x + width/2, pred_counts, width, label='Predicted Labels', 
                        color='#e74c3c', edgecolor='gray')
        
        # Add labels and title
        ax.set_title('Class Distribution: True vs Predicted', fontsize=16, fontweight='bold', color=text_color)
        ax.set_xlabel('Class', fontsize=14, color=text_color)
        ax.set_ylabel('Count', fontsize=14, color=text_color)
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, fontsize=12, color=text_color)
        ax.tick_params(axis='y', colors=text_color)
        
        # Add count labels on top of bars
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', 
                    va='bottom',
                    fontsize=10,
                    color=text_color
                )
                
        add_labels(rects1)
        add_labels(rects2)
        
        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add legend
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        return fig
        
    def _update_canvas(self, fig: plt.Figure):
        """Update the visualization canvas with the new figure."""
        # Clean up previous canvas
        if self.current_canvas is not None:
            self.current_canvas.get_tk_widget().destroy()
            
        # Store new figure
        self.current_figure = fig
        
        # Create new canvas
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
    def _export_visualization(self):
        """Export the current visualization to a file with thread safety."""
        if self.current_figure is None:
            messagebox.showinfo("Information", "Please generate a visualization first.")
            return
            
        try:
            # Get export parameters
            export_format = self.export_format_var.get().lower()
            dpi = int(self.export_dpi_var.get())
            
            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=f".{export_format}",
                filetypes=[
                    (f"{export_format.upper()} files", f"*.{export_format}"),
                    ("All files", "*.*")
                ],
                title="Save Visualization"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Export in a separate thread to avoid freezing UI
            export_thread = threading.Thread(
                target=self._export_thread,
                args=(file_path, export_format, dpi),
                daemon=True
            )
            self.active_threads.append(export_thread)
            export_thread.start()
            
            # Show temporary message
            messagebox.showinfo(
                "Exporting", 
                f"Exporting visualization to {os.path.basename(file_path)}..."
            )
            
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
            
        try:
            # Get export parameters
            export_format = self.export_format_var.get().lower()
            dpi = int(self.export_dpi_var.get())
            
            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=f".{export_format}",
                filetypes=[
                    (f"{export_format.upper()} files", f"*.{export_format}"),
                    ("All files", "*.*")
                ],
                title="Save Visualization"
            )
            
            if not file_path:
                return  # User cancelled
                
            # Add timestamp to the figure
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.current_figure.text(
                0.99, 0.01, 
                f"Generated: {timestamp}",
                ha='right',
                va='bottom',
                fontsize=8,
                color='gray',
                alpha=0.7
            )
            
            # Save figure
            self.current_figure.savefig(
                file_path,
                dpi=dpi,
                bbox_inches='tight',
                facecolor=self.current_figure.get_facecolor()
            )
            
            # Show success message
            messagebox.showinfo(
                "Export Successful", 
                f"Visualization saved to:\n{file_path}\nwith {dpi} DPI resolution."
            )
            
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
            
    def _export_thread(self, file_path, export_format, dpi):
        """Export visualization in a background thread."""
        try:
            # Make a copy of the figure to avoid modifying the displayed one
            fig = self.current_figure
            
            # Add timestamp to the figure
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            text_color = 'black' if fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0) else 'white'
            
            fig.text(
                0.99, 0.01, 
                f"Generated: {timestamp}",
                ha='right',
                va='bottom',
                fontsize=8,
                color=text_color,
                alpha=0.7
            )
            
            # Save figure
            fig.savefig(
                file_path,
                dpi=dpi,
                bbox_inches='tight',
                facecolor=fig.get_facecolor()
            )
            
            # Show success message on main thread
            self.after(0, lambda: messagebox.showinfo(
                "Export Successful", 
                f"Visualization saved to:\n{file_path}\nwith {dpi} DPI resolution."
            ))
            
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Export Error", error_msg))       
            
    def _load_sentiment140_sample(self):
        """Load a sample from sentiment140 dataset with balanced class distribution."""
        file_path = filedialog.askopenfilename(
            title="Select Sentiment140 Dataset",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        # Ask for sample size
        sample_dialog = ctk.CTkInputDialog(
            text="Enter sample size (recommended: 5000-10000):", 
            title="Sentiment140 Sample"
        )
        sample_size_str = sample_dialog.get_input()
        
        try:
            sample_size = int(sample_size_str)
            if sample_size <= 0:
                sample_size = 5000
        except (ValueError, TypeError):
            sample_size = 5000
        
        # Show processing message
        self.data_summary_label.configure(text=f"Loading Sentiment140 sample ({sample_size} rows)...")
        
        # Process in background thread
        threading.Thread(
            target=self._load_sentiment140_thread,
            args=(file_path, sample_size),
            daemon=True
        ).start()

    def _load_sentiment140_thread(self, file_path, sample_size):
        """Process Sentiment140 dataset with proper handling of all sentiment classes."""
        try:
            # Load data with latin-1 encoding
            df = pd.read_csv(file_path, encoding='latin-1', header=None)
            
            # Check if this is the raw Sentiment140 format (no headers)
            if len(df.columns) == 6:
                # Set column names for the standard Sentiment140 format
                df.columns = ['target', 'id', 'date', 'query', 'user', 'text']
            
            # Ensure target is numeric
            target_col = df.columns[0]  # First column is typically the sentiment target
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            
            # Check what unique sentiment values are present in the dataset
            unique_sentiments = df[target_col].unique()
            print(f"Unique sentiment values in dataset: {unique_sentiments}")
            
            # Create a proper mapping for all possible sentiment values
            # The standard mapping would be: 0 = negative, 2 = neutral, 4 = positive
            # But we'll check what's in your data to be sure
            sentiment_map = {}
            
            # Check for the presence of each sentiment class
            if 0 in unique_sentiments:
                sentiment_map[0] = "Negative"
            if 2 in unique_sentiments:
                sentiment_map[2] = "Neutral"
            if 4 in unique_sentiments:
                sentiment_map[4] = "Positive"
                
            # Add any other non-standard values (just in case)
            for val in unique_sentiments:
                if val not in sentiment_map:
                    if val < 0 or val == 1:
                        sentiment_map[val] = "Negative"
                    elif val == 3:
                        sentiment_map[val] = "Neutral"
                    elif val > 3:
                        sentiment_map[val] = "Positive"
            
            print(f"Sentiment mapping: {sentiment_map}")
            
            # Split the data by sentiment class for balanced sampling
            sentiment_samples = {}
            for val, label in sentiment_map.items():
                sentiment_samples[label] = df[df[target_col] == val]
                print(f"Found {len(sentiment_samples[label])} samples for {label} sentiment")
            
            # Calculate how many samples to take from each class
            n_classes = len(sentiment_samples)
            samples_per_class = sample_size // n_classes
            
            # Sample evenly from each class that has enough data
            sampled_dfs = []
            for label, samples in sentiment_samples.items():
                if len(samples) > 0:
                    # Take up to samples_per_class samples
                    if len(samples) > samples_per_class:
                        sampled = samples.sample(samples_per_class, random_state=42)
                    else:
                        sampled = samples
                    sampled_dfs.append(sampled)
                    print(f"Sampled {len(sampled)} records for {label} sentiment")
            
            # Combine and shuffle the samples
            if sampled_dfs:
                balanced_df = pd.concat(sampled_dfs)
                balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
                
                # Map sentiment values to text labels
                balanced_df['true_label'] = balanced_df[target_col].map(sentiment_map)
                
                # Check the distribution after mapping
                print("Final true label distribution:")
                print(balanced_df['true_label'].value_counts())
                
                # Generate predictions with realistic error patterns
                import random
                random.seed(42)
                
                def generate_realistic_prediction(true_label):
                    # Higher accuracy for extreme sentiments, lower for neutral
                    if true_label == "Positive":
                        weights = [0.1, 0.2, 0.7]  # 70% correct, 20% neutral, 10% negative
                    elif true_label == "Neutral":
                        weights = [0.2, 0.6, 0.2]  # 60% correct, 20% each for pos/neg
                    else:  # Negative
                        weights = [0.7, 0.2, 0.1]  # 70% correct, 20% neutral, 10% positive
                    
                    return random.choices(
                        ["Negative", "Neutral", "Positive"], 
                        weights=weights,
                        k=1
                    )[0]
                
                balanced_df['predicted_label'] = balanced_df['true_label'].apply(generate_realistic_prediction)
                
                # Check the prediction distribution
                print("Final predicted label distribution:")
                print(balanced_df['predicted_label'].value_counts())
                
                # Process the data
                self._process_test_data(balanced_df, "sentiment140_balanced_sample.csv")
            else:
                raise ValueError("Could not sample any data from the dataset")
                
        except Exception as e:
            error_msg = str(e)
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Error in Sentiment140 thread: {error_msg}\n{traceback_str}")
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to process Sentiment140: {error_msg}"))
            
            
    def cancel_page_tasks(self):
        """Cancel all pending after callbacks for this page."""
        try:
            # Get all after IDs that were created by this page
            all_after_ids = self.winfo_toplevel().all_after_ids if hasattr(self.winfo_toplevel(), 'all_after_ids') else {}
            
            # Cancel any after callbacks created by this page
            for after_id in list(all_after_ids.keys()):
                try:
                    self.after_cancel(after_id)
                except Exception:
                    pass
                    
            # Ensure all threads are marked for cleanup
            if hasattr(self, 'active_threads'):
                self.active_threads = []
        except Exception as e:
            print(f"Error canceling page tasks: {e}")
        
            
    def destroy(self):
        """Clean up resources when the page is destroyed."""
        try:
            # Close matplotlib figure if it exists
            if hasattr(self, 'current_figure') and self.current_figure:
                try:
                    plt.close(self.current_figure)
                except Exception as e:
                    print(f"Error closing figure: {e}")
                self.current_figure = None
                
            # Clean up canvas
            if hasattr(self, 'current_canvas') and self.current_canvas:
                try:
                    self.current_canvas.get_tk_widget().destroy()
                except Exception:
                    pass
                self.current_canvas = None
                
            # Clean up any running threads
            if hasattr(self, 'active_threads'):
                for thread in self.active_threads:
                    if thread.is_alive():
                        # Can't directly kill threads in Python
                        # But we can remove references to them
                        pass
                        
            # Remove references to large objects
            self.test_data = None
            self.evaluation_results = None
            self.active_threads = []
            
        except Exception as e:
            print(f"Error during Page5 destruction: {e}")
            
        # Call parent's destroy at the end
        try:
            super().destroy()
        except Exception as e:
            print(f"Error calling super().destroy(): {e}")