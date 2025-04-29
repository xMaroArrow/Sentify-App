import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import os
import re
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Optional, Tuple
import threading
import io
from PIL import Image

# Try to import plotly if available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class Page4(ctk.CTkFrame):
    """
    Advanced Visualization Page for Sentiment Analysis Results.
    
    This page provides comprehensive visualizations for sentiment analysis data
    including distribution charts, time series analysis, comparison visualizations,
    and model performance metrics.
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # Initialize state variables
        self.current_data = None
        self.comparison_data = None
        self.model_eval_data = None
        self.visualization_threads = []
        
        # Color schemes for consistent visualization
        self.sentiment_colors = {
            "Positive": "#4CAF50",  # Green
            "Neutral": "#FFC107",   # Amber
            "Negative": "#F44336"   # Red
        }
        
        # Create the main UI
        self._create_ui()
        
    def _create_ui(self):
        """Create the user interface for the visualization page."""
        # Create a master scrollable frame for all content
        self.master_scroll = ctk.CTkScrollableFrame(self, width=950, height=800)
        self.master_scroll.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Create header section
        self._create_header()
        
        # Create data loading section
        self._create_data_section()
        
        # Create visualization tabs
        self._create_visualization_tabs()
        
        # Create export section
        self._create_export_section()
        
    def _create_header(self):
        """Create the header section with title and description."""
        header_frame = ctk.CTkFrame(self.master_scroll)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="Advanced Sentiment Visualizations",
            font=("Arial", 24, "bold")
        )
        title_label.pack(pady=(10, 5))
        
        # Description
        desc_label = ctk.CTkLabel(
            header_frame,
            text="Generate publication-ready visualizations for sentiment analysis results",
            font=("Arial", 14),
            text_color="#888888"
        )
        desc_label.pack(pady=(0, 10))
        
    def _create_data_section(self):
        """Create the data loading section."""
        data_frame = ctk.CTkFrame(self.master_scroll)
        data_frame.pack(fill="x", padx=5, pady=5)
        
        # Title
        section_label = ctk.CTkLabel(
            data_frame,
            text="Data Sources",
            font=("Arial", 18, "bold")
        )
        section_label.pack(pady=(10, 5), padx=10, anchor="w")
        
        # Primary data selection
        primary_frame = ctk.CTkFrame(data_frame)
        primary_frame.pack(fill="x", padx=10, pady=5)
        
        primary_label = ctk.CTkLabel(
            primary_frame,
            text="Primary Data Source:",
            font=("Arial", 14)
        )
        primary_label.pack(side="left", padx=(10, 5), pady=10)
        
        self.primary_path_var = ctk.StringVar(value="No file selected")
        primary_path_label = ctk.CTkLabel(
            primary_frame,
            textvariable=self.primary_path_var,
            font=("Arial", 12),
            width=400,
            anchor="w"
        )
        primary_path_label.pack(side="left", padx=5, pady=10, fill="x", expand=True)
        
        primary_button = ctk.CTkButton(
            primary_frame,
            text="Load Primary Data",
            command=self._load_primary_data,
            width=150
        )
        primary_button.pack(side="right", padx=10, pady=10)
        
        # Comparison data selection
        compare_frame = ctk.CTkFrame(data_frame)
        compare_frame.pack(fill="x", padx=10, pady=5)
        
        compare_label = ctk.CTkLabel(
            compare_frame,
            text="Comparison Data (Optional):",
            font=("Arial", 14)
        )
        compare_label.pack(side="left", padx=(10, 5), pady=10)
        
        self.compare_path_var = ctk.StringVar(value="No file selected")
        compare_path_label = ctk.CTkLabel(
            compare_frame,
            textvariable=self.compare_path_var,
            font=("Arial", 12),
            width=400,
            anchor="w"
        )
        compare_path_label.pack(side="left", padx=5, pady=10, fill="x", expand=True)
        
        compare_button = ctk.CTkButton(
            compare_frame,
            text="Load Comparison Data",
            command=self._load_comparison_data,
            width=150
        )
        compare_button.pack(side="right", padx=10, pady=10)
        
        # Data summary
        self.data_summary_frame = ctk.CTkFrame(data_frame)
        self.data_summary_frame.pack(fill="x", padx=10, pady=5)
        
        self.data_summary_label = ctk.CTkLabel(
            self.data_summary_frame,
            text="Load data to see summary statistics",
            font=("Arial", 12),
            justify="left",
            wraplength=900
        )
        self.data_summary_label.pack(pady=10, padx=10, fill="x")
        
    def _create_visualization_tabs(self):
        """Create tabbed interface for different visualization categories."""
        # Create the tabview for visualizations
        self.tab_view = ctk.CTkTabview(self.master_scroll)
        self.tab_view.pack(fill="both", expand=True, padx=5, pady=10)
        
        # Add tabs for different visualization categories
        self.tab_view.add("Distribution")
        self.tab_view.add("Time Series")
        self.tab_view.add("Comparisons")
        self.tab_view.add("Correlations")
        self.tab_view.add("Model Performance")
        
        # Set up each tab
        self._setup_distribution_tab()
        self._setup_time_series_tab()
        self._setup_comparison_tab()
        self._setup_correlation_tab()
        self._setup_model_performance_tab()
        
    def _setup_distribution_tab(self):
        """Set up the distribution visualization tab."""
        tab = self.tab_view.tab("Distribution")
        
        # Controls frame
        controls_frame = ctk.CTkFrame(tab)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        # Visualization type selection
        type_label = ctk.CTkLabel(
            controls_frame,
            text="Visualization Type:",
            font=("Arial", 14)
        )
        type_label.pack(side="left", padx=10, pady=10)
        
        self.dist_type_var = ctk.StringVar(value="Pie Chart")
        dist_type_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["Pie Chart", "Bar Chart", "Donut Chart", "Horizontal Bar", "Stacked Bar"],
            variable=self.dist_type_var,
            width=150
        )
        dist_type_menu.pack(side="left", padx=10, pady=10)
        
        # Group by selection
        group_label = ctk.CTkLabel(
            controls_frame,
            text="Group By:",
            font=("Arial", 14)
        )
        group_label.pack(side="left", padx=(20, 10), pady=10)
        
        self.dist_group_var = ctk.StringVar(value="None")
        self.dist_group_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["None", "Date", "Username"],
            variable=self.dist_group_var,
            width=150
        )
        self.dist_group_menu.pack(side="left", padx=10, pady=10)
        
        # Generate button
        self.dist_generate_btn = ctk.CTkButton(
            controls_frame,
            text="Generate Visualization",
            command=self._generate_distribution_viz,
            width=170
        )
        self.dist_generate_btn.pack(side="right", padx=10, pady=10)
        
        # Canvas for visualization
        self.dist_canvas_frame = ctk.CTkFrame(tab)
        self.dist_canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Placeholder text
        self.dist_placeholder = ctk.CTkLabel(
            self.dist_canvas_frame,
            text="Select data and visualization options above, then click 'Generate Visualization'",
            font=("Arial", 14),
            text_color="#888888"
        )
        self.dist_placeholder.pack(expand=True)
        
        # Canvas for matplotlib
        self.dist_canvas = None
        
    def _setup_time_series_tab(self):
        """Set up the time series visualization tab."""
        tab = self.tab_view.tab("Time Series")
        
        # Controls frame
        controls_frame = ctk.CTkFrame(tab)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        # Visualization type selection
        type_label = ctk.CTkLabel(
            controls_frame,
            text="Visualization Type:",
            font=("Arial", 14)
        )
        type_label.pack(side="left", padx=10, pady=10)
        
        self.time_type_var = ctk.StringVar(value="Line Chart")
        time_type_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["Line Chart", "Area Chart", "Stacked Area", "Bar Timeline"],
            variable=self.time_type_var,
            width=150
        )
        time_type_menu.pack(side="left", padx=10, pady=10)
        
        # Time interval selection
        interval_label = ctk.CTkLabel(
            controls_frame,
            text="Time Interval:",
            font=("Arial", 14)
        )
        interval_label.pack(side="left", padx=(20, 10), pady=10)
        
        self.time_interval_var = ctk.StringVar(value="Day")
        time_interval_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["Hour", "Day", "Week", "Month"],
            variable=self.time_interval_var,
            width=100
        )
        time_interval_menu.pack(side="left", padx=10, pady=10)
        
        # Generate button
        self.time_generate_btn = ctk.CTkButton(
            controls_frame,
            text="Generate Visualization",
            command=self._generate_time_series_viz,
            width=170
        )
        self.time_generate_btn.pack(side="right", padx=10, pady=10)
        
        # Canvas for visualization
        self.time_canvas_frame = ctk.CTkFrame(tab)
        self.time_canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Placeholder text
        self.time_placeholder = ctk.CTkLabel(
            self.time_canvas_frame,
            text="Select data and visualization options above, then click 'Generate Visualization'",
            font=("Arial", 14),
            text_color="#888888"
        )
        self.time_placeholder.pack(expand=True)
        
        # Canvas for matplotlib
        self.time_canvas = None
        
    def _setup_comparison_tab(self):
        """Set up the comparison visualization tab."""
        tab = self.tab_view.tab("Comparisons")
        
        # Controls frame
        controls_frame = ctk.CTkFrame(tab)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        # Visualization type selection
        type_label = ctk.CTkLabel(
            controls_frame,
            text="Visualization Type:",
            font=("Arial", 14)
        )
        type_label.pack(side="left", padx=10, pady=10)
        
        self.comp_type_var = ctk.StringVar(value="Side-by-Side Bars")
        comp_type_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["Side-by-Side Bars", "Grouped Bars", "Radar Chart", "Heatmap"],
            variable=self.comp_type_var,
            width=150
        )
        comp_type_menu.pack(side="left", padx=10, pady=10)
        
        # Comparison factor
        factor_label = ctk.CTkLabel(
            controls_frame,
            text="Compare By:",
            font=("Arial", 14)
        )
        factor_label.pack(side="left", padx=(20, 10), pady=10)
        
        self.comp_factor_var = ctk.StringVar(value="Dataset")
        self.comp_factor_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["Dataset", "Date", "Source"],
            variable=self.comp_factor_var,
            width=100
        )
        self.comp_factor_menu.pack(side="left", padx=10, pady=10)
        
        # Generate button
        self.comp_generate_btn = ctk.CTkButton(
            controls_frame,
            text="Generate Visualization",
            command=self._generate_comparison_viz,
            width=170
        )
        self.comp_generate_btn.pack(side="right", padx=10, pady=10)
        
        # Canvas for visualization
        self.comp_canvas_frame = ctk.CTkFrame(tab)
        self.comp_canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Placeholder text
        self.comp_placeholder = ctk.CTkLabel(
            self.comp_canvas_frame,
            text="Load both primary and comparison data, then select visualization options",
            font=("Arial", 14),
            text_color="#888888"
        )
        self.comp_placeholder.pack(expand=True)
        
        # Canvas for matplotlib
        self.comp_canvas = None
        
    def _setup_correlation_tab(self):
        """Set up the correlation visualization tab."""
        tab = self.tab_view.tab("Correlations")
        
        # Controls frame
        controls_frame = ctk.CTkFrame(tab)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        # Visualization type selection
        type_label = ctk.CTkLabel(
            controls_frame,
            text="Visualization Type:",
            font=("Arial", 14)
        )
        type_label.pack(side="left", padx=10, pady=10)
        
        self.corr_type_var = ctk.StringVar(value="Scatter Plot")
        corr_type_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["Scatter Plot", "Bubble Chart", "Heatmap", "Correlation Matrix"],
            variable=self.corr_type_var,
            width=150
        )
        corr_type_menu.pack(side="left", padx=10, pady=10)
        
        # X-axis variable
        x_label = ctk.CTkLabel(
            controls_frame,
            text="X Variable:",
            font=("Arial", 14)
        )
        x_label.pack(side="left", padx=(20, 10), pady=10)
        
        self.corr_x_var = ctk.StringVar(value="Time")
        self.corr_x_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["Time", "Positive", "Negative", "Neutral"],
            variable=self.corr_x_var,
            width=100
        )
        self.corr_x_menu.pack(side="left", padx=10, pady=10)
        
        # Y-axis variable
        y_label = ctk.CTkLabel(
            controls_frame,
            text="Y Variable:",
            font=("Arial", 14)
        )
        y_label.pack(side="left", padx=(20, 10), pady=10)
        
        self.corr_y_var = ctk.StringVar(value="Positive")
        self.corr_y_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["Time", "Positive", "Negative", "Neutral"],
            variable=self.corr_y_var,
            width=100
        )
        self.corr_y_menu.pack(side="left", padx=10, pady=10)
        
        # Generate button
        self.corr_generate_btn = ctk.CTkButton(
            controls_frame,
            text="Generate Visualization",
            command=self._generate_correlation_viz,
            width=170
        )
        self.corr_generate_btn.pack(side="right", padx=10, pady=10)
        
        # Canvas for visualization
        self.corr_canvas_frame = ctk.CTkFrame(tab)
        self.corr_canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Placeholder text
        self.corr_placeholder = ctk.CTkLabel(
            self.corr_canvas_frame,
            text="Select data and correlation variables, then click 'Generate Visualization'",
            font=("Arial", 14),
            text_color="#888888"
        )
        self.corr_placeholder.pack(expand=True)
        
        # Canvas for matplotlib
        self.corr_canvas = None
        
    def _setup_model_performance_tab(self):
        """Set up the model performance visualization tab."""
        tab = self.tab_view.tab("Model Performance")
        
        # Controls frame
        controls_frame = ctk.CTkFrame(tab)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        # Visualization type selection
        type_label = ctk.CTkLabel(
            controls_frame,
            text="Visualization Type:",
            font=("Arial", 14)
        )
        type_label.pack(side="left", padx=10, pady=10)
        
        self.model_type_var = ctk.StringVar(value="Confusion Matrix")
        model_type_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["Confusion Matrix", "Precision-Recall", "F1 Scores", "Class Balance"],
            variable=self.model_type_var,
            width=150
        )
        model_type_menu.pack(side="left", padx=10, pady=10)
        
        # Load test data option
        self.load_test_button = ctk.CTkButton(
            controls_frame,
            text="Load Test Results",
            command=self._load_test_data,
            width=150
        )
        self.load_test_button.pack(side="left", padx=20, pady=10)
        
        # Generate button
        self.model_generate_btn = ctk.CTkButton(
            controls_frame,
            text="Generate Visualization",
            command=self._generate_model_viz,
            width=170
        )
        self.model_generate_btn.pack(side="right", padx=10, pady=10)
        
        # Canvas for visualization
        self.model_canvas_frame = ctk.CTkFrame(tab)
        self.model_canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Placeholder text
        self.model_placeholder = ctk.CTkLabel(
            self.model_canvas_frame,
            text="Load test data and select visualization type, then click 'Generate Visualization'",
            font=("Arial", 14),
            text_color="#888888"
        )
        self.model_placeholder.pack(expand=True)
        
        # Canvas for matplotlib
        self.model_canvas = None
        
    def _create_export_section(self):
        """Create controls for exporting visualizations."""
        export_frame = ctk.CTkFrame(self.master_scroll)
        export_frame.pack(fill="x", padx=5, pady=5)
        
        # Title
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
            values=["PNG", "PDF", "SVG", "EPS"],
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
            values=["72", "150", "300", "600"],
            variable=self.export_dpi_var,
            width=100
        )
        export_dpi_menu.pack(side="left", padx=10, pady=10)
        
        # Export current visualization button
        self.export_current_btn = ctk.CTkButton(
            controls_frame,
            text="Export Current Visualization",
            command=self._export_current_visualization,
            width=200
        )
        self.export_current_btn.pack(side="right", padx=10, pady=10)
        
        # Export all button
        self.export_all_btn = ctk.CTkButton(
            controls_frame,
            text="Export All Visualizations",
            command=self._export_all_visualizations,
            width=170
        )
        self.export_all_btn.pack(side="right", padx=10, pady=10)
        
    # ----- Data loading methods -----
    
    def _load_primary_data(self):
        """Load primary data from CSV file."""
        file_path = filedialog.askopenfilename(
            title="Select Primary Data File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Update UI
            self.primary_path_var.set(os.path.basename(file_path))
            
            # Load data in a background thread
            thread = threading.Thread(
                target=self._load_data_thread,
                args=(file_path, "primary"),
                daemon=True
            )
            thread.start()
            self.visualization_threads.append(thread)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            
    def _load_comparison_data(self):
        """Load comparison data from CSV file."""
        file_path = filedialog.askopenfilename(
            title="Select Comparison Data File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Update UI
            self.compare_path_var.set(os.path.basename(file_path))
            
            # Load data in a background thread
            thread = threading.Thread(
                target=self._load_data_thread,
                args=(file_path, "comparison"),
                daemon=True
            )
            thread.start()
            self.visualization_threads.append(thread)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            
    def _load_test_data(self):
        """Load test results for model evaluation."""
        file_path = filedialog.askopenfilename(
            title="Select Model Test Results",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Process test data in a background thread
            thread = threading.Thread(
                target=self._load_data_thread,
                args=(file_path, "test"),
                daemon=True
            )
            thread.start()
            self.visualization_threads.append(thread)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load test data: {str(e)}")
            
    def _load_data_thread(self, file_path: str, data_type: str):
        """Load and process data in a background thread."""
        try:
            # Load data using pandas
            df = pd.read_csv(file_path)
            
            # Process based on data type
            if data_type == "primary":
                self._process_primary_data(df)
            elif data_type == "comparison":
                self._process_comparison_data(df)
            elif data_type == "test":
                self._process_test_data(df)
                
        except Exception as e:
            # Show error on main thread
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to process {data_type} data: {str(e)}"))
            
    def _process_primary_data(self, df: pd.DataFrame):
        """Process primary sentiment analysis data."""
        # Basic data cleaning
        df = self._clean_sentiment_data(df)
        
        # Store the processed data
        self.current_data = df
        
        # Update UI with data summary
        summary = self._generate_data_summary(df)
        self.after(0, lambda: self._update_data_summary(summary))
        
        # Update dropdown options based on available columns
        self._update_column_options(df)
        
    def _process_comparison_data(self, df: pd.DataFrame):
        """Process comparison sentiment analysis data."""
        # Basic data cleaning
        df = self._clean_sentiment_data(df)
        
        # Store the processed data
        self.comparison_data = df
        
        # Update UI with comparison availability
        if self.current_data is not None:
            summary = "Comparison data loaded. You can now generate comparative visualizations."
            self.after(0, lambda: self._update_data_summary_append(summary))
            
    def _process_test_data(self, df: pd.DataFrame):
        """Process model test results data."""
        # Check for required columns
        required_cols = ['true_label', 'predicted_label']
        alt_cols = [['actual', 'predicted'], ['y_true', 'y_pred'], ['Truth', 'Prediction']]
        
        # Try to identify columns
        found_cols = None
        for cols in [required_cols] + alt_cols:
            if all(col in df.columns for col in cols):
                found_cols = cols
                break
                
        if found_cols:
            # Rename columns to standard format
            col_map = {found_cols[0]: 'true_label', found_cols[1]: 'predicted_label'}
            df = df.rename(columns=col_map)
        else:
            # Try to infer from data structure
            if len(df.columns) >= 2:
                # Assume first column is true and second is predicted
                df = df.rename(columns={df.columns[0]: 'true_label', df.columns[1]: 'predicted_label'})
            else:
                raise ValueError("Test data must contain columns for true labels and predicted labels")
        
        # Store processed data
        self.model_eval_data = df
        
        # Update UI
        summary = "Model evaluation data loaded. You can now generate model performance visualizations."
        self.after(0, lambda: self._update_data_summary_append(summary))
        
        def _clean_sentiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize sentiment data for visualization."""
        # Handle different column naming conventions
        # Normalize Date/Time column
        date_cols = [col for col in df.columns if col.lower() in ['date', 'datetime', 'created_at', 'timestamp']]
        if date_cols:
            df = df.rename(columns={date_cols[0]: 'Date'})
            
        # Normalize Sentiment column
        sentiment_cols = [col for col in df.columns if col.lower() in ['sentiment', 'label', 'polarity', 'class']]
        if sentiment_cols:
            df = df.rename(columns={sentiment_cols[0]: 'Sentiment'})
        
        # Normalize text column
        text_cols = [col for col in df.columns if col.lower() in ['text', 'tweet', 'content', 'message']]
        if text_cols:
            df = df.rename(columns={text_cols[0]: 'Text'})
            
        # Normalize username column
        user_cols = [col for col in df.columns if col.lower() in ['username', 'user', 'author', 'screen_name']]
        if user_cols:
            df = df.rename(columns={user_cols[0]: 'Username'})
            
        # Process Date column
        if 'Date' in df.columns:
            # Try different date formats
            date_formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%d/%m/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M"
            ]
            
            for date_format in date_formats:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], format=date_format)
                    break
                except ValueError:
                    continue
                    
            # If none of the formats worked, try pandas default parser
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except:
                    # If all else fails, create a dummy date column
                    df['Date'] = pd.date_range(
                        start=datetime.now() - timedelta(days=len(df)),
                        periods=len(df),
                        freq='D'
                    )
        
        # Standardize sentiment labels
        if 'Sentiment' in df.columns:
            # Map various sentiment notations to standard format
            sentiment_map = {
                # Positive variations
                'positive': 'Positive',
                'pos': 'Positive',
                '1': 'Positive',
                'p': 'Positive',
                'good': 'Positive',
                
                # Neutral variations
                'neutral': 'Neutral',
                'neu': 'Neutral',
                '0': 'Neutral',
                'n': 'Neutral',
                'ok': 'Neutral',
                
                # Negative variations
                'negative': 'Negative',
                'neg': 'Negative',
                '-1': 'Negative',
                'bad': 'Negative'
            }
            
            # Apply mapping for string labels
            if df['Sentiment'].dtype == object:
                df['Sentiment'] = df['Sentiment'].str.lower().map(sentiment_map).fillna(df['Sentiment'])
            
            # For numeric labels, try to map to sentiment categories
            elif pd.api.types.is_numeric_dtype(df['Sentiment']):
                # Typical mapping: negative < 0, neutral = 0, positive > 0
                df['Sentiment'] = df['Sentiment'].apply(
                    lambda x: 'Negative' if x < 0 else ('Positive' if x > 0 else 'Neutral')
                )
        
        return df
        
    def _generate_data_summary(self, df: pd.DataFrame) -> str:
        """Generate a summary of the loaded data."""
        try:
            # Basic statistics
            total_rows = len(df)
            date_range = "Unknown"
            
            # Get date range if available
            if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
                start_date = df['Date'].min().strftime("%Y-%m-%d")
                end_date = df['Date'].max().strftime("%Y-%m-%d")
                date_range = f"{start_date} to {end_date}"
            
            # Get sentiment distribution if available
            sentiment_dist = ""
            if 'Sentiment' in df.columns:
                sentiment_counts = df['Sentiment'].value_counts()
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / total_rows) * 100
                    sentiment_dist += f"{sentiment}: {count} ({percentage:.1f}%), "
                sentiment_dist = sentiment_dist.rstrip(", ")
            
            # Compile summary
            summary = (
                f"Data Summary: {total_rows} records, spanning {date_range}.\n"
                f"Available columns: {', '.join(df.columns.tolist())}.\n"
            )
            
            if sentiment_dist:
                summary += f"Sentiment distribution: {sentiment_dist}."
                
            return summary
            
        except Exception as e:
            return f"Error generating data summary: {str(e)}"
        
    def _update_data_summary(self, summary: str):
        """Update the data summary text."""
        self.data_summary_label.configure(text=summary)
        
    def _update_data_summary_append(self, text: str):
        """Append text to the existing data summary."""
        current_text = self.data_summary_label.cget("text")
        self.data_summary_label.configure(text=f"{current_text}\n{text}")
        
    def _update_column_options(self, df: pd.DataFrame):
        """Update dropdown options based on available columns."""
        # Get column names
        columns = df.columns.tolist()
        
        # Update group by options for distribution tab
        group_options = ["None"]
        if "Date" in columns:
            group_options.append("Date")
        if "Username" in columns:
            group_options.append("Username")
            
        self.dist_group_menu.configure(values=group_options)
        
        # Update correlation tab options
        corr_options = []
        if "Date" in columns:
            corr_options.append("Time")
        
        # Add numeric columns as correlation options
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        corr_options.extend(numeric_cols)
        
        # Add sentiment-related options if not already present
        for sentiment in ["Positive", "Negative", "Neutral"]:
            if sentiment not in corr_options:
                corr_options.append(sentiment)
                
        if corr_options:
            self.corr_x_menu.configure(values=corr_options)
            self.corr_y_menu.configure(values=corr_options)
            
    # ----- Visualization generation methods -----
    
    def _generate_distribution_viz(self):
        """Generate sentiment distribution visualization."""
        if self.current_data is None:
            messagebox.showinfo("Information", "Please load primary data first.")
            return
            
        # Get visualization parameters
        viz_type = self.dist_type_var.get()
        group_by = self.dist_group_var.get()
        
        # Generate visualization in a background thread
        thread = threading.Thread(
            target=self._distribution_viz_thread,
            args=(viz_type, group_by),
            daemon=True
        )
        thread.start()
        self.visualization_threads.append(thread)
        
    def _distribution_viz_thread(self, viz_type: str, group_by: str):
        """Generate distribution visualization in a background thread."""
        try:
            # Prepare data
            df = self.current_data.copy()
            
            # Get sentiment distribution
            if 'Sentiment' not in df.columns:
                raise ValueError("Data does not contain sentiment information")
                
            # Remove placeholder
            self.after(0, lambda: self.dist_placeholder.pack_forget())
            
            # Create figure
            plt.close('all')  # Close any existing figures
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            
            # Set dark theme
            plt.style.use('dark_background')
            
            # Process based on group by option
            if group_by == "None":
                # Simple distribution of all data
                sentiment_counts = df['Sentiment'].value_counts()
                
                if viz_type == "Pie Chart":
                    # Create pie chart
                    wedges, texts, autotexts = ax.pie(
                        sentiment_counts,
                        labels=sentiment_counts.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=[self.sentiment_colors.get(s, '#999999') for s in sentiment_counts.index],
                        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
                    )
                    
                    # Enhance text visibility
                    for text in texts:
                        text.set_fontsize(12)
                        text.set_fontweight('bold')
                    
                    for autotext in autotexts:
                        autotext.set_fontsize(10)
                        autotext.set_fontweight('bold')
                        
                    ax.set_title('Sentiment Distribution', fontsize=16)
                    
                elif viz_type == "Bar Chart":
                    # Create bar chart
                    bars = ax.bar(
                        sentiment_counts.index,
                        sentiment_counts.values,
                        color=[self.sentiment_colors.get(s, '#999999') for s in sentiment_counts.index],
                        edgecolor='white',
                        linewidth=1
                    )
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width()/2., 
                            height + 0.1,
                            f'{height:,}',
                            ha='center', 
                            va='bottom',
                            fontsize=10,
                            fontweight='bold'
                        )
                        
                    ax.set_title('Sentiment Distribution', fontsize=16)
                    ax.set_xlabel('Sentiment', fontsize=12)
                    ax.set_ylabel('Count', fontsize=12)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                elif viz_type == "Donut Chart":
                    # Create donut chart (pie chart with a hole)
                    wedges, texts, autotexts = ax.pie(
                        sentiment_counts,
                        labels=sentiment_counts.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=[self.sentiment_colors.get(s, '#999999') for s in sentiment_counts.index],
                        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
                    )
                    
                    # Add a circle at the center to create a donut chart
                    centre_circle = plt.Circle((0, 0), 0.70, fc='black')
                    ax.add_patch(centre_circle)
                    
                    # Add total count in the center
                    ax.text(
                        0, 0,
                        f'Total:\n{sum(sentiment_counts):,}',
                        ha='center',
                        va='center',
                        fontsize=14,
                        fontweight='bold'
                    )
                    
                    # Enhance text visibility
                    for text in texts:
                        text.set_fontsize(12)
                        text.set_fontweight('bold')
                    
                    for autotext in autotexts:
                        autotext.set_fontsize(10)
                        autotext.set_fontweight('bold')
                        
                    ax.set_title('Sentiment Distribution', fontsize=16)
                    
                elif viz_type == "Horizontal Bar":
                    # Create horizontal bar chart
                    bars = ax.barh(
                        sentiment_counts.index,
                        sentiment_counts.values,
                        color=[self.sentiment_colors.get(s, '#999999') for s in sentiment_counts.index],
                        edgecolor='white',
                        linewidth=1
                    )
                    
                    # Add value labels to the right of bars
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(
                            width + 0.5, 
                            bar.get_y() + bar.get_height()/2.,
                            f'{width:,}',
                            ha='left', 
                            va='center',
                            fontsize=10,
                            fontweight='bold'
                        )
                        
                    ax.set_title('Sentiment Distribution', fontsize=16)
                    ax.set_xlabel('Count', fontsize=12)
                    ax.set_ylabel('Sentiment', fontsize=12)
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    
                elif viz_type == "Stacked Bar":
                    # For a single dataset, stacked bar is the same as regular bar
                    # Here we'll create a percentage stacked bar
                    total = sum(sentiment_counts)
                    percentages = [(count/total) * 100 for count in sentiment_counts]
                    
                    # Create stacked bar
                    ax.bar(
                        ['All Data'],
                        [100],
                        color='none',
                        edgecolor='white',
                        linewidth=1
                    )
                    
                    # Track the bottom position for each stacked segment
                    bottom = 0
                    
                    # Add each sentiment as a stack segment
                    for i, (sentiment, percentage) in enumerate(zip(sentiment_counts.index, percentages)):
                        bar = ax.bar(
                            ['All Data'],
                            [percentage],
                            bottom=bottom,
                            color=self.sentiment_colors.get(sentiment, '#999999'),
                            edgecolor='white',
                            linewidth=1
                        )
                        
                        # Add text label in the middle of each segment
                        if percentage > 5:  # Only add text if segment is large enough
                            ax.text(
                                0,  # Bar is at x=0
                                bottom + percentage/2,  # Middle of segment
                                f'{sentiment}\n{percentage:.1f}%',
                                ha='center',
                                va='center',
                                fontsize=10,
                                fontweight='bold'
                            )
                        
                        # Update bottom for next segment
                        bottom += percentage
                        
                    ax.set_title('Sentiment Distribution (Percentage)', fontsize=16)
                    ax.set_ylabel('Percentage', fontsize=12)
                    ax.set_ylim(0, 100)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            else:  # Group by Date or Username
                if group_by == "Date" and "Date" in df.columns:
                    # Group by date (use day level for simplicity)
                    if pd.api.types.is_datetime64_any_dtype(df['Date']):
                        # Extract date component
                        df['DateGroup'] = df['Date'].dt.date
                    else:
                        # Try to convert to datetime
                        df['DateGroup'] = pd.to_datetime(df['Date']).dt.date
                        
                    # Group by date and sentiment
                    grouped = df.groupby(['DateGroup', 'Sentiment']).size().unstack(fill_value=0)
                    
                elif group_by == "Username" and "Username" in df.columns:
                    # Group by username and sentiment
                    # Limit to top 10 usernames by volume for readability
                    top_users = df['Username'].value_counts().nlargest(10).index
                    df_filtered = df[df['Username'].isin(top_users)]
                    grouped = df_filtered.groupby(['Username', 'Sentiment']).size().unstack(fill_value=0)
                    
                else:
                    raise ValueError(f"Cannot group by {group_by}")
                
                # Ensure all sentiment columns exist
                for sentiment in ['Positive', 'Neutral', 'Negative']:
                    if sentiment not in grouped.columns:
                        grouped[sentiment] = 0
                
                # Create appropriate visualization
                if viz_type in ["Bar Chart", "Stacked Bar"]:
                    # Set up the plot
                    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
                    
                    # Set dark theme
                    plt.style.use('dark_background')
                    
                    if viz_type == "Bar Chart":
                        # Create grouped bar chart
                        grouped.plot(
                            kind='bar',
                            ax=ax,
                            color=[
                                self.sentiment_colors.get('Positive', '#4CAF50'),
                                self.sentiment_colors.get('Neutral', '#FFC107'),
                                self.sentiment_colors.get('Negative', '#F44336')
                            ],
                            edgecolor='white',
                            linewidth=0.5
                        )
                        
                    else:  # Stacked Bar
                        # Create stacked bar chart
                        grouped.plot(
                            kind='bar',
                            ax=ax,
                            stacked=True,
                            color=[
                                self.sentiment_colors.get('Positive', '#4CAF50'),
                                self.sentiment_colors.get('Neutral', '#FFC107'),
                                self.sentiment_colors.get('Negative', '#F44336')
                            ],
                            edgecolor='white',
                            linewidth=0.5
                        )
                        
                    # Set labels and title
                    if group_by == "Date":
                        ax.set_title(f'Sentiment Distribution by Date', fontsize=16)
                        ax.set_xlabel('Date', fontsize=12)
                    else:
                        ax.set_title(f'Sentiment Distribution by Username', fontsize=16)
                        ax.set_xlabel('Username', fontsize=12)
                        
                    ax.set_ylabel('Count', fontsize=12)
                    ax.legend(title='Sentiment')
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Rotate x labels for better readability
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                else:
                    # For other chart types, revert to simple distribution
                    # since they don't make sense for grouped data
                    messagebox.showinfo(
                        "Visualization Changed",
                        f"{viz_type} isn't suitable for grouped data. Using Bar Chart instead."
                    )
                    
                    # Create grouped bar chart
                    grouped.plot(
                        kind='bar',
                        ax=ax,
                        color=[
                            self.sentiment_colors.get('Positive', '#4CAF50'),
                            self.sentiment_colors.get('Neutral', '#FFC107'),
                            self.sentiment_colors.get('Negative', '#F44336')
                        ],
                        edgecolor='white',
                        linewidth=0.5
                    )
                    
                    # Set labels and title
                    if group_by == "Date":
                        ax.set_title(f'Sentiment Distribution by Date', fontsize=16)
                        ax.set_xlabel('Date', fontsize=12)
                    else:
                        ax.set_title(f'Sentiment Distribution by Username', fontsize=16)
                        ax.set_xlabel('Username', fontsize=12)
                        
                    ax.set_ylabel('Count', fontsize=12)
                    ax.legend(title='Sentiment')
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Rotate x labels for better readability
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
            
            # Adjust layout and display
            plt.tight_layout()
            
            # Update canvas on the main thread
            self.after(0, lambda: self._update_dist_canvas(fig))
            
        except Exception as e:
            # Show error on main thread
            self.after(0, lambda: messagebox.showerror("Visualization Error", str(e)))
            
    def _update_dist_canvas(self, fig):
        """Update distribution visualization canvas with the new figure."""
        # Clear previous figure
        if self.dist_canvas is not None:
            self.dist_canvas.get_tk_widget().destroy()
            
        # Create new canvas with the figure
        self.dist_canvas = FigureCanvasTkAgg(fig, master=self.dist_canvas_frame)
        self.dist_canvas.draw()
        self.dist_canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def _generate_time_series_viz(self):
        """Generate time series visualization for sentiment trends."""
        if self.current_data is None:
            messagebox.showinfo("Information", "Please load primary data first.")
            return
            
        # Check if Date column is available
        if 'Date' not in self.current_data.columns:
            messagebox.showinfo("Information", "Data does not contain date information.")
            return
            
        # Get visualization parameters
        viz_type = self.time_type_var.get()
        interval = self.time_interval_var.get()
        
        # Generate visualization in a background thread
        thread = threading.Thread(
            target=self._time_series_viz_thread,
            args=(viz_type, interval),
            daemon=True
        )
        thread.start()
        self.visualization_threads.append(thread)
        
    def _time_series_viz_thread(self, viz_type: str, interval: str):
        """Generate time series visualization in a background thread."""
        try:
            # Prepare data
            df = self.current_data.copy()
            
            # Ensure Date is datetime and Sentiment column exists
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
            if 'Sentiment' not in df.columns:
                raise ValueError("Data does not contain sentiment information")
                
            # Remove rows with NaT dates
            df = df.dropna(subset=['Date'])
            
            # Remove placeholder
            self.after(0, lambda: self.time_placeholder.pack_forget())
            
            # Create figure
            plt.close('all')  # Close any existing figures
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            
            # Set dark theme
            plt.style.use('dark_background')
            
            # Group by time interval
            if interval == "Hour":
                df['TimeGroup'] = df['Date'].dt.floor('H')
                time_format = '%Y-%m-%d %H:00'
            elif interval == "Day":
                df['TimeGroup'] = df['Date'].dt.floor('D')
                time_format = '%Y-%m-%d'
            elif interval == "Week":
                df['TimeGroup'] = df['Date'].dt.to_period('W').dt.start_time
                time_format = '%Y-%m-%d'
            else:  # Month
                df['TimeGroup'] = df['Date'].dt.to_period('M').dt.start_time
                time_format = '%Y-%m'
                
            # Group by time and sentiment
            time_sentiment = df.groupby(['TimeGroup', 'Sentiment']).size().unstack(fill_value=0)
            
            # Ensure all sentiment columns exist
            for sentiment in ['Positive', 'Neutral', 'Negative']:
                if sentiment not in time_sentiment.columns:
                    time_sentiment[sentiment] = 0
                    
            # Sort by time
            time_sentiment = time_sentiment.sort_index()
            
            # Create appropriate visualization
            if viz_type == "Line Chart":
                # Line chart for each sentiment
                for sentiment in ['Positive', 'Neutral', 'Negative']:
                    ax.plot(
                        time_sentiment.index,
                        time_sentiment[sentiment],
                        label=sentiment,
                        color=self.sentiment_colors.get(sentiment, '#999999'),
                        marker='o',
                        markersize=4,
                        linewidth=2
                    )
                    
                ax.set_title(f'Sentiment Trends Over Time (by {interval})', fontsize=16)
                ax.set_xlabel('Time', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend(title='Sentiment')
                
            elif viz_type == "Area Chart":
                # Area chart for each sentiment
                for sentiment in ['Positive', 'Neutral', 'Negative']:
                    ax.fill_between(
                        time_sentiment.index,
                        time_sentiment[sentiment],
                        alpha=0.7,
                        label=sentiment,
                        color=self.sentiment_colors.get(sentiment, '#999999')
                    )
                    
                ax.set_title(f'Sentiment Trends Over Time (by {interval})', fontsize=16)
                ax.set_xlabel('Time', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend(title='Sentiment')
                
            elif viz_type == "Stacked Area":
                # Stacked area chart
                ax.stackplot(
                    time_sentiment.index,
                    [time_sentiment['Positive'], time_sentiment['Neutral'], time_sentiment['Negative']],
                    labels=['Positive', 'Neutral', 'Negative'],
                    colors=[
                        self.sentiment_colors.get('Positive', '#4CAF50'),
                        self.sentiment_colors.get('Neutral', '#FFC107'),
                        self.sentiment_colors.get('Negative', '#F44336')
                    ],
                    alpha=0.8
                )
                
                ax.set_title(f'Cumulative Sentiment Over Time (by {interval})', fontsize=16)
                ax.set_xlabel('Time', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend(title='Sentiment')
                
            elif viz_type == "Bar Timeline":
                # Bar chart timeline
                width = 0.25  # Width of bars
                
                # Position bars for each sentiment
                indices = np.arange(len(time_sentiment.index))
                
                # Plot each sentiment as separate bars
                ax.bar(
                    indices - width,
                    time_sentiment['Positive'],
                    width,
                    label='Positive',
                    color=self.sentiment_colors.get('Positive', '#4CAF50'),
                    edgecolor='white',
                    linewidth=0.5
                )
                
                ax.bar(
                    indices,
                    time_sentiment['Neutral'],
                    width,
                    label='Neutral',
                    color=self.sentiment_colors.get('Neutral', '#FFC107'),
                    edgecolor='white',
                    linewidth=0.5
                )
                
                ax.bar(
                    indices + width,
                    time_sentiment['Negative'],
                    width,
                    label='Negative',
                    color=self.sentiment_colors.get('Negative', '#F44336'),
                    edgecolor='white',
                    linewidth=0.5
                )
                
                # Set x-axis labels
                ax.set_xticks(indices)
                ax.set_xticklabels([d.strftime(time_format) for d in time_sentiment.index], rotation=45, ha='right')
                
                ax.set_title(f'Sentiment Trends Over Time (by {interval})', fontsize=16)
                ax.set_xlabel('Time', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.legend(title='Sentiment')
                
            # Format x-axis dates
            if viz_type != "Bar Timeline":
                date_format = mdates.DateFormatter(time_format)
                ax.xaxis.set_major_formatter(date_format)
                
                # Rotate date labels for better readability
                plt.xticks(rotation=45, ha='right')
                
            # Adjust layout and display
            plt.tight_layout()
            
            # Update canvas on the main thread
            self.after(0, lambda: self._update_time_canvas(fig))
            
        except Exception as e:
            # Show error on main thread
            self.after(0, lambda: messagebox.showerror("Visualization Error", str(e)))
            
    def _update_time_canvas(self, fig):
        """Update time series visualization canvas with the new figure."""
        # Clear previous figure
        if self.time_canvas is not None:
            self.time_canvas.get_tk_widget().destroy()
            
        # Create new canvas with the figure
        self.time_canvas = FigureCanvasTkAgg(fig, master=self.time_canvas_frame)
        self.time_canvas.draw()
        self.time_canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def _generate_comparison_viz(self):
        """Generate comparison visualization between datasets."""
        if self.current_data is None:
            messagebox.showinfo("Information", "Please load primary data first.")
            return
            
        # Check if comparison data is available for dataset comparison
        if self.comp_factor_var.get() == "Dataset" and self.comparison_data is None:
            messagebox.showinfo("Information", "Please load comparison data for dataset comparison.")
            return
            
        # Get visualization parameters
        viz_type = self.comp_type_var.get()
        compare_by = self.comp_factor_var.get()
        
        # Generate visualization in a background thread
        thread = threading.Thread(
            target=self._comparison_viz_thread,
            args=(viz_type, compare_by),
            daemon=True
        )
        thread.start()
        self.visualization_threads.append(thread)
        
    def _comparison_viz_thread(self, viz_type: str, compare_by: str):
        try:
        """Generate comparison visualization in a background thread."""
        # Prepare primary data
        df1 = self.current_data.copy()
        
        # Get sentiment distribution
        if 'Sentiment' not in df1.columns:
            raise ValueError("Primary data does not contain sentiment information")
            
        # Remove placeholder
        self.after(0, lambda: self.comp_placeholder.pack_forget())
        
        # Create figure
        plt.close('all')  # Close any existing figures
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        
        # Set dark theme
        plt.style.use('dark_background')
        

        
        