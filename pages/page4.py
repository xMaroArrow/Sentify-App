# page4.py
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import threading
import time
import os
from datetime import datetime, timedelta
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import the tweepy-based Twitter collector
try:
    from addons.collector import TweetCollector  # Tweepy API version
except ImportError:
    try:
        from addons.bypass_collector import TweetCollector  # Nitter fallback
    except ImportError:
        TweetCollector = None

# Import custom components
from addons.xlm_roberta_analyzer import MultilingualSentimentAnalyzer
from addons.twitter_data_manager import TwitterDataManager

class Page4(ctk.CTkFrame):
    """
    Multilingual Twitter Opinion Polling Page.
    
    This page provides multilingual sentiment analysis of Twitter data using XLM-RoBERTa.
    It supports real-time monitoring and visualization of sentiment across different languages.
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # Initialize the multilingual sentiment analyzer
        self.analyzer = MultilingualSentimentAnalyzer()
        
        # Initialize Twitter data manager
        self.data_manager = TwitterDataManager()
        
        # UI state variables
        self.running = False
        self.countdown_job = None
        self.update_thread = None
        
        # Configuration options
        self.config = {
            "cooldown": 60,  # Seconds between API calls
            "language_filter": None,  # No filter by default
            "confidence_threshold": 60.0,  # Minimum confidence threshold
            "max_tweets": 1000,  # Maximum number of tweets to keep in memory
        }

        # List of language options for the dropdown
        self.language_options = [
            "All Languages",
            "English (en)", 
            "Arabic (ar)", 
            "French (fr)",
            "Spanish (es)",
            "German (de)",
            "Italian (it)",
            "Japanese (ja)",
            "Chinese (zh)",
            "Russian (ru)",
            "Dutch (nl)",
            "Portuguese (pt)",
            "Hindi (hi)",
            "Urdu (ur)",
            "Persian (fa)",
            "Turkish (tr)"
        ]
        
        
        # Language dictionary for display names
        self.language_names = {
            "en": "English",
            "ar": "Arabic",
            "fr": "French",
            "es": "Spanish",
            "de": "German",
            "it": "Italian",
            "ja": "Japanese",
            "ko": "Korean",
            "nl": "Dutch",
            "pt": "Portuguese",
            "ru": "Russian",
            "tr": "Turkish",
            "zh": "Chinese",
            "hi": "Hindi",
            "fa": "Persian",
            "ur": "Urdu"
        }
        
        # Create the UI
        self._create_ui()
        
    def _create_ui(self):
        """Create the user interface."""
        # Use a scrollable frame as the main container
        self.scrollable = ctk.CTkScrollableFrame(self, width=950, height=800)
        self.scrollable.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Header section
        self._create_header()
        
        # Input section
        self._create_input_section()
        
        # Settings section
        self._create_settings_section()
        
        # Stats section
        self._create_stats_section()
        
        # Visualization section
        self._create_visualization_tabs()
        
        # Sample tweets section
        self._create_samples_section()
        
    def _create_header(self):
        """Create the header section."""
        header_frame = ctk.CTkFrame(self.scrollable)
        header_frame.pack(fill="x", pady=10)
        
        title = ctk.CTkLabel(
            header_frame, 
            text="Multilingual Twitter Opinion Polling", 
            font=("Arial", 24, "bold")
        )
        title.pack(side="left", padx=20, pady=10)
        
        # Status indicator
        self.status_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        self.status_frame.pack(side="right", padx=20, pady=10)
        
        status_label = ctk.CTkLabel(
            self.status_frame, 
            text="Status:", 
            font=("Arial", 14)
        )
        status_label.pack(side="left", padx=(0, 5))
        
        self.status_indicator = ctk.CTkLabel(
            self.status_frame,
            text="Idle",
            font=("Arial", 14),
            text_color="#FFA500"  # Orange for idle
        )
        self.status_indicator.pack(side="left")
        
        # Description text
        description = ctk.CTkLabel(
            self.scrollable,
            text="Monitor and analyze Twitter sentiment in multiple languages using XLM-RoBERTa.",
            font=("Arial", 14),
            wraplength=900
        )
        description.pack(pady=(0, 10))
        
    def _create_input_section(self):
        """Create the input section for Twitter searches."""
        input_frame = ctk.CTkFrame(self.scrollable)
        input_frame.pack(fill="x", pady=10)
        
        # Search term input
        search_label = ctk.CTkLabel(
            input_frame,
            text="Search Term:",
            font=("Arial", 14)
        )
        search_label.pack(side="left", padx=(20, 5), pady=10)
        
        self.search_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Enter hashtag or keyword",
            width=300
        )
        self.search_entry.pack(side="left", padx=5, pady=10)
        
        # Action buttons
        self.start_button = ctk.CTkButton(
            input_frame,
            text="Start Monitoring",
            command=self.start_monitoring,
            width=150
        )
        self.start_button.pack(side="left", padx=10, pady=10)
        
        self.stop_button = ctk.CTkButton(
            input_frame,
            text="Stop",
            command=self.stop_monitoring,
            width=100,
            fg_color="#E74C3C"  # Red color for stop button
        )
        self.stop_button.pack(side="left", padx=5, pady=10)
        
        # Load CSV option
        self.load_csv_button = ctk.CTkButton(
            input_frame,
            text="Load CSV",
            command=self.load_csv_file,
            width=100,
            fg_color="#27AE60"  # Green color
        )
        self.load_csv_button.pack(side="left", padx=5, pady=10)
        
        # Clear data button
        self.clear_button = ctk.CTkButton(
            input_frame,
            text="Clear Data",
            command=self.clear_data,
            width=100,
            fg_color="#95A5A6"  # Gray color
        )
        self.clear_button.pack(side="left", padx=5, pady=10)
        
        # Export results button
        self.export_button = ctk.CTkButton(
            input_frame,
            text="Export Results",
            command=self.export_results,
            width=120,
            state="disabled"
        )
        self.export_button.pack(side="right", padx=20, pady=10)
        
        # Progress bar
        self.progress_frame = ctk.CTkFrame(self.scrollable)
        self.progress_frame.pack(fill="x", pady=(0, 10))
        
        self.countdown_label = ctk.CTkLabel(
            self.progress_frame,
            text="Next request in: 0s",
            font=("Arial", 12)
        )
        self.countdown_label.pack(side="left", padx=20, pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, width=400)
        self.progress_bar.pack(side="left", padx=10, pady=5)
        self.progress_bar.set(0)
        
    def _create_settings_section(self):
        """Create the settings section for configuration options."""
        settings_frame = ctk.CTkFrame(self.scrollable)
        settings_frame.pack(fill="x", pady=10)
        
        settings_label = ctk.CTkLabel(
            settings_frame,
            text="Settings",
            font=("Arial", 16, "bold")
        )
        settings_label.pack(anchor="w", padx=20, pady=(10, 5))
        
        # Create two columns for settings
        settings_columns = ctk.CTkFrame(settings_frame, fg_color="transparent")
        settings_columns.pack(fill="x", padx=20, pady=5)
        
        # Left column
        left_column = ctk.CTkFrame(settings_columns, fg_color="transparent")
        left_column.pack(side="left", fill="both", expand=True)
        
        # Cooldown setting
        cooldown_frame = ctk.CTkFrame(left_column)
        cooldown_frame.pack(fill="x", pady=5)
        
        cooldown_label = ctk.CTkLabel(
            cooldown_frame,
            text="API Cooldown (seconds):",
            font=("Arial", 12)
        )
        cooldown_label.pack(side="left", padx=10, pady=5)
        
        cooldown_values = ["30", "60", "120", "300"]
        self.cooldown_var = ctk.StringVar(value=str(self.config["cooldown"]))
        cooldown_dropdown = ctk.CTkOptionMenu(
            cooldown_frame,
            values=cooldown_values,
            variable=self.cooldown_var,
            command=self._update_cooldown
        )
        cooldown_dropdown.pack(side="left", padx=5, pady=5)
        
        # Confidence threshold setting
        confidence_frame = ctk.CTkFrame(left_column)
        confidence_frame.pack(fill="x", pady=5)
        
        confidence_label = ctk.CTkLabel(
            confidence_frame,
            text="Confidence Threshold (%):",
            font=("Arial", 12)
        )
        confidence_label.pack(side="left", padx=10, pady=5)
        
        self.confidence_slider = ctk.CTkSlider(
            confidence_frame,
            from_=0,
            to=100,
            number_of_steps=20,
            command=self._update_confidence_threshold
        )
        self.confidence_slider.pack(side="left", padx=5, pady=5, fill="x", expand=True)
        self.confidence_slider.set(self.config["confidence_threshold"])
        
        self.confidence_label = ctk.CTkLabel(
            confidence_frame,
            text=f"{self.config['confidence_threshold']:.0f}%",
            font=("Arial", 12)
        )
        self.confidence_label.pack(side="left", padx=5, pady=5)
        
        # Right column
        right_column = ctk.CTkFrame(settings_columns, fg_color="transparent")
        right_column.pack(side="left", fill="both", expand=True)
        
        # Language filter
        language_frame = ctk.CTkFrame(right_column)
        language_frame.pack(fill="x", pady=5)

        language_label = ctk.CTkLabel(
            language_frame,
            text="Language Filter:",
            font=("Arial", 12)
        )
        language_label.pack(side="left", padx=10, pady=5)

        # Create list of common languages with their codes
        language_options = [
            "All Languages",
            "English (en)", 
            "Arabic (ar)", 
            "French (fr)",
            "Spanish (es)",
            "German (de)",
            "Italian (it)",
            "Japanese (ja)",
            "Chinese (zh)",
            "Russian (ru)",
            "Dutch (nl)",
            "Portuguese (pt)",
            "Hindi (hi)",
            "Urdu (ur)",
            "Persian (fa)",
            "Turkish (tr)"
        ]

        self.lang_filter_var = ctk.StringVar(value="All Languages")
        language_dropdown = ctk.CTkOptionMenu(
            language_frame,
            values=language_options,
            variable=self.lang_filter_var,
            command=self._update_language_filter,
            width=180
        )
        language_dropdown.pack(side="left", padx=5, pady=5)

        # Multi-select button - opens a dialog to select multiple languages
        self.multi_select_button = ctk.CTkButton(
            language_frame,
            text="Multi-Select",
            command=self._open_language_selection,
            width=100
        )
        self.multi_select_button.pack(side="left", padx=5, pady=5)
                
        # Max tweets setting
        max_tweets_frame = ctk.CTkFrame(right_column)
        max_tweets_frame.pack(fill="x", pady=5)
        
        max_tweets_label = ctk.CTkLabel(
            max_tweets_frame,
            text="Max Tweets to Store:",
            font=("Arial", 12)
        )
        max_tweets_label.pack(side="left", padx=10, pady=5)
        
        max_tweets_values = ["100", "500", "1000", "5000", "10000"]
        self.max_tweets_var = ctk.StringVar(value=str(self.config["max_tweets"]))
        max_tweets_dropdown = ctk.CTkOptionMenu(
            max_tweets_frame,
            values=max_tweets_values,
            variable=self.max_tweets_var,
            command=self._update_max_tweets
        )
        max_tweets_dropdown.pack(side="left", padx=5, pady=5)
        
    def _create_stats_section(self):
        """Create the statistics section."""
        stats_frame = ctk.CTkFrame(self.scrollable)
        stats_frame.pack(fill="x", pady=10)
        
        stats_label = ctk.CTkLabel(
            stats_frame,
            text="Statistics",
            font=("Arial", 16, "bold")
        )
        stats_label.pack(anchor="w", padx=20, pady=(10, 5))
        
        # Create two columns for statistics
        stats_columns = ctk.CTkFrame(stats_frame, fg_color="transparent")
        stats_columns.pack(fill="x", padx=20, pady=5)
        
        # Left column - Overall stats
        left_column = ctk.CTkFrame(stats_columns)
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        overall_label = ctk.CTkLabel(
            left_column,
            text="Overall Sentiment",
            font=("Arial", 14, "bold")
        )
        overall_label.pack(anchor="w", padx=10, pady=5)
        
        # Overall statistics
        self.overall_stats = ctk.CTkLabel(
            left_column,
            text="No data available",
            font=("Arial", 12),
            justify="left",
            wraplength=400
        )
        self.overall_stats.pack(anchor="w", padx=10, pady=5)
        
        # Right column - Language stats
        right_column = ctk.CTkFrame(stats_columns)
        right_column.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        language_stats_label = ctk.CTkLabel(
            right_column,
            text="Language Distribution",
            font=("Arial", 14, "bold")
        )
        language_stats_label.pack(anchor="w", padx=10, pady=5)
        
        # Language statistics
        self.language_stats = ctk.CTkLabel(
            right_column,
            text="No data available",
            font=("Arial", 12),
            justify="left",
            wraplength=400
        )
        self.language_stats.pack(anchor="w", padx=10, pady=5)
        
    def _create_visualization_tabs(self):
        """Create the visualization section with tabs for different charts."""
        viz_frame = ctk.CTkFrame(self.scrollable)
        viz_frame.pack(fill="x", pady=10)
        
        viz_label = ctk.CTkLabel(
            viz_frame,
            text="Visualizations",
            font=("Arial", 16, "bold")
        )
        viz_label.pack(anchor="w", padx=20, pady=(10, 5))
        
        # Create tabview for different visualizations
        self.viz_tabview = ctk.CTkTabview(viz_frame, height=400)
        self.viz_tabview.pack(fill="x", padx=20, pady=10)
        
        # Add tabs
        self.viz_tabview.add("Sentiment Distribution")
        self.viz_tabview.add("Time Trends")
        self.viz_tabview.add("Language Analysis")
        self.viz_tabview.add("Correlations")
        
        # Initialize placeholders for the canvases
        self.dist_canvas = None
        self.trend_canvas = None
        self.lang_canvas = None
        self.corr_canvas = None
        
        # Set up each tab
        self._setup_distribution_tab()
        self._setup_time_series_tab()
        self._setup_language_tab()
        self._setup_correlation_tab()
        
    def _setup_distribution_tab(self):
        """Set up the distribution visualization tab."""
        tab = self.viz_tabview.tab("Sentiment Distribution")
        
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
            values=["Pie Chart", "Bar Chart", "Donut Chart", "Horizontal Bar"],
            variable=self.dist_type_var,
            width=150
        )
        dist_type_menu.pack(side="left", padx=10, pady=10)
        
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
            text="Start monitoring or load data to see sentiment distribution",
            font=("Arial", 14),
            text_color="#888888"
        )
        self.dist_placeholder.pack(expand=True)
    
    def _setup_time_series_tab(self):
        """Set up the time series visualization tab."""
        tab = self.viz_tabview.tab("Time Trends")
        
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
        self.trend_placeholder = ctk.CTkLabel(
            self.time_canvas_frame,
            text="Start monitoring or load data to see sentiment trends over time",
            font=("Arial", 14),
            text_color="#888888"
        )
        self.trend_placeholder.pack(expand=True)
        
    def _setup_language_tab(self):
        """Set up the language analysis visualization tab."""
        tab = self.viz_tabview.tab("Language Analysis")
        
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
        
        self.lang_type_var = ctk.StringVar(value="Combined View")
        lang_type_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["Combined View", "Pie Chart", "Bar Chart", "Stacked Bars"],
            variable=self.lang_type_var,
            width=150
        )
        lang_type_menu.pack(side="left", padx=10, pady=10)
        
        # Top languages selection
        top_label = ctk.CTkLabel(
            controls_frame,
            text="Top Languages:",
            font=("Arial", 14)
        )
        top_label.pack(side="left", padx=(20, 10), pady=10)
        
        self.top_langs_var = ctk.StringVar(value="5")
        top_langs_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["3", "5", "8", "All"],
            variable=self.top_langs_var,
            width=80
        )
        top_langs_menu.pack(side="left", padx=10, pady=10)
        
        # Generate button
        self.lang_generate_btn = ctk.CTkButton(
            controls_frame,
            text="Generate Visualization",
            command=self._generate_language_viz,
            width=170
        )
        self.lang_generate_btn.pack(side="right", padx=10, pady=10)
        
        # Canvas for visualization
        self.lang_canvas_frame = ctk.CTkFrame(tab)
        self.lang_canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Placeholder text
        self.lang_placeholder = ctk.CTkLabel(
            self.lang_canvas_frame,
            text="Start monitoring or load data to see language distribution",
            font=("Arial", 14),
            text_color="#888888"
        )
        self.lang_placeholder.pack(expand=True)
        
    def _setup_correlation_tab(self):
        """Set up the correlation visualization tab."""
        tab = self.viz_tabview.tab("Correlations")
        
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
            values=["Scatter Plot", "Bubble Chart", "Heatmap"],
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
        
    def _create_samples_section(self):
        """Create the section for sample tweets."""
        samples_frame = ctk.CTkFrame(self.scrollable)
        samples_frame.pack(fill="x", pady=10)
        
        samples_label = ctk.CTkLabel(
            samples_frame,
            text="Sample Tweets",
            font=("Arial", 16, "bold")
        )
        samples_label.pack(anchor="w", padx=20, pady=(10, 5))
        
        # Create container for sample tweets
        self.samples_container = ctk.CTkFrame(samples_frame)
        self.samples_container.pack(fill="x", padx=20, pady=10)
        
        # Create sections for each sentiment
        sentiments = ["Positive", "Neutral", "Negative"]
        self.sample_frames = {}
        
        for sentiment in sentiments:
            frame = ctk.CTkFrame(self.samples_container)
            frame.pack(fill="x", pady=5)
            
            # Label with sentiment name
            sentiment_label = ctk.CTkLabel(
                frame,
                text=f"{sentiment} Tweets",
                font=("Arial", 14, "bold")
            )
            sentiment_label.pack(anchor="w", padx=10, pady=5)
            
            # Sample text area
            text_area = ctk.CTkTextbox(
                frame,
                height=70,
                wrap="word"
            )
            text_area.pack(fill="x", padx=10, pady=(0, 10))
            text_area.insert("1.0", f"No {sentiment.lower()} tweets available yet.")
            text_area.configure(state="disabled")
            
            self.sample_frames[sentiment] = text_area
            
    # ---- Event handlers and functionality ----
    
    def start_monitoring(self):
        """Start monitoring Twitter for the specified search term."""
        # Check if analyzer is ready
        if not self.analyzer.initialized:
            messagebox.showinfo("Model Loading", "XLM-RoBERTa model is still loading. Please wait a moment and try again.")
            return
            
        # Check if collector is available
        if TweetCollector is None:
            messagebox.showerror("Error", "Twitter collector is not available. Please check your installation.")
            return
            
        # Get search term
        search_term = self.search_entry.get().strip()
        if not search_term:
            messagebox.showinfo("Input Required", "Please enter a hashtag or keyword to search for.")
            return
            
    # Update UI
        self.update_status("Initializing...", "#0078D7")  # Blue for processing
        self.start_button.configure(state="disabled")
        self.progress_bar.set(0.2)
        
        # Start collection in a background thread
        threading.Thread(target=self._start_collection_thread, args=(search_term,), daemon=True).start()
        
    def _start_collection_thread(self, search_term):
        """Start Twitter collection in a background thread."""
        try:
            # Start collector with current cooldown setting
            cooldown = int(self.cooldown_var.get())
            success = self.data_manager.start_collection(search_term, cooldown=cooldown)
            
            if not success:
                self.after(0, lambda: messagebox.showerror("Collection Error", "Failed to start tweet collection. Check if TweetCollector is available."))
                self.after(0, lambda: self.update_status("Error starting collection", "#E74C3C"))  # Red for error
                self.after(0, lambda: self.start_button.configure(state="normal"))
                self.after(0, lambda: self.progress_bar.set(0))
                return
                
            # Clear previous data
            self.data_manager.clear_data()
            
            # Update UI
            self.after(0, lambda: self.update_status("Collecting tweets...", "#27AE60"))  # Green for success
            self.after(0, lambda: self.progress_bar.set(1.0))
            self.after(0, lambda: self.export_button.configure(state="normal"))
            self.after(0, lambda: self.start_button.configure(state="normal"))
            
            # Start monitoring loop
            self.running = True
            self.after(0, self._schedule_update)
            
            # Reset progress after delay
            self.after(2000, lambda: self.progress_bar.set(0))
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to start tweet collection: {str(e)}"))
            self.after(0, lambda: self.update_status("Error", "#E74C3C"))  # Red for error
            self.after(0, lambda: self.start_button.configure(state="normal"))
            self.after(0, lambda: self.progress_bar.set(0))
            
    def _schedule_update(self):
        """Schedule the next data update."""
        if not self.running:
            return
            
        # Calculate seconds until next request
        sec = self.data_manager.seconds_until_next_request()
        
        # Update UI with countdown
        if sec > 0:
            cooldown = int(self.cooldown_var.get())
            progress = max(0, min(1.0, 1 - (sec / cooldown)))
            
            self.countdown_label.configure(text=f"Next request in: {sec}s")
            self.progress_bar.set(progress)
        else:
            self.countdown_label.configure(text="Processing...")
            self.progress_bar.set(1.0)
            
            # Process new tweets
            self._process_new_tweets()
            
        # Schedule next update
        self.countdown_job = self.after(1000, self._schedule_update)
        
    def _process_new_tweets(self):
        """Process new tweets from the collector."""
        if not self.running:
            return
            
        # Start processing in a background thread
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_thread = threading.Thread(target=self._process_tweets_thread, daemon=True)
            self.update_thread.start()
            
    def _process_tweets_thread(self):
        """Process tweets in a background thread."""
        try:
            # Get new tweets
            new_tweets = self.data_manager.get_new_tweets()
            
            if not new_tweets:
                self.after(0, lambda: self.update_status("Waiting for tweets...", "#FFA500"))  # Orange for waiting
                return
                
            # Get language filter
            language_filter = self._get_language_filter()
            
            # Get confidence threshold
            confidence_threshold = self.config["confidence_threshold"]
            
            # Update status
            self.after(0, lambda: self.update_status("Analyzing tweets...", "#0078D7"))  # Blue for processing
            
            # Analyze new tweets
            count = self.data_manager.update_sentiment_data(
                self.analyzer, 
                new_tweets, 
                language_filter=language_filter,
                confidence_threshold=confidence_threshold
            )
            
            # Update UI
            if count > 0:
                self.after(0, lambda: self.update_status(f"Analyzed {count} new tweets", "#27AE60"))  # Green for success
                self.after(0, self._update_stats)
                self.after(0, self._update_visualizations)
                self.after(0, self._update_samples)
            else:
                self.after(0, lambda: self.update_status("No new tweets to analyze", "#FFA500"))  # Orange for waiting
                
        except Exception as e:
            self.after(0, lambda: self.update_status(f"Error: {str(e)}", "#E74C3C"))  # Red for error
            print(f"Error processing tweets: {e}")
            
    def stop_monitoring(self):
        """Stop monitoring Twitter."""
        self.running = False
        
        # Stop collector
        self.data_manager.stop_collection()
        
        # Cancel countdown timer
        if self.countdown_job:
            self.after_cancel(self.countdown_job)
            self.countdown_job = None
            
        # Update UI
        self.update_status("Stopped", "#95A5A6")  # Gray for stopped
        self.countdown_label.configure(text="Next request in: 0s")
        self.progress_bar.set(0)
        
    def load_csv_file(self):
        """Load tweet data from a CSV file."""
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Tweet CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        # Update UI
        self.update_status("Loading CSV...", "#0078D7")  # Blue for processing
        self.progress_bar.set(0.2)
        
        # Stop any ongoing monitoring
        self.stop_monitoring()
        
        # Clear previous data
        self.data_manager.clear_data()
        
        # Load CSV in background thread
        threading.Thread(target=self._load_csv_thread, args=(file_path,), daemon=True).start()
        
    def _load_csv_thread(self, file_path):
        """Load and process CSV in a background thread."""
        try:
            # Load the CSV file
            success = self.data_manager.load_csv(file_path)
            
            if not success:
                self.after(0, lambda: messagebox.showerror("Load Error", "Failed to load CSV file."))
                self.after(0, lambda: self.update_status("Error loading CSV", "#E74C3C"))  # Red for error
                self.after(0, lambda: self.progress_bar.set(0))
                return
                
            # Update progress
            self.after(0, lambda: self.progress_bar.set(0.4))
            self.after(0, lambda: self.update_status("Processing tweets...", "#0078D7"))  # Blue for processing
            
            # Get all tweets
            tweets = self.data_manager.get_new_tweets()
            
            if not tweets:
                self.after(0, lambda: messagebox.showinfo("Empty File", "No tweets found in the CSV file."))
                self.after(0, lambda: self.update_status("No tweets found", "#FFA500"))  # Orange for warning
                self.after(0, lambda: self.progress_bar.set(0))
                return
                
            # Get language filter and confidence threshold
            language_filter = self._get_language_filter()
            confidence_threshold = self.config["confidence_threshold"]
            
            # Process tweets in batches to avoid blocking UI
            total_tweets = len(tweets)
            batch_size = 50
            processed = 0
            
            for i in range(0, total_tweets, batch_size):
                batch = tweets[i:i+batch_size]
                processed += self.data_manager.update_sentiment_data(
                    self.analyzer,
                    batch,
                    language_filter=language_filter,
                    confidence_threshold=confidence_threshold
                )
                
                # Update progress
                progress = 0.4 + (0.6 * min(1.0, (i + batch_size) / total_tweets))
                self.after(0, lambda p=progress: self.progress_bar.set(p))
                self.after(0, lambda p=processed: self.update_status(f"Processed {p}/{total_tweets} tweets", "#0078D7"))
                
                # Give UI a chance to update
                time.sleep(0.1)
                
            # Update UI
            self.after(0, lambda: self.update_status(f"Analyzed {processed} tweets", "#27AE60"))  # Green for success
            self.after(0, lambda: self.export_button.configure(state="normal"))
            self.after(0, self._update_stats)
            self.after(0, self._update_visualizations)
            self.after(0, self._update_samples)
            
            # Reset progress after delay
            self.after(2000, lambda: self.progress_bar.set(0))
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to process CSV: {str(e)}"))
            self.after(0, lambda: self.update_status("Error", "#E74C3C"))  # Red for error
            self.after(0, lambda: self.progress_bar.set(0))
            
    def _generate_correlation_viz(self):
        """Generate correlation visualization between selected variables."""
        if not self.data_manager.sentiment_data:
            messagebox.showinfo("Information", "Please load primary data first.")
            return
            
        try:
            # Get selected variables
            x_var = self.corr_x_var.get()
            y_var = self.corr_y_var.get()
            
            # Remove placeholder
            self.corr_placeholder.pack_forget()
            
            # Create figure
            plt.close('all')  # Close any existing figures
            fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
            
            # Set dark theme
            plt.style.use('dark_background')
            
            # Prepare data
            x_data = []
            y_data = []
            colors = []
            
            for item in self.data_manager.sentiment_data:
                # Get x data
                if x_var == "Time":
                    x_value = item["timestamp"]
                else:
                    x_value = item["confidence"][x_var]
                    
                # Get y data
                if y_var == "Time":
                    y_value = item["timestamp"]
                else:
                    y_value = item["confidence"][y_var]
                
                x_data.append(x_value)
                y_data.append(y_value)
                colors.append({
                    "Positive": "#4CAF50",
                    "Neutral": "#FFC107",
                    "Negative": "#F44336"
                }.get(item["sentiment"], "#FFFFFF"))
            
            # Create scatter plot
            scatter = ax.scatter(
                x_data, 
                y_data,
                alpha=0.6,
                c=colors,
                s=50
            )
            
            # Set labels and title
            ax.set_title(f"Correlation: {x_var} vs {y_var}", color="white", fontsize=14)
            ax.set_xlabel(x_var, color="white")
            ax.set_ylabel(y_var, color="white")
            
            # Add grid
            ax.grid(True, linestyle="--", alpha=0.3)
            
            # Add legend for sentiment colors
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor="#4CAF50", label='Positive', markersize=8),
                Line2D([0], [0], marker='o', color='w', markerfacecolor="#FFC107", label='Neutral', markersize=8),
                Line2D([0], [0], marker='o', color='w', markerfacecolor="#F44336", label='Negative', markersize=8)
            ]
            ax.legend(handles=legend_elements, title="Sentiment")
            
            # Format x-axis if it's a time variable
            if x_var == "Time":
                fig.autofmt_xdate()
            
            # Adjust layout
            plt.tight_layout()
            
            # Update canvas
            if self.corr_canvas:
                self.corr_canvas.get_tk_widget().destroy()
                
            self.corr_canvas = FigureCanvasTkAgg(fig, master=self.corr_canvas_frame)
            self.corr_canvas.draw()
            self.corr_canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Error generating correlation visualization: {str(e)}")
    
    def _generate_distribution_viz(self):
        """Generate sentiment distribution visualization."""
        if not self.data_manager.sentiment_data:
            messagebox.showinfo("Information", "Please load data first.")
            return
            
        # Get visualization type
        viz_type = self.dist_type_var.get()
        
        try:
            # Remove placeholder
            self.dist_placeholder.pack_forget()
            
            # Calculate sentiment counts
            sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
            
            for item in self.data_manager.sentiment_data:
                sentiment_counts[item["sentiment"]] += 1
                
            # Create figure
            plt.close('all')  # Close any existing figures
            fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
            
            # Set dark theme
            plt.style.use('dark_background')
            
            # Create colors
            colors = ["#4CAF50", "#FFC107", "#F44336"]  # Green, Amber, Red
            
            if viz_type == "Pie Chart":
                # Create pie chart
                wedges, texts, autotexts = ax.pie(
                    sentiment_counts.values(),
                    labels=sentiment_counts.keys(),
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors,
                    explode=(0.05, 0, 0),  # Slightly emphasize positive
                    wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}
                )
                
                # Enhance text styling
                for text in texts:
                    text.set_color('white')
                    text.set_fontweight('bold')
                    
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(8)
                    
            elif viz_type == "Bar Chart":
                # Create bar chart
                bars = ax.bar(
                    sentiment_counts.keys(),
                    sentiment_counts.values(),
                    color=colors,
                    edgecolor='white',
                    linewidth=0.5
                )
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.1,
                        f'{height:,}',
                        ha='center',
                        va='bottom',
                        color='white'
                    )
                    
                ax.set_xlabel("Sentiment", color="white")
                ax.set_ylabel("Count", color="white")
                ax.grid(True, linestyle="--", alpha=0.3, axis="y")
                
            elif viz_type == "Donut Chart":
                # Create pie chart with a hole in the middle
                wedges, texts, autotexts = ax.pie(
                    sentiment_counts.values(),
                    labels=sentiment_counts.keys(),
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}
                )
                
                # Add a circle at the center to create a donut chart
                centre_circle = plt.Circle((0, 0), 0.70, fc='#2B2B2B')
                ax.add_patch(centre_circle)
                
                # Add total count in the center
                total = sum(sentiment_counts.values())
                ax.text(
                    0, 0,
                    f'Total\n{total:,}',
                    ha='center',
                    va='center',
                    fontsize=12,
                    fontweight='bold',
                    color='white'
                )
                
                # Enhance text styling
                for text in texts:
                    text.set_color('white')
                    text.set_fontweight('bold')
                    
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(8)
                
            elif viz_type == "Horizontal Bar":
                # Create horizontal bar chart
                bars = ax.barh(
                    list(sentiment_counts.keys()),
                    list(sentiment_counts.values()),
                    color=colors,
                    edgecolor='white',
                    linewidth=0.5
                )
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(
                        width + 0.1,
                        bar.get_y() + bar.get_height()/2.,
                        f'{width:,}',
                        va='center',
                        color='white'
                    )
                    
                ax.set_xlabel("Count", color="white")
                ax.set_ylabel("Sentiment", color="white")
                ax.grid(True, linestyle="--", alpha=0.3, axis="x")
            
            # Add title with timestamp
            ax.set_title(
                "Sentiment Distribution\n" + 
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                color='white',
                fontsize=14
            )
            
            # Adjust layout
            plt.tight_layout()
            
            # Update canvas
            if self.dist_canvas:
                self.dist_canvas.get_tk_widget().destroy()
                
            self.dist_canvas = FigureCanvasTkAgg(fig, master=self.dist_canvas_frame)
            self.dist_canvas.draw()
            self.dist_canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Error generating distribution visualization: {str(e)}")
    
    def _generate_time_series_viz(self):
        """Generate time series visualization for sentiment trends."""
        if not self.data_manager.sentiment_data:
            messagebox.showinfo("Information", "Please load data first.")
            return
            
        # Get visualization parameters
        viz_type = self.time_type_var.get()
        interval = self.time_interval_var.get()
        
        # Map UI interval to pandas interval
        interval_map = {
            "Hour": "1H",
            "Day": "1D",
            "Week": "1W",
            "Month": "1M"
        }
        pandas_interval = interval_map.get(interval, "1D")
        
        try:
            # Get sentiment trend data
            sentiment_trend = self.data_manager.get_sentiment_trend(
                time_interval=pandas_interval, 
                last_hours=24 if interval == "Hour" else 720  # 30 days for non-hourly
            )
            
            if sentiment_trend.empty:
                messagebox.showinfo("No Trend Data", "Not enough time data points for the selected interval.")
                return
                
            # Remove placeholder
            self.trend_placeholder.pack_forget()
            
            # Create figure
            plt.close('all')  # Close any existing figures
            fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
            
            # Set dark theme
            plt.style.use('dark_background')
            
            if viz_type == "Line Chart":
                # Plot lines for each sentiment
                sentiment_trend.plot(
                    ax=ax,
                    color=["#4CAF50", "#FFC107", "#F44336"],  # Green, Amber, Red
                    linewidth=2,
                    marker='o',
                    markersize=4
                )
                
            elif viz_type == "Area Chart":
                # Area charts for each sentiment
                for col in sentiment_trend.columns:
                    color = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}.get(col, "#999999")
                    ax.fill_between(
                        sentiment_trend.index,
                        sentiment_trend[col],
                        alpha=0.7,
                        label=col,
                        color=color
                    )
                    
            elif viz_type == "Stacked Area":
                # Stacked area chart
                sentiment_trend.plot.area(
                    ax=ax,
                    stacked=True,
                    color=["#4CAF50", "#FFC107", "#F44336"],  # Green, Amber, Red
                    alpha=0.7
                )
                
            elif viz_type == "Bar Timeline":
                # Get indices for bar positions
                indices = np.arange(len(sentiment_trend.index))
                bar_width = 0.25
                
                # Plot bars for each sentiment
                ax.bar(
                    indices - bar_width,
                    sentiment_trend["Positive"].values,
                    bar_width,
                    label="Positive",
                    color="#4CAF50",
                    edgecolor='white',
                    linewidth=0.5
                )
                
                ax.bar(
                    indices,
                    sentiment_trend["Neutral"].values,
                    bar_width,
                    label="Neutral",
                    color="#FFC107",
                    edgecolor='white',
                    linewidth=0.5
                )
                
                ax.bar(
                    indices + bar_width,
                    sentiment_trend["Negative"].values,
                    bar_width,
                    label="Negative",
                    color="#F44336",
                    edgecolor='white',
                    linewidth=0.5
                )
                
                # Set x-ticks to show dates
                ax.set_xticks(indices)
                if interval == "Hour":
                    x_labels = [dt.strftime("%H:%M") for dt in sentiment_trend.index]
                elif interval == "Day":
                    x_labels = [dt.strftime("%d %b") for dt in sentiment_trend.index]
                else:
                    x_labels = [dt.strftime("%b %d") for dt in sentiment_trend.index]
                    
                ax.set_xticklabels(x_labels, rotation=45, ha="right")
            
            # Set labels and title
            ax.set_title(f"Sentiment Trends Over Time (by {interval})", color='white', fontsize=14)
            ax.set_xlabel("Time", color='white')
            ax.set_ylabel("Tweet Count", color='white')
            
            # Format x-axis dates
            if viz_type != "Bar Timeline":
                fig.autofmt_xdate()
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add legend
            ax.legend(title="Sentiment")
            
            # Adjust layout
            plt.tight_layout()
            
            # Update canvas
            if self.trend_canvas:
                self.trend_canvas.get_tk_widget().destroy()
                
            self.trend_canvas = FigureCanvasTkAgg(fig, master=self.time_canvas_frame)
            self.trend_canvas.draw()
            self.trend_canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Error generating time series visualization: {str(e)}")
    
    def _generate_language_viz(self):
        """Generate language distribution visualization."""
        if not self.data_manager.sentiment_data:
            messagebox.showinfo("Information", "Please load data first.")
            return
            
        # Get visualization parameters
        viz_type = self.lang_type_var.get()
        top_langs = self.top_langs_var.get()
        
        try:
            # Get language distribution data
            language_distribution = self.data_manager.get_language_distribution()
            
            if language_distribution.empty:
                messagebox.showinfo("No Language Data", "No language data available to visualize.")
                return
                
            # Remove placeholder
            self.lang_placeholder.pack_forget()
            
            # Limit to top languages if specified
            if top_langs != "All":
                language_distribution = language_distribution.head(int(top_langs))
                
            # Create figure
            plt.close('all')  # Close any existing figures
            
            if viz_type == "Combined View":
                # Create figure with two subplots
                fig = plt.figure(figsize=(8, 5), dpi=100)
                gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])
                
                # Set dark theme
                plt.style.use('dark_background')
                
                # Left subplot: Language pie chart
                ax1 = fig.add_subplot(gs[0])
                
                # Get language counts
                langs = language_distribution['language'].tolist()
                counts = language_distribution['total'].tolist()
                lang_names = [self.language_names.get(l, l) for l in langs]
                
                # Create custom colormap for languages
                colors = plt.cm.tab10(np.linspace(0, 1, len(langs)))
                
                # Create pie chart
                wedges, texts = ax1.pie(
                    counts,
                    labels=None,  # No labels on pie chart
                    startangle=90,
                    colors=colors,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}
                )
                
                # Add legend
                ax1.legend(
                    wedges,
                    [f"{name} ({count})" for name, count in zip(lang_names, counts)],
                    loc="center left",
                    bbox_to_anchor=(0, 0.5),
                    fontsize=8
                )
                
                ax1.set_title("Language Distribution", color='white', fontsize=12)
                
                # Right subplot: Sentiment by language
                ax2 = fig.add_subplot(gs[1])
                
                # Create data for stacked bar chart
                data = []
                for lang in langs:
                    row = language_distribution[language_distribution['language'] == lang].iloc[0]
                    lang_name = self.language_names.get(lang, lang)
                    data.append({
                        "language": lang_name,
                        "Positive": row["Positive"],
                        "Neutral": row["Neutral"],
                        "Negative": row["Negative"]
                    })
                
                # Set up bar chart
                langs = [d["language"] for d in data]
                pos = np.arange(len(langs))
                bar_width = 0.6
                
                # Plot stacked bars
                ax2.bar(
                    pos, [d["Positive"] for d in data], bar_width,
                    color="#4CAF50", label="Positive", edgecolor='white', linewidth=0.5
                )
                
                ax2.bar(
                    pos, [d["Neutral"] for d in data], bar_width,
                    bottom=[d["Positive"] for d in data],
                    color="#FFC107", label="Neutral", edgecolor='white', linewidth=0.5
                )
                
                # Calculate bottom position for negative sentiment
                bottoms = [d["Positive"] + d["Neutral"] for d in data]
                
                ax2.bar(
                    pos, [d["Negative"] for d in data], bar_width,
                    bottom=bottoms,
                    color="#F44336", label="Negative", edgecolor='white', linewidth=0.5
                )
                
                # Set labels and title
                ax2.set_title("Sentiment by Language", color='white', fontsize=12)
                ax2.set_xlabel("Language", color='white')
                ax2.set_ylabel("Tweet Count", color='white')
                
                # Set x-tick labels
                ax2.set_xticks(pos)
                ax2.set_xticklabels(langs, rotation=45, ha="right")
                
                # Add legend and grid
                ax2.legend(title="Sentiment")
                ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
                
            elif viz_type == "Pie Chart":
                # Create single pie chart of languages
                fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
                plt.style.use('dark_background')
                
                # Get language counts
                langs = language_distribution['language'].tolist()
                counts = language_distribution['total'].tolist()
                lang_names = [self.language_names.get(l, l) for l in langs]
                
                # Create custom colormap for languages
                colors = plt.cm.rainbow(np.linspace(0, 1, len(langs)))
                
                # Create pie chart
                wedges, texts, autotexts = ax.pie(
                    counts,
                    labels=lang_names,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}
                )
                
                # Enhance text styling
                for text in texts:
                    text.set_color('white')
                    text.set_fontsize(8)
                    
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(8)
                    
                ax.set_title("Language Distribution", color='white', fontsize=14)
                
            elif viz_type in ["Bar Chart", "Stacked Bars"]:
                # Create bar chart or stacked bar chart
                fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
                plt.style.use('dark_background')
                
                # Get language data
                langs = language_distribution['language'].tolist()
                lang_names = [self.language_names.get(l, l) for l in langs]
                
                if viz_type == "Bar Chart":
                    # Simple bar chart of total counts
                    counts = language_distribution['total'].tolist()
                    
                    # Create custom colormap for languages
                    colors = plt.cm.rainbow(np.linspace(0, 1, len(langs)))
                    
                    bars = ax.bar(
                        lang_names,
                        counts,
                        color=colors,
                        edgecolor='white',
                        linewidth=0.5
                    )
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width()/2.,
                            height + 0.1,
                            f'{height:,}',
                            ha='center',
                            va='bottom',
                            color='white',
                            fontsize=8
                        )
                        
                    ax.set_title("Language Distribution", color='white', fontsize=14)
                    ax.set_xlabel("Language", color='white')
                    ax.set_ylabel("Tweet Count", color='white')
                    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
                    
                else:  # Stacked Bars
                    # Stacked bar chart showing sentiment distribution by language
                    bar_width = 0.6
                    pos = np.arange(len(lang_names))
                    
                    ax.bar(
                        pos,
                        language_distribution["Positive"].tolist(),
                        bar_width,
                        label="Positive",
                        color="#4CAF50",
                        edgecolor='white',
                        linewidth=0.5
                    )
                    
                    ax.bar(
                        pos,
                        language_distribution["Neutral"].tolist(),
                        bar_width,
                        bottom=language_distribution["Positive"].tolist(),
                        label="Neutral",
                        color="#FFC107",
                        edgecolor='white',
                        linewidth=0.5
                    )
                    
                    bottoms = (language_distribution["Positive"] + language_distribution["Neutral"]).tolist()
                    
                    ax.bar(
                        pos,
                        language_distribution["Negative"].tolist(),
                        bar_width,
                        bottom=bottoms,
                        label="Negative",
                        color="#F44336",
                        edgecolor='white',
                        linewidth=0.5
                    )
                    
                    # Set labels and title
                    ax.set_title("Sentiment by Language", color='white', fontsize=14)
                    ax.set_xlabel("Language", color='white')
                    ax.set_ylabel("Tweet Count", color='white')
                    
                    # Set x-tick labels
                    ax.set_xticks(pos)
                    ax.set_xticklabels(lang_names, rotation=45, ha="right")
                    
                    # Add legend and grid
                    ax.legend(title="Sentiment")
                    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
                
                # Rotate x-labels for better readability
                plt.xticks(rotation=45, ha="right")
            
            # Adjust layout
            plt.tight_layout()
            
            # Update canvas
            if self.lang_canvas:
                self.lang_canvas.get_tk_widget().destroy()
                
            self.lang_canvas = FigureCanvasTkAgg(fig, master=self.lang_canvas_frame)
            self.lang_canvas.draw()
            self.lang_canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Error generating language visualization: {str(e)}")
    
    def clear_data(self):
        """Clear all collected data."""
        # Confirm with the user
        if messagebox.askyesno("Clear Data", "Are you sure you want to clear all collected data?"):
            # Stop monitoring
            self.stop_monitoring()
            
            # Clear data
            self.data_manager.clear_data()
            
            # Update UI
            self.update_status("Data cleared", "#95A5A6")  # Gray for cleared
            self._reset_visualizations()
            self._update_stats()
            self._update_samples()
            
            # Disable export button
            self.export_button.configure(state="disabled")
            
    def export_results(self):
        """Export analysis results to files."""
        if len(self.data_manager.sentiment_data) == 0:
            messagebox.showinfo("No Data", "No data available to export.")
            return
            
        # Ask for export directory
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
            
        try:
            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            search_term = self.search_entry.get().strip()
            
            if not search_term:
                search_term = "twitter_sentiment"
                
            base_filename = f"{search_term.replace('#', '').replace(' ', '_')}_{timestamp}"
            
            # Export summary CSV
            self._export_summary_csv(export_dir, base_filename)
            
            # Export visualizations
            self._export_visualizations(export_dir, base_filename)
            
            # Export detailed results
            self._export_detailed_csv(export_dir, base_filename)
            
            # Confirm success
            messagebox.showinfo("Export Complete", f"Results exported to {export_dir}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
            
    def _export_summary_csv(self, export_dir, base_filename):
        """Export summary statistics to CSV."""
        # Create summary data
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        language_counts = {}
        
        for item in self.data_manager.sentiment_data:
            sentiment_counts[item["sentiment"]] += 1
            
            lang = item["language"]
            if lang not in language_counts:
                language_counts[lang] = 0
            language_counts[lang] += 1
            
        # Create summary CSV
        summary_path = os.path.join(export_dir, f"{base_filename}_summary.csv")
        
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            import csv
            writer = csv.writer(f)
            
            # Write sentiment summary
            writer.writerow(["Sentiment Summary"])
            writer.writerow(["Sentiment", "Count", "Percentage"])
            
            total = sum(sentiment_counts.values())
            for sentiment, count in sentiment_counts.items():
                percentage = (count / total * 100) if total > 0 else 0
                writer.writerow([sentiment, count, f"{percentage:.2f}%"])
                
            writer.writerow([])  # Empty row
            
            # Write language summary
            writer.writerow(["Language Summary"])
            writer.writerow(["Language", "Count", "Percentage"])
            
            for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total * 100) if total > 0 else 0
                writer.writerow([lang, count, f"{percentage:.2f}%"])
                
    def _export_detailed_csv(self, export_dir, base_filename):
        """Export detailed results to CSV."""
        detailed_path = os.path.join(export_dir, f"{base_filename}_detailed.csv")
        
        with open(detailed_path, "w", encoding="utf-8", newline="") as f:
            import csv
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "Timestamp", "Text", "Sentiment", "Language", 
                "Positive_Confidence", "Neutral_Confidence", "Negative_Confidence"
            ])
            
            # Write data
            for item in self.data_manager.sentiment_data:
                writer.writerow([
                    item["timestamp"].isoformat(),
                    item["text"],
                    item["sentiment"],
                    item["language"],
                    f"{item['confidence']['Positive']:.2f}",
                    f"{item['confidence']['Neutral']:.2f}",
                    f"{item['confidence']['Negative']:.2f}"
                ])
                
    def _export_visualizations(self, export_dir, base_filename):
        """Export visualizations as image files."""
        # Export sentiment distribution
        if self.dist_canvas:
            dist_path = os.path.join(export_dir, f"{base_filename}_distribution.png")
            self.dist_canvas.figure.savefig(dist_path, dpi=300, bbox_inches="tight")
            
        # Export time trends
        if self.trend_canvas:
            trend_path = os.path.join(export_dir, f"{base_filename}_trends.png")
            self.trend_canvas.figure.savefig(trend_path, dpi=300, bbox_inches="tight")
            
        # Export language analysis
        if self.lang_canvas:
            lang_path = os.path.join(export_dir, f"{base_filename}_languages.png")
            self.lang_canvas.figure.savefig(lang_path, dpi=300, bbox_inches="tight")
            
        # Export correlation visualization
        if self.corr_canvas:
            corr_path = os.path.join(export_dir, f"{base_filename}_correlation.png")
            self.corr_canvas.figure.savefig(corr_path, dpi=300, bbox_inches="tight")
            
    def update_status(self, message, color="#FFA500"):
        """Update the status indicator with the given message and color."""
        self.status_indicator.configure(text=message, text_color=color)
        
    def _update_stats(self):
        """Update the statistics displays."""
        if not self.data_manager.sentiment_data:
            self.overall_stats.configure(text="No data available")
            self.language_stats.configure(text="No data available")
            return
            
        # Calculate sentiment statistics
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        
        for item in self.data_manager.sentiment_data:
            # Make sure we only count valid sentiments
            if item["sentiment"] in sentiment_counts:
                sentiment_counts[item["sentiment"]] += 1
            else:
                # Default to neutral for any errors
                print(f"Warning: Unknown sentiment value '{item['sentiment']}' encountered")
                sentiment_counts["Neutral"] += 1
                
        total = sum(sentiment_counts.values())
        
        # Update overall stats
        overall_text = f"Total tweets analyzed: {total}\n\n"
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            overall_text += f"{sentiment}: {count} ({percentage:.1f}%)\n"
            
        self.overall_stats.configure(text=overall_text)
        
        # Calculate language statistics
        language_distribution = self.data_manager.get_language_distribution()
        
        if language_distribution.empty:
            self.language_stats.configure(text="No language data available")
            return
            
        # Format language statistics
        lang_text = ""
        for _, row in language_distribution.iterrows():
            lang = row["language"]
            total_lang = row["total"]
            percentage = (total_lang / total * 100) if total > 0 else 0
            
            # Get language name if available
            lang_name = self.language_names.get(lang, lang)
            
            lang_text += f"{lang_name} ({lang}): {total_lang} ({percentage:.1f}%)\n"
            
            # If there are many languages, only show top 5
            if len(lang_text.split("\n")) > 5 and len(language_distribution) > 5:
                remaining = len(language_distribution) - 5
                lang_text += f"... and {remaining} more languages"
                break
                
        self.language_stats.configure(text=lang_text)
        
    def _update_visualizations(self):
        """Update all visualizations if data is available."""
        if self.data_manager.sentiment_data:
            if self.viz_tabview.get() == "Sentiment Distribution":
                self._generate_distribution_viz()
            elif self.viz_tabview.get() == "Time Trends":
                self._generate_time_series_viz()
            elif self.viz_tabview.get() == "Language Analysis":
                self._generate_language_viz()
            # Don't automatically update correlation tab as it requires specific variable selections
            
    def _update_samples(self):
        """Update the sample tweets display."""
        for sentiment in ["Positive", "Neutral", "Negative"]:
            # Get samples for this sentiment
            samples = self.data_manager.get_top_samples(sentiment, count=3)
            
            # Update text area
            text_area = self.sample_frames[sentiment]
            text_area.configure(state="normal")
            text_area.delete("1.0", "end")
            
            if samples:
                for i, sample in enumerate(samples):
                    # Format text
                    text = sample["text"]
                    lang = sample["language"]
                    conf = sample["confidence"]
                    
                    # Add sentiment emoji
                    emoji = "✅" if sentiment == "Positive" else "⚠️" if sentiment == "Neutral" else "❌"
                    
                    # Format the sample
                    text_area.insert("end", f"{emoji} {text}\n")
                    text_area.insert("end", f"   Language: {self.language_names.get(lang, lang)} ({lang}) • Confidence: {conf:.1f}%\n\n")
            else:
                text_area.insert("end", f"No {sentiment.lower()} tweets available yet.")
                
            text_area.configure(state="disabled")
    
    def _reset_visualizations(self):
        """Reset all visualizations to their placeholder state."""
        # Close any existing figures
        plt.close('all')
        
        # Clear canvases
        if self.dist_canvas:
            self.dist_canvas.get_tk_widget().destroy()
            self.dist_canvas = None
            self.dist_placeholder.pack(expand=True)
            
        if self.trend_canvas:
            self.trend_canvas.get_tk_widget().destroy()
            self.trend_canvas = None
            self.trend_placeholder.pack(expand=True)
            
        if self.lang_canvas:
            self.lang_canvas.get_tk_widget().destroy()
            self.lang_canvas = None
            self.lang_placeholder.pack(expand=True)
            
        if self.corr_canvas:
            self.corr_canvas.get_tk_widget().destroy()
            self.corr_canvas = None
            self.corr_placeholder.pack(expand=True)
            
        # Reset sample displays
        for sentiment in ["Positive", "Neutral", "Negative"]:
            text_area = self.sample_frames[sentiment]
            text_area.configure(state="normal")
            text_area.delete("1.0", "end")
            text_area.insert("1.0", f"No {sentiment.lower()} tweets available yet.")
            text_area.configure(state="disabled")
            
    # ---- Settings handlers ----
    
    def _update_cooldown(self, value):
        """Update the cooldown setting."""
        try:
            self.config["cooldown"] = int(value)
        except ValueError:
            self.config["cooldown"] = 60
            
    def _update_confidence_threshold(self, value):
        """Update the confidence threshold setting."""
        self.config["confidence_threshold"] = float(value)
        self.confidence_label.configure(text=f"{value:.0f}%")
        
    def _update_max_tweets(self, value):
        """Update the maximum tweets setting."""
        try:
            self.config["max_tweets"] = int(value)
        except ValueError:
            self.config["max_tweets"] = 1000
                
    def _update_language_filter(self, selection):
        """Update language filter based on dropdown selection."""
        if selection == "All Languages":
            self.config["language_filter"] = None
        else:
            # Extract language code from the selection (format: "Language (code)")
            lang_code = selection.split("(")[1].strip(")")
            self.config["language_filter"] = [lang_code]
            
    def _open_language_selection(self):
        """Open a dialog to select multiple languages."""
        # Create a toplevel window for language selection
        selection_window = ctk.CTkToplevel(self)
        selection_window.title("Select Languages")
        selection_window.geometry("300x400")
        selection_window.grab_set()  # Make window modal
        
        # Frame for selections
        select_frame = ctk.CTkScrollableFrame(selection_window, width=280, height=320)
        select_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Create dictionary of language checkboxes and variables
        lang_vars = {}
        
        # Language mappings for display names and codes
        language_list = [
            ("English", "en"),
            ("Arabic", "ar"),
            ("French", "fr"),
            ("Spanish", "es"),
            ("German", "de"),
            ("Italian", "it"),
            ("Japanese", "ja"),
            ("Chinese", "zh"),
            ("Russian", "ru"),
            ("Dutch", "nl"),
            ("Portuguese", "pt"),
            ("Hindi", "hi"),
            ("Urdu", "ur"),
            ("Persian", "fa"),
            ("Turkish", "tr"),
            ("Korean", "ko"),
            ("Swedish", "sv"),
            ("Norwegian", "no"),
            ("Danish", "da"),
            ("Finnish", "fi"),
            ("Greek", "el"),
            ("Polish", "pl"),
            ("Czech", "cs"),
            ("Hungarian", "hu"),
            ("Romanian", "ro"),
            ("Bulgarian", "bg"),
            ("Ukrainian", "uk"),
            ("Thai", "th"),
            ("Vietnamese", "vi"),
            ("Indonesian", "id"),
            ("Malay", "ms"),
            ("Filipino", "tl")
        ]
        
        # Get current language filter
        current_filter = self.config["language_filter"] or []
        
        # Create "Select All" option
        select_all_var = tk.BooleanVar(value=False)
        select_all_cb = ctk.CTkCheckBox(
            select_frame,
            text="Select All Languages",
            variable=select_all_var,
            command=lambda: self._toggle_all_languages(lang_vars, select_all_var),
            onvalue=True,
            offvalue=False
        )
        select_all_cb.pack(anchor="w", pady=(0, 10))
        
        # Add individual language checkboxes
        for lang_name, lang_code in language_list:
            var = tk.BooleanVar(value=lang_code in current_filter)
            lang_vars[lang_code] = var
            
            checkbox = ctk.CTkCheckBox(
                select_frame,
                text=f"{lang_name} ({lang_code})",
                variable=var,
                onvalue=True,
                offvalue=False
            )
            checkbox.pack(anchor="w", pady=2)
        
        # Buttons frame
        button_frame = ctk.CTkFrame(selection_window)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        # Apply button
        apply_button = ctk.CTkButton(
            button_frame,
            text="Apply",
            command=lambda: self._apply_language_selection(lang_vars, selection_window)
        )
        apply_button.pack(side="left", padx=10, pady=5, fill="x", expand=True)
        
        # Cancel button
        cancel_button = ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=selection_window.destroy,
            fg_color="#E74C3C"
        )
        cancel_button.pack(side="left", padx=10, pady=5, fill="x", expand=True)

    def _toggle_all_languages(self, lang_vars, select_all_var):
        """Toggle all language checkboxes."""
        value = select_all_var.get()
        for var in lang_vars.values():
            var.set(value)

    def _apply_language_selection(self, lang_vars, window):
        """Apply the selected languages to the filter."""
        # Get selected languages
        selected_langs = [code for code, var in lang_vars.items() if var.get()]
        
        # Update the filter
        if not selected_langs:
            self.config["language_filter"] = None
            self.lang_filter_var.set("All Languages")
        else:
            self.config["language_filter"] = selected_langs
            if len(selected_langs) == 1:
                # Show the single language in the dropdown
                lang_name = next((name for name, code in [item.split("(") for item in self.language_options if "(" in item] 
                                if code.strip(")") == selected_langs[0]), "")
                if lang_name:
                    self.lang_filter_var.set(f"{lang_name}({selected_langs[0]})")
            else:
                # Show "Multiple" in the dropdown
                self.lang_filter_var.set(f"Multiple ({len(selected_langs)})")
        
        # Close the window
        window.destroy()

    def _get_language_filter(self):
        """Get the current language filter."""
        return self.config["language_filter"]