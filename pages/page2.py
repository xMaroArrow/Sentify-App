# Enhanced page2.py
import customtkinter as ctk
from tkinter import filedialog
from datetime import datetime
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patheffects as path_effects
import threading
import pandas as pd
from typing import Dict, List, Optional, Any

# Import the shared sentiment analyzer
from addons.sentiment_analyzer import SentimentAnalyzer
from utils import theme

# Choose ONE collector implementation
try:
    from addons.bypass_collector import TweetCollector  # Nitter fallback
except ImportError:
    try:
        from addons.collector import TweetCollector  # Tweepy API version
    except ImportError:
        TweetCollector = None  # Offline mode only

# Optional Reddit collector
try:
    from addons.reddit_collector import RedditCollector
except Exception:
    RedditCollector = None  # Reddit streaming unavailable

class Page2(ctk.CTkFrame):
    """
    Opinion‑polling page with live or offline (CSV) mode.
    
    This page enables:
    1. Real-time monitoring of tweets with sentiment analysis
    2. Loading and analyzing CSV files with sentiment data
    3. Visualization of sentiment trends over time
    4. Sample tweet display for each sentiment category
    """

    def __init__(self, parent):
        super().__init__(parent)

        # Initialize the shared sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()

        # Initialize canvas variables BEFORE they are used
        self.canvas = None
        self.trend_canvas = None
        self.example_textbox = None

        # Use grid layout for better centering and control
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create a master scrollable container that adapts to parent size
        self.master_scroll = ctk.CTkScrollableFrame(
            self,
            corner_radius=0,  # Remove rounded corners for better space usage
        )
        self.master_scroll.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # --- State holders -------------------------------------------
        # Use a broad type here because TweetCollector may be provided
        # by different optional backends at runtime.
        self.collector: Optional[Any] = None
        self.countdown_job: Optional[int] = None
        self.analysis_thread: Optional[threading.Thread] = None
        self.running = False

        # Data storage
        self.sentiment_counts = {"Positive": [], "Neutral": [], "Negative": []}
        self.time_stamps: List[str] = []
        self.sample_tweets: Dict[str, List[str]] = {"Positive": [], "Neutral": [], "Negative": []}
        self.current_csv_path: Optional[str] = None

        # --- UI ------------------------------------------------------
       # Title and description - now in the scrollable container
        title_frame = ctk.CTkFrame(self.master_scroll)
        title_frame.pack(fill="x", pady=(10, 0))
        
        ctk.CTkLabel(
            title_frame, 
            text="Real‑Time Opinion Polling", 
            font=("Arial", 20)
        ).pack(side="left", padx=10)
        
        # Info button with tooltip popup
        self.info_button = ctk.CTkButton(
            title_frame, 
            text="ⓘ", 
            width=30, 
            command=self.show_info,
            fg_color="#444444",
            hover_color="#666666"
        )
        self.info_button.pack(side="right", padx=10)
        
        # Description
        desc = ctk.CTkLabel(
            self.master_scroll,  # Changed parent to master_scroll
            text="Monitor Twitter sentiment for hashtags in real-time or analyze existing data",
            font=("Arial", 12),
            text_color=theme.subtle_text_color(),
            wraplength=850  # Add wraplength to handle narrow windows
        )
        desc.pack(pady=(0, 10))

        # Input area
        input_frame = ctk.CTkFrame(self.master_scroll)  # Changed parent
        input_frame.pack(fill="x", padx=20, pady=10)
        
        # Rest of your UI components - update their parent to self.master_scroll
        # Source selector
        self.source_var = ctk.StringVar(value="Twitter")
        ctk.CTkLabel(input_frame, text="Source:").pack(side="left", padx=(0, 4))
        self.source_menu = ctk.CTkOptionMenu(
            input_frame,
            values=["Twitter", "Reddit"],
            variable=self.source_var,
            width=110,
        )
        self.source_menu.pack(side="left", padx=(0, 10))

        self.entry = ctk.CTkEntry(
            input_frame, 
            placeholder_text="#Hashtag / search / Reddit URL", 
            width=300
        )
        self.entry.pack(side="left", padx=(0, 10))

        # Batch size and interval controls
        ctk.CTkLabel(input_frame, text="Batch:").pack(side="left", padx=(0, 4))
        self.batch_entry = ctk.CTkEntry(input_frame, width=50)
        self.batch_entry.insert(0, "5")
        self.batch_entry.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(input_frame, text="Every (min):").pack(side="left", padx=(0, 4))
        self.interval_entry = ctk.CTkEntry(input_frame, width=60)
        self.interval_entry.insert(0, "1")
        self.interval_entry.pack(side="left", padx=(0, 10))
        
        self.status = ctk.CTkLabel(
            input_frame, 
            text="Ready", 
            text_color=theme.subtle_text_color(),
            width=150
        )
        self.status.pack(side="right")

        # Action buttons
        btn_row = ctk.CTkFrame(self.master_scroll)  # Changed parent
        btn_row.pack(pady=6)
        
        self.start_button = ctk.CTkButton(
            btn_row, 
            text="Start Monitoring", 
            command=self.start_monitoring
        )
        self.start_button.pack(side="left", padx=4)
        
        self.stop_button = ctk.CTkButton(
            btn_row, 
            text="Stop", 
            fg_color="#8b0000", 
            command=self.stop_monitoring
        )
        self.stop_button.pack(side="left", padx=4)
        
        self.load_csv_button = ctk.CTkButton(
            btn_row, 
            text="Load CSV", 
            fg_color="#006400", 
            command=self.load_csv
        )
        self.load_csv_button.pack(side="left", padx=4)
        
        self.export_button = ctk.CTkButton(
            btn_row, 
            text="Export Results", 
            command=self.export_results,
            state="disabled"
        )
        self.export_button.pack(side="left", padx=4)

        # Progress indicator
        self.progress = ctk.CTkProgressBar(self.master_scroll, width=400)  # Changed parent
        self.progress.pack(pady=(10, 0))
        self.progress.set(0)

        # Visualization container - fill available space
        self.viz_frame = ctk.CTkFrame(self.master_scroll)
        self.viz_frame.pack(expand=True, fill="both", padx=20, pady=(10, 0))

        # Chart container - responsive grid (side-by-side or stacked)
        self.fig_frame = ctk.CTkFrame(self.viz_frame)
        self.fig_frame.pack(pady=5, fill="both", expand=True)
        try:
            self.fig_frame.grid_columnconfigure(0, weight=1, uniform="figs")
            self.fig_frame.grid_columnconfigure(1, weight=1, uniform="figs")
            self.fig_frame.grid_rowconfigure(0, weight=1)
            self.fig_frame.grid_rowconfigure(1, weight=1)
        except Exception:
            pass
        try:
            self.viz_frame.configure(border_width=1, border_color=theme.border_color())
            self.fig_frame.configure(border_width=1, border_color=theme.border_color())
        except Exception:
            pass

        # Dedicated containers for each chart for better layout control
        self.pie_container = ctk.CTkFrame(self.fig_frame, fg_color=theme.panel_bg())
        self.trend_container = ctk.CTkFrame(self.fig_frame, fg_color=theme.panel_bg())
        try:
            self.pie_container.configure(border_width=1, border_color=theme.border_color())
            self.trend_container.configure(border_width=1, border_color=theme.border_color())
        except Exception:
            pass

        # Track layout mode and arrange on resize
        self._charts_stacked = None  # unknown initially
        self.fig_frame.bind("<Configure>", self._arrange_chart_grid)
        
        # Sample tweets container
        self.samples_frame = ctk.CTkFrame(self.viz_frame)
        self.samples_frame.pack(side="bottom", fill="both", expand=True, pady=5)
        
        self.example_textbox = None
        
        # Stats container
        self.stats_frame = ctk.CTkFrame(self.viz_frame)
        self.stats_frame.pack(side="bottom", fill="x", pady=5)
        
        stats_label = ctk.CTkLabel(
            self.stats_frame, 
            text="Statistics", 
            font=("Arial", 14)
        )
        stats_label.pack(pady=(5, 0))
        
        self.stats_text = ctk.CTkLabel(
            self.stats_frame,
            text="No data available",
            font=("Arial", 12),
            text_color=theme.subtle_text_color(),
            justify="left",
            anchor="w",
            wraplength=1000,
        )
        self.stats_text.pack(pady=(0, 5))
        
        # Initialize empty visualizations
        try:
            self.update_pie_chart()
            self.update_trend_chart()
        except Exception as e:
            print(f"Error initializing visualizations: {e}")
        

    # ------------------------------------------------------------------
    # Info popup
    # ------------------------------------------------------------------
    def show_info(self):
        """Display information about this page's functionality."""
        popup = ctk.CTkToplevel(self)
        popup.title("Real-Time Opinion Polling - Help")
        popup.geometry("500x400")
        popup.resizable(False, False)
        
        frame = ctk.CTkScrollableFrame(popup, width=480, height=380)
        frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        ctk.CTkLabel(
            frame, 
            text="Real-Time Opinion Polling", 
            font=("Arial", 18, "bold")
        ).pack(pady=(0, 10))
        
        info_text = """
This page allows you to analyze sentiment from Twitter in two ways:

1. Real-time monitoring:
   • Enter a hashtag or search term in the input field
   • Click "Start Monitoring" to begin collecting tweets
   • The system will analyze sentiment every minute
   • Results update automatically in the charts

2. Offline analysis:
   • Click "Load CSV" to import previously collected data
   • Select a CSV file with columns: Date, Username, Sentiment, Tweet
   • Results will be aggregated and displayed

The visualization includes:
   • Pie chart showing overall sentiment distribution
   • Trend chart showing sentiment changes over time
   • Sample tweets for each sentiment category
   • Basic statistics about the data

Requirements:
   • Internet connection for real-time monitoring
   • Twitter API access or fallback to public scraping
   • For offline mode, properly formatted CSV files
        """
        
        ctk.CTkLabel(
            frame, 
            text=info_text, 
            font=("Arial", 12),
            justify="left"
        ).pack(pady=10, padx=10)
        
        ctk.CTkButton(
            frame, 
            text="Close", 
            command=popup.destroy
        ).pack(pady=10)

    # ------------------------------------------------------------------
    # Live monitoring (streaming)
    # ------------------------------------------------------------------
    def _set_controls_state(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for w in [self.entry, self.source_menu, self.batch_entry, self.interval_entry, self.load_csv_button, self.export_button, self.start_button]:
            try:
                w.configure(state=state)
            except Exception:
                pass

    def start_monitoring(self):
        """Start real-time monitoring for Twitter or Reddit with manual cadence."""
        source = (getattr(self, 'source_var', None).get() if hasattr(self, 'source_var') else 'Twitter')
        query = self.entry.get().strip()
        if not query:
            self.status.configure(text="Enter a hashtag or query", text_color="red")
            return

        try:
            batch_size = int((self.batch_entry.get() or "").strip() or 0)
            interval_min = float((self.interval_entry.get() or "").strip() or 0)
        except Exception:
            self.status.configure(text="Batch/interval must be numeric", text_color="red")
            return
        if batch_size <= 0 or interval_min <= 0:
            self.status.configure(text="Batch/interval must be > 0", text_color="red")
            return
        interval_seconds = max(1, int(interval_min * 60))

        # Stop any existing monitoring
        self.stop_monitoring()

        # Reset in-memory data
        self.sentiment_counts = {"Positive": [], "Neutral": [], "Negative": []}
        self.time_stamps = []
        self.sample_tweets = {"Positive": [], "Neutral": [], "Negative": []}
        self.progress.set(0.1)
        self.status.configure(text="Initializing...", text_color="orange")
        self._set_controls_state(False)

        def _init_and_start():
            try:
                if str(source).lower() == "reddit":
                    if RedditCollector is None:
                        raise RuntimeError("Reddit not configured (install praw / set credentials)")
                    self.collector = RedditCollector(cooldown=interval_seconds)
                    self.current_csv_path = None
                    self.collector.start_streaming(self._on_stream_batch, query, batch_size, interval_seconds)
                else:
                    if TweetCollector is None:
                        raise RuntimeError("Twitter collector unavailable")
                    self.collector = TweetCollector(query, cooldown=interval_seconds)
                    self.current_csv_path = getattr(self.collector, 'csv_path', None)
                    # Use the pluggable streaming API we added
                    try:
                        self.collector.start_streaming(self._on_stream_batch, query, batch_size, interval_seconds)
                    except Exception:
                        # Fallback: start legacy collector (won't provide streaming batches)
                        self.collector.start()

                self.running = True
                self.after(0, lambda: self.status.configure(text="Collecting...", text_color="orange"))
                self.after(0, lambda: self.progress.set(0.2))
                self.after(0, self._schedule_countdown)
            except Exception as e:
                self.after(0, lambda: self.status.configure(text=f"Error: {e}", text_color="red"))
                self.after(0, lambda: self.progress.set(0))
                self.after(0, lambda: self._set_controls_state(True))
                self.collector = None

        threading.Thread(target=_init_and_start, daemon=True).start()

    def stop_monitoring(self):
        """Stop all monitoring and analysis threads."""
        self.running = False
        
        # Stop collector
        if self.collector:
            self.collector.stop()
            self.collector = None
        
        # Cancel countdown timer
        if self.countdown_job:
            self.after_cancel(self.countdown_job)
            self.countdown_job = None
        
        # Update UI
        self.status.configure(text="Stopped", text_color=theme.subtle_text_color())
        self.progress.set(0)
        try:
            self._set_controls_state(True)
        except Exception:
            pass

    def _schedule_countdown(self):
        """Update countdown display for next data collection."""
        if not self.running or not self.collector:
            return
            
        # Calculate seconds until next request
        try:
            sec = int(self.collector.seconds_until_next_request())
        except Exception:
            sec = 0
        
        # Update status with countdown
        if sec > 0:
            self.status.configure(text=f"Collecting… (⏳ {sec}s)", text_color="orange")
            self.progress.set(max(0, min(1.0, 1 - (sec / 60))))  # Progress based on 60s cycle
        else:
            self.status.configure(text="Collecting…", text_color="orange")
            self.progress.set(1.0)
            
        # Schedule next update
        self.countdown_job = self.after(1000, self._schedule_countdown)

    # ------------------------------------------------------------------
    # Streaming callback (live mode)
    # ------------------------------------------------------------------
    def _on_stream_batch(self, texts, timestamps):
        if not self.running or not texts:
            return

        def _analyze_batch():
            pos = neu = neg = 0
            samples_added = {"Positive": 0, "Neutral": 0, "Negative": 0}
            for text in texts:
                result = self.sentiment_analyzer.analyze_text(text)
                label = max(result.items(), key=lambda x: x[1])[0] if result else "Neutral"
                if label == "Positive":
                    pos += 1
                elif label == "Negative":
                    neg += 1
                else:
                    neu += 1
                if samples_added.get(label, 0) < 3:
                    truncated = (text[:120] + "…") if len(text) > 120 else text
                    self.sample_tweets[label].append(truncated)
                    samples_added[label] = samples_added.get(label, 0) + 1

            ts_label = datetime.now().strftime("%H:%M")
            self.time_stamps.append(ts_label)
            self.sentiment_counts["Positive"].append(pos)
            self.sentiment_counts["Neutral"].append(neu)
            self.sentiment_counts["Negative"].append(neg)

            # UI updates on main thread
            self.after(0, self.update_pie_chart)
            self.after(0, self.update_trend_chart)
            self.after(0, self._update_stats_enhanced)
            self.after(0, self._show_samples)
            self.after(0, lambda: self.status.configure(text="Analyzed batch", text_color="green"))

        # Run analysis off the UI thread
        self.analysis_thread = threading.Thread(target=_analyze_batch, daemon=True)
        self.analysis_thread.start()

    # ------------------------------------------------------------------
    # Analyzer callback (live mode)
    # ------------------------------------------------------------------
    def on_new_analysis(self, new_rows):
        """Process newly analyzed tweets from the analyzer."""
        if not self.running:
            return
            
        # Count sentiments
        pos = sum(r["sentiment"] == "Positive" for r in new_rows)
        neu = sum(r["sentiment"] == "Neutral" for r in new_rows)
        neg = sum(r["sentiment"] == "Negative" for r in new_rows)
        
        # Get current timestamp
        ts = datetime.now().strftime("%H:%M:%S")
        
        # Store data
        self.time_stamps.append(ts)
        self.sentiment_counts["Positive"].append(pos)
        self.sentiment_counts["Neutral"].append(neu)
        self.sentiment_counts["Negative"].append(neg)
        
        # Collect sample tweets for each sentiment
        for row in new_rows:
            sentiment = row["sentiment"]
            text = row["text"]
            if len(self.sample_tweets[sentiment]) < 3:
                self.sample_tweets[sentiment].append(text[:120] + "…" if len(text) > 120 else text)
        
        # Update visualizations
        self.update_pie_chart()
        self.update_trend_chart()
        self._update_stats_enhanced()
        self._show_samples()
        
        # Update status
        self.status.configure(text="Analyzing…", text_color="green")

    # ------------------------------------------------------------------
    # Offline CSV loader
    # ------------------------------------------------------------------
    def load_csv(self):
        """Load and analyze sentiment data from a CSV file."""
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return

        try:
            # Update UI
            self.status.configure(text="Loading CSV...", text_color="blue")
            self.progress.set(0.2)
            self.current_csv_path = path
            
            # Stop live threads if running
            self.stop_monitoring()
            
            # Process in background thread
            threading.Thread(target=self._process_csv, args=(path,), daemon=True).start()
            
        except Exception as e:
            self.status.configure(text=f"Error: {str(e)}", text_color="red")
            self.progress.set(0)

    def _process_csv(self, path):
        """Process CSV file in background thread."""
        try:
            # Reset state
            self.sentiment_counts = {"Positive": [], "Neutral": [], "Negative": []}
            self.time_stamps = []
            self.sample_tweets = {"Positive": [], "Neutral": [], "Negative": []}
            
            # Update progress
            self.after(0, lambda: self.progress.set(0.4))
            
            # Read CSV with pandas for better performance and error handling
            try:
                df = pd.read_csv(path, encoding="utf-8")
                self.after(0, lambda: self.progress.set(0.6))
            except UnicodeDecodeError:
                # Try alternative encoding if UTF-8 fails
                df = pd.read_csv(path, encoding="latin1")
                self.after(0, lambda: self.progress.set(0.6))
                
            # Validate required columns
            required_cols = ["Date", "Tweet"]
            if "Sentiment" not in df.columns:
                # Perform sentiment analysis if sentiment column is missing
                self.after(0, lambda: self.status.configure(
                    text="Analyzing sentiment...", 
                    text_color="blue"
                ))
                
                # Add sentiment column
                df["Sentiment"] = df["Tweet"].apply(
                    lambda text: max(
                        self.sentiment_analyzer.analyze_text(text).items(), 
                        key=lambda x: x[1]
                    )[0]
                )
                
            # Normalize column names
            df.columns = [c.capitalize() for c in df.columns]
            
            # Ensure required columns exist
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
                
            # Parse timestamps
            def parse_date(date_str):
                """Parse dates with multiple possible formats."""
                formats = [
                    "%Y-%m-%d %H:%M:%S",  # Standard format
                    "%Y-%m-%d %H:%M",     # Without seconds
                    "%d/%m/%Y %H:%M:%S",  # European format
                    "%d/%m/%Y %H:%M",     # European without seconds
                    "%m/%d/%Y %H:%M:%S",  # US format
                    "%m/%d/%Y %H:%M",     # US without seconds
                ]
                
                for fmt in formats:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
                return None
                
            # Group by minute
            df["DateTime"] = df["Date"].apply(parse_date)
            df = df.dropna(subset=["DateTime"])  # Remove rows with invalid dates
            df["Minute"] = df["DateTime"].apply(lambda dt: dt.strftime("%H:%M"))
            
            # Aggregate by minute and sentiment
            pivoted = pd.pivot_table(
                df, 
                index="Minute",
                columns="Sentiment",
                aggfunc="size",
                fill_value=0
            )
            
            # Fill missing sentiment columns
            for sentiment in ["Positive", "Neutral", "Negative"]:
                if sentiment not in pivoted.columns:
                    pivoted[sentiment] = 0
            
            # Convert to time series
            for minute in sorted(pivoted.index):
                self.time_stamps.append(minute)
                self.sentiment_counts["Positive"].append(pivoted.loc[minute].get("Positive", 0))
                self.sentiment_counts["Neutral"].append(pivoted.loc[minute].get("Neutral", 0))
                self.sentiment_counts["Negative"].append(pivoted.loc[minute].get("Negative", 0))
            
            # Collect sample tweets
            for sentiment in ["Positive", "Neutral", "Negative"]:
                sentiment_tweets = df[df["Sentiment"] == sentiment]["Tweet"].tolist()
                samples = sentiment_tweets[:3] if sentiment_tweets else []
                self.sample_tweets[sentiment] = [
                    t[:120] + "…" if len(t) > 120 else t for t in samples
                ]
            
            # Update UI on main thread
            self.after(0, lambda: self.progress.set(0.8))
            self.after(0, self.update_pie_chart)
            self.after(0, self.update_trend_chart)
            self.after(0, self._update_stats_enhanced)
            self.after(0, self._show_samples)
            self.after(0, lambda: self.export_button.configure(state="normal"))
            self.after(0, lambda: self.progress.set(1.0))
            self.after(0, lambda: self.status.configure(
                text=f"Loaded {len(df)} entries from {os.path.basename(path)}", 
                text_color="green"
            ))
            
            # Reset progress after delay
            self.after(2000, lambda: self.progress.set(0))
            
        except Exception as e:
            self.after(0, lambda: self.status.configure(text=f"Error: {str(e)}", text_color="red"))
            self.after(0, lambda: self.progress.set(0))

    # ------------------------------------------------------------------
    # Export functionality
    # ------------------------------------------------------------------
    def export_results(self):
        """Export analysis results to CSV and charts to images."""
        if not self.time_stamps:
            self.status.configure(text="No data to export", text_color="red")
            return
            
        try:
            # Ask for export directory
            export_dir = filedialog.askdirectory(title="Select Export Directory")
            if not export_dir:
                return
                
            # Create export directory if it doesn't exist
            os.makedirs(export_dir, exist_ok=True)
            
            # Generate base filename
            base_name = os.path.splitext(os.path.basename(self.current_csv_path))[0] if self.current_csv_path else "sentiment_analysis"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{base_name}_{timestamp}"
            
            # Export time series data
            time_series_path = os.path.join(export_dir, f"{base_filename}_timeseries.csv")
            with open(time_series_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Positive", "Neutral", "Negative"])
                for i, ts in enumerate(self.time_stamps):
                    writer.writerow([
                        ts,
                        self.sentiment_counts["Positive"][i],
                        self.sentiment_counts["Neutral"][i],
                        self.sentiment_counts["Negative"][i]
                    ])
            
            # Export pie chart
            if self.canvas:
                pie_path = os.path.join(export_dir, f"{base_filename}_pie.png")
                self.canvas.figure.savefig(pie_path, bbox_inches="tight", dpi=300)
                
            # Export trend chart
            if self.trend_canvas:
                trend_path = os.path.join(export_dir, f"{base_filename}_trend.png")
                self.trend_canvas.figure.savefig(trend_path, bbox_inches="tight", dpi=300)
                
            # Export summary report
            summary_path = os.path.join(export_dir, f"{base_filename}_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("SENTIMENT ANALYSIS SUMMARY\n")
                f.write("==========================\n\n")
                
                # Overall statistics
                totals = {k: sum(v) for k, v in self.sentiment_counts.items()}
                total_tweets = sum(totals.values())
                f.write(f"Total tweets analyzed: {total_tweets}\n")
                f.write("Sentiment distribution:\n")
                for sentiment, count in totals.items():
                    percentage = (count / total_tweets * 100) if total_tweets > 0 else 0
                    f.write(f"  {sentiment}: {count} ({percentage:.1f}%)\n")
                
                # Time range
                if self.time_stamps:
                    f.write(f"\nTime range: {self.time_stamps[0]} to {self.time_stamps[-1]}\n")
                    
                # Sample tweets
                f.write("\nSAMPLE TWEETS\n")
                f.write("=============\n\n")
                for sentiment, tweets in self.sample_tweets.items():
                    f.write(f"{sentiment} examples:\n")
                    for tweet in tweets:
                        f.write(f"- {tweet}\n")
                    f.write("\n")
            
            # Confirm success
            self.status.configure(
                text=f"Exported to {export_dir}", 
                text_color="green"
            )
            
        except Exception as e:
            self.status.configure(text=f"Export error: {str(e)}", text_color="red")

    # ------------------------------------------------------------------
    # Visualization helpers (reuse for both modes)
    # ------------------------------------------------------------------
    def update_pie_chart(self):
        """Create or update the pie chart with smooth animated transitions."""
        # Calculate totals
        totals = {k: sum(v) for k, v in self.sentiment_counts.items()}
        labels = ["Positive", "Neutral", "Negative"]
        new_vals = [totals.get("Positive", 0), totals.get("Neutral", 0), totals.get("Negative", 0)]

        # Lazy init figure/canvas once
        if not getattr(self, "canvas", None):
            bg = theme.plot_bg()
            txt = theme.text_color()
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
            ax.set_title("Sentiment Distribution", color=txt, fontsize=14, pad=12)
            self.canvas = FigureCanvasTkAgg(fig, master=self.pie_container)
            self.canvas.draw()
            w = self.canvas.get_tk_widget()
            w.pack(fill="both", expand=True, padx=8, pady=8)
            self.pie_fig = fig
            self.pie_ax = ax
            self._pie_prev = [0, 0, 0]

        # If no data at all, show placeholder text
        if sum(new_vals) == 0:
            self.pie_ax.clear()
            bg = theme.plot_bg()
            txt = theme.text_color()
            self.pie_fig.patch.set_facecolor(bg)
            self.pie_ax.set_facecolor(bg)
            self.pie_ax.text(0.5, 0.5, "No data available", ha="center", va="center", color=txt, fontsize=12)
            self.pie_ax.axis("equal")
            self.pie_ax.set_title("Sentiment Distribution", color=txt, fontsize=14, pad=12)
            self.canvas.draw_idle()
            self._pie_prev = [0, 0, 0]
            self._ensure_chart_layout()
            return

        # Animate from previous to new values
        start = getattr(self, "_pie_prev", [0, 0, 0])
        steps = 12
        colors = ["#4caf50", "#9e9e9e", "#e53935"]
        bg = theme.plot_bg()
        txt = theme.text_color()

        def frame(i=1):
            t = i / float(steps)
            vals = [s + (n - s) * t for s, n in zip(start, new_vals)]
            self.pie_ax.clear()
            self.pie_fig.patch.set_facecolor(bg)
            self.pie_ax.set_facecolor(bg)
            wedges, _texts, autotexts = self.pie_ax.pie(
                vals,
                labels=labels,
                colors=colors,
                autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",
                startangle=90,
                textprops={"color": txt, "fontweight": "bold"},
                wedgeprops={"edgecolor": bg, "linewidth": 1},
            )
            for at in autotexts:
                at.set_fontsize(9)
            self.pie_ax.axis("equal")
            self.pie_ax.set_title("Sentiment Distribution", color=txt, fontsize=14, pad=12)
            self.canvas.draw_idle()
            if i < steps:
                self.after(16, lambda: frame(i + 1))
            else:
                self._pie_prev = new_vals
                self._ensure_chart_layout()

        frame(1)

    def update_trend_chart(self):
        """Create or update a responsive line chart with smooth animation."""
        labels = ["Positive", "Neutral", "Negative"]
        colors = ["#4caf50", "#9e9e9e", "#e53935"]
        bg = theme.plot_bg()
        txt = theme.text_color()

        # Lazy init figure/canvas and line artists
        if not getattr(self, "trend_canvas", None):
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
            ax.set_title("Sentiment Trend Over Time", color=txt, fontsize=14, pad=12)
            ax.set_xlabel("Time", color=txt, fontsize=10)
            ax.set_ylabel("Count", color=txt, fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.25, color=txt)
            for spine in ax.spines.values():
                spine.set_visible(False)
            self.trend_lines = []
            for c in colors:
                line, = ax.plot([], [], color=c, linewidth=2.2)
                self.trend_lines.append(line)
            ax.tick_params(axis="x", rotation=45, labelcolor=txt, labelsize=8)
            ax.tick_params(axis="y", labelcolor=txt, labelsize=8)
            self.trend_canvas = FigureCanvasTkAgg(fig, master=self.trend_container)
            self.trend_canvas.draw()
            self.trend_fig = fig
            self.trend_ax = ax
            w = self.trend_canvas.get_tk_widget()
            w.pack(fill="both", expand=True, padx=8, pady=8)
            self._trend_prev = [[], [], []]

        # If no data yet, render placeholder
        if not self.time_stamps:
            self.trend_ax.clear()
            self.trend_fig.patch.set_facecolor(bg)
            self.trend_ax.set_facecolor(bg)
            self.trend_ax.set_title("Sentiment Trend Over Time", color=txt, fontsize=14, pad=12)
            self.trend_ax.set_xlabel("Time", color=txt, fontsize=10)
            self.trend_ax.set_ylabel("Count", color=txt, fontsize=10)
            self.trend_ax.text(0.5, 0.5, "No time-series data available", ha="center", va="center", color=txt, fontsize=12)
            self.trend_ax.grid(True, linestyle="--", alpha=0.25, color=txt)
            self.trend_ax.tick_params(axis="x", rotation=45, labelcolor=txt, labelsize=8)
            self.trend_ax.tick_params(axis="y", labelcolor=txt, labelsize=8)
            self.trend_canvas.draw_idle()
            self._ensure_chart_layout()
            return

        # Prepare new data arrays
        new_series = [self.sentiment_counts[lbl][:] for lbl in labels]
        prev_series = getattr(self, "_trend_prev", [[], [], []])
        # Pad previous arrays to new length
        for idx in range(3):
            if not prev_series[idx]:
                prev_series[idx] = [0] * len(new_series[idx])
            elif len(prev_series[idx]) < len(new_series[idx]):
                pad = [prev_series[idx][-1]] * (len(new_series[idx]) - len(prev_series[idx]))
                prev_series[idx] = prev_series[idx] + pad

        steps = 12
        xs = list(range(len(self.time_stamps)))

        # Ensure line artists exist (axes may have been cleared)
        if not getattr(self, "trend_lines", None) or len(self.trend_lines) != 3 or any(getattr(l, 'axes', None) is not self.trend_ax for l in self.trend_lines):
            self.trend_ax.cla()
            self.trend_lines = []
            for c in colors:
                line, = self.trend_ax.plot([], [], color=c, linewidth=2.2)
                self.trend_lines.append(line)

        # Ensure axis cosmetics
        self.trend_ax.set_facecolor(bg)
        self.trend_fig.patch.set_facecolor(bg)
        self.trend_ax.set_title("Sentiment Trend Over Time", color=txt, fontsize=14, pad=12)
        self.trend_ax.set_xlabel("Time", color=txt, fontsize=10)
        self.trend_ax.set_ylabel("Count", color=txt, fontsize=10)
        self.trend_ax.grid(True, linestyle="--", alpha=0.25, color=txt)
        self.trend_ax.tick_params(axis="x", rotation=45, labelcolor=txt, labelsize=8)
        self.trend_ax.tick_params(axis="y", labelcolor=txt, labelsize=8)

        # Legend (recreate once)
        if not getattr(self, "_trend_legend", None):
            self._trend_legend = self.trend_ax.legend(labels, loc="upper left", facecolor=bg, edgecolor=theme.border_color(), labelcolor=txt, fontsize=8)

        def frame(i=1):
            t = i / float(steps)
            for idx, line in enumerate(self.trend_lines):
                prev = prev_series[idx]
                new = new_series[idx]
                interp = [p + (n - p) * t for p, n in zip(prev, new)]
                line.set_data(xs, interp)
            # Adjust view limits
            self.trend_ax.relim()
            self.trend_ax.autoscale_view()
            # Thin x ticks and apply labels from timestamps
            if len(xs) > 0:
                if len(xs) > 12:
                    step = max(len(xs) // 12, 1)
                    idxs = xs[::step]
                else:
                    idxs = xs
                labels = [self.time_stamps[i] for i in idxs]
                self.trend_ax.set_xticks(idxs)
                self.trend_ax.set_xticklabels(labels, rotation=45, color=txt, fontsize=8)
            self.trend_canvas.draw_idle()
            if i < steps:
                self.after(16, lambda: frame(i + 1))
            else:
                self._trend_prev = [arr[:] for arr in new_series]
                self._ensure_chart_layout()

        frame(1)

    def update_theme(self, mode):
        """Redraw charts to adapt to current theme."""
        try:
            self.update_pie_chart()
            self.update_trend_chart()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Responsive layout helpers
    # ------------------------------------------------------------------
    def _arrange_chart_grid(self, event=None):
        try:
            w = max(self.fig_frame.winfo_width(), 0)
        except Exception:
            w = 0
        stacked = w < 800  # stack on smaller widths
        if stacked == self._charts_stacked:
            return

        # Clear any existing grid placement
        try:
            self.pie_container.grid_forget()
            self.trend_container.grid_forget()
        except Exception:
            pass

        if stacked:
            try:
                self.fig_frame.grid_rowconfigure(0, weight=1)
                self.fig_frame.grid_rowconfigure(1, weight=1)
                self.pie_container.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
                self.trend_container.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
            except Exception:
                self.pie_container.pack(fill="both", expand=True, padx=8, pady=8)
                self.trend_container.pack(fill="both", expand=True, padx=8, pady=8)
        else:
            try:
                self.fig_frame.grid_rowconfigure(0, weight=1)
                self.fig_frame.grid_rowconfigure(1, weight=0)
                self.pie_container.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
                self.trend_container.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
            except Exception:
                self.pie_container.pack(side="left", fill="both", expand=True, padx=8, pady=8)
                self.trend_container.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self._charts_stacked = stacked

    def _ensure_chart_layout(self):
        # Make sure containers are placed (in case called before first configure)
        if self._charts_stacked is None:
            self._arrange_chart_grid()

    def _show_samples(self):
        """Display sample items in three columns (Positive/Neutral/Negative).
        Clears old widgets to avoid growing empty space.
        """
        # Clear previous widgets inside samples_frame
        try:
            for child in self.samples_frame.winfo_children():
                child.destroy()
        except Exception:
            pass

        # Grid layout with three equal columns
        try:
            self.samples_frame.grid_columnconfigure(0, weight=1, uniform="scols")
            self.samples_frame.grid_columnconfigure(1, weight=1, uniform="scols")
            self.samples_frame.grid_columnconfigure(2, weight=1, uniform="scols")
        except Exception:
            pass

        sentiments = [
            ("Positive", "#4caf50"),
            ("Neutral", "#9e9e9e"),
            ("Negative", "#e53935"),
        ]

        for idx, (label, color) in enumerate(sentiments):
            col = ctk.CTkFrame(self.samples_frame)
            try:
                col.grid(row=0, column=idx, padx=6, pady=6, sticky="nsew")
            except Exception:
                col.pack(side="left", expand=True, fill="both", padx=6, pady=6)

            title = ctk.CTkLabel(col, text=f"{label} Examples", text_color=color, font=("Arial", 13, "bold"))
            title.pack(anchor="w", padx=6, pady=(6, 4))

            tb = ctk.CTkTextbox(col, height=180, wrap="word")
            tb.pack(fill="both", expand=True, padx=6, pady=(0, 6))
            items = self.sample_tweets.get(label, [])
            if items:
                text = "\n\n".join(f"• {t}" for t in items[:5])
            else:
                text = "(no examples yet)"
            tb.insert("1.0", text)
            tb.configure(state="disabled")
        
    def _update_stats(self):
        """Update the statistics display."""
        if not self.time_stamps:
            self.stats_text.configure(text="No data available")
            return
            
        try:
            # Calculate overall stats
            totals = {k: sum(v) for k, v in self.sentiment_counts.items()}
            total_tweets = sum(totals.values())
            
            # Calculate percentages
            percentages = {}
            for sentiment, count in totals.items():
                percentages[sentiment] = (count / total_tweets * 100) if total_tweets > 0 else 0
                
            # Find dominant sentiment
            dominant = max(totals.items(), key=lambda x: x[1])[0] if total_tweets > 0 else "None"
            
            # Calculate time range
            first_time = self.time_stamps[0]
            last_time = self.time_stamps[-1]
            
            # Prepare stats text
            stats = (
                f"Total tweets: {total_tweets} • "
                f"Time range: {first_time} - {last_time} • "
                f"Dominant sentiment: {dominant} ({percentages[dominant]:.1f}%)"
            )
            
            self.stats_text.configure(text=stats)
            
        except Exception as e:
            print(f"Error updating stats: {e}")
            self.stats_text.configure(text="Error calculating statistics")

    def _update_stats_enhanced(self):
        """Enhanced statistics with date and key info."""
        if not self.time_stamps:
            self.stats_text.configure(text="No data available")
            return

        try:
            totals = {k: sum(v) for k, v in self.sentiment_counts.items()}
            total_tweets = sum(totals.values())
            percentages = {s: ((totals.get(s, 0) / total_tweets * 100) if total_tweets > 0 else 0)
                           for s in ["Positive", "Neutral", "Negative"]}

            dominant = max(totals.items(), key=lambda x: x[1])[0] if total_tweets > 0 else "None"
            first_time = self.time_stamps[0]
            last_time = self.time_stamps[-1]
            intervals = len(self.time_stamps)

            totals_by_ts = [
                (self.time_stamps[i], self.sentiment_counts["Positive"][i]
                 + self.sentiment_counts["Neutral"][i]
                 + self.sentiment_counts["Negative"][i])
                for i in range(intervals)
            ]
            peak_ts, peak_val = max(totals_by_ts, key=lambda t: t[1]) if totals_by_ts else ("-", 0)
            avg_per_interval = (total_tweets / intervals) if intervals > 0 else 0

            if getattr(self, "current_csv_path", None):
                context = f"Source: CSV · {os.path.basename(self.current_csv_path)}"
            else:
                src = getattr(self, "source_var", None).get() if hasattr(self, "source_var") else "Live"
                query = self.entry.get().strip() if hasattr(self, "entry") else ""
                context = f"Source: Live · {src}{f' · {query}' if query else ''}"

            today = datetime.now().strftime("%Y-%m-%d")
            now = datetime.now().strftime("%H:%M:%S")

            stats_lines = [
                f"Date: {today} · Last update: {now}",
                context,
                f"Time range: {first_time} - {last_time} ({intervals} intervals)",
                f"Total tweets: {total_tweets} · Avg/interval: {avg_per_interval:.2f} · Peak: {peak_ts} ({peak_val})",
                (
                    f"Positive: {totals.get('Positive', 0)} ({percentages['Positive']:.1f}%)   "
                    f"Neutral: {totals.get('Neutral', 0)} ({percentages['Neutral']:.1f}%)   "
                    f"Negative: {totals.get('Negative', 0)} ({percentages['Negative']:.1f}%)"
                ),
                f"Dominant: {dominant} ({percentages.get(dominant, 0):.1f}%)",
            ]
            self.stats_text.configure(text="\n".join(stats_lines))

        except Exception as e:
            print(f"Error updating stats: {e}")
            self.stats_text.configure(text="Error calculating statistics")
            
            
    def cancel_page_tasks(self):
        """Cancel all pending after callbacks for this page."""
        try:
            # Cancel countdown timer
            if hasattr(self, 'countdown_job') and self.countdown_job:
                try:
                    self.after_cancel(self.countdown_job)
                except Exception:
                    pass
                self.countdown_job = None
                
            # Do not stop collection here; keep running state.
        except Exception as e:
            print(f"Error canceling page tasks: {e}")
            

    def destroy(self):
        """Clean up resources when page is destroyed."""
        try:
            # Stop all monitoring processes first
            try:
                self.running = False
                
                # Stop collector
                if hasattr(self, 'collector') and self.collector:
                    self.collector.stop()
                    self.collector = None
                
                # Cancel countdown timer
                if hasattr(self, 'countdown_job') and self.countdown_job:
                    try:
                        self.after_cancel(self.countdown_job)
                    except Exception:
                        pass
                    self.countdown_job = None
            except Exception as e:
                print(f"Error stopping monitoring: {e}")
                
            # Close matplotlib figures to prevent memory leaks
            try:
                if hasattr(self, 'canvas') and self.canvas:
                    try:
                        self.canvas.get_tk_widget().destroy()
                    except Exception:
                        pass
                    self.canvas = None
                    
                if hasattr(self, 'trend_canvas') and self.trend_canvas:
                    try:
                        self.trend_canvas.get_tk_widget().destroy()
                    except Exception:
                        pass
                    self.trend_canvas = None
            except Exception as e:
                print(f"Error closing figures: {e}")
                
        except Exception as e:
            print(f"Error during Page2 destruction: {e}")
            
        # Call parent's destroy method
        try:
            super().destroy()
        except Exception as e:
            print(f"Error calling super().destroy(): {e}")
