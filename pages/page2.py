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
        
         # Create a master scrollable container with fixed width to preserve graph sizes
        self.master_scroll = ctk.CTkScrollableFrame(
            self, 
            width=950,  # Fixed width to accommodate full-size graphs
            height=800,  # Reasonable height that will scroll if needed
            corner_radius=0,  # Remove rounded corners for better space usage
        )
        self.master_scroll.pack(expand=True, fill="both", padx=0, pady=0)

        # --- State holders -------------------------------------------
        self.collector: Optional[TweetCollector] = None
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
            text_color="#aaaaaa",
            wraplength=850  # Add wraplength to handle narrow windows
        )
        desc.pack(pady=(0, 10))

        # Input area
        input_frame = ctk.CTkFrame(self.master_scroll)  # Changed parent
        input_frame.pack(fill="x", padx=20, pady=10)
        
        # Rest of your UI components - update their parent to self.master_scroll
        self.entry = ctk.CTkEntry(
            input_frame, 
            placeholder_text="#Hashtag or search term", 
            width=300
        )
        self.entry.pack(side="left", padx=(0, 10))
        
        self.status = ctk.CTkLabel(
            input_frame, 
            text="Ready", 
            text_color="gray",
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

        # Visualization container
        self.viz_frame = ctk.CTkFrame(self.master_scroll)  # Changed parent
        self.viz_frame.pack(expand=True, fill="both", padx=20, pady=(10, 0))
        
        # Chart container - preserve original sizes
        self.fig_frame = ctk.CTkFrame(self.viz_frame)
        self.fig_frame.pack(side="top", pady=5, fill="x")
        # Ensure fixed minimum width for the fig_frame to keep graphs from shrinking
        self.fig_frame.configure(width=900, height=350)
        self.fig_frame.grid_propagate(False)  # Prevent frame from shrinking
        try:
            self.viz_frame.configure(border_width=1, border_color=theme.border_color())
            self.fig_frame.configure(border_width=1, border_color=theme.border_color())
        except Exception:
            pass
        
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
            text_color="#aaaaaa"
        )
        self.stats_text.pack(pady=(0, 5))
        
        # Initialize empty visualizations
        self.update_pie_chart()
        self.update_trend_chart()
        
        # At the end of __init__
        try:
            # Initialize empty visualizations
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
    # Live monitoring
    # ------------------------------------------------------------------
    def start_monitoring(self):
        """Start real-time monitoring of tweets for a hashtag."""
        if TweetCollector is None:
            self.status.configure(text="Collector unavailable", text_color="red")
            return

        hashtag = self.entry.get().strip()
        if not hashtag:
            self.status.configure(text="Enter a hashtag first!", text_color="red")
            return

        # Stop any existing monitoring
        self.stop_monitoring()

        # Start collection process
        try:
            # Update UI
            self.start_button.configure(state="disabled")
            self.status.configure(text="Initializing...", text_color="orange")
            self.progress.set(0.1)
            
            # Initialize collector in background thread to keep UI responsive
            threading.Thread(target=self._initialize_collection, args=(hashtag,), daemon=True).start()
        except Exception as e:
            self.status.configure(text=f"Error: {str(e)}", text_color="red")
            self.start_button.configure(state="normal")
            self.progress.set(0)

    def _initialize_collection(self, hashtag):
        """Initialize collection in background thread."""
        try:
            # Create collector
            self.collector = TweetCollector(hashtag, cooldown=60)
            self.after(0, lambda: self.status.configure(text="Starting collector...", text_color="orange"))
            self.after(0, lambda: self.progress.set(0.3))
            
            # Start collector
            self.collector.start()
            self.after(0, lambda: self.progress.set(0.6))
            
            # Store path for export functionality
            self.current_csv_path = self.collector.csv_path
            
            # Reset data storage
            self.sentiment_counts = {"Positive": [], "Neutral": [], "Negative": []}
            self.time_stamps = []
            self.sample_tweets = {"Positive": [], "Neutral": [], "Negative": []}
            
            # Create analyzer with direct callback to this instance
            from addons.analyzer import SentimentAnalyzer as FileAnalyzer
            analyzer = FileAnalyzer(
                self.collector.csv_path,
                update_callback=self.on_new_analysis,
                reload_every=30,
            )
            analyzer.start()
            
            # Update UI on main thread
            self.after(0, lambda: self.progress.set(1.0))
            self.after(0, lambda: self.status.configure(text="Collecting...", text_color="orange"))
            self.after(0, lambda: self.export_button.configure(state="normal"))
            self.after(0, lambda: self.start_button.configure(state="normal"))
            
            # Start countdown
            self.running = True
            self.after(0, self._schedule_countdown)
            
            # Reset progress after delay
            self.after(2000, lambda: self.progress.set(0))
            
        except Exception as e:
            self.after(0, lambda: self.status.configure(text=f"Error: {str(e)}", text_color="red"))
            self.after(0, lambda: self.start_button.configure(state="normal"))
            self.after(0, lambda: self.progress.set(0))
            self.collector = None

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
        self.status.configure(text="Stopped", text_color="gray")
        self.progress.set(0)

    def _schedule_countdown(self):
        """Update countdown display for next data collection."""
        if not self.running or not self.collector:
            return
            
        # Calculate seconds until next request
        sec = self.collector.seconds_until_next_request()
        
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
        self._update_stats()
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
            self.after(0, self._update_stats)
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
        """Create dark‑theme pie chart with inline labels."""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        # Calculate totals
        totals = {k: sum(v) for k, v in self.sentiment_counts.items()}
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))
        bg = theme.plot_bg()
        txt = theme.text_color()
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        
        # Check if we have data
        if any(totals.values()):
            # Sort by sentiment for consistent ordering
            labels, values = zip(*sorted(totals.items()))
            
            # Create pie chart with enhanced styling 
            colors = ["#e91e63", "#ffb300", "#4caf50"]  # Positive, Neutral, Negative
            wedges, texts, autotexts = ax.pie(
                values, 
                labels=labels, 
                colors=colors,
                autopct=lambda p: f"{p:.1f}%\n({int(p*sum(values)/100)})" if p > 5 else "",
                startangle=90, 
                textprops={"color": txt, "fontweight": "bold"},
                wedgeprops={"edgecolor": bg, "linewidth": 1}
            )
            
            # Enhanced text styling
            for autotext in autotexts:
                autotext.set_fontsize(9)
        else:
            # Empty pie chart with message
            ax.text(
                0.5, 0.5, 
                "No data available", 
                ha="center", 
                va="center", 
                color="white",
                fontsize=12
            )
            
        # Title and styling
        ax.set_title("Sentiment Distribution", color="white", fontsize=14, pad=20)
        
        # Create canvas with fixed size
        self.canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.config(width=450, height=320)  # Fix the canvas size
        canvas_widget.pack(side="left", padx=5)

    def update_trend_chart(self):
        """Create dark‑theme line chart with inline labels at latest points."""
        # Remove previous chart if it exists
        if self.trend_canvas:
            self.trend_canvas.get_tk_widget().destroy()
            
        # Skip if no data
        if not self.time_stamps:
            # Create empty figure with message
            fig, ax = plt.subplots(figsize=(6, 4))
            bg = theme.plot_bg()
            txt = theme.text_color()
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
            
            ax.text(
                0.5, 0.5, 
                "No time-series data available", 
                ha="center", 
                va="center", 
                color="white",
                fontsize=12
            )
            
            ax.set_title("Sentiment Trend Over Time", color=txt, fontsize=14, pad=20)
            
            # Set up empty axes
            ax.set_xlabel("Time", color=txt)
            ax.set_ylabel("Tweet Count", color=txt)
            ax.tick_params(axis="x", colors=txt)
            ax.tick_params(axis="y", colors=txt)
            
            # Create canvas with fixed size
            self.trend_canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
            self.trend_canvas.draw()
            canvas_widget = self.trend_canvas.get_tk_widget()
            canvas_widget.config(width=450, height=320)  # Fix the canvas size
            canvas_widget.pack(side="right", padx=5)
            
            return

        # Create figure for trend chart
        fig, ax = plt.subplots(figsize=(6, 4))
        bg = theme.plot_bg()
        txt = theme.text_color()
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        
        # Prepare color scheme and sentiment labels
        colors = {
            "Positive": "#4caf50",  # green
            "Neutral": "#ffb300",   # amber
            "Negative": "#e91e63"   # pink
        }
        
        # Plot each sentiment series with enhanced styling
        for sentiment, color in colors.items():
            series = self.sentiment_counts[sentiment]
            if not series:
                continue
                
            # Plot line with enhanced styling
            line, = ax.plot(
                self.time_stamps,
                series,
                marker="o",
                color=color,
                linewidth=2,
                markersize=4,
                alpha=0.8,
                label=sentiment
            )
            
            # Add label at last data point if value is non-zero
            if series[-1] > 0:
                ax.text(
                    self.time_stamps[-1], 
                    series[-1], 
                    f" {sentiment}: {series[-1]}", 
                    color=color,
                    va="center", 
                    fontsize=8,
                    fontweight="bold",
                    path_effects=[
                        path_effects.withStroke(
                            linewidth=2, foreground=dark
                        )
                    ]
                )
        
        # Enhanced formatting
        ax.set_title("Sentiment Trend Over Time", color=txt, fontsize=14, pad=20)
        ax.set_xlabel("Time", color=txt, fontsize=10)
        ax.set_ylabel("Tweet Count", color=txt, fontsize=10)
        
        # Improve tick formatting
        ax.tick_params(axis="x", rotation=45, labelcolor=txt, labelsize=8)
        ax.tick_params(axis="y", labelcolor=txt, labelsize=8)
        
        # Show fewer x-axis ticks if many data points
        if len(self.time_stamps) > 10:
            step = max(len(self.time_stamps) // 10, 1)
            ax.set_xticks(self.time_stamps[::step])
            
        # Add grid for readability
        ax.grid(True, linestyle="--", alpha=0.3, color=txt)
        
        # Add legend
        if any(sum(self.sentiment_counts[s]) > 0 for s in colors.keys()):
            ax.legend(
                loc="upper left", 
                facecolor=bg, 
                edgecolor="#555555",
                labelcolor=txt,
                fontsize=8
            )
        
        # Ensure tight layout
        fig.tight_layout()
        
        # Create canvas
        self.trend_canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        self.trend_canvas.draw()
        self.trend_canvas.get_tk_widget().pack(side="right", padx=5)

    def update_theme(self, mode):
        """Redraw charts to adapt to current theme."""
        try:
            self.update_pie_chart()
            self.update_trend_chart()
        except Exception:
            pass

    def _show_samples(self):
        """Display sample tweets for each sentiment category."""
        # Clear previous samples
        if self.example_textbox:
            self.example_textbox.destroy()
            
        # Create frame to hold textbox and scrollbar
        text_frame = ctk.CTkFrame(self.samples_frame)
        text_frame.pack(pady=8, fill="both", expand=True)
        
        # Create new text display with horizontal scrolling
        self.example_textbox = ctk.CTkTextbox(
            text_frame, 
            width=700, 
            height=180,
            wrap="none"  # Disable text wrapping to enable horizontal scrolling
        )
        self.example_textbox.pack(side="left", fill="both", expand=True)
        
        # Insert text without using tag_config with font option
        self.example_textbox.insert("end", "Sample Tweets\n\n")
        
        # Add samples for each sentiment
        for sentiment, tweets in self.sample_tweets.items():
            # Insert sentiment heading
            self.example_textbox.insert("end", f"{sentiment} examples:\n")
            
            # Add sample tweets or placeholder
            if tweets:
                for tweet in tweets:
                    self.example_textbox.insert("end", f"  • {tweet}\n")
            else:
                self.example_textbox.insert("end", "  No examples available\n")
                
            self.example_textbox.insert("end", "\n")
            
        # Make read-only
        self.example_textbox.configure(state="disabled")
        
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
                
            # Mark as not running
            self.running = False
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
