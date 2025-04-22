"""
page2.py
------------
CustomTkinter Page that wires the Collector and Analyzer together.
Add this Page2 to your main app's pages dict.
"""
import customtkinter as ctk
from addons.bypass_collector import TweetCollector
from addons.analyzer import SentimentAnalyzer
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Page2(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.collector: TweetCollector | None = None
        self.analyzer: SentimentAnalyzer | None = None
        self.countdown_job = None

        self.sentiment_counts = {"Positive": [], "Neutral": [], "Negative": []}
        self.time_stamps: list[str] = []

        title = ctk.CTkLabel(
            self, text="Real‑Time Opinion Polling",
            font=("Arial", 20))
        title.pack(pady=10)

        self.entry = ctk.CTkEntry(
            self, placeholder_text="#Hashtag to monitor", width=240)
        self.entry.pack(pady=6)

        self.status = ctk.CTkLabel(self, text="Idle", text_color="gray")
        self.status.pack()

        btn_row = ctk.CTkFrame(self)
        btn_row.pack(pady=6)
        ctk.CTkButton(btn_row, text="Start Monitoring",
                      command=self.start_monitoring).pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Stop", fg_color="#8b0000",
                      command=self.stop_monitoring).pack(side="left", padx=4)

        # Pie + trend placeholders
        self.fig_frame = ctk.CTkFrame(self)
        self.fig_frame.pack(pady=12)
        self.canvas: FigureCanvasTkAgg | None = None
        self.trend_canvas: FigureCanvasTkAgg | None = None

    # ------------------------------------------------------------------ #
    # Start / Stop
    # ------------------------------------------------------------------ #
    def start_monitoring(self):
        hashtag = self.entry.get().strip()
        if not hashtag:
            self.status.configure(text="Enter a hashtag first!", text_color="red")
            return

        # Collector
        self.collector = TweetCollector(hashtag, cooldown=60)
        self.collector.start()

        # Analyzer
        self.analyzer = SentimentAnalyzer(
            self.collector.csv_path,
            update_callback=self.on_new_analysis,
            reload_every=30
        )
        self.analyzer.start()

        self.status.configure(text="Collecting…", text_color="orange")
        self._schedule_countdown()          # ← NEW

    def stop_monitoring(self):
        if self.collector:
            self.collector.stop()
        if self.analyzer:
            self.analyzer.stop()
        if self.countdown_job:              # ← NEW
            self.after_cancel(self.countdown_job)
        self.status.configure(text="Stopped", text_color="gray")
        
    # ------------------------------------------------------------------ #
    # Countdown updater  ❱  runs every second
    # ------------------------------------------------------------------ #
    def _schedule_countdown(self):
        """Poll collector for remaining‑seconds and update status label."""
        if not self.collector:
            return
        sec = self.collector.seconds_until_next_request()
        if sec > 0:
            self.status.configure(
                text=f"Collecting… (⏳ {sec}s)",
                text_color="orange"
            )
        else:
            # once sec == 0 the collector is about to call again
            self.status.configure(text="Collecting…", text_color="orange")

        self.countdown_job = self.after(1000, self._schedule_countdown)

    # ------------------------------------------------------------------ #
    # Callback from analyzer
    # ------------------------------------------------------------------ #
    def on_new_analysis(self, new_rows: list[dict]):
        # Aggregate counts
        pos = sum(r["sentiment"] == "Positive" for r in new_rows)
        neu = sum(r["sentiment"] == "Neutral" for r in new_rows)
        neg = sum(r["sentiment"] == "Negative" for r in new_rows)
        total = pos + neu + neg

        if total == 0:
            return

        self.sentiment_counts["Positive"].append(pos)
        self.sentiment_counts["Neutral"].append(neu)
        self.sentiment_counts["Negative"].append(neg)
        self.time_stamps.append(datetime.now().strftime("%H:%M:%S"))

        self.update_pie_chart()
        self.update_trend_chart()
        self.status.configure(text="Analyzing…", text_color="green")

    # ------------------------------------------------------------------ #
    # Matplotlib helpers
    # ------------------------------------------------------------------ #
    def update_pie_chart(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        sums = {k: sum(vs) for k, vs in self.sentiment_counts.items()}
        labels, values = zip(*sums.items())

        fig, ax = plt.subplots(figsize=(4, 3))
        wedges, _, _ = ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title("Overall Sentiment Share")
        self.canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="left")
        plt.close(fig)

    def update_trend_chart(self):
        if self.trend_canvas:
            self.trend_canvas.get_tk_widget().destroy()

        history = np.array([self.sentiment_counts["Positive"],
                            self.sentiment_counts["Neutral"],
                            self.sentiment_counts["Negative"]])
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(self.time_stamps, history[0], label="Positive")
        ax.plot(self.time_stamps, history[1], label="Neutral")
        ax.plot(self.time_stamps, history[2], label="Negative")
        ax.set_title("Sentiment Trend")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        self.trend_canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        self.trend_canvas.draw()
        self.trend_canvas.get_tk_widget().pack(side="left")
        plt.close(fig)
