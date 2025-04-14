import customtkinter as ctk
import tweepy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import csv
import ssl
import os
import time

from datetime import datetime

class Page2(ctk.CTkFrame):
    def __init__(self, parent):
        self.auto_refresh_interval = 60  # seconds
        self.auto_refresh_enabled = True
        self.cooldown_active = False
        super().__init__(parent)

        self.scrollable = ctk.CTkScrollableFrame(self, width=800, height=800)
        self.scrollable.pack(expand=True, fill="both", padx=10, pady=10)

        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model.eval()

        title = ctk.CTkLabel(self.scrollable, text="Opinion Polling via Twitter", font=("Arial", 20))
        title.pack(pady=10)

        desc = ctk.CTkLabel(self.scrollable, text="Enter a Twitter hashtag or keyword to analyze public sentiment.", font=("Arial", 14))
        desc.pack(pady=5)

        self.entry = ctk.CTkEntry(self.scrollable, placeholder_text="e.g. #AI, climate change", width=400)
        self.entry.pack(pady=10)

        self.result_label = ctk.CTkLabel(self.scrollable, text="", font=("Arial", 12))
        self.status_label = ctk.CTkLabel(self.scrollable, text="Status: Idle", font=("Arial", 11), text_color="gray")
        self.result_label.pack(pady=5)
        self.status_label.pack(pady=2)

        submit_btn = ctk.CTkButton(self.scrollable, text="Analyze", command=self.run_polling_thread)
        submit_btn.pack(pady=10)

        self.canvas_frame = ctk.CTkFrame(self.scrollable)
        self.canvas_frame.pack(pady=20)

        self.trend_frame = ctk.CTkFrame(self.scrollable)
        self.trend_frame.pack(pady=20)

        export_btn = ctk.CTkButton(self.scrollable, text="Export Trend Chart", command=self.export_trend_chart)
        export_btn.pack(pady=5)

        reset_btn = ctk.CTkButton(self.scrollable, text="Reset Trend Data", command=self.reset_trend_data)
        reset_btn.pack(pady=5)

        # Start auto-refresh
        self.schedule_auto_refresh()

        self.sentiment_history = []
        self.time_stamps = []

    def run_polling_thread(self):
        if self.cooldown_active:
            self.result_label.configure(text="Please wait a few seconds before submitting again.")
            return
        self.cooldown_active = True
        self.result_label.configure(text="Running... Please wait.")
        self.status_label.configure(text="Status: Connecting to Twitter API...", text_color="orange")
        self.after(10000, self.reset_cooldown)  # 10-second cooldown

        threading.Thread(target=self.poll_opinion, daemon=True).start()

    def start_countdown(self, seconds):
        """Show a countdown in status_label for rate-limit wait time."""
        if seconds <= 0:
            self.status_label.configure(text="Status: Idle", text_color="gray")
            return
        self.status_label.configure(text=f"Retry in {seconds}s", text_color="orange")
        self.after(1000, lambda: self.start_countdown(seconds - 1))

    def schedule_auto_refresh(self):
        """Automatically re-run polling every self.auto_refresh_interval seconds."""
        if self.auto_refresh_enabled:
            self.after(self.auto_refresh_interval * 1000, self.run_polling_thread)

    def reset_cooldown(self):
        self.cooldown_active = False

    def poll_opinion(self):
        """Fetch tweets from Twitter, analyze sentiment, and update charts."""
        ssl._create_default_https_context = ssl._create_unverified_context
        keyword = self.entry.get().strip()
        if not keyword:
            self.result_label.configure(text="Please enter a hashtag or keyword.")
            return

        self.result_label.configure(text="Collecting tweets and analyzing sentiment... Please wait.")
        tweets = []
        tweet_data = []

        try:
            client = tweepy.Client(bearer_token="AAAAAAAAAAAAAAAAAAAAAONjxwEAAAAAj74TclHPqXhKgRmuSlsIRJSXF9g%3DdsDgbh7xAa0apGZjGtkfFWYKWVIZO0Hd2Y1Hi9uqXobjSrzFw1")  # <-- Insert your real token
            query = f"{keyword} -is:retweet lang:en"
            response = client.search_recent_tweets(query=query, tweet_fields=["created_at", "author_id", "text"], max_results=100)
            if not response.data:
                self.result_label.configure(text="No tweets found for this keyword. Try another.")
                self.status_label.configure(text="Status: No data", text_color="gray")
                return
            for tweet in response.data:
                tweets.append(tweet.text)
                tweet_data.append((tweet.created_at.strftime("%Y-%m-%d %H:%M"), tweet.author_id, tweet.text))

        except tweepy.TooManyRequests as e:
            # Rate limit error
            reset_time = int(e.response.headers.get('x-rate-limit-reset', time.time() + 60))
            wait_seconds = max(0, reset_time - int(time.time()))
            self.result_label.configure(text=f"Rate limit reached. Try again in {wait_seconds} seconds.")
            self.status_label.configure(text=f"Status: Rate limit hit ({wait_seconds}s cooldown)", text_color="red")
            self.start_countdown(wait_seconds)
            return

        except Exception as e:
            self.result_label.configure(text=f"Failed to fetch tweets via Tweepy. Details: {str(e)}")
            self.status_label.configure(text="Status: API error", text_color="red")
            return

        # Sentiment analysis
        sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}
        results = []
        for date, user, text in tweet_data:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            pred = torch.argmax(probs).item()
            label = ["Negative", "Neutral", "Positive"][pred]
            sentiments[label] += 1
            results.append([date, user, label, text])

        self.result_label.configure(text=f"Analyzed {len(tweets)} tweets. Result below:")
        self.status_label.configure(text="Status: Analysis complete", text_color="green")
        self.display_example_tweets(results)
        self.plot_pie_chart(sentiments)
        self.save_to_csv(results, keyword)
        self.update_trend_plot(sentiments)
        self.schedule_auto_refresh()

    def display_example_tweets(self, tweet_data):
        """Show up to 2 example tweets per sentiment."""
        if hasattr(self, 'example_textbox'):
            self.example_textbox.destroy()

        self.example_textbox = ctk.CTkTextbox(self.scrollable, width=700, height=200)
        self.example_textbox.pack(pady=10)
        self.example_textbox.insert("end", "Sample Tweets by Sentiment:\n\n")

        grouped = {"Positive": [], "Neutral": [], "Negative": []}
        for date, user, label, text in tweet_data:
            if len(grouped[label]) < 2:
                grouped[label].append(f"@{user} ({date}): {text[:120]}...")

        for sentiment, examples in grouped.items():
            self.example_textbox.insert("end", f"{sentiment} Tweets:\n")
            for ex in examples:
                self.example_textbox.insert("end", f"  - {ex}\n")
            self.example_textbox.insert("end", "\n")
        self.example_textbox.configure(state="disabled")

    def plot_pie_chart(self, sentiments):
        """Display pie chart of positive, neutral, negative tweet counts."""
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.get_tk_widget().destroy()

        labels = list(sentiments.keys())
        values = list(sentiments.values())

        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor("#2B2B2B")
        ax.set_facecolor("#2B2B2B")
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": "white"}
        )
        ax.set_title("Sentiment Distribution", color="white")

        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        plt.close(fig)

    def update_trend_plot(self, sentiments):
        """Update line chart of sentiment over time with new counts."""
        self.time_stamps.append(datetime.now().strftime("%H:%M:%S"))
        self.sentiment_history.append([
            sentiments["Positive"],
            sentiments["Neutral"],
            sentiments["Negative"]
        ])

        if hasattr(self, 'trend_canvas') and self.trend_canvas:
            self.trend_canvas.get_tk_widget().destroy()

        history = np.array(self.sentiment_history).T
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor("#2B2B2B")
        ax.set_facecolor("#2B2B2B")
        ax.plot(self.time_stamps, history[0], label="Positive", marker='o')
        ax.plot(self.time_stamps, history[1], label="Neutral", marker='o')
        ax.plot(self.time_stamps, history[2], label="Negative", marker='o')
        ax.set_title("Sentiment Trend Over Time", color="white")
        ax.set_ylabel("Tweet Count", color="white")
        ax.set_xlabel("Time", color="white")
        ax.tick_params(axis='x', rotation=45, labelcolor="white")
        ax.tick_params(axis='y', labelcolor="white")
        ax.legend()

        self.trend_canvas = FigureCanvasTkAgg(fig, master=self.trend_frame)
        self.trend_canvas.draw()
        self.trend_canvas.get_tk_widget().pack()
        plt.close(fig)

    def export_trend_chart(self):
        """Save the current sentiment trend line chart to a PNG."""
        if not self.time_stamps or not self.sentiment_history:
            return

        history = np.array(self.sentiment_history).T
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(self.time_stamps, history[0], label="Positive", marker='o')
        ax.plot(self.time_stamps, history[1], label="Neutral", marker='o')
        ax.plot(self.time_stamps, history[2], label="Negative", marker='o')
        ax.set_title("Sentiment Trend Over Time")
        ax.set_ylabel("Tweet Count")
        ax.set_xlabel("Time")
        ax.legend()
        fig.autofmt_xdate()
        os.makedirs("outputs", exist_ok=True)
        fig.savefig("outputs/sentiment_trend.png")
        plt.close(fig)

    def reset_trend_data(self):
        """Clear the entire history of sentiment data and remove the line chart."""
        self.time_stamps.clear()
        self.sentiment_history.clear()
        if hasattr(self, 'trend_canvas') and self.trend_canvas:
            self.trend_canvas.get_tk_widget().destroy()

    def save_to_csv(self, results, keyword):
        """Save tweet data with sentiments to a CSV under outputs/"""
        os.makedirs("outputs", exist_ok=True)
        filename = f"outputs/{keyword.replace('#', '').replace(' ', '_')}_sentiment.csv"
        with open(filename, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Username", "Sentiment", "Tweet"])
            for row in results:
                writer.writerow(row)
