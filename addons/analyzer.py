"""
analyzer.py
-----------
Reads newly appended tweets from the collector's CSV and performs
sentiment analysis with cardiffnlp/twitter-roberta-base-sentiment.

Usage:
    analyzer = SentimentAnalyzer(csv_path, callback, reload_every=60)
    analyzer.start()
    analyzer.stop()
"""
import threading
import time
import csv
from typing import Callable, List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalyzer:
    def __init__(self, csv_path: str,
                 update_callback: Callable[[List[Dict]], None],
                 reload_every: int = 60):
        """
        Parameters
        ----------
        csv_path : str
            Path to CSV produced by TweetCollector.
        update_callback : callable
            Function that receives a list of rows with sentiment results.
        reload_every : int
            Seconds between reloads.
        """
        self.csv_path = csv_path
        self.reload_every = reload_every
        self.update_callback = update_callback
        self.stop_event = threading.Event()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment")
        self.model.eval()

        self._last_seen_line = 0  # File pointer for incremental reading

    # ------------------------------------------------------------------ #
    # Thread management
    # ------------------------------------------------------------------ #
    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()

    # ------------------------------------------------------------------ #
    # Internal loop
    # ------------------------------------------------------------------ #
    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self._process_new_rows()
            except Exception as exc:
                print(f"[Analyzer] Error: {exc}")
            time.sleep(self.reload_every)

    def _process_new_rows(self):
        if not torch.is_grad_enabled():  # ensures no Autograd overhead
            torch.set_grad_enabled(False)

        new_results = []
        try:
            with open(self.csv_path, encoding="utf‑8") as f:
                reader = list(csv.reader(f))
        except FileNotFoundError:
            return

        rows = reader[self._last_seen_line + 1:]  # skip header + already processed
        if not rows:
            return

        texts = [r[2] for r in rows]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).tolist()

        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        for r, p in zip(rows, preds):
            new_results.append({
                "created_at": r[0],
                "username": r[1],
                "text": r[2],
                "sentiment": label_map[p]
            })

        self._last_seen_line += len(rows)
        if new_results:
            self.update_callback(new_results)
