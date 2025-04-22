"""collector.py – Tweepy API version (revived)
------------------------------------------------
Official‑API TweetCollector with built‑in cooldown and GUI countdown
support.  Replace your **bearer token** below.  This file completely
replaces the previous Nitter/scrape variant so Analyzer and GUI can work
with reliable Twitter data—as long as the key has recent‑search access.

CSV schema (unchanged): created_at, username, text, tweet_id
"""
from __future__ import annotations

import threading, time, csv, os
from datetime import datetime
from typing import Optional

import tweepy

class TweetCollector:
    """Collect tweets via Twitter/X v2 *recent search* endpoint.

    Parameters
    ----------
    keyword : str
        Hashtag or search term (e.g. "#AI", "gaza").
    cooldown : int
        Seconds to wait between API calls—keeps us within free‑tier quota
        (default 60 s).  GUI polls `seconds_until_next_request()` for the
        live countdown.
    bearer_token : str | None
        Twitter/X API bearer token.  If None, raise ValueError.
    """

    def __init__(self, keyword: str, cooldown: int = 60,
                 bearer_token: str | None = "AAAAAAAAAAAAAAAAAAAAAONjxwEAAAAAj74TclHPqXhKgRmuSlsIRJSXF9g%3DdsDgbh7xAa0apGZjGtkfFWYKWVIZO0Hd2Y1Hi9uqXobjSrzFw1"):
        if not bearer_token or bearer_token.startswith("YOUR_"):
            raise ValueError("Please set a valid Twitter bearer token.")

        self.keyword = keyword
        self.cooldown = cooldown
        self.client = tweepy.Client(bearer_token=bearer_token,
                                    wait_on_rate_limit=False)

        self.stop_event = threading.Event()
        self._next_request_time: Optional[float] = None
        self._ids_seen: set[str] = set()

        # CSV path (outputs/<hashtag>_YYYYMMDD.csv)
        os.makedirs("outputs", exist_ok=True)
        date_part = datetime.now().strftime("%Y%m%d")
        fname = keyword.lstrip("#").replace(" ", "_")
        self.csv_path = f"outputs/{fname}_{date_part}.csv"
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(["created_at", "username", "text", "tweet_id"])

    # ------------------------------------------------------------------
    # Public helper for GUI countdown
    # ------------------------------------------------------------------
    def seconds_until_next_request(self) -> int:
        if self._next_request_time is None:
            return 0
        return max(0, int(self._next_request_time - time.time()))

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------
    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------
    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self._collect_once()
            except tweepy.TooManyRequests as e:
                reset_epoch = int(e.response.headers.get("x-rate-limit-reset", time.time() + 60))
                sleep_for = max(0, reset_epoch - int(time.time())) + 2
                print(f"[TweepyCollector] Rate limit hit – sleeping {sleep_for}s")
                self._next_request_time = time.time() + sleep_for
                time.sleep(sleep_for)
                continue
            except Exception as exc:
                print(f"[TweepyCollector] Error: {exc}")

            self._next_request_time = time.time() + self.cooldown
            time.sleep(self.cooldown)

    # ------------------------------------------------------------------
    def _collect_once(self):
        query = f"{self.keyword} -is:retweet lang:en"
        resp = self.client.search_recent_tweets(
            query=query,
            tweet_fields=["created_at", "author_id", "text", "id"],
            max_results=100,
        )
        if not resp.data:
            print("[TweepyCollector] No tweets returned in this batch.")
            return

        new_rows: list[list[str]] = []
        for tw in resp.data:
            if tw.id in self._ids_seen:
                continue
            self._ids_seen.add(tw.id)
            new_rows.append([
                tw.created_at.isoformat(timespec="seconds"),
                str(tw.author_id),
                tw.text.replace("\n", " "),
                str(tw.id),
            ])

        if new_rows:
            with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerows(new_rows)
            print(f"[TweepyCollector] Saved {len(new_rows)} tweets → {self.csv_path}")
        else:
            print("[TweepyCollector] All tweets were duplicates – nothing new.")
