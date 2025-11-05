"""bypass_collector.py – Nitter scraper
------------------------------------------------
Fallback collector that avoids both the Twitter API **and** snscrape by
scraping public **Nitter** front‑ends.  Updated to skip down servers and
rotate through a pool until a healthy instance responds.

Dependencies
------------
    pip install requests beautifulsoup4 html5lib

Public interface identical to the original TweetCollector so Analyzer and
GUI need no change.

CSV schema: created_at, username, text, tweet_id
"""
from __future__ import annotations

import threading, time, csv, os, random
from datetime import datetime
from typing import Optional
import urllib.parse as ul
import requests
import re
from bs4 import BeautifulSoup

# ---- Pool of known‑good Nitter mirrors (add/remove as needed) --------
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.cz",
    "https://nitter.poast.org",
    "https://nitter.moomoo.me",
    "https://xcancel.com",
    "https://nitter.privacyredirect.com",
    
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

class TweetCollector:
    """Scrape tweets from a Nitter instance every <cooldown> seconds."""

    def __init__(self, keyword: str, cooldown: int = 60):
        self.keyword = keyword
        self.cooldown = cooldown
        self.stop_event = threading.Event()
        self._next_request_time: Optional[float] = None
        self._ids_seen: set[str] = set()

        # CSV path
        os.makedirs("outputs", exist_ok=True)
        date_part = datetime.now().strftime("%Y%m%d")
        fname = keyword.lstrip("#").replace(" ", "_")
        self.csv_path = f"outputs/{fname}_{date_part}.csv"
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(["created_at", "username", "text", "tweet_id"])

    # ------------------------------------------------------------------
    def seconds_until_next_request(self) -> int:
        if self._next_request_time is None:
            return 0
        return max(0, int(self._next_request_time - time.time()))

    # ------------------------------------------------------------------
    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()

    # ------------------------------------------------------------------
    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self._scrape_once()
            except Exception as exc:
                print(f"[NitterCollector] Error: {exc}")
            self._next_request_time = time.time() + self.cooldown
            time.sleep(self.cooldown)

    # ------------------------------------------------------------------
    def _build_search_url(self, base: str) -> str:
        q = f"{self.keyword} lang:en"
        return f"{base}/search?f=tweets&q={ul.quote(q)}"

    def _scrape_once(self):
        """Try each Nitter mirror until one succeeds or all fail."""
        for base in random.sample(NITTER_INSTANCES, len(NITTER_INSTANCES)):
            url = self._build_search_url(base)
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15)
            except requests.RequestException as e:
                print(f"[NitterCollector] {base} unreachable: {e}")
                continue

            if resp.status_code == 404:
                print(f"[NitterCollector] {base} returned 404 – skipping instance")
                continue
            if resp.status_code != 200:
                print(f"[NitterCollector] HTTP {resp.status_code} from {base}")
                continue

            new_rows = self._parse_html(resp.text)
            if new_rows is None:
                # Structure changed or empty page, try next instance
                continue
            if new_rows:
                with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
                    csv.writer(f).writerows(new_rows)
                print(f"[NitterCollector] Saved {len(new_rows)} tweets → {self.csv_path}")
            else:
                print("[NitterCollector] No new tweets scraped in this batch.")
            return  # Success or no new tweets → exit loop

        # All instances failed
        raise RuntimeError("All Nitter mirrors failed (429/404/unreachable)")

    def _parse_html(self, html: str):
        soup = BeautifulSoup(html, "html5lib")
        tweets = soup.select("article.timeline-item")[:100]
        if not tweets:
            return None  # signal to try another mirror

        new_rows = []
        for art in tweets:
            try:
                tw_id = art["data-id"]
                if tw_id in self._ids_seen:
                    continue
                self._ids_seen.add(tw_id)

                username = art.select_one("a.username").text.lstrip("@")
                created = art.select_one("span.tweet-date > a")
                created_at = created["title"] if created else datetime.utcnow().isoformat()
                text = " ".join(t.strip() for t in art.select_one("div.tweet-content").stripped_strings)
                text = re.sub(r"https?://\S+", "", text).strip()

                new_rows.append([created_at, username, text, tw_id])
            except Exception:
                continue  # skip malformed block
        return new_rows

    # ------------------------------------------------------------------
    # Streaming (manual cadence) API for UI
    # ------------------------------------------------------------------
    def start_streaming(self, callback, query: str, batch_size: int = 5, interval_seconds: int = 60):
        """
        Periodically scrape up to `batch_size` tweets matching `query` every
        `interval_seconds` and invoke `callback(texts, timestamps)`.

        Also appends fetched tweets to this instance's CSV for continuity.
        """
        self.stop_event.clear()
        self._ids_seen.clear()
        self.cooldown = max(1, int(interval_seconds))
        self._stream_batch_size = max(1, int(batch_size))
        self.keyword = query
        self.thread = threading.Thread(target=self._stream_loop, args=(callback,), daemon=True)
        self.thread.start()

    def _stream_loop(self, callback):
        while not self.stop_event.is_set():
            try:
                texts, times, rows = self._stream_collect_once(self._stream_batch_size)
                if rows:
                    try:
                        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
                            csv.writer(f).writerows(rows)
                    except Exception:
                        pass
                if texts:
                    try:
                        callback(texts, times)
                    except Exception:
                        pass
            except Exception as exc:
                print(f"[NitterCollector] Streaming error: {exc}")
            self._next_request_time = time.time() + self.cooldown
            time.sleep(self.cooldown)

    def _stream_collect_once(self, batch_size: int):
        texts, times, rows = [], [], []
        for base in random.sample(NITTER_INSTANCES, len(NITTER_INSTANCES)):
            url = self._build_search_url(base)
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15)
                if resp.status_code != 200:
                    continue
                data = self._parse_html(resp.text)
                if not data:
                    continue
                for created_at, username, text, tw_id in data:
                    if tw_id in self._ids_seen:
                        continue
                    self._ids_seen.add(tw_id)
                    txt = (text or "").replace("\n", " ")
                    txt = re.sub(r"https?://\S+", "", txt).strip()
                    texts.append(txt)
                    times.append(created_at)
                    rows.append([created_at, username, txt, tw_id])
                    if len(texts) >= batch_size:
                        return texts, times, rows
            except Exception:
                continue
        return texts, times, rows
