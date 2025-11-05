"""
RedditCollector
----------------
Simple Reddit collector that fetches recent submissions matching a query
on a manual cadence and calls back with raw texts and timestamps.

Dependencies: praw (see addons/reddit_client.py). If credentials or praw
are missing, the collector raises RedditNotConfigured with guidance.

Public API:
    rc = RedditCollector(cooldown=60)
    rc.start_streaming(callback, query, batch_size=5, interval_seconds=60)
    rc.stop()
    rc.seconds_until_next_request()

Callback signature:
    callback(texts: list[str], timestamps: list[str])

Notes:
    - Uses subreddit("all").search(..., sort="new") each interval.
    - Tracks IDs to avoid duplicates across batches.
    - Only submissions are fetched (title + selftext). Comments search is
      not included here; integrate Pushshift or alternative if needed.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
import re
from typing import Callable, Optional, Set, List

try:
    from addons.reddit_client import get_reddit_client, RedditNotConfigured, fetch_thread_items
except Exception:
    # Lazy import errors will be raised on initialization below
    get_reddit_client = None  # type: ignore
    class RedditNotConfigured(Exception):  # type: ignore
        pass


class RedditCollector:
    def __init__(self, cooldown: int = 60):
        self.cooldown = int(max(1, cooldown))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._next_request_time: Optional[float] = None
        self._ids_seen: Set[str] = set()
        self._client = None

    # ------------------------------------------------------------------
    def seconds_until_next_request(self) -> int:
        if self._next_request_time is None:
            return 0
        return max(0, int(self._next_request_time - time.time()))

    # ------------------------------------------------------------------
    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    # ------------------------------------------------------------------
    def start_streaming(
        self,
        callback: Callable[[List[str], List[str]], None],
        query: str,
        batch_size: int = 5,
        interval_seconds: int = 60,
    ):
        """
        Periodically fetch up to batch_size new submissions matching query
        and invoke callback(texts, timestamps).

        Authentication:
          - Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment vars,
            or use app config via ConfigManager (see addons/reddit_client.py).
        """
        if get_reddit_client is None:
            raise RedditNotConfigured(
                "Missing addons.reddit_client; ensure praw is installed."
            )

        self._stop_event.clear()
        self.cooldown = int(max(1, interval_seconds))
        self._next_request_time = None
        self._thread = threading.Thread(
            target=self._loop,
            args=(callback, query, int(max(1, batch_size))),
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------
    def _ensure_client(self):
        if self._client is None:
            # This may raise RedditNotConfigured with helpful message
            self._client = get_reddit_client()
        return self._client

    def _loop(self, callback, query: str, batch_size: int):
        while not self._stop_event.is_set():
            try:
                texts, times = self._fetch_batch(query, batch_size)
                if texts:
                    try:
                        callback(texts, times)
                    except Exception:
                        pass
            except RedditNotConfigured as e:
                # Surface configuration issues quickly; stop further attempts
                print(f"[RedditCollector] Configuration error: {e}")
                break
            except Exception as e:
                print(f"[RedditCollector] Error: {e}")

            self._next_request_time = time.time() + self.cooldown
            time.sleep(self.cooldown)

    def _fetch_batch(self, query: str, batch_size: int):
        reddit = self._ensure_client()

        # If query looks like a Reddit URL, fetch that thread (submission + comments)
        if query.startswith("http://") or query.startswith("https://"):
            items = fetch_thread_items(reddit, query, include_submission=True, max_comments=None)
            results: List[tuple[str, str, str]] = []  # (id, text, ts)
            for it in items:
                iid = it.get("id")
                if not iid or iid in self._ids_seen:
                    continue
                self._ids_seen.add(iid)
                text = (it.get("text") or "").strip().replace("\n", " ")
                text = re.sub(r"https?://\S+", "", text).strip()
                if not text:
                    continue
                # We don’t receive timestamps from helper; use now for labeling
                ts = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
                results.append((iid, text, ts))
                if len(results) >= batch_size:
                    break
            texts = [t for _, t, _ in results]
            times = [ts for _, _, ts in results]
            return texts, times

        # Otherwise, treat as a keyword search across r/all
        sub = reddit.subreddit("all")
        results2 = []
        count = 0
        for s in sub.search(query, sort="new", limit=max(batch_size * 3, batch_size)):
            sid = getattr(s, "id", None)
            if not sid or sid in self._ids_seen:
                continue
            self._ids_seen.add(sid)
            title = getattr(s, "title", "") or ""
            body = getattr(s, "selftext", "") or ""
            text = f"{title}\n\n{body}".strip()
            if not text:
                continue
            created_utc = getattr(s, "created_utc", None)
            ts = (
                datetime.utcfromtimestamp(created_utc).isoformat(sep=" ", timespec="seconds")
                if created_utc
                else datetime.utcnow().isoformat(sep=" ", timespec="seconds")
            )
            clean = re.sub(r"https?://\S+", "", text.replace("\n", " ")).strip()
            results2.append((clean, ts))
            count += 1
            if count >= batch_size:
                break
        texts = [t for t, _ in results2]
        times = [ts for _, ts in results2]
        return texts, times
