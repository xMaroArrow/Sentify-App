import os
from typing import List, Dict, Any, Optional
from utils.config_manager import ConfigManager


class RedditNotConfigured(Exception):
    pass


def _require_praw():
    try:
        import praw  # type: ignore
        return praw
    except Exception as e:
        raise RedditNotConfigured(
            "Missing dependency 'praw'. Install with: pip install praw"
        ) from e


def get_reddit_client():
    """
    Create a Reddit client using environment variables:
      REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
    """
    praw = _require_praw()

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "Sentify-App/1.0")

    # Fallback to app config if env vars are not set
    if not client_id or not client_secret:
        cfg = ConfigManager()
        client_id = client_id or cfg.get("reddit_client_id")
        client_secret = client_secret or cfg.get("reddit_client_secret")
        user_agent = user_agent or cfg.get("reddit_user_agent", "Sentify-App/1.0")

    if not client_id or not client_secret:
        raise RedditNotConfigured(
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables."
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )


def fetch_thread_items(
    reddit,
    url_or_id: str,
    include_submission: bool = True,
    max_comments: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch the submission and comments for a Reddit thread.

    Returns a list of dicts with keys: type (submission|comment), id, author, text
    """
    if url_or_id.startswith("http"):
        submission = reddit.submission(url=url_or_id)
    else:
        submission = reddit.submission(id=url_or_id)

    # Flatten comments
    submission.comments.replace_more(limit=0)

    items: List[Dict[str, Any]] = []
    if include_submission:
        items.append(
            {
                "type": "submission",
                "id": submission.id,
                "author": str(submission.author) if submission.author else "[deleted]",
                "text": f"{submission.title}\n\n{submission.selftext or ''}".strip(),
            }
        )

    count = 0
    for c in submission.comments.list():
        if max_comments is not None and count >= max_comments:
            break
        body = getattr(c, "body", "") or ""
        if not body.strip():
            continue
        items.append(
            {
                "type": "comment",
                "id": c.id,
                "author": str(c.author) if c.author else "[deleted]",
                "text": body.strip(),
            }
        )
        count += 1

    return items
