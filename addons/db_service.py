# db_service.py
import sqlite3
import os
from datetime import datetime

class DatabaseService:
    def __init__(self, db_path="sentiment_analysis.db"):
        self.db_path = db_path
        self.initialize_db()
        
    def initialize_db(self):
        """Create tables if they don't exist."""
        if not os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tweets table
            cursor.execute('''
            CREATE TABLE tweets (
                id INTEGER PRIMARY KEY,
                tweet_id TEXT UNIQUE,
                created_at TIMESTAMP,
                username TEXT,
                text TEXT,
                positive_score REAL,
                neutral_score REAL,
                negative_score REAL,
                sentiment TEXT,
                hashtag TEXT,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create hashtag_trends table
            cursor.execute('''
            CREATE TABLE hashtag_trends (
                id INTEGER PRIMARY KEY,
                hashtag TEXT,
                timestamp TIMESTAMP,
                positive_count INTEGER,
                neutral_count INTEGER,
                negative_count INTEGER,
                total_count INTEGER
            )
            ''')
            
            conn.commit()
            conn.close()
    
    def save_tweet(self, tweet_data):
        """Save a single tweet analysis result."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR IGNORE INTO tweets (
            tweet_id, created_at, username, text, 
            positive_score, neutral_score, negative_score, sentiment, hashtag
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            tweet_data["tweet_id"],
            tweet_data["created_at"],
            tweet_data["username"],
            tweet_data["text"],
            tweet_data["scores"]["Positive"],
            tweet_data["scores"]["Neutral"],
            tweet_data["scores"]["Negative"],
            tweet_data["sentiment"],
            tweet_data.get("hashtag", None)
        ))
        
        conn.commit()
        conn.close()
    
    def save_trend_snapshot(self, hashtag, positive, neutral, negative):
        """Save a snapshot of sentiment trend for a hashtag."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total = positive + neutral + negative
        
        cursor.execute('''
        INSERT INTO hashtag_trends (
            hashtag, timestamp, positive_count, neutral_count, 
            negative_count, total_count
        ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (hashtag, timestamp, positive, neutral, negative, total))
        
        conn.commit()
        conn.close()
    
    def get_hashtag_trend(self, hashtag, days=7):
        """Get sentiment trend for a hashtag over time."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT timestamp, positive_count, neutral_count, negative_count, total_count
        FROM hashtag_trends
        WHERE hashtag = ? AND timestamp >= datetime('now', ?)
        ORDER BY timestamp
        ''', (hashtag, f'-{days} days'))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{
            "timestamp": row[0],
            "positive": row[1],
            "neutral": row[2],
            "negative": row[3],
            "total": row[4]
        } for row in results]