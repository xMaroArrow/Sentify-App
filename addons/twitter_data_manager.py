# addons/twitter_data_manager.py
import os
from datetime import datetime, timedelta
import pandas as pd

# Try to import the Twitter collector
try:
    from addons.collector import TweetCollector  # Tweepy API version
except ImportError:
    try:
        from addons.bypass_collector import TweetCollector  # Nitter fallback
    except ImportError:
        TweetCollector = None

class TwitterDataManager:
    """
    Manages Twitter data collection and processing.
    Interfaces with the TweetCollector to fetch data and tracks historical results.
    """
    
    def __init__(self):
        self.collector = None
        self.csv_path = None
        self.last_processed_line = 0
        self.sentiment_data = []
        self.sentiment_by_language = {}
        self.sentiment_by_time = []
        
    def start_collection(self, keyword, cooldown=60):
        """Start collecting tweets for the given keyword."""
        if TweetCollector is None:
            return False
            
        try:
            self.collector = TweetCollector(keyword, cooldown=cooldown)
            self.collector.start()
            self.csv_path = self.collector.csv_path
            self.last_processed_line = 0
            return True
        except Exception as e:
            print(f"Error starting tweet collection: {e}")
            return False
            
    def stop_collection(self):
        """Stop the tweet collection process."""
        if self.collector:
            self.collector.stop()
            self.collector = None
            
    def seconds_until_next_request(self):
        """Get seconds until the next API request."""
        if self.collector:
            return self.collector.seconds_until_next_request()
        return 0
        
    def load_csv(self, file_path):
        """Load tweet data from a CSV file."""
        try:
            self.csv_path = file_path
            self.last_processed_line = 0
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
            
    def get_new_tweets(self):
        """Get new tweets that have been collected since the last check."""
        if not self.csv_path or not os.path.exists(self.csv_path):
            return []
            
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            # Skip header if this is the first read
            start_line = 1 if self.last_processed_line == 0 else self.last_processed_line
            new_lines = lines[start_line:]
            
            if not new_lines:
                return []
                
            # Update the last processed line count
            self.last_processed_line = len(lines)
            
            # Parse CSV lines and extract tweet text
            import csv
            from io import StringIO
            
            tweets = []
            for line in new_lines:
                try:
                    reader = csv.reader(StringIO(line))
                    for row in reader:
                        if len(row) >= 3:  # created_at, username, text, [id]
                            tweets.append({
                                "created_at": row[0],
                                "username": row[1],
                                "text": row[2],
                                "tweet_id": row[3] if len(row) > 3 else ""
                            })
                except Exception as e:
                    print(f"Error parsing tweet: {e}")
                    
            return tweets
            
        except Exception as e:
            print(f"Error reading tweets: {e}")
            return []
            
    def update_sentiment_data(self, analyzer, new_tweets, 
                              language_filter=None, confidence_threshold=50):
        """Update sentiment data with newly analyzed tweets."""
        if not new_tweets:
            return 0
            
        try:
            processed_count = 0
            
            for tweet in new_tweets:
                text = tweet["text"]
                created_at = tweet["created_at"]
                
                # Detect language
                language = analyzer.detect_language(text)
                
                # Apply language filter if specified
                if language_filter and language not in language_filter:
                    continue
                    
                # Analyze sentiment
                sentiment, confidence = analyzer.analyze_sentiment(text, return_confidence=True)
                
                # Check confidence threshold
                max_confidence = max(confidence.values())
                if max_confidence < confidence_threshold:
                    # Skip low-confidence results
                    continue
                    
                # Parse timestamp
                try:
                    timestamp = datetime.fromisoformat(created_at)
                except ValueError:
                    # Fall back to current time if parsing fails
                    timestamp = datetime.now()
                    
                # Store result
                result = {
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "language": language,
                    "timestamp": timestamp,
                    "username": tweet["username"],
                    "tweet_id": tweet["tweet_id"]
                }
                
                self.sentiment_data.append(result)
                processed_count += 1
                
                # Update language statistics
                if language not in self.sentiment_by_language:
                    self.sentiment_by_language[language] = {
                        "Positive": 0, "Neutral": 0, "Negative": 0, "total": 0
                    }
                    
                self.sentiment_by_language[language][sentiment] += 1
                self.sentiment_by_language[language]["total"] += 1
                
                # Store time-based data for trends
                self.sentiment_by_time.append({
                    "timestamp": timestamp, "sentiment": sentiment, "language": language
                })
                
            return processed_count
                
        except Exception as e:
            print(f"Error updating sentiment data: {e}")
            return 0
            
    def get_language_distribution(self):
        """Get the distribution of languages in the analyzed tweets."""
        if not self.sentiment_by_language:
            return pd.DataFrame()
            
        data = []
        for lang, counts in self.sentiment_by_language.items():
            data.append({
                "language": lang,
                "Positive": counts["Positive"],
                "Neutral": counts["Neutral"],
                "Negative": counts["Negative"],
                "total": counts["total"]
            })
            
        df = pd.DataFrame(data)
        return df.sort_values(by="total", ascending=False)
        
    def get_sentiment_trend(self, time_interval='1D', last_hours=24):
        """Get sentiment trends over time."""
        if not self.sentiment_by_time:
            return pd.DataFrame()
            
        # Create DataFrame with timezone handling
        data_with_fixed_times = []
        for item in self.sentiment_by_time:
            # Handle timezone-aware datetimes by converting to naive
            timestamp = item["timestamp"]
            if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                # Convert to naive datetime by removing timezone info
                timestamp = timestamp.replace(tzinfo=None)
                
            data_with_fixed_times.append({
                "timestamp": timestamp,
                "sentiment": item["sentiment"],
                "language": item["language"]
            })
        
        # Create DataFrame
        df = pd.DataFrame(data_with_fixed_times)
        
        if df.empty:
            return pd.DataFrame()
            
        # Filter for recent data
        cutoff_time = datetime.now() - timedelta(hours=last_hours)
        df = df[df["timestamp"] > cutoff_time]
        
        if df.empty:
            return pd.DataFrame()
            
        # Handle different time intervals
        try:
            # Group by time interval and sentiment
            df["time_bucket"] = df["timestamp"].dt.floor(time_interval)
            grouped = df.groupby(["time_bucket", "sentiment"]).size().unstack(fill_value=0)
        except Exception as e:
            print(f"Error grouping by time: {e}")
            # Fallback to daily grouping if there's an error
            df["time_bucket"] = df["timestamp"].dt.floor("1D")
            grouped = df.groupby(["time_bucket", "sentiment"]).size().unstack(fill_value=0)
        
        # Ensure all sentiment columns exist
        for sentiment in ["Positive", "Neutral", "Negative"]:
            if sentiment not in grouped.columns:
                grouped[sentiment] = 0
                
        return grouped
        
    def get_top_samples(self, sentiment, count=3):
        """Get top samples for a given sentiment based on confidence."""
        samples = []
        
        # Filter for the requested sentiment
        matching = [item for item in self.sentiment_data if item["sentiment"] == sentiment]
        
        # Sort by confidence
        matching.sort(key=lambda x: x["confidence"][sentiment], reverse=True)
        
        # Take the top N samples
        for item in matching[:count]:
            # Truncate text if too long
            text = item["text"]
            if len(text) > 100:
                text = text[:97] + "..."
                
            samples.append({
                "text": text,
                "language": item["language"],
                "confidence": item["confidence"][sentiment]
            })
            
        return samples
        
    def clear_data(self):
        """Clear all collected sentiment data."""
        self.sentiment_data = []
        self.sentiment_by_language = {}
        self.sentiment_by_time = []
        self.last_processed_line = 0