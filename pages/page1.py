# Updated page1.py
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tweepy
import time
from tweepy import TooManyRequests
import threading

# Import the new shared sentiment analyzer
from addons.sentiment_analyzer import SentimentAnalyzer
try:
    from addons.reddit_client import (
        get_reddit_client,
        fetch_thread_items,
        RedditNotConfigured,
    )
except Exception:
    get_reddit_client = None
    fetch_thread_items = None
    RedditNotConfigured = Exception
from utils import theme

class Page1(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Initialize the shared sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()

        self.scrollable = ctk.CTkScrollableFrame(self)
        self.scrollable.pack(expand=True, fill="both", padx=10, pady=10)
        # Subtle border for separation
        try:
            self.scrollable.configure(border_width=1, border_color=theme.border_color())
        except Exception:
            pass

        # Initialize Tweepy Client
        self.twitter_client = self.initialize_twitter_client()

        label = ctk.CTkLabel(self.scrollable, text="Sentiment Analysis", font=("Arial", 24, "bold"))
        label.pack(pady=(10, 2))
        intro = ctk.CTkLabel(
            self.scrollable,
            text=(
                "Analyze a single tweet, free text, hashtags, or accounts. "
                "Choose an input type below, paste your content, and view the sentiment breakdown."
            ),
            font=("Arial", 14),
            wraplength=850,
            justify="left",
            text_color=theme.subtle_text_color(),
        )
        intro.pack(pady=(0, 12), padx=4, anchor="w")

        # Default to Text and limit options
        self.option_var = ctk.StringVar(value="Text")
        self.option_menu = ctk.CTkOptionMenu(
            self.scrollable,
            variable=self.option_var,
            values=["Text", "Tweet", "Reddit Thread"],
            command=self.update_description
        )
        self.option_menu.pack(pady=10)

        self.description_label = ctk.CTkLabel(
            self.scrollable,
            text="Enter your Text:",
            font=("Arial", 14),
            wraplength=850,
            justify="left",
            text_color=theme.subtle_text_color(),
        )
        self.description_label.pack(pady=5, padx=4, anchor="w")

        self.url_entry = ctk.CTkEntry(self.scrollable, placeholder_text="Enter your text here...")
        self.url_entry.pack(pady=10)

        self.error_label = ctk.CTkLabel(self.scrollable, text="", font=("Arial", 12), text_color="red")
        self.error_label.pack(pady=5)

        self.tweet_text_area = ctk.CTkTextbox(self.scrollable, width=400, height=100)
        self.tweet_text_area.configure(state="disabled")
        self.tweet_text_area.pack(pady=10)

        # Cache for tweet data to prevent redundant API calls
        self.tweet_cache = {}

        # Submit button with loading state support
        self.submit_button = ctk.CTkButton(
            self.scrollable, 
            text="Submit", 
            command=self.submit_action
        )
        self.submit_button.pack(pady=10)

        # Chart container
        self.canvas = None
        self.matplotlib_figure = None
        self.canvas_frame = ctk.CTkFrame(self.scrollable)
        self.canvas_frame.pack(pady=10)
        try:
            self.canvas_frame.configure(border_width=1, border_color=theme.border_color())
        except Exception:
            pass

        # Display initial chart with a short animation
        self.last_counts = [40, 30, 30]
        self._pie_anim_after = None
        self.animate_pie(self.last_counts)

    def initialize_twitter_client(self):
        """Initialize the Twitter API client with proper error handling."""
        try:
            bearer_token = "AAAAAAAAAAAAAAAAAAAAAONjxwEAAAAAj74TclHPqXhKgRmuSlsIRJSXF9g%3DdsDgbh7xAa0apGZjGtkfFWYKWVIZO0Hd2Y1Hi9uqXobjSrzFw1"
            return tweepy.Client(bearer_token=bearer_token)
        except Exception as e:
            print(f"Error initializing Twitter client: {e}")
            self.error_label.configure(
                text="Failed to initialize Twitter API connection. Some features may be limited."
            )
            return None

    def create_pie_chart(self, counts):
        """Create a professional donut pie chart with theme-aware styling."""
        sentiments = ["Neutral", "Negative", "Positive"]

        # Remember last counts
        try:
            self.last_counts = counts
        except Exception:
            pass

        # Theme-aware colors
        txt = theme.text_color()
        bg = theme.plot_bg()

        plt.rcParams.update({
            "text.color": txt,
            "axes.labelcolor": txt,
            "xtick.color": txt,
            "ytick.color": txt,
        })

        # Clean up previous figure if it exists
        if self.matplotlib_figure:
            plt.close(self.matplotlib_figure)

        # Create new square figure for cleaner donut
        self.matplotlib_figure, ax = plt.subplots(figsize=(5.2, 5.2))
        self.matplotlib_figure.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        # Define palette (neutral gray, red, green)
        palette = ["#6C757D", "#E74C3C", "#2ECC71"]

        # Small separation for readability
        explode = [0.02, 0.02, 0.02]

        def _fmt_pct(pct: float) -> str:
            return f"{pct:.1f}%" if pct >= 3 else ""

        wedges, texts, autotexts = ax.pie(
            counts,
            labels=sentiments,
            autopct=_fmt_pct,
            pctdistance=0.8,
            startangle=90,
            explode=explode,
            textprops={"color": txt, "fontsize": 11},
            colors=palette,
            wedgeprops={"linewidth": 1.0, "edgecolor": bg},
        )

        # Donut hole
        centre_circle = plt.Circle((0, 0), 0.55, fc=bg)
        ax.add_artist(centre_circle)
        ax.axis("equal")  # Equal aspect ratio for a perfect circle

        # Center label showing dominant sentiment
        try:
            labels_order = sentiments
            dominant_idx = int(max(range(len(counts)), key=lambda i: counts[i]))
            dominant = labels_order[dominant_idx]
            total = sum(counts) if sum(counts) else 1
            dom_pct = counts[dominant_idx] * 100.0 / total
            ax.text(0, 0, f"{dominant}\n{dom_pct:.0f}%", ha="center", va="center", fontsize=14, color=txt)
        except Exception:
            pass

        ax.set_title("Sentiment Distribution", color=txt, fontsize=14, pad=14)

        # Update canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(self.matplotlib_figure, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        plt.close(self.matplotlib_figure)

    def update_theme(self, mode):
        """Refresh widgets and chart when theme changes."""
        try:
            # Update borders to current theme
            self.scrollable.configure(border_width=1, border_color=theme.border_color())
            self.canvas_frame.configure(border_width=1, border_color=theme.border_color())
        except Exception:
            pass
        # Redraw chart with last known counts
        try:
            self.create_pie_chart(self.last_counts)
        except Exception:
            pass

    def update_description(self, selected_option):
        """Update UI based on the selected input type."""
        self.url_entry.delete(0, "end")
        if selected_option == "Text":
            self.description_label.configure(text="Enter your Text:")
            self.url_entry.configure(placeholder_text="Enter your text here...")
        elif selected_option == "Tweet":
            self.description_label.configure(text="Enter Tweet URL:")
            self.url_entry.configure(placeholder_text="Enter tweet URL here...")
        elif selected_option == "Reddit Thread":
            self.description_label.configure(text="Enter Reddit Thread URL or ID:")
            self.url_entry.configure(placeholder_text="Enter Reddit thread URL or base36 ID...")
        else:
            self.description_label.configure(text="Enter URL:")
            self.url_entry.configure(placeholder_text="Enter URL here...")

    def submit_action(self):
        """Process the submission with a loading indicator."""
        # Reset previous error messages
        self.error_label.configure(text="")
        
        # Get input type and content
        selected_option = self.option_var.get()
        user_input = self.url_entry.get().strip()
        
        # Validate input
        if not user_input:
            self.error_label.configure(text="Please enter text or a URL to analyze")
            return
            
        # Show loading state
        self.submit_button.configure(state="disabled", text="Processing...")
        
        # Clear previous results
        self.tweet_text_area.configure(state="normal")
        self.tweet_text_area.delete("1.0", "end")
        self.tweet_text_area.configure(state="disabled")
        
        # Process in background thread to keep UI responsive
        threading.Thread(
            target=self._process_submission,
            args=(selected_option, user_input),
            daemon=True
        ).start()

    def _process_submission(self, selected_option, user_input):
        """Process the submission in a background thread."""
        try:
            if selected_option == "Text":
                # Direct text analysis
                self._update_ui_with_text(user_input, user_input)
            elif selected_option == "Tweet":
                # Fetch and analyze tweet
                tweet_text = self.fetch_tweet_text(user_input)
                if tweet_text:
                    self._update_ui_with_text(tweet_text, user_input)
                else:
                    self.after(0, lambda: self.error_label.configure(
                        text="Unable to fetch tweet. Please check the URL or try again later."
                    ))
            elif selected_option == "Reddit Thread":
                self._process_reddit_thread(user_input)
            else:
                # Future implementation for other options
                self.after(0, lambda: self.error_label.configure(
                    text=f"{selected_option} analysis is not implemented yet"
                ))
        except Exception as e:
            error_message = f"Error processing request: {e}"
            self.after(0, lambda msg=error_message: self.error_label.configure(
                text=msg
            ))
        finally:
            # Reset button state
            self.after(0, lambda: self.submit_button.configure(
                state="normal", text="Submit"
            ))

    def _update_ui_with_text(self, text, source):
        """Update UI with analysis results."""
        # Update text display
        def update_text_area():
            self.tweet_text_area.configure(state="normal")
            self.tweet_text_area.delete("1.0", "end")
            self.tweet_text_area.insert("1.0", text)
            self.tweet_text_area.configure(state="disabled")
        
        # Perform sentiment analysis
        sentiment = self.sentiment_analyzer.analyze_text(text)
        
        # Get ordered values for visualization
        counts = self.sentiment_analyzer.get_ordered_sentiment_values(sentiment)
        
        # Update UI on main thread
        self.after(0, update_text_area)
        self.after(0, lambda: self.animate_pie(counts))

    def _process_reddit_thread(self, url_or_id: str):
        """Fetch a Reddit thread and update UI with aggregate sentiment."""
        try:
            if get_reddit_client is None or fetch_thread_items is None:
                raise RedditNotConfigured("Reddit client not available. Install praw and set env vars.")

            reddit = get_reddit_client()
            items = fetch_thread_items(reddit, url_or_id, include_submission=True, max_comments=None)
            if not items:
                self.after(0, lambda: self.error_label.configure(text="No content found in thread."))
                return

            pos = neu = neg = 0
            for it in items:
                s = self.sentiment_analyzer.analyze_text(it.get("text", ""))
                p_pos = s.get("Positive", 0.0)
                p_neu = s.get("Neutral", 0.0)
                p_neg = s.get("Negative", 0.0)
                if p_pos >= p_neu and p_pos >= p_neg:
                    pos += 1
                elif p_neg >= p_pos and p_neg >= p_neu:
                    neg += 1
                else:
                    neu += 1

            total = max(1, pos + neu + neg)
            pct_pos = round(pos * 100.0 / total, 1)
            pct_neu = round(neu * 100.0 / total, 1)
            pct_neg = round(neg * 100.0 / total, 1)

            # Order: [Neutral, Negative, Positive]
            counts = [pct_neu, pct_neg, pct_pos]

            summary_text = (
                f"Reddit thread analyzed ({total} items)\n"
                f"Positive: {pct_pos}% ({pos})\n"
                f"Neutral:  {pct_neu}% ({neu})\n"
                f"Negative: {pct_neg}% ({neg})\n"
            )

            def update_text_area():
                self.tweet_text_area.configure(state="normal")
                self.tweet_text_area.delete("1.0", "end")
                self.tweet_text_area.insert("1.0", summary_text)
                self.tweet_text_area.configure(state="disabled")

            self.after(0, update_text_area)
            self.after(0, lambda: self.animate_pie(counts))
        except RedditNotConfigured as e:
            msg = str(e)
            self.after(0, lambda m=msg: self.error_label.configure(text=m))
        except Exception as e:
            msg = f"Reddit error: {e}"
            self.after(0, lambda m=msg: self.error_label.configure(text=m))

    def fetch_tweet_text(self, tweet_url):
        """Fetch tweet text from URL with enhanced error handling and caching."""
        if not self.twitter_client:
            return None
            
        try:
            # Extract tweet ID from URL
            tweet_id = tweet_url.split("/")[-1]
            
            # Check cache first
            if tweet_id in self.tweet_cache:
                return self.tweet_cache[tweet_id]

            # Implement exponential backoff for API rate limits
            retry_delay = 5
            for attempt in range(3):
                try:
                    tweet = self.twitter_client.get_tweet(
                        tweet_id, 
                        tweet_fields=["text", "created_at", "author_id"]
                    )
                    
                    if not tweet or not tweet.data:
                        return None
                        
                    text = tweet.data["text"]
                    self.tweet_cache[tweet_id] = text
                    return text
                    
                except TooManyRequests:
                    if attempt < 2:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise Exception("Twitter API rate limit exceeded. Please try again later.")
                except Exception as e:
                    if "Not Found" in str(e):
                        raise Exception("Tweet not found. Please check the URL.")
                    raise
                    
            return None
            
        except Exception as e:
            print(f"Error fetching tweet: {e}")
            raise
        
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
            # Cancel pie animation
            if hasattr(self, '_pie_anim_after') and self._pie_anim_after:
                try:
                    self.after_cancel(self._pie_anim_after)
                except Exception:
                    pass
                self._pie_anim_after = None
                
            # Mark as not running
            self.running = False
        except Exception as e:
            print(f"Error canceling page tasks: {e}")

    def destroy(self):
        """Clean up resources when page is destroyed."""
        try:
            # Cancel any pending animation callbacks
            if hasattr(self, '_pie_anim_after') and self._pie_anim_after:
                try:
                    self.after_cancel(self._pie_anim_after)
                except Exception:
                    pass
                self._pie_anim_after = None

            # Destroy canvas widget if present
            if self.canvas:
                try:
                    self.canvas.get_tk_widget().destroy()
                except Exception:
                    pass
                self.canvas = None

            # Close any lingering matplotlib figure
            if getattr(self, 'matplotlib_figure', None):
                try:
                    plt.close(self.matplotlib_figure)
                except Exception:
                    pass
                self.matplotlib_figure = None
        finally:
            # Ensure base class resources are released
            try:
                super().destroy()
            except Exception:
                pass

    def animate_pie(self, target_counts, duration_ms: int = 700, steps: int = 20):
        """Animate the pie chart from current counts to target counts."""
        try:
            # Cancel any existing animation
            if hasattr(self, '_pie_anim_after') and self._pie_anim_after:
                try:
                    self.after_cancel(self._pie_anim_after)
                except Exception:
                    pass
                self._pie_anim_after = None

            start = getattr(self, 'last_counts', [0, 0, 0])
            # Guard against size mismatch
            if len(start) != len(target_counts):
                start = [0] * len(target_counts)

            step_time = max(int(duration_ms / max(steps, 1)), 1)
            state = {"i": 0}

            def ease_out(t: float) -> float:
                # Smooth easing for nicer feel
                return 1 - (1 - t) * (1 - t)

            def tick():
                i = state["i"]
                t = ease_out(i / float(steps)) if steps > 0 else 1.0
                interp = [s + (e - s) * t for s, e in zip(start, target_counts)]
                try:
                    self.create_pie_chart(interp)
                except Exception:
                    pass
                if i < steps:
                    state["i"] = i + 1
                    self._pie_anim_after = self.after(step_time, tick)
                else:
                    # Snap to final and store
                    try:
                        self.create_pie_chart(target_counts)
                    except Exception:
                        pass
                    self.last_counts = target_counts
                    self._pie_anim_after = None

            tick()
        except Exception:
            # Fallback: render final state
            try:
                self.create_pie_chart(target_counts)
            except Exception:
                pass
            self.canvas = None
