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

        label = ctk.CTkLabel(self.scrollable, text="Sentiment Analysis Inputs", font=("Arial", 20))
        label.pack(pady=10)

        self.option_var = ctk.StringVar(value="Tweet")
        self.option_menu = ctk.CTkOptionMenu(
            self.scrollable,
            variable=self.option_var,
            values=["Tweet", "Text", "Hashtag", "Account"],
            command=self.update_description
        )
        self.option_menu.pack(pady=10)

        self.description_label = ctk.CTkLabel(self.scrollable, text="Enter Tweet URL:", font=("Arial", 16))
        self.description_label.pack(pady=5)

        self.url_entry = ctk.CTkEntry(self.scrollable, placeholder_text="Enter tweet URL here...")
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

        # Display initial chart
        self.last_counts = [40, 30, 30]
        self.create_pie_chart(self.last_counts)

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
        """Create a pie chart visualization from sentiment counts."""
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

        # Create new figure
        self.matplotlib_figure, ax = plt.subplots()
        self.matplotlib_figure.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        # Create pie chart
        ax.pie(
            counts,
            labels=sentiments,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": txt},
            colors=["#ffb300", "#e91e63", "#4caf50"]  # amber, pink, green
        )
        ax.set_title("Sentiment Analysis Results", color=txt)

        # Update canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(self.matplotlib_figure, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
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
        self.after(0, lambda: self.create_pie_chart(counts))

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
                
            # Mark as not running
            self.running = False
        except Exception as e:
            print(f"Error canceling page tasks: {e}")

    def destroy(self):
        """Clean up resources when page is destroyed."""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.matplotlib_figure:
            plt.close(self.matplotlib_figure)
            self.matplotlib_figure = None
        super().destroy()
