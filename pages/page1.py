import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tweepy
import time
from tweepy import TooManyRequests


class Page1(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        # Load Hugging Face model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

        # Initialize Tweepy Client
        self.twitter_client = self.initialize_twitter_client()

        # Page Label
        label = ctk.CTkLabel(self, text="Sentiment Analysis Inputs", font=("Arial", 20))
        label.pack(pady=10)

        # Dropdown Menu
        self.option_var = ctk.StringVar(value="Tweet")  # Default value
        self.option_menu = ctk.CTkOptionMenu(
            self,
            variable=self.option_var,
            values=["Tweet", "Text", "Hashtag", "Account"],
            command=self.update_description
        )
        self.option_menu.pack(pady=10)

        # Description Label
        self.description_label = ctk.CTkLabel(self, text="Enter Tweet URL:", font=("Arial", 16))
        self.description_label.pack(pady=5)

        # Entry Box
        self.url_entry = ctk.CTkEntry(self, placeholder_text="Enter tweet URL here...")
        self.url_entry.pack(pady=10)

        # Error Message Label
        self.error_label = ctk.CTkLabel(self, text="", font=("Arial", 12), text_color="red")
        self.error_label.pack(pady=5)

        # Tweet Display Area (Read-Only)
        self.tweet_text_area = ctk.CTkTextbox(self, width=400, height=100)
        self.tweet_text_area.configure(state="disabled")  # Make the textbox read-only
        self.tweet_text_area.pack(pady=10)
        
        # Add a cache for tweet data
        self.tweet_cache = {}

        # Submit Button
        submit_button = ctk.CTkButton(self, text="Submit", command=self.submit_action)
        submit_button.pack(pady=10)

        # Create the initial pie chart
        self.canvas = None
        self.matplotlib_figure = None
        self.create_pie_chart([40, 30, 30])  # Initial chart: Neutral, Negative, Positive

    def initialize_twitter_client(self):
        """Initialize the Twitter API v2 client."""
        bearer_token = "AAAAAAAAAAAAAAAAAAAAAONjxwEAAAAAj74TclHPqXhKgRmuSlsIRJSXF9g%3DdsDgbh7xAa0apGZjGtkfFWYKWVIZO0Hd2Y1Hi9uqXobjSrzFw1"
        return tweepy.Client(bearer_token=bearer_token)

    def create_pie_chart(self, counts):
        """Create and embed a pie chart."""
        sentiments = ["Neutral", "Negative", "Positive"]

        # Apply dark theme to Matplotlib
        plt.rcParams.update({
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        })

        # Create a new Matplotlib figure
        if self.matplotlib_figure:
            plt.close(self.matplotlib_figure)  # Close the old figure

        self.matplotlib_figure, ax = plt.subplots()
        self.matplotlib_figure.patch.set_facecolor("#2B2B2B")
        ax.set_facecolor("#2B2B2B")

        # Create the pie chart
        ax.pie(
            counts,
            labels=sentiments,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": "white"}
        )

        ax.set_title("Sentiment Analysis Results", color="white")

        # Embed the Matplotlib figure in Tkinter
        if self.canvas:
            self.canvas.get_tk_widget().destroy()  # Destroy old canvas

        self.canvas = FigureCanvasTkAgg(self.matplotlib_figure, master=self)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(pady=10)

        # Close the figure immediately after embedding
        plt.close(self.matplotlib_figure)

    def update_description(self, selected_option):
        """Update the description label and placeholder text based on the selected option."""
        # Reset the entry box content
        self.url_entry.delete(0, "end")  # Clear any existing text in the entry box

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
        """Handle the submit action."""
        selected_option = self.option_var.get()
        user_input = self.url_entry.get()
        self.error_label.configure(text="")  # Clear any previous error messages

        # Clear the tweet display area
        self.tweet_text_area.configure(state="normal")  # Temporarily enable text area
        self.tweet_text_area.delete("1.0", "end")  # Clear existing text
        self.tweet_text_area.configure(state="disabled")  # Disable again

        if selected_option == "Text":
            counts = self.analyze_sentiment(user_input)
            self.tweet_text_area.configure(state="normal")  # Temporarily enable
            self.tweet_text_area.insert("1.0", user_input)  # Show the input text
            self.tweet_text_area.configure(state="disabled")  # Disable again
            self.create_pie_chart(counts)  # Update the pie chart
        elif selected_option == "Tweet":
            tweet_text = self.fetch_tweet_text(user_input)
            if tweet_text:
                self.tweet_text_area.configure(state="normal")  # Temporarily enable
                self.tweet_text_area.insert("1.0", tweet_text)  # Show the fetched tweet
                self.tweet_text_area.configure(state="disabled")  # Disable again
                counts = self.analyze_sentiment(tweet_text)
                self.create_pie_chart(counts)  # Update the pie chart
            else:
                self.error_label.configure(
                    text="Unable to fetch tweet. Please try again later or check the URL."
                )
        else:
            print(f"Selected Option: {selected_option}")
            print(f"Entered Input: {user_input}")


    def fetch_tweet_text(self, tweet_url):
        """Fetch tweet text from a Tweet URL using Tweepy with caching."""
        try:
            # Extract the tweet ID from the URL
            tweet_id = tweet_url.split("/")[-1]

            # Check if the tweet is already in the cache
            if tweet_id in self.tweet_cache:
                print("Fetching tweet from cache.")
                return self.tweet_cache[tweet_id]

            # Retry logic with exponential backoff
            retry_delay = 5
            for attempt in range(3):  # Retry up to 3 times
                try:
                    tweet = self.twitter_client.get_tweet(tweet_id, tweet_fields=["text"])
                    self.tweet_cache[tweet_id] = tweet.data["text"]  # Cache the tweet text
                    return tweet.data["text"]
                except TooManyRequests:
                    if attempt < 2:  # If not the last attempt
                        print(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/3)")
                        time.sleep(retry_delay)  # Wait and retry
                        retry_delay *= 2  # Exponential backoff
            print("Too many requests. Try again later.")
            return None
        except Exception as e:
            print(f"Error fetching tweet: {e}")
            return None


    def analyze_sentiment(self, text):
        """Analyze sentiment using Hugging Face model."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

        neutral = probabilities[1].item() * 100
        negative = probabilities[0].item() * 100
        positive = probabilities[2].item() * 100

        return [neutral, negative, positive]  # Order matches the pie chart

    def destroy(self):
        """Clean up resources."""
        # Destroy the Matplotlib canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()  # Remove canvas from the GUI
            self.canvas = None

        # Close the Matplotlib figure
        if self.matplotlib_figure:
            plt.close(self.matplotlib_figure)
            self.matplotlib_figure = None

        super().destroy()
