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

        self.scrollable = ctk.CTkScrollableFrame(self, width=800, height=850)
        self.scrollable.pack(expand=True, fill="both", padx=10, pady=10)

        # Load Hugging Face model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

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

        self.tweet_cache = {}

        submit_button = ctk.CTkButton(self.scrollable, text="Submit", command=self.submit_action)
        submit_button.pack(pady=10)

        self.canvas = None
        self.matplotlib_figure = None
        self.canvas_frame = ctk.CTkFrame(self.scrollable)
        self.canvas_frame.pack(pady=10)
        self.create_pie_chart([40, 30, 30])

    def initialize_twitter_client(self):
        bearer_token = "AAAAAAAAAAAAAAAAAAAAAONjxwEAAAAAj74TclHPqXhKgRmuSlsIRJSXF9g%3DdsDgbh7xAa0apGZjGtkfFWYKWVIZO0Hd2Y1Hi9uqXobjSrzFw1"
        return tweepy.Client(bearer_token=bearer_token)

    def create_pie_chart(self, counts):
        sentiments = ["Neutral", "Negative", "Positive"]
        plt.rcParams.update({
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        })

        if self.matplotlib_figure:
            plt.close(self.matplotlib_figure)

        self.matplotlib_figure, ax = plt.subplots()
        self.matplotlib_figure.patch.set_facecolor("#2B2B2B")
        ax.set_facecolor("#2B2B2B")

        ax.pie(
            counts,
            labels=sentiments,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": "white"}
        )
        ax.set_title("Sentiment Analysis Results", color="white")

        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(self.matplotlib_figure, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        plt.close(self.matplotlib_figure)

    def update_description(self, selected_option):
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
        selected_option = self.option_var.get()
        user_input = self.url_entry.get()
        self.error_label.configure(text="")

        self.tweet_text_area.configure(state="normal")
        self.tweet_text_area.delete("1.0", "end")
        self.tweet_text_area.configure(state="disabled")

        if selected_option == "Text":
            counts = self.analyze_sentiment(user_input)
            self.tweet_text_area.configure(state="normal")
            self.tweet_text_area.insert("1.0", user_input)
            self.tweet_text_area.configure(state="disabled")
            self.create_pie_chart(counts)
        elif selected_option == "Tweet":
            tweet_text = self.fetch_tweet_text(user_input)
            if tweet_text:
                self.tweet_text_area.configure(state="normal")
                self.tweet_text_area.insert("1.0", tweet_text)
                self.tweet_text_area.configure(state="disabled")
                counts = self.analyze_sentiment(tweet_text)
                self.create_pie_chart(counts)
            else:
                self.error_label.configure(
                    text="Unable to fetch tweet. Please try again later or check the URL."
                )

    def fetch_tweet_text(self, tweet_url):
        try:
            tweet_id = tweet_url.split("/")[-1]
            if tweet_id in self.tweet_cache:
                return self.tweet_cache[tweet_id]

            retry_delay = 5
            for attempt in range(3):
                try:
                    tweet = self.twitter_client.get_tweet(tweet_id, tweet_fields=["text"])
                    self.tweet_cache[tweet_id] = tweet.data["text"]
                    return tweet.data["text"]
                except TooManyRequests:
                    if attempt < 2:
                        time.sleep(retry_delay)
                        retry_delay *= 2
            return None
        except Exception as e:
            print(f"Error fetching tweet: {e}")
            return None

    def analyze_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        return [probabilities[1].item() * 100, probabilities[0].item() * 100, probabilities[2].item() * 100]

    def destroy(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.matplotlib_figure:
            plt.close(self.matplotlib_figure)
            self.matplotlib_figure = None
        super().destroy()
