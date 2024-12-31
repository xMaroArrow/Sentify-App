import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class Page1(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        # Load Hugging Face model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

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
        self.description_label = ctk.CTkLabel(self, text="Enter URL:", font=("Arial", 16))
        self.description_label.pack(pady=5)

        # Entry Box
        self.url_entry = ctk.CTkEntry(self, placeholder_text="Enter URL here...")
        self.url_entry.pack(pady=10)

        # Submit Button
        submit_button = ctk.CTkButton(self, text="Submit", command=self.submit_action)
        submit_button.pack(pady=10)

        # Create the initial pie chart
        self.canvas = None
        self.matplotlib_figure = None
        self.create_pie_chart([40, 30, 30])  # Initial chart: Neutral, Negative, Positive

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
        if selected_option == "Text":
            self.description_label.configure(text="Enter your Text:")
            self.url_entry.configure(placeholder_text="Enter your text here...")
        else:
            self.description_label.configure(text="Enter URL:")
            self.url_entry.configure(placeholder_text="Enter URL here...")

    def submit_action(self):
        """Handle the submit action."""
        selected_option = self.option_var.get()
        user_input = self.url_entry.get()

        if selected_option == "Text":
            counts = self.analyze_sentiment(user_input)
            self.create_pie_chart(counts)  # Update the pie chart
        else:
            print(f"Selected Option: {selected_option}")
            print(f"Entered Input: {user_input}")

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
