import customtkinter as ctk
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import shap

class Page2(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        # Page Label
        label = ctk.CTkLabel(self, text="Explainable Sentiment Analysis", font=("Arial", 20))
        label.pack(pady=10)

        # Load Hugging Face model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

        # Entry Box
        entry_label = ctk.CTkLabel(self, text="Enter your text or tweet:", font=("Arial", 16))
        entry_label.pack(pady=5)

        self.text_entry = ctk.CTkEntry(self, placeholder_text="Enter text here...", width=400)
        self.text_entry.pack(pady=10)

        # Submit Button
        submit_button = ctk.CTkButton(self, text="Analyze and Explain", command=self.submit_action)
        submit_button.pack(pady=10)

        # Explanation Display Area
        self.explanation_text_area = ctk.CTkTextbox(self, width=400, height=150)
        self.explanation_text_area.configure(state="disabled")  # Make it read-only
        self.explanation_text_area.pack(pady=10)

        # Placeholder for SHAP bar chart
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.pack(pady=10)
        self.shap_figure = None
        self.shap_canvas = None

    def submit_action(self):
        """Handle the submit action and perform explainable sentiment analysis."""
        user_input = self.text_entry.get().strip()
        if not user_input:
            self.display_explanation("Please enter some text.")
            return

        # Analyze sentiment and explain the result
        sentiment_scores, token_shap_map = self.analyze_and_explain(user_input)

        # Display the SHAP explanation and sentiment scores
        self.display_explanation(self.format_explanation(sentiment_scores, token_shap_map))

        # Display the SHAP bar chart
        self.plot_shap_values(token_shap_map)

    def analyze_and_explain(self, text):
        """Perform sentiment analysis and explain it using SHAP."""
        # Tokenize input and convert to tensor
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)

        # Calculate probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        sentiment_scores = {
            "Negative": probabilities[0].item() * 100,
            "Neutral": probabilities[1].item() * 100,
            "Positive": probabilities[2].item() * 100
        }

        # Define the SHAP explainer with a wrapper function
        def forward_with_preprocessing(token_ids):
            token_ids = torch.tensor(token_ids, dtype=torch.long).to(self.model.device)  # Fix: Convert to torch.long
            attention_mask = (token_ids != self.tokenizer.pad_token_id).long()  # Create attention mask
            inputs = {"input_ids": token_ids, "attention_mask": attention_mask}
            outputs = self.model(**inputs)
            return torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()

        # SHAP explainer for Hugging Face models
        explainer = shap.Explainer(forward_with_preprocessing, inputs["input_ids"].numpy())
        shap_values = explainer(inputs["input_ids"].numpy())[0].values.flatten()

        # Tokenize input text to get corresponding tokens
        input_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Aggregate SHAP values for subwords and skip special tokens
        token_shap_map = self.aggregate_shap_values(input_tokens, shap_values)

        return sentiment_scores, token_shap_map

    def aggregate_shap_values(self, tokens, shap_values):
        """Aggregate SHAP values by combining subwords and skipping special tokens."""
        token_shap_map = {}
        current_word = ""
        current_shap_value = 0.0

        for token, shap_value in zip(tokens, shap_values):
            # Skip special tokens
            if token in ["<s>", "</s>", "<pad>"]:
                continue

            # Handle subwords (identified by leading "Ġ" for RoBERTa or "##" for BERT)
            if token.startswith("Ġ") or current_word == "":
                # If a new word starts, save the previous one
                if current_word:
                    token_shap_map[current_word] = current_shap_value
                # Start a new word
                current_word = token.replace("Ġ", "")
                current_shap_value = shap_value
            else:
                # Continue appending to the current word
                current_word += token.replace("##", "")
                current_shap_value += shap_value

        # Save the last word
        if current_word:
            token_shap_map[current_word] = current_shap_value

        return token_shap_map

    def format_explanation(self, sentiment_scores, token_shap_map):
        """Format the sentiment scores and SHAP explanation for display."""
        explanation = "Sentiment Scores:\n"
        explanation += f"  Positive: {sentiment_scores['Positive']:.2f}%\n"
        explanation += f"  Neutral: {sentiment_scores['Neutral']:.2f}%\n"
        explanation += f"  Negative: {sentiment_scores['Negative']:.2f}%\n\n"

        explanation += "Most Important Words (by SHAP):\n"
        # Sort by absolute SHAP values and display the top 3
        important_words = sorted(token_shap_map.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for word, value in important_words:
            explanation += f"  {word} (Impact: {value:.2f})\n"

        return explanation

    def display_explanation(self, explanation):
        """Display the explanation in the text area."""
        self.explanation_text_area.configure(state="normal")  # Temporarily enable text area
        self.explanation_text_area.delete("1.0", "end")  # Clear existing text
        self.explanation_text_area.insert("1.0", explanation)  # Display the explanation
        self.explanation_text_area.configure(state="disabled")  # Make it read-only again

    def plot_shap_values(self, token_shap_map):
        """Visualize SHAP values as a bar chart embedded in the GUI."""
        # Close previous figure if it exists
        if self.shap_figure:
            plt.close(self.shap_figure)

        # Create new figure
        self.shap_figure = plt.figure(figsize=(8, 4))

        # Sort tokens by SHAP values and display the top 10
        sorted_tokens_shap = sorted(token_shap_map.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        tokens, shap_values = zip(*sorted_tokens_shap)

        plt.barh(tokens, shap_values, color='skyblue')
        plt.xlabel("SHAP Value")
        plt.title("Top 10 Word Contributions to Sentiment")
        plt.gca().invert_yaxis()  # Ensure the most important words are at the top
        plt.tight_layout()

        # Embed the plot in the GUI
        if self.shap_canvas:
            self.shap_canvas.get_tk_widget().destroy()  # Clear previous canvas if exists

        self.shap_canvas = FigureCanvasTkAgg(self.shap_figure, master=self.canvas_frame)
        self.shap_canvas.draw()
        self.shap_canvas.get_tk_widget().pack()
