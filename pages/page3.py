import customtkinter as ctk
import pyperclip
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.rcParams['toolbar'] = 'None'
import warnings
warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread")
import matplotlib.pyplot as plt
import time
import keyboard
import threading
import pyautogui
import os

class Page3(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        # Create scrollable frame
        self.scrollable = ctk.CTkScrollableFrame(self, width=800, height=800)
        self.scrollable.pack(expand=True, fill="both", padx=10, pady=10)

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model.eval()

        # UI Elements
        title = ctk.CTkLabel(self.scrollable, text="Clipboard Sentiment Analyzer", font=("Arial", 20))
        title.pack(pady=10)

        self.status_label = ctk.CTkLabel(self.scrollable, text="Status: Waiting for input...", font=("Arial", 14))
        self.status_label.pack(pady=10)

        self.result_textbox = ctk.CTkTextbox(self.scrollable, width=600, height=300)
        self.result_textbox.configure(state="disabled")
        self.result_textbox.pack(pady=10)

        analyze_button = ctk.CTkButton(self.scrollable, text="Analyze Selected Text", command=self.analyze_selected)
        analyze_button.pack(pady=5)

        # Chart placeholder
        self.canvas_frame = ctk.CTkFrame(self.scrollable)
        self.canvas_frame.pack(pady=10)
        self.shap_figure = None
        self.shap_canvas = None

        # Start hotkey monitoring thread
        self.listen_to_hotkey()

    def get_selected_text(self):
        try:
            text = pyperclip.paste().strip()
            return text if text else None
        except Exception as e:
            return f"Error accessing clipboard: {e}"

    def analyze_selected(self):
        import tkinter as tk
        from tkinter import Toplevel, Label
        text = self.get_selected_text()
        if text:
            sentiment = self.analyze_sentiment(text)
            self.update_output(text, sentiment)
            self.plot_sentiment(sentiment)
            self.log_result(text, sentiment)
            # Show popup with highest sentiment
            if sentiment:
                highest = max(sentiment.items(), key=lambda x: x[1])
                popup = Toplevel(self)
                popup.overrideredirect(True)
                popup.attributes("-topmost", True)

                # Color based on sentiment
                colors = {
                    "Positive": "#28a745",
                    "Neutral": "#ffc107",
                    "Negative": "#dc3545"
                }
                bg_color = colors.get(highest[0], "#2B2B2B")
                popup.configure(bg=bg_color)
              
                x, y = pyautogui.position()
                popup.geometry(f"250x70+{x}+{y}")

                label = Label(
                    popup,
                    text=f"{highest[0]}: {highest[1]:.2f}%",
                font=("Helvetica", 16, "bold"),
                fg="black",
                bg=bg_color,
                padx=20,
                pady=15,
                bd=3,
                relief="ridge")
                label.pack(expand=True, fill="both")


                # Optional sound feedback
                try:
                    import winsound
                    if highest[0] == "Positive":
                        winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
                    elif highest[0] == "Neutral":
                        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                    elif highest[0] == "Negative":
                        winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
                except:
                    pass

                popup.after(3000, popup.destroy)
        else:
            self.update_output("No text found in clipboard.", None)

    def analyze_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        labels = ["Negative", "Neutral", "Positive"]
        sentiment = {label: float(probs[i]) * 100 for i, label in enumerate(labels)}
        return sentiment

    def update_output(self, text, sentiment):
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", "end")
        self.result_textbox.insert("end", f"Detected Text:\n{text}\n\n")

        if sentiment:
            self.result_textbox.insert("end", "Sentiment:\n")
            for k, v in sentiment.items():
                self.result_textbox.insert("end", f"  {k}: {v:.2f}%\n")
        self.result_textbox.configure(state="disabled")

        self.status_label.configure(text="Status: Updated at " + time.strftime("%H:%M:%S"))

    def plot_sentiment(self, sentiment):
        if self.shap_figure:
            plt.close(self.shap_figure)

        self.shap_figure, ax = plt.subplots(figsize=(4, 3))
        sentiments = ["Neutral", "Negative", "Positive"]
        values = [sentiment.get(s, 0) for s in sentiments]

        self.shap_figure.patch.set_facecolor("#2B2B2B")
        ax.set_facecolor("#2B2B2B")
        plt.rcParams.update({
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        })

        wedges, texts, autotexts = ax.pie(
            values,
            labels=sentiments,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": "white"}
        )

        ax.set_title("Sentiment Distribution", color="white")

        if self.shap_canvas:
            self.shap_canvas.get_tk_widget().destroy()

        self.shap_canvas = FigureCanvasTkAgg(self.shap_figure, master=self.canvas_frame)
        self.shap_canvas.draw()
        self.shap_canvas.get_tk_widget().pack()

        plt.close(self.shap_figure)

    def log_result(self, text, sentiment):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}]\n{text}\n"
        if sentiment:
            for k, v in sentiment.items():
                log_entry += f"  {k}: {v:.2f}%\n"
        log_entry += "\n"

        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = time.strftime("sentiment_log_%Y-%m-%d.txt")
        log_path = os.path.join(log_dir, log_filename)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def listen_to_hotkey(self):
        def hotkey_loop():
            while True:
                keyboard.wait("ctrl+c")
                time.sleep(0.2)      
                self.analyze_selected()
        threading.Thread(target=hotkey_loop, daemon=True).start()

    def destroy(self):
        super().destroy()
