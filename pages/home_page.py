import customtkinter as ctk

class HomePage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        # Add widgets for the Home Page
        label = ctk.CTkLabel(self, text="Welcome to the Sentiment Analysis Applaction!", font=("Arial", 32))
        label.pack(pady=20)
