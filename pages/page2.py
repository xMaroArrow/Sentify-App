import customtkinter as ctk

class Page2(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        label = ctk.CTkLabel(self, text="This is Page 2", font=("Arial", 20))
        label.pack(pady=20)
