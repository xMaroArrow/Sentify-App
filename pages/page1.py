import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Page1(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        # Page Label
        label = ctk.CTkLabel(self, text="Sentiment Analysis Inputs", font=("Arial", 20))
        label.pack(pady=10)

        # Dropdown Menu
        self.option_var = ctk.StringVar(value="Tweet")
        self.option_menu = ctk.CTkOptionMenu(self, variable=self.option_var, values=["Tweet", "Text", "Hashtag", "Account"])
        self.option_menu.pack(pady=10)

        # Entry Box
        entry_label = ctk.CTkLabel(self, text="Enter URL:", font=("Arial", 16))
        entry_label.pack(pady=5)

        self.url_entry = ctk.CTkEntry(self, placeholder_text="Enter URL here...")
        self.url_entry.pack(pady=10)

        # Submit Button
        submit_button = ctk.CTkButton(self, text="Submit", command=self.submit_action)
        submit_button.pack(pady=10)

        # Create the pie chart
        self.canvas = None  # Track the Matplotlib canvas
        self.matplotlib_figure = None  # Track the Matplotlib figure
        self.create_pie_chart()
        
        plt.close()

    def create_pie_chart(self):
        """Create and embed a pie chart."""
        sentiments = ["Neutral", "Negative", "Positive"]
        counts = [40, 30, 30]

        # Apply dark theme to Matplotlib
        plt.rcParams.update({
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        })
        
        # Create a Matplotlib figure
        self.matplotlib_figure, ax = plt.subplots()
        
         # Customize the figure and axes for a dark theme
        self.matplotlib_figure.patch.set_facecolor("#2B2B2B")  # Dark background for the figure
        ax.set_facecolor("#2B2B2B")  # Dark background for the plot area
        
         # Create the pie chart
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=sentiments,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": "white"},  # White text for labels
        )
        
        # Customize title
        ax.set_title("Sentiment Analysis Results", color="white")
        

        # Embed the Matplotlib figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.matplotlib_figure, master=self)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(pady=10)
        

    def submit_action(self):
        """Handle the submit action."""
        print(f"Selected Option: {self.option_var.get()}")
        print(f"Entered URL: {self.url_entry.get()}")

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
