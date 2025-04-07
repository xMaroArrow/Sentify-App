# import libraries
import customtkinter as ctk
from pages.page1 import Page1  
from pages.home_page import HomePage
from pages.page2 import Page2  
from pages.page3 import Page3  

class MyApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure the main window
        self.title("Sentiment Analysis for Social Media")
        self.geometry("1000x850")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Sidebar and content
        self.sidebar_open = False
        self.burger_button = ctk.CTkButton(self, text="☰", width=50, command=self.toggle_sidebar)
        self.burger_button.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        self.sidebar_frame = ctk.CTkFrame(self, width=150)
        self.sidebar_frame.grid(row=0, column=1, rowspan=5, sticky="ns")
        self.sidebar_frame.grid_remove()  # Hide sidebar initially

        # Add sidebar items
        self.add_sidebar_items()

        # Main content area
        self.container = ctk.CTkFrame(self)
        self.container.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(2, weight=1)

        # Initialize pages
        self.pages = {}
        self.initialize_pages()

        # Show the first page
        self.show_page("Home")

        # Track global `after` tasks
        self.global_after_tasks = []

        # Protocol for safe closing
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def toggle_sidebar(self):
        """Toggle the visibility of the sidebar."""
        if self.sidebar_open:
            self.sidebar_frame.grid_remove()  # Hide sidebar
            self.sidebar_open = False
        else:
            self.sidebar_frame.grid(row=0, column=1, rowspan=5, sticky="ns")
            self.sidebar_open = True

    def add_sidebar_items(self):
        """Add items to the sidebar."""
        menu_label = ctk.CTkLabel(self.sidebar_frame, text="Menu", font=("Arial", 14))
        menu_label.pack(pady=10)

        home_button = ctk.CTkButton(self.sidebar_frame, text="Home", command=lambda: self.show_page("Home"))
        home_button.pack(pady=5)

        page1_button = ctk.CTkButton(self.sidebar_frame, text="Page 1", command=lambda: self.show_page("Page1"))
        page1_button.pack(pady=5)

        page2_button = ctk.CTkButton(self.sidebar_frame, text="Page 2", command=lambda: self.show_page("Page2"))
        page2_button.pack(pady=5)

        page3_button = ctk.CTkButton(self.sidebar_frame, text="Page 3", command=lambda: self.show_page("Page3"))  
        page3_button.pack(pady=5)

    def initialize_pages(self):
        """Initialize all pages."""
        self.pages["Home"] = HomePage(self.container)
        self.pages["Page1"] = Page1(self.container)
        self.pages["Page2"] = Page2(self.container)
        self.pages["Page3"] = Page3(self.container)  

        # Hide all pages initially
        for page in self.pages.values():
            page.grid_remove()

    def show_page(self, page_name):
        """Display the selected page."""
        for page in self.pages.values():
            page.grid_remove()  # Hide all pages
        self.pages[page_name].grid(row=0, column=0, sticky="nsew")  # Show selected page

    def schedule_task(self, func, delay):
        """Schedule a global task and track it."""
        task_id = self.after(delay, func)
        self.global_after_tasks.append(task_id)

    def cancel_all_tasks(self):
        """Cancel all global scheduled tasks."""
        for task_id in self.global_after_tasks:
            try:
                self.after_cancel(task_id)
            except Exception:
                pass
        self.global_after_tasks = []

    def on_closing(self):
        """Handle closing of the application."""
        self.cancel_all_tasks()  # Cancel global tasks
        for page in self.pages.values():
            page.destroy()  # Ensure all pages are cleaned up
        self.destroy()

# Run the application
if __name__ == "__main__":
    app = MyApp()
    app.mainloop()
