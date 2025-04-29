import customtkinter as ctk
import webbrowser
from PIL import Image, ImageTk
import os
import platform
from datetime import datetime

class HomePage(ctk.CTkFrame):
    """
    Enhanced home page for the Sentiment Analysis Application.
    
    This page serves as the entry point for users, providing:
    - Welcome information
    - Application overview and feature highlights
    - Quick navigation to key features
    - Basic instructions and getting started guide
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # Set up scrollable container
        self.main_container = ctk.CTkScrollableFrame(self, width=950, height=800)
        self.main_container.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Create sections
        self._create_header()
        self._create_welcome()
        self._create_features()
        self._create_navigation()
        self._create_system_info()
        self._create_footer()
        
    def _create_header(self):
        """Create the application header section."""
        header_frame = ctk.CTkFrame(self.main_container)
        header_frame.pack(fill="x", padx=10, pady=(0, 20))
        
        # Logo/Title section
        title = ctk.CTkLabel(
            header_frame, 
            text="Sentify", 
            font=("Arial", 48, "bold"),
            text_color="#0078D7"
        )
        title.pack(pady=(20, 0))
        
        subtitle = ctk.CTkLabel(
            header_frame, 
            text="AI-Powered Sentiment Analysis Platform",
            font=("Arial", 18)
        )
        subtitle.pack(pady=(0, 10))
        
        # Try to load app logo image if available
        try:
            # Adjust the path to where your logo is located
            logo_path = os.path.join("assets", "logo.png")
            if os.path.exists(logo_path):
                logo_image = Image.open(logo_path)
                logo_image = logo_image.resize((150, 150))
                logo_photo = ImageTk.PhotoImage(logo_image)
                
                logo_label = ctk.CTkLabel(header_frame, image=logo_photo, text="")
                logo_label.image = logo_photo  # Keep a reference
                logo_label.pack(pady=10)
        except Exception as e:
            print(f"Could not load logo: {e}")
            
        # Add a brief description
        description = ctk.CTkLabel(
            header_frame,
            text="Analyze sentiment in social media, text, and more.",
            font=("Arial", 14),
            wraplength=700
        )
        description.pack(pady=(0, 20))
        
    def _create_welcome(self):
        """Create welcome message with application overview."""
        welcome_frame = ctk.CTkFrame(self.main_container)
        welcome_frame.pack(fill="x", padx=10, pady=10)
        
        welcome_title = ctk.CTkLabel(
            welcome_frame,
            text="Welcome to Sentify",
            font=("Arial", 24, "bold")
        )
        welcome_title.pack(pady=(15, 10), padx=15, anchor="w")
        
        welcome_text = """
Sentify is a powerful sentiment analysis tool that helps you understand the emotional tone behind text content. 
Whether you're analyzing social media feedback, customer reviews, or market research, 
Sentify provides accurate sentiment classification and visualization.

This application uses state-of-the-art natural language processing to classify text as Positive, Neutral, or Negative, 
giving you valuable insights into how people feel about your products, services, or topics of interest.
        """
        
        welcome_message = ctk.CTkLabel(
            welcome_frame,
            text=welcome_text,
            font=("Arial", 14),
            wraplength=850,
            justify="left"
        )
        welcome_message.pack(pady=(0, 15), padx=15)
        
    def _create_features(self):
        """Create feature highlights section."""
        features_frame = ctk.CTkFrame(self.main_container)
        features_frame.pack(fill="x", padx=10, pady=10)
        
        features_title = ctk.CTkLabel(
            features_frame,
            text="Key Features",
            font=("Arial", 24, "bold")
        )
        features_title.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Create grid for features
        features_grid = ctk.CTkFrame(features_frame, fg_color="transparent")
        features_grid.pack(fill="x", padx=15, pady=(0, 15))
        
        # Feature 1: Single Text Analysis
        feature1_frame = self._create_feature_card(
            features_grid,
            "Single Text Analysis",
            "Analyze individual tweets, posts, or text segments with detailed sentiment breakdown.",
            "Page 1"
        )
        feature1_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Feature 2: Real-time Monitoring
        feature2_frame = self._create_feature_card(
            features_grid,
            "Real-time Monitoring",
            "Track sentiment trends in real-time for hashtags, keywords, or topics of interest.",
            "Page 2"
        )
        feature2_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Feature 3: Clipboard Analysis
        feature3_frame = self._create_feature_card(
            features_grid,
            "Clipboard Analysis",
            "Quickly analyze text from anywhere with clipboard integration and keyboard shortcuts.",
            "Page 3"
        )
        feature3_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # Feature 4: Visual Analytics
        feature4_frame = self._create_feature_card(
            features_grid,
            "Visual Analytics",
            "Understand sentiment data with intuitive charts, graphs, and visualizations.",
            None
        )
        feature4_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        # Configure grid columns to be equal width
        features_grid.grid_columnconfigure(0, weight=1)
        features_grid.grid_columnconfigure(1, weight=1)
        
    def _create_feature_card(self, parent, title, description, page_name=None):
        """Create a feature card with title, description and optional navigation button."""
        card = ctk.CTkFrame(parent)
        
        # Title
        card_title = ctk.CTkLabel(
            card,
            text=title,
            font=("Arial", 18, "bold")
        )
        card_title.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Description
        card_desc = ctk.CTkLabel(
            card,
            text=description,
            font=("Arial", 13),
            wraplength=350,
            justify="left"
        )
        card_desc.pack(pady=(0, 15), padx=15, anchor="w")
        
        # Add navigation button if page name is provided
        if page_name:
            card_button = ctk.CTkButton(
                card,
                text=f"Go to {page_name}",
                command=lambda p=page_name: self._navigate_to_page(p),
                width=150
            )
            card_button.pack(pady=(0, 15), padx=15, anchor="e")
        
        return card
    
    def _navigate_to_page(self, page_name):
        """Navigate to the specified page in the application."""
        # This function will be connected to the main app's navigation
        # You'll need to implement this in your main.py
        try:
            # Access the parent window
            main_app = self.winfo_toplevel()
            
            # If the main app has a show_page method, use it
            if hasattr(main_app, 'show_page'):
                main_app.show_page(page_name)
                
            # Alternative approach for more complex app structures
            elif hasattr(main_app, 'pages') and page_name in main_app.pages:
                for page in main_app.pages.values():
                    page.grid_remove()
                main_app.pages[page_name].grid(row=0, column=0, sticky="nsew")
        except Exception as e:
            print(f"Navigation error: {e}")
    
    def _create_navigation(self):
        """Create quick navigation section with buttons for all pages."""
        nav_frame = ctk.CTkFrame(self.main_container)
        nav_frame.pack(fill="x", padx=10, pady=10)
        
        nav_title = ctk.CTkLabel(
            nav_frame,
            text="Quick Navigation",
            font=("Arial", 24, "bold")
        )
        nav_title.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Create a row of navigation buttons
        buttons_frame = ctk.CTkFrame(nav_frame, fg_color="transparent")
        buttons_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        # Button for Page 1
        page1_button = ctk.CTkButton(
            buttons_frame,
            text="Single Text Analysis",
            command=lambda: self._navigate_to_page("Page1"),
            width=200,
            height=40,
            font=("Arial", 14)
        )
        page1_button.pack(side="left", padx=10, pady=10)
        
        # Button for Page 2
        page2_button = ctk.CTkButton(
            buttons_frame,
            text="Real-time Monitoring",
            command=lambda: self._navigate_to_page("Page2"),
            width=200,
            height=40,
            font=("Arial", 14)
        )
        page2_button.pack(side="left", padx=10, pady=10)
        
        # Button for Page 3
        page3_button = ctk.CTkButton(
            buttons_frame,
            text="Clipboard Analysis",
            command=lambda: self._navigate_to_page("Page3"),
            width=200,
            height=40,
            font=("Arial", 14)
        )
        page3_button.pack(side="left", padx=10, pady=10)
        
        # Add optional settings button if you have a settings page
        if hasattr(self, 'settings_page_exists') and self.settings_page_exists:
            settings_button = ctk.CTkButton(
                buttons_frame,
                text="Settings",
                command=lambda: self._navigate_to_page("Settings"),
                width=150,
                height=40,
                font=("Arial", 14)
            )
            settings_button.pack(side="left", padx=10, pady=10)
    
    def _create_system_info(self):
        """Create system information section."""
        sysinfo_frame = ctk.CTkFrame(self.main_container)
        sysinfo_frame.pack(fill="x", padx=10, pady=10)
        
        sysinfo_title = ctk.CTkLabel(
            sysinfo_frame,
            text="System Information",
            font=("Arial", 18, "bold")
        )
        sysinfo_title.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Get system information
        system = platform.system()
        python_version = platform.python_version()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Display system information
        info_text = f"""
• Application: Sentify - Sentiment Analysis Platform
• System: {system}
• Python: {python_version}
• Current Time: {current_time}
• Model: cardiffnlp/twitter-roberta-base-sentiment
        """
        
        sysinfo_label = ctk.CTkLabel(
            sysinfo_frame,
            text=info_text,
            font=("Arial", 13),
            justify="left"
        )
        sysinfo_label.pack(pady=(0, 15), padx=15, anchor="w")
    
    def _create_footer(self):
        """Create footer with credits and links."""
        footer_frame = ctk.CTkFrame(self.main_container)
        footer_frame.pack(fill="x", padx=10, pady=10)
        
        # Credits
        credits_text = "© 2025 Sentify - Developed as part of Master's Thesis at University of Bahrain"
        credits_label = ctk.CTkLabel(
            footer_frame,
            text=credits_text,
            font=("Arial", 12)
        )
        credits_label.pack(pady=(15, 5), padx=15)
        
        # Links frame
        links_frame = ctk.CTkFrame(footer_frame, fg_color="transparent")
        links_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        # Documentation link
        docs_button = ctk.CTkButton(
            links_frame,
            text="Documentation",
            command=lambda: self._open_browser("https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment"),
            width=150,
            height=30,
            font=("Arial", 12)
        )
        docs_button.pack(side="left", padx=10, pady=10)
        
        # GitHub link
        github_button = ctk.CTkButton(
            links_frame,
            text="View on GitHub",
            command=lambda: self._open_browser("https://github.com/yourusername/sentify"),
            width=150,
            height=30,
            font=("Arial", 12)
        )
        github_button.pack(side="left", padx=10, pady=10)
        
        # Feedback link
        feedback_button = ctk.CTkButton(
            links_frame,
            text="Provide Feedback",
            command=lambda: self._open_browser("mailto:youremail@example.com"),
            width=150,
            height=30,
            font=("Arial", 12)
        )
        feedback_button.pack(side="left", padx=10, pady=10)
    
    def _open_browser(self, url):
        """Open the specified URL in the default web browser."""
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Error opening URL: {e}")