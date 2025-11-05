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
            text="AI-Powered Sentiment Intelligence Platform",
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
            text="Effortlessly explore overviews, trends, insights, reports, and exports to understand how people feel.",
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
        welcome_title.pack(pady=(15, 4), padx=20, anchor="w")

        welcome_subtitle = ctk.CTkLabel(
            welcome_frame,
            text="Your command center for sentiment intelligence across every workflow.",
            font=("Arial", 15),
            text_color="#0078D7"
        )
        welcome_subtitle.pack(pady=(0, 12), padx=20, anchor="w")

        welcome_text = (
            "Sentify brings together real-time monitoring, granular insights, and polished reporting so your team "
            "can respond with confidence. Navigate through the sections below to review a quick overview, discover "
            "trends as they develop, unlock deeper insights, compile polished reports, and manage exports that keep "
            "stakeholders informed."
        )

        welcome_message = ctk.CTkLabel(
            welcome_frame,
            text=welcome_text,
            font=("Arial", 14),
            wraplength=850,
            justify="left"
        )
        welcome_message.pack(pady=(0, 18), padx=20)
        
    def _create_features(self):
        """Create feature highlights section."""
        features_frame = ctk.CTkFrame(self.main_container)
        features_frame.pack(fill="x", padx=10, pady=10)

        features_title = ctk.CTkLabel(
            features_frame,
            text="Discover What's Possible",
            font=("Arial", 24, "bold")
        )
        features_title.pack(pady=(15, 4), padx=20, anchor="w")

        features_subtitle = ctk.CTkLabel(
            features_frame,
            text="Each workspace is tailored to help you move from raw feedback to actionable intelligence.",
            font=("Arial", 14)
        )
        features_subtitle.pack(pady=(0, 12), padx=20, anchor="w")

        features_grid = ctk.CTkFrame(features_frame, fg_color="transparent")
        features_grid.pack(fill="x", padx=20, pady=(0, 15))

        feature_definitions = [
            {
                "title": "Overview",
                "description": "Summarize recent sentiment at a glance and monitor highlights the moment you log in.",
                "page": "Page1",
                "cta": "Open Overview"
            },
            {
                "title": "Trends",
                "description": "Visualize evolving conversations, spikes in activity, and shifts in audience emotion.",
                "page": "Page2",
                "cta": "View Trends"
            },
            {
                "title": "Insights",
                "description": "Dive deeper into topics, keywords, and drivers influencing positive or negative reactions.",
                "page": "Page3",
                "cta": "Explore Insights"
            },
            {
                "title": "Reports",
                "description": "Build curated summaries with charts ready to share across your organization.",
                "page": "Page4",
                "cta": "Build Reports"
            },
            {
                "title": "Exports",
                "description": "Download datasets, tables, and visuals to integrate with your existing workflows.",
                "page": "Page5",
                "cta": "Manage Exports"
            }
        ]

        for index, feature in enumerate(feature_definitions):
            row, column = divmod(index, 3)
            card = self._create_feature_card(
                features_grid,
                feature["title"],
                feature["description"],
                feature.get("page"),
                feature.get("cta")
            )
            card.grid(row=row, column=column, padx=10, pady=10, sticky="nsew")

        for column in range(3):
            features_grid.grid_columnconfigure(column, weight=1)

    def _create_feature_card(self, parent, title, description, page_name=None, cta_label=None):
        """Create a feature card with title, description and optional navigation button."""
        card = ctk.CTkFrame(parent, corner_radius=12)

        card_title = ctk.CTkLabel(
            card,
            text=title,
            font=("Arial", 18, "bold")
        )
        card_title.pack(pady=(18, 6), padx=18, anchor="w")

        card_desc = ctk.CTkLabel(
            card,
            text=description,
            font=("Arial", 13),
            wraplength=320,
            justify="left"
        )
        card_desc.pack(pady=(0, 18), padx=18, anchor="w")

        if page_name:
            button_text = cta_label or f"Open {title}"
            card_button = ctk.CTkButton(
                card,
                text=button_text,
                command=lambda p=page_name: self._navigate_to_page(p),
                height=38,
                corner_radius=10,
                font=("Arial", 13, "bold"),
                fg_color="#1F6AA5",
                hover_color="#155a8a"
            )
            card_button.pack(pady=(0, 18), padx=18, anchor="e")

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
            text="Navigate the Workspace",
            font=("Arial", 24, "bold")
        )
        nav_title.pack(pady=(15, 4), padx=20, anchor="w")

        nav_subtitle = ctk.CTkLabel(
            nav_frame,
            text="Jump directly into the area you need with streamlined, consistent controls.",
            font=("Arial", 14)
        )
        nav_subtitle.pack(pady=(0, 12), padx=20, anchor="w")

        buttons_frame = ctk.CTkFrame(nav_frame, fg_color="transparent")
        buttons_frame.pack(fill="x", padx=20, pady=(0, 20))

        buttons_frame.grid_columnconfigure((0, 1, 2), weight=1)

        nav_items = [
            ("Overview", "Page1"),
            ("Trends", "Page2"),
            ("Insights", "Page3"),
            ("Reports", "Page4"),
            ("Exports", "Page5")
        ]

        if hasattr(self, 'settings_page_exists') and self.settings_page_exists:
            nav_items.append(("Settings", "Settings"))

        button_style = {
            "height": 48,
            "corner_radius": 12,
            "font": ("Arial", 15, "bold"),
            "fg_color": "#1F6AA5",
            "hover_color": "#155a8a"
        }

        for index, (label, page) in enumerate(nav_items):
            row, column = divmod(index, 3)
            button = ctk.CTkButton(
                buttons_frame,
                text=label,
                command=lambda p=page: self._navigate_to_page(p),
                **button_style
            )
            button.grid(row=row, column=column, padx=10, pady=10, sticky="ew")
    
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
            command=lambda: self._open_browser("https://github.com/xMaroArrow/Sentify-App"),
            width=150,
            height=30,
            font=("Arial", 12)
        )
        github_button.pack(side="left", padx=10, pady=10)
        
        # Feedback link
        feedback_button = ctk.CTkButton(
            links_frame,
            text="Provide Feedback",
            command=lambda: self._open_browser("mailto:SentifyApp@gmail.com"),
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

