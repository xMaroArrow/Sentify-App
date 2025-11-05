import customtkinter as ctk
import webbrowser
from PIL import Image, ImageTk
import os
import platform
from datetime import datetime

from utils.config_manager import ConfigManager
from addons.sentiment_analyzer import SentimentAnalyzer
try:
    from utils import theme
except Exception:
    class theme:  # fallback colors
        @staticmethod
        def border_color():
            return "#3a3a3a"

        @staticmethod
        def subtle_text_color():
            return "#a0a0a0"


class HomePage(ctk.CTkFrame):
    """
    Feature-rich home page for a compelling first-run experience.

    Sections inside a scrollable frame:
    - Hero banner with CTA
    - Feature highlights
    - Live stats (metrics + model/system grid)
    - Navigation grid (Page1–Page6, Settings)
    - Quick Start checklist
    - Testimonials carousel (placeholder)
    - Footer (links + Release Notes)
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.config = ConfigManager()
        self._testimonials = [
            "“Sentify helped us understand our users at scale.”",
            "“Great visuals and quick insights for our team.”",
            "“Simple workflow from training to analysis.”",
        ]
        self._testimonial_idx = 0

        # Scrollable surface
        self.main_container = ctk.CTkScrollableFrame(self, width=950, height=800)
        self.main_container.pack(expand=True, fill="both", padx=10, pady=10)

        # Sections
        self._create_hero_banner()
        self._create_feature_highlights()
        self._create_live_stats()
        self._create_navigation_grid()
        self._create_quick_start()
        self._create_testimonials()
        self._create_footer()

    # -------------------- Hero --------------------
    def _create_hero_banner(self):
        header = ctk.CTkFrame(self.main_container)
        header.pack(fill="x", padx=10, pady=(0, 14))
        try:
            header.configure(border_width=1, border_color=theme.border_color())
        except Exception:
            pass

        title = ctk.CTkLabel(header, text="Sentify", font=("Arial", 42, "bold"))
        title.pack(pady=(16, 2))

        subtitle = ctk.CTkLabel(
            header,
            text="AI‑powered sentiment analysis for text and social media",
            font=("Arial", 16),
            text_color=theme.subtle_text_color(),
        )
        subtitle.pack(pady=(0, 10))

        # Optional logo
        try:
            logo_path = os.path.join("assets", "logo.png")
            if os.path.exists(logo_path):
                img = Image.open(logo_path).resize((96, 96))
                photo = ImageTk.PhotoImage(img)
                logo = ctk.CTkLabel(header, image=photo, text="")
                logo.image = photo
                logo.pack(pady=(0, 10))
        except Exception:
            pass

        # Call‑to‑action
        cta = ctk.CTkButton(header, text="Analyze Your First Text", width=220,
                            command=lambda: self._navigate_to_page("Page1"))
        cta.pack(pady=(4, 12))

    # -------------------- Feature highlights --------------------
    def _create_feature_highlights(self):
        block = ctk.CTkFrame(self.main_container)
        block.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(block, text="Highlights", font=("Arial", 20, "bold")).pack(
            pady=(12, 6), padx=12, anchor="w"
        )

        grid = ctk.CTkFrame(block, fg_color="transparent")
        grid.pack(fill="x", padx=12, pady=(0, 4))

        self._feature_card(grid, "Single Text Analysis",
                           "Analyze any text and visualize Positive/Neutral/Negative.", "Page1").pack(side="left", expand=True, fill="x", padx=6, pady=6)
        self._feature_card(grid, "Real‑time Monitoring",
                           "Track hashtags or topics and see trends.", "Page2").pack(side="left", expand=True, fill="x", padx=6, pady=6)
        self._feature_card(grid, "Clipboard Analyzer",
                           "Copy text anywhere, analyze with a hotkey.", "Page3").pack(side="left", expand=True, fill="x", padx=6, pady=6)

    def _feature_card(self, parent, title, desc, page=None):
        card = ctk.CTkFrame(parent)
        try:
            card.configure(border_width=1, border_color=theme.border_color())
        except Exception:
            pass
        # Accent bar
        bar = ctk.CTkFrame(card, width=6, height=80)
        bar.pack(side="left", fill="y")
        try:
            bar.configure(fg_color="#3B82F6")
        except Exception:
            pass
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(side="left", fill="both", expand=True, padx=10, pady=8)
        ctk.CTkLabel(body, text=title, font=("Arial", 16, "bold")).pack(anchor="w")
        ctk.CTkLabel(body, text=desc, font=("Arial", 13), wraplength=360, justify="left",
                     text_color=theme.subtle_text_color()).pack(anchor="w", pady=(2, 6))
        if page:
            ctk.CTkButton(body, text="Open", width=100, command=lambda: self._navigate_to_page(page)).pack(anchor="e")
        return card

    # -------------------- Live stats --------------------
    def _create_live_stats(self):
        block = ctk.CTkFrame(self.main_container)
        block.pack(fill="x", padx=10, pady=(0, 10))
        try:
            block.configure(border_width=1, border_color=theme.border_color())
        except Exception:
            pass

        ctk.CTkLabel(block, text="Live Stats", font=("Arial", 20, "bold")).pack(
            pady=(12, 6), padx=12, anchor="w"
        )

        # Metric cards
        metrics = ctk.CTkFrame(block, fg_color="transparent")
        metrics.pack(fill="x", padx=12, pady=(0, 8))

        total_models = self._count_available_models()
        analyses_today = int(self.config.get("analyses_today", 0) or 0)

        self._metric_card(metrics, "Models Available", str(total_models), "#3B82F6").pack(side="left", expand=True, fill="x", padx=6, pady=6)
        self._metric_card(metrics, "Analyses Today", str(analyses_today), "#10B981").pack(side="left", expand=True, fill="x", padx=6, pady=6)

        # Model + system grid
        grid = ctk.CTkFrame(block, fg_color="transparent")
        grid.pack(fill="x", padx=12, pady=(4, 10))
        try:
            grid.grid_columnconfigure(0, weight=1)
            grid.grid_columnconfigure(1, weight=2)
        except Exception:
            pass

        source = (self.config.get("model_source", "huggingface") or "").lower()
        model_value = self.config.get("model", "") if source == "huggingface" else self.config.get("local_model_path", "")
        local_type = self.config.get("local_model_type", "transformer") if source == "local" else ""

        analyzer = SentimentAnalyzer()
        active = analyzer.is_initialized()
        status_text = "Active" if active else "Inactive"

        self._kv_row(grid, 0, "Model Source", source.capitalize())
        self._kv_row(grid, 1, "Current Model", model_value or "—")
        if source == "local":
            self._kv_row(grid, 2, "Local Type", local_type)
        self._kv_row(grid, 3, "Status", status_text)
        self._kv_row(grid, 4, "Python", platform.python_version())
        self._kv_row(grid, 5, "System", platform.system())
        self._kv_row(grid, 6, "Time", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        ctk.CTkButton(grid, text="Change Model", width=140, command=lambda: self._navigate_to_page("Settings")).grid(row=0, column=1, sticky="e", padx=6, pady=6)

    # -------------------- Navigation grid --------------------
    def _create_navigation_grid(self):
        block = ctk.CTkFrame(self.main_container)
        block.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(block, text="Navigate", font=("Arial", 20, "bold")).pack(
            pady=(12, 6), padx=12, anchor="w"
        )

        grid = ctk.CTkFrame(block)
        grid.pack(fill="x", padx=12, pady=(0, 8))
        try:
            for col in range(3):
                grid.grid_columnconfigure(col, weight=1, uniform="nav")
        except Exception:
            pass

        items = [
            ("Page1", "Analyze Text"),
            ("Page2", "Real‑time"),
            ("Page3", "Clipboard"),
            ("Page4", "Experiments"),
            ("Page5", "Evaluation"),
            ("Page6", "Comparison"),
        ]
        # Add Settings if present
        try:
            main_app = self.winfo_toplevel()
            if hasattr(main_app, 'pages') and 'Settings' in getattr(main_app, 'pages', {}):
                items.append(("Settings", "Settings"))
        except Exception:
            items.append(("Settings", "Settings"))

        for idx, (page, label) in enumerate(items):
            r, c = divmod(idx, 3)
            cell = ctk.CTkFrame(grid)
            cell.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
            try:
                cell.configure(border_width=1, border_color=theme.border_color())
            except Exception:
                pass
            ctk.CTkLabel(cell, text=label, font=("Arial", 14, "bold")).pack(pady=(10, 4))
            ctk.CTkButton(cell, text="Open", command=lambda p=page: self._navigate_to_page(p)).pack(pady=(0, 10))

    # -------------------- Quick Start --------------------
    def _create_quick_start(self):
        block = ctk.CTkFrame(self.main_container)
        block.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(block, text="Getting Started", font=("Arial", 20, "bold")).pack(
            pady=(12, 6), padx=12, anchor="w"
        )

        steps = [
            "Open Settings and choose a model (Hugging Face or Local)",
            "Go to Analyze Text and paste a sample sentence",
            "Check the sentiment distribution and details",
            "Try Real‑time Monitoring with a hashtag",
            "Optional: Fine‑tune on your dataset",
        ]
        for i, step in enumerate(steps, 1):
            ctk.CTkLabel(block, text=f"{i}. {step}", font=("Arial", 13)).pack(padx=12, pady=2, anchor="w")

    # -------------------- Testimonials --------------------
    def _create_testimonials(self):
        block = ctk.CTkFrame(self.main_container)
        block.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(block, text="What People Say", font=("Arial", 20, "bold")).pack(
            pady=(12, 6), padx=12, anchor="w"
        )

        body = ctk.CTkFrame(block)
        body.pack(fill="x", padx=12, pady=6)
        try:
            body.configure(border_width=1, border_color=theme.border_color())
        except Exception:
            pass

        self.testimonial_label = ctk.CTkLabel(body, text=self._testimonials[self._testimonial_idx],
                                              font=("Arial", 14), wraplength=820, justify="left")
        self.testimonial_label.pack(padx=12, pady=10)

        controls = ctk.CTkFrame(body, fg_color="transparent")
        controls.pack(pady=(0, 10))
        ctk.CTkButton(controls, text="◀", width=40, command=self._prev_testimonial).pack(side="left", padx=4)
        self._dots_label = ctk.CTkLabel(controls, text=self._testimonial_dots(), font=("Arial", 12))
        self._dots_label.pack(side="left", padx=6)
        ctk.CTkButton(controls, text="▶", width=40, command=self._next_testimonial).pack(side="left", padx=4)

    def _prev_testimonial(self):
        self._testimonial_idx = (self._testimonial_idx - 1) % len(self._testimonials)
        self._refresh_testimonial()

    def _next_testimonial(self):
        self._testimonial_idx = (self._testimonial_idx + 1) % len(self._testimonials)
        self._refresh_testimonial()

    def _refresh_testimonial(self):
        try:
            self.testimonial_label.configure(text=self._testimonials[self._testimonial_idx])
            self._dots_label.configure(text=self._testimonial_dots())
        except Exception:
            pass

    def _testimonial_dots(self):
        dots = []
        for i in range(len(self._testimonials)):
            dots.append("●" if i == self._testimonial_idx else "○")
        return " ".join(dots)

    # -------------------- Footer --------------------
    def _create_footer(self):
        footer = ctk.CTkFrame(self.main_container)
        footer.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(footer, text="Links", font=("Arial", 18, "bold")).pack(pady=(12, 6), padx=12, anchor="w")

        row = ctk.CTkFrame(footer, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkButton(row, text="Documentation", width=160,
                      command=lambda: self._open_browser("https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment")).pack(side="left", padx=6, pady=6)
        ctk.CTkButton(row, text="GitHub", width=160,
                      command=lambda: self._open_browser("https://github.com/xMaroArrow/Sentify-App")).pack(side="left", padx=6, pady=6)
        ctk.CTkButton(row, text="Feedback", width=160,
                      command=lambda: self._open_browser("mailto:SentifyApp@gmail.com")).pack(side="left", padx=6, pady=6)
        ctk.CTkButton(row, text="Release Notes", width=160,
                      command=lambda: self._open_browser("https://github.com/xMaroArrow/Sentify-App/releases")).pack(side="left", padx=6, pady=6)

        credits = ctk.CTkLabel(
            footer,
            text="© 2025 Sentify — Master's Thesis, University of Bahrain",
            font=("Arial", 12),
            text_color=theme.subtle_text_color(),
        )
        credits.pack(pady=(2, 8))

    # -------------------- Helpers --------------------
    def _navigate_to_page(self, page_name):
        try:
            app = self.winfo_toplevel()
            if hasattr(app, 'show_page'):
                app.show_page(page_name)
            elif hasattr(app, 'pages') and page_name in app.pages:
                for p in app.pages.values():
                    p.grid_remove()
                app.pages[page_name].grid(row=0, column=0, sticky="nsew")
        except Exception as e:
            print(f"Navigation error: {e}")

    def _open_browser(self, url):
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Error opening URL: {e}")

    # helpers: metrics/grid
    def _metric_card(self, parent, title, value, color="#3B82F6"):
        card = ctk.CTkFrame(parent)
        try:
            card.configure(border_width=1, border_color=theme.border_color())
        except Exception:
            pass
        top = ctk.CTkFrame(card, fg_color=color, height=6)
        top.pack(fill="x")
        ctk.CTkLabel(card, text=title, font=("Arial", 12), text_color=theme.subtle_text_color()).pack(padx=10, pady=(8, 0), anchor="w")
        ctk.CTkLabel(card, text=value, font=("Arial", 20, "bold")).pack(padx=10, pady=(0, 8), anchor="w")
        return card

    def _kv_row(self, parent, r, key, value):
        ctk.CTkLabel(parent, text=key, font=("Arial", 12), text_color=theme.subtle_text_color()).grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ctk.CTkLabel(parent, text=value, font=("Arial", 13)).grid(row=r, column=1, sticky="w", padx=6, pady=4)

    def _count_available_models(self) -> int:
        base = self._models_base_dir()
        seen = set()
        if os.path.isdir(base):
            for root, dirs, files in os.walk(base):
                rel = os.path.relpath(root, base)
                rel_parts = [] if rel == '.' else rel.split(os.sep)
                if rel_parts and rel_parts[0].lower() == 'archive':
                    continue
                if 'config.json' in files or ('model.pt' in files and 'config.json' not in files):
                    seen.add(os.path.normpath(root))
        return len(seen)

    def _models_base_dir(self) -> str:
        try:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(root, 'models')
        except Exception:
            return os.path.abspath('models')

