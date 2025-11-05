# import libraries
import customtkinter as ctk
from utils import theme
from utils.config_manager import ConfigManager
from pages.page1 import Page1  
from pages.home_page import HomePage
from pages.page2 import Page2  
from pages.page3 import Page3  
from pages.page4 import Page4
from pages.page5 import Page5
from pages.page6 import Page6
from pages.page_finetune import PageFineTune
from pages.settings_page import SettingsPage

class MyApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure the main window
        self.title("Sentiment Analysis for Social Media")
        self.geometry("1150x850")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Sidebar and content
        self.sidebar_open = False
        self.burger_button = ctk.CTkButton(self, text="☰", width=50, command=self.toggle_sidebar)
        self.burger_button.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        # Settings toggle button just below the burger
        self.settings_button = ctk.CTkButton(self, text="Settings", width=80, command=self.toggle_settings_panel)
        self.settings_button.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nw")

        # Settings panel (initially hidden)
        self.settings_panel = ctk.CTkFrame(self)
        self.settings_panel.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nw")
        self.settings_panel.grid_remove()

        settings_title = ctk.CTkLabel(self.settings_panel, text="Settings", font=("Arial", 13, "bold"))
        settings_title.pack(anchor="w", pady=(8, 4), padx=8)

        # Light mode switch
        self.light_mode_var = ctk.BooleanVar(value=False)
        self.light_mode_switch = ctk.CTkSwitch(
            self.settings_panel,
            text="Light mode",
            variable=self.light_mode_var,
            command=self.toggle_light_mode,
        )
        self.light_mode_switch.pack(anchor="w", padx=8, pady=(0, 8))

        # Header bar across the top (for a cleaner, modern layout)
        self.header = ctk.CTkFrame(self)
        self.header.grid(row=0, column=0, columnspan=3, sticky="new")
        self.header.grid_columnconfigure(1, weight=1)

        # Sidebar toggle in header
        self.header_burger = ctk.CTkButton(self.header, text="≡", width=40, command=self.toggle_sidebar)
        self.header_burger.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Title centered
        self.app_title = ctk.CTkLabel(self.header, text="Sentify", font=("Arial", 18, "bold"))
        self.app_title.grid(row=0, column=1, pady=10)

        # Theme switch on right, using the same variable
        self.header_light_switch = ctk.CTkSwitch(
            self.header,
            text="Light Mode",
            variable=self.light_mode_var,
            command=self.toggle_light_mode,
        )
        self.header_light_switch.grid(row=0, column=2, padx=10, pady=10, sticky="e")

        # Hide the old standalone controls (we use the header now)
        try:
            self.burger_button.grid_remove()
            self.settings_button.grid_remove()
            self.settings_panel.grid_remove()
        except Exception:
            pass

        self.sidebar_frame = ctk.CTkFrame(self, width=170)
        self.sidebar_frame.grid(row=1, column=0, sticky="nsw", padx=(10, 0), pady=(10, 10))
        self.sidebar_frame.grid_remove()  # Hide sidebar initially

        # Create config manager and add sidebar items
        self.config_manager = ConfigManager()
        # Add sidebar items
        self.add_sidebar_items()

        # Main content area
        self.container = ctk.CTkFrame(self)
        # Content under header in row 1, grows fully
        self.container.grid(row=1, column=1, sticky="nsew", padx=10, pady=(10, 10))
        # Grid weights: header fixed (row 0), content grows (row 1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        # Sidebar fixed width at col 0, content grows at col 1
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Initialize pages
        self.pages = {}
        self.initialize_pages()

        # Show the first page
        self.show_page("Home")

        # Track global `after` tasks
        self.global_after_tasks = []
        
        # Create a dictionary to track all after IDs in the application
        self.all_after_ids = {}

        # Protocol for safe closing
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Override the after method to track all after calls
        self._original_after = self.after
        self.after = self._tracked_after

        # Apply initial theming for surfaces and borders
        self.apply_theme()

    def toggle_sidebar(self):
        """Toggle the visibility of the sidebar."""
        if self.sidebar_open:
            self.sidebar_frame.grid_remove()  # Hide sidebar
            self.sidebar_open = False
        else:
            self.sidebar_frame.grid(row=1, column=0, sticky="nsw", padx=(10, 0), pady=(10, 10))
            self.sidebar_open = True

    def add_sidebar_items(self):
        """Add items to the sidebar."""
        menu_label = ctk.CTkLabel(self.sidebar_frame, text="Menu", font=("Arial", 14))
        menu_label.pack(pady=10)

        btn_kwargs = {"height": 28, "font": ("Arial", 12)}

        home_button = ctk.CTkButton(self.sidebar_frame, text="Home", command=lambda: self.show_page("Home"), **btn_kwargs)
        home_button.pack(pady=4, padx=10, fill="x")

        page1_button = ctk.CTkButton(self.sidebar_frame, text="Analyze", command=lambda: self.show_page("Page1"), **btn_kwargs)
        page1_button.pack(pady=4, padx=10, fill="x")

        page2_button = ctk.CTkButton(self.sidebar_frame, text="Trends", command=lambda: self.show_page("Page2"), **btn_kwargs)
        page2_button.pack(pady=4, padx=10, fill="x")

        page3_button = ctk.CTkButton(self.sidebar_frame, text="Clipboard", command=lambda: self.show_page("Page3"), **btn_kwargs)  
        page3_button.pack(pady=4, padx=10, fill="x")
        
        page4_button = ctk.CTkButton(self.sidebar_frame, text="Multilingual", command=lambda: self.show_page("Page4"), **btn_kwargs)  
        page4_button.pack(pady=4, padx=10, fill="x")
        
        page5_button = ctk.CTkButton(self.sidebar_frame, text="Training", command=lambda: self.show_page("Page5"), **btn_kwargs)  
        page5_button.pack(pady=4, padx=10, fill="x")
        
        # Page6 intentionally hidden from burger menu
        
        # Separator and Settings entry
        spacer = ctk.CTkLabel(self.sidebar_frame, text="")
        spacer.pack(pady=(8, 0))
        settings_button = ctk.CTkButton(self.sidebar_frame, text="Settings", command=lambda: self.show_page("Settings"), **btn_kwargs)
        settings_button.pack(pady=4, padx=10, fill="x")


        finetune_button = ctk.CTkButton(self.sidebar_frame, text="Fine-Tune", command=lambda: self.show_page("FineTune"))
        finetune_button.pack(pady=5)
        

    def initialize_pages(self):
        """Initialize all pages."""
        self.pages["Home"] = HomePage(self.container)
        self.pages["Page1"] = Page1(self.container)
        self.pages["Page2"] = Page2(self.container)
        self.pages["Page3"] = Page3(self.container)  
        self.pages["Page4"] = Page4(self.container) 
        self.pages["Page5"] = Page5(self.container)
        self.pages["Page6"] = Page6(self.container)
        self.pages["FineTune"] = PageFineTune(self.container)
        
        # Hide all pages initially
        self.pages["Settings"] = SettingsPage(self.container, self.config_manager)
        
        # Hide all pages initially (only if already managed by grid)
        for page in self.pages.values():
            try:
                if hasattr(page, "winfo_manager") and page.winfo_manager() == "grid":
                    page.grid_remove()
            except Exception:
                # If a page isn't yet managed or was destroyed, skip safely
                pass

    def show_page(self, page_name):
        """Display the selected page."""
        # Hide only pages that exist and are currently managed by grid
        for page in self.pages.values():
            try:
                # Proactively cancel any scheduled tasks on the page before hiding
                if hasattr(page, 'cancel_page_tasks'):
                    try:
                        page.cancel_page_tasks()
                    except Exception:
                        pass
                if page.winfo_exists() and getattr(page, "winfo_manager", lambda: "")() == "grid":
                    page.grid_remove()
            except Exception:
                # If a page isn't yet managed or was destroyed, skip safely
                pass

        # Show the selected page if it exists
        selected = self.pages.get(page_name)
        if selected is not None and selected.winfo_exists():
            selected.grid(row=0, column=0, sticky="nsew")

    def toggle_settings_panel(self):
        """Show/Hide the small settings panel under the menu button."""
        if str(self.settings_panel.grid_info()) != "{}":
            # Currently visible -> hide it
            self.settings_panel.grid_remove()
        else:
            self.settings_panel.grid()

    def toggle_light_mode(self):
        """Toggle CustomTkinter appearance between light and dark and refresh UI."""
        if self.light_mode_var.get():
            ctk.set_appearance_mode("light")
        else:
            ctk.set_appearance_mode("dark")
        # Re-apply panel colors and borders
        self.apply_theme()
        # Ask pages to refresh any charts to update backgrounds
        for page in self.pages.values():
            if hasattr(page, "update_theme"):
                try:
                    page.update_theme(ctk.get_appearance_mode())
                except Exception:
                    pass

    def apply_theme(self):
        """Apply subtle light/dark colors and borders for separation."""
        try:
            bg = theme.surface_bg()
            panel = theme.panel_bg()
            border = theme.border_color()

            # Window background
            self.configure(fg_color=bg)

            # Panels
            try:
                self.header.configure(fg_color=panel, border_width=0)
            except Exception:
                pass
            try:
                self.sidebar_frame.configure(fg_color=panel, border_width=1, border_color=border)
            except Exception:
                pass
            try:
                self.container.configure(fg_color=panel, border_width=1, border_color=border)
            except Exception:
                pass

            # Buttons text color tweak
            try:
                self.settings_button.configure(text_color=theme.text_color())
            except Exception:
                pass
        except Exception:
            pass

    def _tracked_after(self, ms, func, *args, **kwargs):
        """Track all after calls for proper cleanup."""
        after_id = self._original_after(ms, func, *args, **kwargs)
        self.all_after_ids[after_id] = True
        return after_id


    def _tracked_after(self, ms, func, *args, **kwargs):
        """Track all after calls for proper cleanup."""
        after_id = self._original_after(ms, func, *args, **kwargs)
        self.all_after_ids[after_id] = True
        return after_id


    def schedule_task(self, func, delay):
        """Schedule a global task and track it."""
        task_id = self.after(delay, func)
        self.global_after_tasks.append(task_id)
        return task_id

    def cancel_all_tasks(self):
        """Cancel all global scheduled tasks."""
        # Cancel tracked global tasks
        for task_id in self.global_after_tasks:
            try:
                self.after_cancel(task_id)
                if task_id in self.all_after_ids:
                    del self.all_after_ids[task_id]
            except Exception:
                pass
        self.global_after_tasks = []
        
        # Cancel ALL remaining after callbacks
        for after_id in list(self.all_after_ids.keys()):
            try:
                self.after_cancel(after_id)
            except Exception:
                pass
        self.all_after_ids.clear()
        
        # Cancel any pending after calls at the Tcl level
        try:
            # This attempts to clear all "after" events at the Tcl interpreter level
            self.eval('after cancel [after info]')
        except Exception as e:
            print(f"Error clearing Tcl after events: {e}")

    def on_closing(self):
        """Handle closing of the application."""
        try:
            # First cancel all pending after callbacks
            self.cancel_all_tasks()
            
            # Ask each page to clean up
            for page_name, page in list(self.pages.items()):
                try:
                    if page is not None:
                        # Cancel any page-specific after callbacks
                        if hasattr(page, 'cancel_page_tasks'):
                            page.cancel_page_tasks()
                        page.destroy()
                    self.pages[page_name] = None
                except Exception as e:
                    print(f"Error destroying page {page_name}: {e}")
            
            # Final cleanup of Tcl after events
            try:
                self.eval('after cancel [after info]')
            except Exception:
                pass
                
            # Destroy the main window
            self.quit()
            self.destroy()
        except Exception as e:
            print(f"Error during application closing: {e}")
            # Force exit as a last resort
            import os
            os._exit(0)

# Run the application
if __name__ == "__main__":
    app = MyApp()
    app.mainloop()
