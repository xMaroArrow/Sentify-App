import customtkinter as ctk
import pyperclip
import torch
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.rcParams['toolbar'] = 'None'
import warnings
warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread")
import time
import keyboard
import os
import tkinter as tk
from tkinter import Toplevel, Label, filedialog
import pyautogui

# Import the shared sentiment analyzer
from addons.sentiment_analyzer import SentimentAnalyzer
from utils import theme

class Page3(ctk.CTkFrame):
    """
    Clipboard Sentiment Analyzer - Monitors and analyzes text from the clipboard.
    
    Features:
    - Real-time sentiment analysis of clipboard text
    - Hotkey support (Ctrl+C) for analyzing selected text
    - Visualization of sentiment distribution
    - Detailed results display
    - Results logging to file
    - Desktop notifications for analysis results
    """
    
    def __init__(self, parent):
        super().__init__(parent)

        # Initialize the shared sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Create scrollable main frame
        self.scrollable = ctk.CTkScrollableFrame(self, width=850, height=800)
        self.scrollable.pack(expand=True, fill="both", padx=10, pady=10)
        try:
            self.scrollable.configure(border_width=1, border_color=theme.border_color())
        except Exception:
            pass

        # UI Elements - Headers
        self._create_header()
        
        # Result display area
        self._create_result_display()
        
        # Action buttons
        self._create_action_buttons()
        
        # Chart placeholder
        self._create_chart_area()
        
        # Settings section
        self._create_settings_area()
        
        # Initialize tracking variables
        self.clipboard_history = []
        self.last_analyzed_text = ""
        
        # Start hotkey monitoring thread
        self.listen_to_hotkey()

    def _create_header(self):
        """Create the header section of the UI."""
        # Title with improved styling
        title_frame = ctk.CTkFrame(self.scrollable)
        title_frame.pack(fill="x", pady=(0, 10))
        
        title = ctk.CTkLabel(
            title_frame, 
            text="Clipboard Sentiment Analyzer", 
            font=("Arial", 24, "bold")
        )
        title.pack(side="left", pady=10, padx=10)
        
        # Help button
        help_button = ctk.CTkButton(
            title_frame,
            text="?",
            width=30,
            height=30,
            command=self.show_help,
            fg_color="#444444",
            hover_color="#666666"
        )
        help_button.pack(side="right", padx=10, pady=10)
        
        # Description text
        desc = ctk.CTkLabel(
            self.scrollable,
            text="Analyze the sentiment of text copied to your clipboard. Press Ctrl+C to automatically analyze selected text.",
            font=("Arial", 14),
            wraplength=800,
            justify="left"
        )
        desc.pack(pady=(0, 10), fill="x")
        
        # Status indicator with better visual feedback
        status_frame = ctk.CTkFrame(self.scrollable)
        status_frame.pack(fill="x", pady=(0, 10))
        
        status_label = ctk.CTkLabel(
            status_frame, 
            text="Status:", 
            font=("Arial", 14)
        )
        status_label.pack(side="left", padx=10, pady=5)
        
        self.status_indicator = ctk.CTkLabel(
            status_frame,
            text="Waiting for input...",
            font=("Arial", 14),
            text_color="#FFA500"  # Orange for waiting state
        )
        self.status_indicator.pack(side="left", pady=5)
        
        # Timestamp display
        self.timestamp_label = ctk.CTkLabel(
            status_frame,
            text="",
            font=("Arial", 12),
            text_color="#888888"
        )
        self.timestamp_label.pack(side="right", padx=10, pady=5)

    def _create_result_display(self):
        """Create the text display area for analysis results."""
        # Text display with improved styling and scrollbars
        text_frame = ctk.CTkFrame(self.scrollable)
        text_frame.pack(pady=10, fill="both", expand=True)
        
        self.result_textbox = ctk.CTkTextbox(
            text_frame, 
            width=800, 
            height=250,
            font=("Arial", 13),
            wrap="word"
        )
        self.result_textbox.pack(padx=10, pady=10, fill="both", expand=True)
        self.result_textbox.configure(state="disabled")

    def _create_action_buttons(self):
        """Create the action buttons section."""
        # Button container with improved layout
        button_frame = ctk.CTkFrame(self.scrollable)
        button_frame.pack(fill="x", pady=10)
        
        # Primary action button
        analyze_button = ctk.CTkButton(
            button_frame, 
            text="Analyze Clipboard Text", 
            command=self.analyze_selected,
            height=40,
            font=("Arial", 14),
            fg_color="#0078D7",
            hover_color="#005A9E"
        )
        analyze_button.pack(side="left", padx=10, pady=10)
        
        # Secondary action buttons
        clear_button = ctk.CTkButton(
            button_frame,
            text="Clear Results",
            command=self.clear_results,
            height=40,
            font=("Arial", 14),
            fg_color="#555555",
            hover_color="#444444"
        )
        clear_button.pack(side="left", padx=10, pady=10)
        
        export_button = ctk.CTkButton(
            button_frame,
            text="Export Results",
            command=self.export_results,
            height=40,
            font=("Arial", 14),
            fg_color="#006400",
            hover_color="#004d00"
        )
        export_button.pack(side="left", padx=10, pady=10)

    def _create_chart_area(self):
        """Create the visualization area for sentiment charts."""
        # Chart container with fixed size to prevent resizing issues
        self.canvas_frame = ctk.CTkFrame(self.scrollable)
        self.canvas_frame.pack(pady=10, padx=10, fill="x")
        self.canvas_frame.configure(height=400)
        try:
            self.canvas_frame.configure(border_width=1, border_color=theme.border_color())
        except Exception:
            pass
        
        # Initialize chart placeholders
        self.sentiment_figure = None
        self.sentiment_canvas = None

    def _create_settings_area(self):
        """Create settings area for customization options."""
        settings_frame = ctk.CTkFrame(self.scrollable)
        settings_frame.pack(fill="x", pady=10)
        
        settings_label = ctk.CTkLabel(
            settings_frame,
            text="Settings",
            font=("Arial", 16, "bold")
        )
        settings_label.pack(pady=(10, 5))
        
        # Notification settings
        notification_frame = ctk.CTkFrame(settings_frame)
        notification_frame.pack(fill="x", padx=20, pady=5)
        
        notification_label = ctk.CTkLabel(
            notification_frame,
            text="Notification Options:",
            font=("Arial", 14)
        )
        notification_label.pack(side="left", padx=10, pady=5)
        
        self.show_notifications_var = tk.BooleanVar(value=True)
        notification_switch = ctk.CTkSwitch(
            notification_frame,
            text="Show Desktop Notifications",
            variable=self.show_notifications_var
        )
        notification_switch.pack(side="left", padx=10, pady=5)
        
        # Sound feedback settings
        sound_frame = ctk.CTkFrame(settings_frame)
        sound_frame.pack(fill="x", padx=20, pady=5)
        
        sound_label = ctk.CTkLabel(
            sound_frame,
            text="Sound Feedback:",
            font=("Arial", 14)
        )
        sound_label.pack(side="left", padx=10, pady=5)
        
        self.sound_feedback_var = tk.BooleanVar(value=True)
        sound_switch = ctk.CTkSwitch(
            sound_frame,
            text="Enable Sound Feedback",
            variable=self.sound_feedback_var
        )
        sound_switch.pack(side="left", padx=10, pady=5)
        
        # Hotkey settings
        hotkey_frame = ctk.CTkFrame(settings_frame)
        hotkey_frame.pack(fill="x", padx=20, pady=5)
        
        hotkey_label = ctk.CTkLabel(
            hotkey_frame,
            text="Hotkey:",
            font=("Arial", 14)
        )
        hotkey_label.pack(side="left", padx=10, pady=5)
        
        self.hotkey_var = tk.StringVar(value="ctrl+c")
        hotkey_options = ["ctrl+c", "ctrl+shift+a", "alt+s"]
        hotkey_dropdown = ctk.CTkOptionMenu(
            hotkey_frame,
            values=hotkey_options,
            variable=self.hotkey_var,
            command=self.update_hotkey
        )
        hotkey_dropdown.pack(side="left", padx=10, pady=5)
        
        # Auto-log settings
        log_frame = ctk.CTkFrame(settings_frame)
        log_frame.pack(fill="x", padx=20, pady=5)
        
        log_label = ctk.CTkLabel(
            log_frame,
            text="Auto-Logging:",
            font=("Arial", 14)
        )
        log_label.pack(side="left", padx=10, pady=5)
        
        self.auto_log_var = tk.BooleanVar(value=True)
        log_switch = ctk.CTkSwitch(
            log_frame,
            text="Save Results to Log File",
            variable=self.auto_log_var
        )
        log_switch.pack(side="left", padx=10, pady=5)

    def show_help(self):
        """Display help information in a popup window."""
        help_window = ctk.CTkToplevel(self)
        help_window.title("Clipboard Sentiment Analyzer - Help")
        help_window.geometry("600x400")
        help_window.grab_set()  # Make window modal
        
        # Create scrollable frame for help content
        help_frame = ctk.CTkScrollableFrame(help_window, width=580, height=380)
        help_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Help content
        ctk.CTkLabel(
            help_frame,
            text="Clipboard Sentiment Analyzer",
            font=("Arial", 20, "bold")
        ).pack(pady=(0, 10))
        
        help_text = """
This tool analyzes the sentiment of text from your clipboard.

How to use:
1. Copy text (Ctrl+C) from any application
2. The analyzer will automatically process the text
3. View the sentiment analysis results

Alternatively, you can:
- Click the "Analyze Clipboard Text" button to analyze the current clipboard content
- Use the "Clear Results" button to reset the display
- Use the "Export Results" button to save analysis results

Features:
- Real-time sentiment analysis
- Visual sentiment distribution
- Desktop notifications
- Automatic logging of results
- Customizable settings

Tips:
- For best results, analyze complete sentences or paragraphs
- The tool works with text from any application
- Results are automatically saved to logs/sentiment_log_YYYY-MM-DD.txt
        """
        
        ctk.CTkLabel(
            help_frame,
            text=help_text,
            font=("Arial", 14),
            justify="left",
            wraplength=560
        ).pack(pady=10)
        
        # Close button
        ctk.CTkButton(
            help_frame,
            text="Close",
            command=help_window.destroy,
            width=100
        ).pack(pady=10)

    def get_selected_text(self):
        """Get text from clipboard with improved error handling."""
        try:
            text = pyperclip.paste().strip()
            return text if text else None
        except Exception as e:
            self.update_status(f"Error accessing clipboard: {e}", "red")
            return None

    def update_status(self, message, color="#FFA500"):
        """Update the status indicator with the given message and color."""
        self.status_indicator.configure(text=message, text_color=color)
        self.timestamp_label.configure(text=time.strftime("%H:%M:%S"))

    def analyze_selected(self):
        """Analyze text from clipboard with enhanced visualization and feedback."""
        text = self.get_selected_text()
        
        if not text:
            self.update_status("No text found in clipboard", "#dc3545")  # Red for error
            self.update_output("No text found in clipboard.", None)
            return
            
        # Check if this is the same text we just analyzed (avoid duplicates)
        if text == self.last_analyzed_text:
            self.update_status("Same text already analyzed", "#FFA500")  # Orange for warning
            return
            
        self.last_analyzed_text = text
        self.clipboard_history.append(text)
        
        # Update status
        self.update_status("Analyzing text...", "#0078D7")  # Blue for processing
        
        # Perform analysis in a background thread to keep UI responsive
        threading.Thread(target=self._analyze_thread, args=(text,), daemon=True).start()

    def _analyze_thread(self, text):
        """Perform analysis in background thread to keep UI responsive."""
        try:
            # Get sentiment analysis results
            sentiment = self.sentiment_analyzer.analyze_text(text)
            
            # Update UI on main thread
            self.after(0, lambda: self.update_output(text, sentiment))
            self.after(0, lambda: self.plot_sentiment(sentiment))
            
            # Log results if enabled
            if self.auto_log_var.get():
                self.log_result(text, sentiment)
            
            # Show notification if enabled
            if self.show_notifications_var.get():
                self.after(0, lambda: self.show_notification(sentiment))
                
            # Update status to success
            self.after(0, lambda: self.update_status("Analysis complete", "#28a745"))  # Green for success
            
        except Exception as e:
            # Handle errors
            error_msg = str(e)
            self.after(0, lambda: self.update_status(f"Error: {error_msg}", "#dc3545"))
            print(f"Analysis error: {e}")

    def update_output(self, text, sentiment):
        """Update the results textbox with analyzed text and sentiment."""
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", "end")
        
        # Insert timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.result_textbox.insert("end", f"Analysis time: {timestamp}\n\n")
        
        # Insert detected text with character count
        char_count = len(text)
        word_count = len(text.split())
        self.result_textbox.insert("end", f"Detected Text ({word_count} words, {char_count} characters):\n")
        self.result_textbox.insert("end", f"{text}\n\n")

        # Insert sentiment results with improved formatting
        if sentiment:
            self.result_textbox.insert("end", "Sentiment Analysis:\n")
            
            # Find dominant sentiment
            dominant = max(sentiment.items(), key=lambda x: x[1])[0]
            
            # Display each sentiment score
            for k, v in sentiment.items():
                prefix = "➤ " if k == dominant else "  "
                self.result_textbox.insert("end", f"{prefix}{k}: {v:.2f}%\n")
                
            # Add interpretation
            self.result_textbox.insert("end", f"\nInterpretation: The text is predominantly {dominant.lower()}\n")
            
        self.result_textbox.configure(state="disabled")

    def plot_sentiment(self, sentiment):
        """Create enhanced pie chart visualization for sentiment distribution."""
        # Skip plotting if no sentiment information is available
        if not sentiment:
            self._display_empty_chart("No sentiment data available")
            return

        # Clean up previous figure
        if self.sentiment_figure:
            plt.close(self.sentiment_figure)

        # Create new figure with improved styling
        self.sentiment_figure, ax = plt.subplots(figsize=(5, 4))

        # Theme-aware background
        bg = theme.plot_bg()
        txt = theme.text_color()
        self.sentiment_figure.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        # Configure text colors
        plt.rcParams.update({
            "text.color": txt,
            "axes.labelcolor": txt,
            "xtick.color": txt,
            "ytick.color": txt,
        })

        # Prepare data for visualization and guard against invalid values
        sentiments = ["Neutral", "Negative", "Positive"]
        values = [max(sentiment.get(s, 0), 0) for s in sentiments]
        total = sum(values)

        if total <= 0:
            plt.close(self.sentiment_figure)
            self._display_empty_chart("No sentiment data available")
            return

        # Define colors with improved visual hierarchy
        colors = ["#ffb300", "#e91e63", "#4caf50"]  # amber, pink, green

        # Create enhanced pie chart
        wedges, texts, autotexts = ax.pie(
            values,
            labels=sentiments,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": txt, "fontweight": "bold"},
            colors=colors,
            wedgeprops={"edgecolor": bg, "linewidth": 1}
        )

        # Improve font size for better readability
        for autotext in autotexts:
            autotext.set_fontsize(9)

        # Add title with the dominant sentiment highlighted
        dominant = max(sentiment.items(), key=lambda x: x[1])[0]
        ax.set_title(f"Sentiment: {dominant}", color=txt, fontweight="bold")

        self.sentiment_figure.tight_layout()

        # Update canvas
        if self.sentiment_canvas:
            self.sentiment_canvas.get_tk_widget().destroy()

        self.sentiment_canvas = FigureCanvasTkAgg(self.sentiment_figure, master=self.canvas_frame)
        self.sentiment_canvas.draw()
        self.sentiment_canvas.get_tk_widget().pack(pady=10, expand=True)

    def _display_empty_chart(self, message: str):
        """Display a placeholder when there is no sentiment data to visualize."""
        if self.sentiment_figure:
            plt.close(self.sentiment_figure)

        self.sentiment_figure, ax = plt.subplots(figsize=(5, 4))
        bg = theme.plot_bg()
        txt = theme.text_color()
        self.sentiment_figure.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.axis("off")
        ax.text(0.5, 0.5, message, color=txt, ha="center", va="center", fontsize=12)
        self.sentiment_figure.tight_layout()
        try:
            self.sentiment_figure.subplots_adjust(top=0.88)
        except Exception:
            pass

        if self.sentiment_canvas:
            self.sentiment_canvas.get_tk_widget().destroy()

        self.sentiment_canvas = FigureCanvasTkAgg(self.sentiment_figure, master=self.canvas_frame)
        self.sentiment_canvas.draw()
        self.sentiment_canvas.get_tk_widget().pack(pady=10, expand=True)

    def update_theme(self, mode):
        """Refresh the chart to reflect current theme."""
        try:
            # If we have a displayed chart, re-run last plot operation by using the existing content
            if hasattr(self, 'last_analyzed_text') and self.last_analyzed_text:
                # Try to re-analyze to refresh chart colors without changing values
                sentiment = self.sentiment_analyzer.analyze_text(self.last_analyzed_text)
                self.plot_sentiment(sentiment)
            else:
                self._display_empty_chart("No sentiment data available")
        except Exception:
            pass

    def log_result(self, text, sentiment):
        """Log analysis results to a daily log file with improved formatting."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Create more structured log entry
        log_entry = f"[{timestamp}]\n"
        
        # Truncate text if too long for the log
        if len(text) > 500:
            log_entry += f"{text[:500]}...\n"
        else:
            log_entry += f"{text}\n"
            
        # Add sentiment information
        if sentiment:
            # Find dominant sentiment
            dominant = max(sentiment.items(), key=lambda x: x[1])[0]
            log_entry += f"Dominant sentiment: {dominant}\n"
            
            # Add detailed scores
            for k, v in sentiment.items():
                log_entry += f"  {k}: {v:.2f}%\n"
                
        log_entry += "\n"  # Add separator between entries

        # Ensure log directory exists
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create daily log file
        log_filename = time.strftime("sentiment_log_%Y-%m-%d.txt")
        log_path = os.path.join(log_dir, log_filename)
        
        # Append to log file
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def show_notification(self, sentiment):
        """Display desktop notification with sentiment results."""
        try:
            # Find the dominant sentiment
            highest = max(sentiment.items(), key=lambda x: x[1])
            
            # Create notification popup
            popup = Toplevel(self)
            popup.overrideredirect(True)  # Remove window decorations
            popup.attributes("-topmost", True)  # Keep on top

            # Set colors based on sentiment
            colors = {
                "Positive": "#28a745",  # Green
                "Neutral": "#ffc107",   # Amber
                "Negative": "#dc3545"   # Red
            }
            bg_color = colors.get(highest[0], "#2B2B2B")
            popup.configure(bg=bg_color)
            
            # Position near cursor
            x, y = pyautogui.position()
            popup.geometry(f"250x70+{x+10}+{y+10}")

            # Create notification content
            label = Label(
                popup,
                text=f"{highest[0]}: {highest[1]:.2f}%",
                font=("Helvetica", 16, "bold"),
                fg="white",
                bg=bg_color,
                padx=20,
                pady=15,
                bd=3,
                relief="ridge"
            )
            label.pack(expand=True, fill="both")

            # Play sound feedback if enabled
            if self.sound_feedback_var.get():
                try:
                    import winsound
                    if highest[0] == "Positive":
                        winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
                    elif highest[0] == "Neutral":
                        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                    elif highest[0] == "Negative":
                        winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
                except ImportError:
                    # Sound not supported on this platform
                    pass

            # Auto-close after delay
            popup.after(3000, popup.destroy)
            
        except Exception as e:
            print(f"Notification error: {e}")

    def clear_results(self):
        """Clear analysis results and reset the display."""
        # Reset text display
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", "end")
        self.result_textbox.configure(state="disabled")
        
        # Clear chart
        if self.sentiment_figure:
            plt.close(self.sentiment_figure)
            self.sentiment_figure = None
            
        if self.sentiment_canvas:
            self.sentiment_canvas.get_tk_widget().destroy()
            self.sentiment_canvas = None
            
        # Reset status
        self.update_status("Ready for new analysis", "#28a745")
        
        # Reset last analyzed text
        self.last_analyzed_text = ""

    def export_results(self):
        """Export analysis results to a file."""
        if not self.last_analyzed_text:
            self.update_status("No analysis to export", "#dc3545")
            return
            
        try:
            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Analysis Results"
            )
            
            if not file_path:
                return  # User cancelled
                
            # Get content from textbox
            self.result_textbox.configure(state="normal")
            content = self.result_textbox.get("1.0", "end")
            self.result_textbox.configure(state="disabled")
            
            # Write to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
                
            # Confirm success
            self.update_status(f"Exported to {os.path.basename(file_path)}", "#28a745")
            
        except Exception as e:
            self.update_status(f"Export error: {str(e)}", "#dc3545")

    def update_hotkey(self, hotkey):
        """Update the keyboard hotkey for analysis."""
        # First stop any existing hotkey listener
        self.stop_hotkey_listener()
        
        # Start new listener with the selected hotkey
        self.listen_to_hotkey(hotkey)
        
        # Update status
        self.update_status(f"Hotkey changed to {hotkey}", "#28a745")

    def listen_to_hotkey(self, hotkey="ctrl+c"):
        """Start a listener thread for the specified hotkey."""
        # Store a reference to the thread so we can stop it later
        self.hotkey_thread = threading.Thread(
            target=self._hotkey_loop,
            args=(hotkey,),
            daemon=True
        )
        self.hotkey_thread.start()

    def stop_hotkey_listener(self):
        """Stop the hotkey listener thread."""
        # We can't directly stop a thread, but we can use a flag
        # The thread is daemon, so it will terminate with the application
        pass

    def _hotkey_loop(self, hotkey):
        """Background loop that waits for hotkey presses."""
        # Store current hotkey for reference
        self.current_hotkey = hotkey
        
        while True:
            try:
                # Wait for the specified hotkey
                keyboard.wait(hotkey)
                
                # Brief delay to allow copy operation to complete
                time.sleep(0.2)
                
                # Analyze the clipboard content
                self.after(0, self.analyze_selected)
                
            except Exception as e:
                print(f"Hotkey error: {e}")
                # If there's an error, sleep briefly to avoid CPU spin
                time.sleep(1)
                
                
    def cancel_page_tasks(self):
        """Cancel all pending after callbacks for this page."""
        try:
            # Cancel countdown timer
            if hasattr(self, 'countdown_job') and self.countdown_job:
                try:
                    self.after_cancel(self.countdown_job)
                except Exception:
                    pass
                self.countdown_job = None
                
            # Mark as not running
            self.running = False
        except Exception as e:
            print(f"Error canceling page tasks: {e}")

    def destroy(self):
        """Clean up resources when the page is destroyed."""
        # Close any open matplotlib figures
        if self.sentiment_figure:
            plt.close(self.sentiment_figure)
        
        # Call parent's destroy method
        super().destroy()
