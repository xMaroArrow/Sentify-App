import customtkinter as ctk

# In a new settings_page.py
class SettingsPage(ctk.CTkFrame):
    def __init__(self, parent, config_manager):
        super().__init__(parent)
        self.config_manager = config_manager
        
        # Confidence threshold slider
        threshold_label = ctk.CTkLabel(self, text="Minimum Confidence Threshold (%)")
        threshold_label.pack(pady=(20, 5))
        
        self.threshold_slider = ctk.CTkSlider(self, from_=50, to=95, 
                                             command=self.update_threshold)
        self.threshold_slider.set(self.config_manager.get("confidence_threshold", 70))
        self.threshold_slider.pack(pady=5)
        
        self.threshold_value = ctk.CTkLabel(self, text=f"{self.threshold_slider.get():.0f}%")
        self.threshold_value.pack(pady=(0, 20))
        
        # Text preprocessing options
        preprocess_frame = ctk.CTkFrame(self)
        preprocess_frame.pack(pady=10, fill="x", padx=20)
        
        ctk.CTkLabel(preprocess_frame, text="Text Preprocessing Options").pack(pady=5)
        
        self.remove_urls = ctk.CTkCheckBox(preprocess_frame, text="Remove URLs", 
                                          command=self.save_settings)
        self.remove_urls.pack(anchor="w", pady=2)
        self.remove_urls.select() if self.config_manager.get("remove_urls", True) else None
        
        self.remove_mentions = ctk.CTkCheckBox(preprocess_frame, text="Remove @mentions", 
                                              command=self.save_settings)
        self.remove_mentions.pack(anchor="w", pady=2)
        self.remove_mentions.select() if self.config_manager.get("remove_mentions", True) else None
        
        # Add more preprocessing options...
        
        # Model selection
        model_frame = ctk.CTkFrame(self)
        model_frame.pack(pady=10, fill="x", padx=20)
        
        ctk.CTkLabel(model_frame, text="Sentiment Model").pack(pady=5)
        
        models = ["cardiffnlp/twitter-roberta-base-sentiment", 
                 "distilbert-base-uncased-finetuned-sst-2-english"]
        self.model_var = ctk.StringVar(value=self.config_manager.get("model", models[0]))
        self.model_dropdown = ctk.CTkOptionMenu(model_frame, values=models, 
                                                variable=self.model_var, 
                                                command=self.change_model)
        self.model_dropdown.pack(pady=5)

        # Reddit API credentials
        reddit_frame = ctk.CTkFrame(self)
        reddit_frame.pack(pady=20, fill="x", padx=20)
        ctk.CTkLabel(reddit_frame, text="Reddit API Credentials").pack(pady=(10, 5), anchor="w")

        # Client ID
        ctk.CTkLabel(reddit_frame, text="Client ID").pack(pady=(4, 2), anchor="w")
        self.reddit_client_id = ctk.CTkEntry(reddit_frame, width=400)
        current_id = str(self.config_manager.get("reddit_client_id", ""))
        if current_id:
            self.reddit_client_id.insert(0, current_id)
        self.reddit_client_id.pack(pady=(0, 6), anchor="w")

        # Client Secret
        ctk.CTkLabel(reddit_frame, text="Client Secret").pack(pady=(4, 2), anchor="w")
        self.reddit_client_secret = ctk.CTkEntry(reddit_frame, show="*", width=400)
        current_secret = str(self.config_manager.get("reddit_client_secret", ""))
        if current_secret:
            self.reddit_client_secret.insert(0, current_secret)
        self.reddit_client_secret.pack(pady=(0, 6), anchor="w")

        # User Agent
        ctk.CTkLabel(reddit_frame, text="User Agent").pack(pady=(4, 2), anchor="w")
        self.reddit_user_agent = ctk.CTkEntry(reddit_frame, width=400)
        current_ua = str(self.config_manager.get("reddit_user_agent", "Sentify-App/1.0 (by u/username)"))
        if current_ua:
            self.reddit_user_agent.insert(0, current_ua)
        self.reddit_user_agent.pack(pady=(0, 10), anchor="w")

        save_btn = ctk.CTkButton(
            reddit_frame,
            text="Save Reddit Credentials",
            command=self._save_reddit_credentials,
            width=220,
        )
        save_btn.pack(pady=(0, 12), anchor="w")
        
    def update_threshold(self, value):
        self.threshold_value.configure(text=f"{value:.0f}%")
        self.config_manager.set("confidence_threshold", value)
        
    def save_settings(self):
        self.config_manager.set("remove_urls", self.remove_urls.get())
        self.config_manager.set("remove_mentions", self.remove_mentions.get())
        # Save other settings...
        
    def change_model(self, value):
        self.config_manager.set("model", value)
        # Show a warning that model will reload on next application start
        # ...

    def _save_reddit_credentials(self):
        self.config_manager.set("reddit_client_id", self.reddit_client_id.get().strip())
        self.config_manager.set("reddit_client_secret", self.reddit_client_secret.get().strip())
        self.config_manager.set("reddit_user_agent", self.reddit_user_agent.get().strip())
