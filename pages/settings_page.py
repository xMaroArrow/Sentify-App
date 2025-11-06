import customtkinter as ctk
import os
from addons.sentiment_analyzer import SentimentAnalyzer
from utils import theme

# In a new settings_page.py
class SettingsPage(ctk.CTkFrame):
    def __init__(self, parent, config_manager):
        super().__init__(parent)
        self.config_manager = config_manager
        self._local_models = self._scan_local_transformer_models()
        
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
        
        # Model selection (source + model)
        model_frame = ctk.CTkFrame(self)
        model_frame.pack(pady=10, fill="x", padx=20)

        ctk.CTkLabel(model_frame, text="Sentiment Model").pack(pady=(10, 5), anchor="w")

        # Source selector
        self.model_source_var = ctk.StringVar(value=self.config_manager.get("model_source", "huggingface").capitalize())
        self.model_source_menu = ctk.CTkOptionMenu(
            model_frame,
            values=["Huggingface", "Local"],
            variable=self.model_source_var,
            command=self._on_source_change,
            width=220,
        )
        self.model_source_menu.pack(pady=(0, 8), anchor="w")

        # Hugging Face models dropdown
        hf_models = [
            "cardiffnlp/twitter-roberta-base-sentiment",
            "distilbert-base-uncased-finetuned-sst-2-english",
            "finiteautomata/bertweet-base-sentiment-analysis",
        ]
        current_hf = self.config_manager.get("model", hf_models[0])
        if current_hf not in hf_models:
            hf_models.insert(0, current_hf)
        self.hf_model_var = ctk.StringVar(value=current_hf)
        self.hf_model_dropdown = ctk.CTkOptionMenu(
            model_frame,
            values=hf_models,
            variable=self.hf_model_var,
            command=self._on_hf_model_change,
            width=440,
        )

        # Custom HuggingFace model input (manual repo id)
        self.hf_custom_frame = ctk.CTkFrame(model_frame, fg_color="transparent")
        ctk.CTkLabel(self.hf_custom_frame, text="Custom HF repo id", anchor="w").pack(anchor="w")
        custom_row = ctk.CTkFrame(self.hf_custom_frame, fg_color="transparent")
        custom_row.pack(fill="x")
        self.hf_custom_entry = ctk.CTkEntry(custom_row, width=340, placeholder_text="e.g. user/model-name or org/model")
        # Prefill with current if using HF
        try:
            if (self.config_manager.get("model_source", "huggingface") or "").lower() == "huggingface":
                self.hf_custom_entry.insert(0, current_hf)
        except Exception:
            pass
        self.hf_custom_entry.pack(side="left", padx=(0, 6), pady=4)
        self.hf_custom_apply = ctk.CTkButton(custom_row, text="Apply", width=80, command=self._apply_custom_hf_model)
        self.hf_custom_apply.pack(side="left", pady=4)

        # Local models dropdown (transformers + non-transformers)
        self._local_transformers = self._scan_local_transformer_models()
        self._local_pytorch = self._scan_local_pytorch_models()
        local_values, self._local_model_map = self._format_local_models_for_menu(
            self._local_transformers,
            self._local_pytorch,
        )
        current_local = self.config_manager.get("local_model_path", "")
        current_type = (self.config_manager.get("local_model_type", "transformer") or "transformer").lower()
        # Validate current selection
        display_value = ""
        if current_local and self._is_valid_models_path(current_local):
            for disp, meta in self._local_model_map.items():
                if meta["path"] == current_local and meta["type"] == current_type:
                    display_value = disp
                    break
        if not display_value and local_values:
            display_value = local_values[0]
        self.local_model_var = ctk.StringVar(value=display_value)
        self.local_model_dropdown = ctk.CTkOptionMenu(
            model_frame,
            values=local_values if local_values else ["<no local models found>"],
            variable=self.local_model_var,
            command=self._on_local_model_change,
            width=440,
        )

        # Place correct dropdown based on source
        self._refresh_model_dropdown_visibility()

        # Info note for local PyTorch models
        self.local_note = ctk.CTkLabel(
            model_frame,
            text="Note: Non-transformer (PyTorch) models require vectorizer.pkl and label_encoder.pkl in the model folder.",
            text_color=theme.subtle_text_color(),
            wraplength=600,
            justify="left",
        )
        self.local_note.pack(pady=(4, 0), anchor="w")

        # Actions: refresh + status indicator
        actions_frame = ctk.CTkFrame(model_frame)
        actions_frame.pack(fill="x", pady=(8, 4))

        self.refresh_btn = ctk.CTkButton(
            actions_frame,
            text="Refresh Models",
            width=140,
            command=self._refresh_local_models_list,
        )
        self.refresh_btn.pack(side="left", padx=(0, 8), pady=4)

        self.model_status_label = ctk.CTkLabel(
            actions_frame,
            text="Model status: —",
            text_color=theme.subtle_text_color()
        )
        self.model_status_label.pack(side="left", padx=4, pady=4)

        # Initial status
        self._update_model_status()

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
        # Backwards compatibility if called by older code paths
        self._on_hf_model_change(value)

    def _on_source_change(self, value: str):
        source = (value or "").strip().lower()
        if source not in ("huggingface", "local"):
            source = "huggingface"
        self.config_manager.set("model_source", source)
        self._refresh_model_dropdown_visibility()
        # Reload analyzer immediately
        SentimentAnalyzer.reload()
        self._update_model_status()

    def _on_hf_model_change(self, value: str):
        if not value:
            return
        self.config_manager.set("model", value)
        # Ensure source is huggingface
        self.config_manager.set("model_source", "huggingface")
        # Reload analyzer immediately
        SentimentAnalyzer.reload()
        self._update_model_status()

    def _apply_custom_hf_model(self):
        """Apply a manually entered Hugging Face repo id."""
        try:
            repo = (self.hf_custom_entry.get() or "").strip()
            if not repo:
                return
            # Persist selection
            self.config_manager.set("model", repo)
            self.config_manager.set("model_source", "huggingface")
            # Update dropdown values to include this repo at the top if missing
            try:
                current_values = list(self.hf_model_dropdown._values) if hasattr(self.hf_model_dropdown, "_values") else []
            except Exception:
                current_values = []
            if repo not in current_values:
                new_values = [repo] + [v for v in current_values if v != repo]
                try:
                    self.hf_model_dropdown.configure(values=new_values)
                except Exception:
                    pass
            # Reflect selection
            try:
                self.hf_model_var.set(repo)
            except Exception:
                pass
            # Reload analyzer
            SentimentAnalyzer.reload()
            self._update_model_status()
            try:
                self.hf_custom_apply.configure(text="Applied")
                self.after(1200, lambda: self.hf_custom_apply.configure(text="Apply"))
            except Exception:
                pass
        except Exception:
            # Keep UI responsive even if reload fails
            try:
                self.hf_custom_apply.configure(text="Failed")
                self.after(1500, lambda: self.hf_custom_apply.configure(text="Apply"))
            except Exception:
                pass

    def _on_local_model_change(self, value: str):
        if not value or value.startswith("<"):
            return
        meta = self._local_model_map.get(value)
        if not meta:
            return
        self.config_manager.set("local_model_path", meta["path"]) 
        self.config_manager.set("local_model_type", meta["type"]) 
        # Ensure source is local
        self.config_manager.set("model_source", "local")
        # Reload analyzer immediately
        SentimentAnalyzer.reload()
        self._update_model_status()

    def _refresh_model_dropdown_visibility(self):
        # Remove both if already placed
        try:
            self.hf_model_dropdown.pack_forget()
        except Exception:
            pass
        try:
            self.local_model_dropdown.pack_forget()
        except Exception:
            pass

        source = (self.model_source_var.get() or "").strip().lower()
        if source == "local":
            self.local_model_dropdown.pack(pady=(0, 5), anchor="w")
        else:
            self.hf_model_dropdown.pack(pady=(0, 5), anchor="w")
            try:
                self.hf_custom_frame.pack(pady=(0, 6), fill="x", anchor="w")
            except Exception:
                pass
        # Hide custom frame when not on HF
        if source == "local":
            try:
                self.hf_custom_frame.pack_forget()
            except Exception:
                pass

    def _scan_local_transformer_models(self):
        """Return a list of local directories (recursively) that look like transformer models."""
        candidates = []
        base = self._models_base_dir()
        if os.path.isdir(base):
            for root, dirs, files in os.walk(base):
                # Skip archive directories under models
                rel = os.path.relpath(root, base)
                rel_parts = [] if rel == '.' else rel.split(os.sep)
                if rel_parts and rel_parts[0].lower() == "archive":
                    continue
                if "config.json" in files:
                    candidates.append(os.path.normpath(root))
        # Deduplicate and sort
        return sorted(set(candidates))

    def _format_local_models_for_menu(self, transformer_paths, pytorch_paths):
        """Build display values for local models and a mapping to their metadata."""
        values = []
        mapping = {}
        base = self._models_base_dir()
        # Transformers first
        for p in transformer_paths:
            rel = os.path.relpath(p, base) if p.startswith(base) else p
            disp = f"[TF] models{os.sep}{rel}"
            values.append(disp)
            mapping[disp] = {"type": "transformer", "path": p}
        # Then PyTorch
        for p in pytorch_paths:
            rel = os.path.relpath(p, base) if p.startswith(base) else p
            disp = f"[PT] models{os.sep}{rel}"
            values.append(disp)
            mapping[disp] = {"type": "pytorch", "path": p}
        return values, mapping

    def _is_valid_models_path(self, path: str) -> bool:
        """Return True if path is inside project models dir but not in models/archive."""
        try:
            norm = os.path.normpath(path)
            base = os.path.normpath(self._models_base_dir())
            # Must be under the models base directory
            if not norm.lower().startswith(base.lower()):
                return False
            rel = os.path.relpath(norm, base)
            rel_parts = [] if rel == '.' else rel.split(os.sep)
            if rel_parts and rel_parts[0].lower() == 'archive':
                return False
            return True
        except Exception:
            return False

    def _models_base_dir(self) -> str:
        """Absolute path to the models directory (project-root/models)."""
        try:
            # pages/settings_page.py -> pages -> project root
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(root, 'models')
        except Exception:
            return os.path.abspath('models')

    def _scan_local_pytorch_models(self):
        """Return a list of local directories (recursively) that look like non-transformer PyTorch models."""
        candidates = []
        base = self._models_base_dir()
        if os.path.isdir(base):
            for root, dirs, files in os.walk(base):
                rel = os.path.relpath(root, base)
                rel_parts = [] if rel == '.' else rel.split(os.sep)
                if rel_parts and rel_parts[0].lower() == "archive":
                    continue
                if ("model.pt" in files) and ("config.json" not in files):
                    candidates.append(os.path.normpath(root))
        return sorted(set(candidates))

    def _refresh_local_models_list(self):
        """Rescan the models directory and update the local models dropdown."""
        try:
            self._local_transformers = self._scan_local_transformer_models()
            self._local_pytorch = self._scan_local_pytorch_models()
            values, mapping = self._format_local_models_for_menu(
                self._local_transformers,
                self._local_pytorch,
            )
            self._local_model_map = mapping

            if not values:
                values = ["<no local models found>"]

            # Keep current selection if still present
            current = self.local_model_var.get()
            self.local_model_dropdown.configure(values=values)
            if current in values:
                pass
            else:
                self.local_model_var.set(values[0])
        except Exception as e:
            # Fallback: show placeholder
            self.local_model_dropdown.configure(values=["<no local models found>"])
            self.local_model_var.set("<no local models found>")

    def _update_model_status(self):
        """Update the on-screen indicator showing current model activation state."""
        try:
            analyzer = SentimentAnalyzer()
            active = analyzer.is_initialized()
            source = (self.config_manager.get("model_source", "huggingface") or "").lower()
            detail = ""
            if source == "huggingface":
                detail = self.config_manager.get("model", "") or ""
            else:
                path = self.config_manager.get("local_model_path", "") or ""
                mtype = self.config_manager.get("local_model_type", "transformer") or "transformer"
                detail = f"{mtype}: {path}"

            # Special case: Local source but no valid selection
            if source == "local":
                path = (self.config_manager.get("local_model_path", "") or "").strip()
                valid = bool(path) and self._is_valid_models_path(path) and os.path.exists(path)
                if not valid:
                    self.model_status_label.configure(text="Model status: No local model selected", text_color="#E74C3C")
                    return

            if active:
                self.model_status_label.configure(text=f"Model status: Active ({source}) {detail}", text_color="#2ECC71")
            else:
                self.model_status_label.configure(text=f"Model status: Inactive ({source}) {detail}", text_color="#E74C3C")
        except Exception:
            # Silent failure
            try:
                self.model_status_label.configure(text="Model status: Unknown", text_color=theme.subtle_text_color())
            except Exception:
                pass

    def _save_reddit_credentials(self):
        self.config_manager.set("reddit_client_id", self.reddit_client_id.get().strip())
        self.config_manager.set("reddit_client_secret", self.reddit_client_secret.get().strip())
        self.config_manager.set("reddit_user_agent", self.reddit_user_agent.get().strip())
