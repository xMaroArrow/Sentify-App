import os
import time
import threading
from typing import Dict, Any, Optional

import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.data_processor import DataProcessor
from utils.visualization import ModelComparisonVisualizer
from models.model_trainer import ModelTrainer


class PageFineTune(ctk.CTkFrame):
    """UI page for fine-tuning Hugging Face transformer models."""

    def __init__(self, parent):
        super().__init__(parent)

        self.dp = DataProcessor()
        self.trainer = ModelTrainer()
        self.viz = ModelComparisonVisualizer()

        self.active_thread: Optional[threading.Thread] = None
        self.results: Optional[Dict[str, Any]] = None
        self.history = []
        self.canvases: Dict[str, FigureCanvasTkAgg] = {}

        self._build_ui()

    def _build_ui(self):
        self.scroll = ctk.CTkScrollableFrame(self, width=980, height=820)
        self.scroll.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header = ctk.CTkFrame(self.scroll)
        header.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(header, text="Transformer Fine-Tuning", font=("Arial", 24, "bold")).pack(pady=(10, 4))
        ctk.CTkLabel(header, text="Fine-tune pre-trained models for social-media sentiment analysis",
                     font=("Arial", 14), text_color="#888888").pack(pady=(0, 10))

        # Dataset
        ds = ctk.CTkFrame(self.scroll)
        ds.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(ds, text="1) Dataset", font=("Arial", 18, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        row = ctk.CTkFrame(ds); row.pack(fill="x", padx=10, pady=5)
        self.ds_path = ctk.StringVar(value="No file selected")
        ctk.CTkLabel(row, textvariable=self.ds_path, width=420, anchor="w").pack(side="left", padx=6)
        ctk.CTkButton(row, text="Load Dataset (CSV)", command=self._load_dataset, width=160).pack(side="left", padx=6)

        pre = ctk.CTkFrame(ds); pre.pack(fill="x", padx=10, pady=5)
        self.pre_opts = {
            "remove_urls": ctk.BooleanVar(value=True),
            "remove_mentions": ctk.BooleanVar(value=True),
            "lowercase": ctk.BooleanVar(value=True),
            "remove_punctuation": ctk.BooleanVar(value=True),
            "remove_hashtags": ctk.BooleanVar(value=False),
            "remove_numbers": ctk.BooleanVar(value=False),
        }
        for k, v in self.pre_opts.items():
            ctk.CTkCheckBox(pre, text=k.replace("_", " ").title(), variable=v).pack(side="left", padx=6)

        split = ctk.CTkFrame(ds); split.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(split, text="Train/Val/Test:").pack(side="left", padx=6)
        self.split_choice = ctk.StringVar(value="80/10/10")
        ctk.CTkOptionMenu(split, values=["80/10/10", "90/5/5", "70/15/15"], variable=self.split_choice, width=120).pack(side="left", padx=6)
        self.stratified = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(split, text="Stratified", variable=self.stratified).pack(side="left", padx=10)
        ctk.CTkButton(split, text="Prepare Split", command=self._prepare_split, width=150).pack(side="right", padx=6)

        self.ds_summary = ctk.CTkTextbox(ds, height=110, font=("Courier", 11))
        self.ds_summary.pack(fill="x", padx=10, pady=5)
        self.ds_summary.configure(state="disabled")

        # Model & Tokenizer
        mt = ctk.CTkFrame(self.scroll)
        mt.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(mt, text="2) Model & Tokenizer", font=("Arial", 18, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        rowm = ctk.CTkFrame(mt); rowm.pack(fill="x", padx=10, pady=5)
        self.model_choices = [
            "cardiffnlp/twitter-roberta-base-sentiment",
            "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
            "distilbert-base-uncased",
            "finiteautomata/bertweet-base-sentiment-analysis",
        ]
        self.model_id = ctk.StringVar(value=self.model_choices[0])
        ctk.CTkLabel(rowm, text="Pretrained model:").pack(side="left", padx=6)
        ctk.CTkOptionMenu(rowm, values=self.model_choices, variable=self.model_id, width=380).pack(side="left", padx=6)
        self.custom_model = ctk.StringVar(value="")
        ctk.CTkEntry(rowm, textvariable=self.custom_model, placeholder_text="Custom HF id or local folder", width=260).pack(side="left", padx=6)
        ctk.CTkButton(rowm, text="Browse...", width=100, command=self._browse_local_model_dir).pack(side="left", padx=6)

        hyp = ctk.CTkFrame(mt); hyp.pack(fill="x", padx=10, pady=5)
        def add_opt(frame, label, values, var, default):
            ctk.CTkLabel(frame, text=label, width=150, anchor="w").pack(side="left", padx=6)
            var.set(str(default))
            ctk.CTkOptionMenu(frame, values=[str(v) for v in values], variable=var, width=90).pack(side="left", padx=6)

        self.max_len = ctk.StringVar(); self.batch = ctk.StringVar(); self.epochs = ctk.StringVar()
        self.lr = ctk.StringVar(); self.wd = ctk.StringVar(); self.warmup = ctk.StringVar(); self.clip = ctk.StringVar()
        self.class_weight = ctk.BooleanVar(value=True)
        self.use_amp = ctk.BooleanVar(value=True)
        self.freeze_base = ctk.BooleanVar(value=False)
        self.unfreeze_n = ctk.StringVar(value="0")
        self.seed = ctk.StringVar(value="42")

        r1 = ctk.CTkFrame(hyp); r1.pack(fill="x", pady=3)
        add_opt(r1, "Max Length", [64, 128, 256], self.max_len, 128)
        add_opt(r1, "Batch Size", [8, 16, 32], self.batch, 16)
        add_opt(r1, "Epochs", [3, 4, 5, 8], self.epochs, 3)
        r2 = ctk.CTkFrame(hyp); r2.pack(fill="x", pady=3)
        add_opt(r2, "Learning Rate", [2e-5, 3e-5, 5e-5], self.lr, 2e-5)
        add_opt(r2, "Weight Decay", [0.0, 0.01, 0.05], self.wd, 0.01)
        add_opt(r2, "Warmup Ratio", [0.0, 0.06, 0.1, 0.2], self.warmup, 0.1)
        add_opt(r2, "Grad Clip", [0.5, 1.0, 2.0], self.clip, 1.0)

        # Manual overrides (optional)
        manual_row = ctk.CTkFrame(hyp)
        manual_row.pack(fill="x", pady=(2, 6))
        ctk.CTkLabel(manual_row, text="Manual Overrides (optional)", font=("Arial", 13, "bold")) \
            .grid(row=0, column=0, columnspan=6, padx=10, pady=(6, 2), sticky="w")
        self.manual_batch = ctk.StringVar(value="")
        self.manual_lr = ctk.StringVar(value="")
        self.manual_epochs = ctk.StringVar(value="")
        ctk.CTkLabel(manual_row, text="Batch Size:").grid(row=1, column=0, padx=10, pady=4, sticky="w")
        ctk.CTkEntry(manual_row, textvariable=self.manual_batch, width=90, placeholder_text="e.g. 24").grid(row=1, column=1, padx=5, pady=4)
        ctk.CTkLabel(manual_row, text="Learning Rate:").grid(row=1, column=2, padx=10, pady=4, sticky="w")
        ctk.CTkEntry(manual_row, textvariable=self.manual_lr, width=120, placeholder_text="e.g. 2e-5").grid(row=1, column=3, padx=5, pady=4)
        ctk.CTkLabel(manual_row, text="Epochs:").grid(row=1, column=4, padx=10, pady=4, sticky="w")
        ctk.CTkEntry(manual_row, textvariable=self.manual_epochs, width=90, placeholder_text="e.g. 4").grid(row=1, column=5, padx=5, pady=4)

        r3 = ctk.CTkFrame(hyp); r3.pack(fill="x", pady=3)
        ctk.CTkCheckBox(r3, text="Class Imbalance (weights)", variable=self.class_weight).pack(side="left", padx=6)
        ctk.CTkCheckBox(r3, text="Mixed Precision (AMP)", variable=self.use_amp).pack(side="left", padx=6)
        ctk.CTkCheckBox(r3, text="Freeze base encoder", variable=self.freeze_base).pack(side="left", padx=6)
        ctk.CTkLabel(r3, text="Unfreeze last N layers:").pack(side="left", padx=6)
        ctk.CTkEntry(r3, textvariable=self.unfreeze_n, width=60).pack(side="left", padx=6)
        ctk.CTkLabel(r3, text="Seed:").pack(side="left", padx=6)
        ctk.CTkEntry(r3, textvariable=self.seed, width=80).pack(side="left", padx=6)

        # Training Controls
        tr = ctk.CTkFrame(self.scroll)
        tr.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(tr, text="3) Training Controls", font=("Arial", 18, "bold")).pack(anchor="w", padx=10, pady=(10, 5))

        # Device chooser and indicator (for training)
        dev_row = ctk.CTkFrame(tr)
        dev_row.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(dev_row, text="Compute Device:").pack(side="left", padx=6)
        self.device_pref_ft = ctk.StringVar(value="Auto")
        ctk.CTkOptionMenu(dev_row, values=["Auto", "GPU", "CPU"], variable=self.device_pref_ft,
                          command=self._on_device_change_ft, width=100).pack(side="left", padx=6)
        self.device_label_ft = ctk.CTkLabel(dev_row, text="Device: " + self.trainer.current_device_info())
        self.device_label_ft.pack(side="left", padx=10)
        ctr = ctk.CTkFrame(tr); ctr.pack(fill="x", padx=10, pady=5)
        self.patience = ctk.StringVar(value="2")
        self.val_metric = ctk.StringVar(value="f1_macro")
        self.ckpt_every = ctk.StringVar(value="1")
        ctk.CTkLabel(ctr, text="Early Stop Patience:").pack(side="left", padx=6)
        ctk.CTkOptionMenu(ctr, values=["1", "2", "3", "4"], variable=self.patience, width=80).pack(side="left", padx=6)
        ctk.CTkLabel(ctr, text="Val Metric:").pack(side="left", padx=6)
        ctk.CTkOptionMenu(ctr, values=["f1_macro", "accuracy"], variable=self.val_metric, width=120).pack(side="left", padx=6)
        ctk.CTkLabel(ctr, text="Checkpoint Every:").pack(side="left", padx=6)
        ctk.CTkOptionMenu(ctr, values=["1", "2", "3"], variable=self.ckpt_every, width=80).pack(side="left", padx=6)

        # Preset buttons
        preset_row = ctk.CTkFrame(tr); preset_row.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(preset_row, text="Presets:", width=80, anchor="w").pack(side="left", padx=6)
        ctk.CTkButton(preset_row, text="Quick", width=90, command=lambda: self._apply_preset("quick")).pack(side="left", padx=4)
        ctk.CTkButton(preset_row, text="Balanced", width=90, command=lambda: self._apply_preset("balanced")).pack(side="left", padx=4)
        ctk.CTkButton(preset_row, text="High-Accuracy", width=120, command=lambda: self._apply_preset("high")).pack(side="left", padx=4)

        br = ctk.CTkFrame(tr); br.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(br, text="Start Fine-Tuning", command=self._start, width=160).pack(side="left", padx=6)
        ctk.CTkButton(br, text="Evaluate Current Model", command=self._eval_current, width=200).pack(side="left", padx=6)

        self.progress = ctk.CTkProgressBar(tr, width=820); self.progress.pack(padx=10, pady=5)
        self.progress.set(0)
        self.progress_text = ctk.CTkLabel(tr, text="Idle."); self.progress_text.pack(padx=10, pady=5)

        self.log = ctk.CTkTextbox(tr, height=120, font=("Courier", 11)); self.log.pack(fill="x", padx=10, pady=5)
        self.log.configure(state="disabled")

        # Visualizations
        viz = ctk.CTkFrame(self.scroll)
        viz.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(viz, text="4) Visualizations", font=("Arial", 18, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        self.viz_tabs = ctk.CTkTabview(viz, width=920, height=420)
        self.viz_tabs.pack(padx=10, pady=10, fill="x")
        for tab in ["Learning Curves", "Confusion Matrix", "ROC", "PR", "Metrics Table"]:
            self.viz_tabs.add(tab)

    # ================= Actions =================
    def _load_dataset(self):
        fp = filedialog.askopenfilename(title="Select Dataset CSV", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not fp:
            return
        self.ds_path.set(fp)
        opts = {k: v.get() for k, v in self.pre_opts.items()}
        self.dp.set_preprocessing_options(opts)

        def worker():
            try:
                self._set_progress("Loading dataset...", 0.05)
                self.dp.load_csv(fp)
                summary = self.dp.get_data_summary()
                self.after(0, lambda: self._set_summary(summary))
                self._set_progress("Dataset loaded", 0.1)
            except Exception as e:
                self._error(f"Load failed: {e}")
        self._run_thread(worker)

    def _prepare_split(self):
        if self.dp.raw_data is None:
            messagebox.showinfo("Info", "Load a dataset first.")
            return
        choice = self.split_choice.get()
        train_size = 0.9 if choice == "80/10/10" else (0.9 if choice == "90/5/5" else 0.85)
        strat = self.stratified.get()

        def worker():
            try:
                self._set_progress("Preparing split...", 0.12)
                self.dp.prepare_train_test_split(train_size=train_size, use_validation=True, stratify=strat)
                summary = self.dp.get_split_summary()
                self.after(0, lambda: self._set_summary(summary))
                self._set_progress("Split prepared", 0.2)
            except Exception as e:
                self._error(f"Split failed: {e}")
        self._run_thread(worker)

    def _start(self):
        if self.dp.train_data is None:
            messagebox.showinfo("Info", "Prepare the split first.")
            return
        if self.active_thread and self.active_thread.is_alive():
            messagebox.showinfo("Training", "Already running.")
            return

        # Determine pretrained source
        model_id = self.custom_model.get().strip() or self.model_id.get()
        # Validate custom id: allow local directory or full HF id like org/model
        if self.custom_model.get().strip():
            cm = self.custom_model.get().strip()
            if not os.path.isdir(cm):
                # If not a folder, require something that looks like an HF id (contains '/')
                if '/' not in cm:
                    messagebox.showerror(
                        "Invalid model identifier",
                        "The custom model must be either a local folder path or a full Hugging Face model id like 'cardiffnlp/twitter-roberta-base-sentiment'.\n\n"
                        "Tip: Use the dropdown for known presets or click 'Browse...' to select a local model directory."
                    )
                    return
        # Resolve hyperparameters with manual overrides if present
        try:
            batch_val = int(float(self.batch.get() or 16))
            if self.manual_batch.get().strip():
                mb = int(float(self.manual_batch.get().strip()))
                if mb > 0:
                    batch_val = mb
        except Exception:
            batch_val = int(float(self.batch.get() or 16))
        try:
            lr_val = float(self.lr.get() or 2e-5)
            if self.manual_lr.get().strip():
                mlr = float(self.manual_lr.get().strip())
                if mlr > 0:
                    lr_val = mlr
        except Exception:
            lr_val = float(self.lr.get() or 2e-5)
        try:
            epochs_val = int(float(self.epochs.get() or 3))
            if self.manual_epochs.get().strip():
                mep = int(float(self.manual_epochs.get().strip()))
                if mep > 0:
                    epochs_val = mep
        except Exception:
            epochs_val = int(float(self.epochs.get() or 3))

        params = dict(
            data_processor=self.dp,
            model_name=f"finetuned_{int(time.time())}",
            pretrained_model=model_id,
            max_length=int(float(self.max_len.get() or 128)),
            batch_size=batch_val,
            lr=lr_val,
            weight_decay=float(self.wd.get() or 0.01),
            warmup_ratio=float(self.warmup.get() or 0.1),
            epochs=epochs_val,
            freeze_base=self.freeze_base.get(),
            unfreeze_last_n_layers=int(float(self.unfreeze_n.get() or 0)),
            use_amp=self.use_amp.get(),
            class_weighting=self.class_weight.get(),
            early_stopping_patience=int(float(self.patience.get() or 2)),
            val_metric=self.val_metric.get(),
            checkpoint_every=int(float(self.ckpt_every.get() or 1)),
            grad_clip_max_norm=float(self.clip.get() or 1.0),
            seed=int(float(self.seed.get() or 42)),
        )

        def on_log(msg: str):
            self.after(0, lambda: self._log(msg))

        def on_epoch_end(epoch: int, hist_ep: Dict[str, Any]):
            self.history.append(hist_ep)
            prog = min(0.2 + (epoch + 1) / max(1, params["epochs"]) * 0.75, 0.95)
            # Format LR if available
            lr_val = hist_ep.get('learning_rate') or hist_ep.get('lr')
            try:
                lr_f = float(lr_val) if lr_val is not None else None
            except Exception:
                lr_f = None
            if lr_f is not None:
                lr_str = f"{lr_f:.6f}" if abs(lr_f) >= 1e-3 else f"{lr_f:.2e}"
                msg = f"Epoch {epoch+1}/{params['epochs']} done. LR: {lr_str}"
            else:
                msg = f"Epoch {epoch+1}/{params['epochs']} done."
            self.after(0, lambda: self._set_progress(msg, prog))
            self.after(0, self._update_learning_curves)

        def on_checkpoint(epoch: int, path: str):
            self.after(0, lambda: self._log(f"Checkpoint: {path}"))

        def on_finished(results: Dict[str, Any]):
            self.results = results
            self.after(0, lambda: self._set_progress("Fine-tuning completed.", 1.0))
            self.after(0, self._update_all_viz)
            self.after(0, lambda: messagebox.showinfo("Finished", "Fine-tuning complete."))

        def worker():
            try:
                self._set_progress("Starting fine-tuning...", 0.25)
                self.trainer.train_transformer_finetune(
                    callbacks={
                        "on_log": on_log,
                        "on_epoch_end": on_epoch_end,
                        "on_checkpoint": on_checkpoint,
                        "on_finished": on_finished,
                    },
                    **params
                )
            except Exception as e:
                # Provide a clearer error if model id is invalid or private
                emsg = str(e)
                if "is not a local folder" in emsg or "not a valid model identifier" in emsg:
                    emsg += ("\n\nResolution:\n- Select a preset from the dropdown, or\n"
                             "- Enter a full model id like 'org/model', or\n"
                             "- Browse to a local model directory containing config.json and weights.\n"
                             "If the repo is private, ensure you are logged in with 'huggingface-cli login' ")
                self._error(f"Training failed: {emsg}")
        self._run_thread(worker)

    def _browse_local_model_dir(self):
        path = filedialog.askdirectory(title="Select local pretrained model directory")
        if path:
            self.custom_model.set(path)

    def _eval_current(self):
        if not self.results:
            messagebox.showinfo("Info", "No in-memory model results. Train first.")
            return
        self._update_all_viz()

    # ================= Viz helpers =================
    def _update_all_viz(self):
        self._update_learning_curves()
        self._update_confusion()
        self._update_roc()
        self._update_pr()
        self._update_table()

    def _embed(self, tab: str, fig: plt.Figure):
        if tab in self.canvases and self.canvases[tab] is not None:
            self.canvases[tab].get_tk_widget().destroy()
            self.canvases[tab] = None
        canvas = FigureCanvasTkAgg(fig, master=self.viz_tabs.tab(tab))
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvases[tab] = canvas

    def _update_learning_curves(self):
        if not self.history:
            return
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8, 5))
        tr = [e.get("train_loss") for e in self.history if "train_loss" in e]
        vl = [e.get("val_loss") for e in self.history if "val_loss" in e]
        ax.plot(range(1, len(tr)+1), tr, 'o-', label='Train Loss')
        if any(v is not None for v in vl):
            ax.plot(range(1, len(vl)+1), vl, 's--', label='Val Loss')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Learning Curves'); ax.grid(alpha=0.3); ax.legend()
        self._embed("Learning Curves", fig)

    def _update_confusion(self):
        if not self.results:
            return
        r = self.results
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        self.viz.plot_confusion_matrix(fig, ax, r['confusion_matrix'], r['classes'],
                                       title=f"Confusion Matrix - {r.get('model_name','')}", publication_ready=True)
        self._embed("Confusion Matrix", fig)

    def _update_roc(self):
        if not self.results:
            return
        r = self.results
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        self.viz.plot_roc_curve(fig, ax, r.get('fpr', {}), r.get('tpr', {}), r.get('roc_auc', {}),
                                classes=r['classes'], title='ROC Curve', publication_ready=True)
        self._embed("ROC", fig)

    def _update_pr(self):
        if not self.results:
            return
        r = self.results
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        self.viz.plot_precision_recall_curve(fig, ax, r.get('precision_curve', {}), r.get('recall_curve', {}),
                                             r.get('average_precision', {}), r['classes'], 'Precision-Recall', True)
        self._embed("PR", fig)

    def _update_table(self):
        if not self.results:
            return
        r = self.results
        fig, ax = plt.subplots(figsize=(8, 5.5))
        self.viz.plot_metrics_table(fig, ax, r['classification_report'], r['accuracy'], r.get('model_name', ''), True)
        self._embed("Metrics Table", fig)

    # ================= Utils =================
    def _run_thread(self, target):
        t = threading.Thread(target=target, daemon=True)
        t.start()
        self.active_thread = t

    def _set_progress(self, text: str, val: float):
        self.after(0, lambda: self.progress_text.configure(text=text))
        # If parent has no progress bar (used minimal one), ignore; else update if exists
        try:
            self.progress.set(max(0.0, min(1.0, val)))
        except Exception:
            pass

    def _set_summary(self, txt: str):
        self.ds_summary.configure(state="normal")
        self.ds_summary.delete("1.0", "end")
        self.ds_summary.insert("1.0", txt)
        self.ds_summary.configure(state="disabled")

    def _log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _error(self, msg: str):
        self._log("ERROR: " + msg)
        messagebox.showerror("Error", msg)

    def _on_device_change_ft(self, choice):
        pref = (choice or 'Auto').lower()
        if pref == 'gpu':
            self.trainer.set_compute_device('gpu')
        elif pref == 'cpu':
            self.trainer.set_compute_device('cpu')
        else:
            self.trainer.set_compute_device('auto')
        try:
            self.device_label_ft.configure(text="Device: " + self.trainer.current_device_info())
        except Exception:
            pass

    def _apply_preset(self, name: str):
        """Apply preset hyperparameters to the UI controls."""
        n = (name or '').lower()
        # Helper to set a StringVar safely
        def setv(var, val):
            try:
                var.set(str(val))
            except Exception:
                pass
        # Detect backbone to pick LR
        model_id = (self.custom_model.get().strip() or self.model_id.get() or '').lower()
        def default_lr():
            return "5e-5" if "distilbert" in model_id else "2e-5"

        if n == "quick":
            setv(self.max_len, 64)
            setv(self.batch, 16)
            setv(self.epochs, 2)
            setv(self.lr, default_lr())
            setv(self.wd, 0.01)
            setv(self.warmup, 0.06)
            setv(self.clip, 1.0)
            self.freeze_base.set(True)
            self.unfreeze_n.set("0")
            self.class_weight.set(True)
            setv(self.patience, 1)
            # Manual overrides (to enforce values)
            setv(self.manual_batch, 16)
            setv(self.manual_epochs, 2)
            setv(self.manual_lr, default_lr())
        elif n == "balanced":
            setv(self.max_len, 128)
            setv(self.batch, 32)
            setv(self.epochs, 4)
            setv(self.lr, default_lr())
            setv(self.wd, 0.01)
            setv(self.warmup, 0.1)
            setv(self.clip, 1.0)
            self.freeze_base.set(False)
            self.unfreeze_n.set("0")
            self.class_weight.set(True)
            setv(self.patience, 2)
            setv(self.manual_batch, 32)
            setv(self.manual_epochs, 4)
            setv(self.manual_lr, default_lr())
        else:  # high accuracy
            setv(self.max_len, 160)
            setv(self.batch, 16)
            setv(self.epochs, 6)
            setv(self.lr, default_lr())
            setv(self.wd, 0.01)
            setv(self.warmup, 0.1)
            setv(self.clip, 1.0)
            self.freeze_base.set(False)
            self.unfreeze_n.set("0")
            self.class_weight.set(True)
            setv(self.patience, 3)
            setv(self.manual_batch, 16)
            setv(self.manual_epochs, 6)
            setv(self.manual_lr, default_lr())

        

