"""
Visualization utilities for model comparison and evaluation.

This module provides functions to create publication-ready visualizations
for model evaluation and comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

class ModelComparisonVisualizer:
    """Creates visualizations for model evaluation and comparison."""
    
    def __init__(self):
        """Initialize the visualizer."""
        # Configure default styles
        self.default_colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58', '#9C27B0', '#3F51B5']
        self.default_markers = ['o', 's', '^', 'D', 'v', '*']
    
    def plot_confusion_matrix(self, fig, ax, confusion_matrix: np.ndarray, 
                             classes: List[str], title: str = "Confusion Matrix",
                             publication_ready: bool = True):
        """
        Plot a confusion matrix.
        
        Args:
            fig: Matplotlib figure or None
            ax: Matplotlib axis
            confusion_matrix: Confusion matrix as numpy array
            classes: List of class names
            title: Plot title
            publication_ready: Whether to use publication-ready styling
        """
        # Configure colors based on publication readiness
        if publication_ready:
            cmap = "Blues"
            text_color_threshold = 0.5
            main_color = "black"
        else:
            cmap = "viridis"
            text_color_threshold = 0.7
            main_color = "white"
        
        # Accept list-backed matrices (JSON) by converting to numpy
        try:
            cm = np.array(confusion_matrix)
        except Exception:
            cm = confusion_matrix
        # Plot the confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        
        # Add colorbar
        if fig is not None:
            fig.colorbar(im, ax=ax)
        
        # Add labels, title, and ticks
        ax.set_title(title, fontsize=12, color=main_color)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=10, color=main_color)
        ax.set_yticklabels(classes, fontsize=10, color=main_color)
        ax.set_xlabel('Predicted Label', fontsize=10, color=main_color)
        ax.set_ylabel('True Label', fontsize=10, color=main_color)
        
        # Add text annotations
        thresh = cm.max() * text_color_threshold
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(int(cm[i, j])),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=8)
        
        # Add grid lines
        ax.set_xticks(np.arange(-.5, len(classes), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(classes), 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.2)
        ax.tick_params(which="minor", size=0)
    
    def plot_precision_recall_curve(self, fig, ax, precision: Dict, recall: Dict,
                                   average_precision: Dict, classes: List[str],
                                   title: str = "Precision-Recall Curve",
                                   publication_ready: bool = True):
        """
        Plot precision-recall curves.
        
        Args:
            fig: Matplotlib figure or None
            ax: Matplotlib axis
            precision: Dictionary mapping class indices to precision values
            recall: Dictionary mapping class indices to recall values
            average_precision: Dictionary mapping class indices to average precision
            classes: List of class names
            title: Plot title
            publication_ready: Whether to use publication-ready styling
        """
        # Configure colors based on publication readiness
        if publication_ready:
            colors = self.default_colors
            main_color = "black"
            grid_alpha = 0.3
        else:
            colors = None  # Use matplotlib defaults
            main_color = "white"
            grid_alpha = 0.2
        
        # Plot precision-recall curve for each class
        for i, class_name in enumerate(classes):
            # Handle JSON-loaded dicts with string keys and list values
            k_i = i if i in precision else (str(i) if str(i) in precision else None)
            k_r = i if i in recall else (str(i) if str(i) in recall else None)
            k_ap = i if i in average_precision else (str(i) if str(i) in average_precision else None)
            if k_i is not None and k_r is not None:
                color = colors[i % len(colors)] if colors else None
                pr = np.array(precision[k_i])
                rc = np.array(recall[k_r])
                ap_val = average_precision.get(k_ap, 0.0)
                try:
                    ap_val = float(ap_val)
                except Exception:
                    ap_val = 0.0
                ax.plot(rc, pr, color=color, lw=2,
                        label=f'{class_name} (AP = {ap_val:.2f})')
        
        # Add labels and grid
        ax.set_xlabel('Recall', fontsize=10, color=main_color)
        ax.set_ylabel('Precision', fontsize=10, color=main_color)
        ax.set_title(title, fontsize=12, color=main_color)
        ax.grid(alpha=grid_alpha)
        
        # Set axis limits
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        # Add legend
        ax.legend(loc="lower left", frameon=True, fontsize=8)
    
    def plot_roc_curve(self, fig, ax, fpr: Dict, tpr: Dict, roc_auc: Dict,
                      classes: List[str], title: str = "ROC Curve",
                      publication_ready: bool = True):
        """
        Plot ROC curves.
        
        Args:
            fig: Matplotlib figure or None
            ax: Matplotlib axis
            fpr: Dictionary mapping class indices to false positive rates
            tpr: Dictionary mapping class indices to true positive rates
            roc_auc: Dictionary mapping class indices to ROC AUC scores
            classes: List of class names
            title: Plot title
            publication_ready: Whether to use publication-ready styling
        """
        # Configure colors based on publication readiness
        if publication_ready:
            colors = self.default_colors
            main_color = "black"
            grid_alpha = 0.3
        else:
            colors = None  # Use matplotlib defaults
            main_color = "white"
            grid_alpha = 0.2
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(classes):
            # JSON-loaded dict may have string keys and list values
            k_f = i if i in fpr else (str(i) if str(i) in fpr else None)
            k_t = i if i in tpr else (str(i) if str(i) in tpr else None)
            k_auc = i if i in roc_auc else (str(i) if str(i) in roc_auc else None)
            if k_f is not None and k_t is not None:
                color = colors[i % len(colors)] if colors else None
                f = np.array(fpr[k_f])
                t = np.array(tpr[k_t])
                auc_val = roc_auc.get(k_auc, 0.0)
                try:
                    auc_val = float(auc_val)
                except Exception:
                    auc_val = 0.0
                ax.plot(f, t, color=color, lw=2,
                        label=f'{class_name} (AUC = {auc_val:.2f})')
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # Add labels and grid
        ax.set_xlabel('False Positive Rate', fontsize=10, color=main_color)
        ax.set_ylabel('True Positive Rate', fontsize=10, color=main_color)
        ax.set_title(title, fontsize=12, color=main_color)
        ax.grid(alpha=grid_alpha)
        
        # Set axis limits
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        # Add legend
        ax.legend(loc="lower right", frameon=True, fontsize=8)
    
    def plot_metrics_table(self, fig, ax, classification_report: Dict, 
                          accuracy: float, model_name: str = "",
                          publication_ready: bool = True):
        """
        Plot a table of classification metrics.
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axis
            classification_report: Dictionary with classification metrics
            accuracy: Overall accuracy score
            model_name: Name of the model
            publication_ready: Whether to use publication-ready styling
        """
        # Configure colors based on publication readiness
        if publication_ready:
            main_color = "black"
            grid_color = "#CCCCCC"
            header_color = "#EFEFEF"
            header_text_color = "black"
        else:
            main_color = "white"
            grid_color = "#555555"
            header_color = "#333333"
            header_text_color = "white"
        
        # Turn off axis and tighten layout to make table fit
        ax.axis('tight')
        ax.axis('off')
        
        # Extract data from classification report
        rows = []
        row_labels = []
        
        # Add header row
        columns = ['Precision', 'Recall', 'F1-score', 'Support']
        
        # Format report data for table
        for label, metrics in classification_report.items():
            if label != 'accuracy' and label != 'macro avg' and label != 'weighted avg':
                row_labels.append(label)
                row = [
                    f"{metrics['precision']:.3f}",
                    f"{metrics['recall']:.3f}",
                    f"{metrics['f1-score']:.3f}",
                    f"{metrics['support']}"
                ]
                rows.append(row)
        
        # Add average rows
        if 'macro avg' in classification_report:
            row_labels.append('Macro Avg')
            metrics = classification_report['macro avg']
            row = [
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                f"{metrics['support']}"
            ]
            rows.append(row)
        
        if 'weighted avg' in classification_report:
            row_labels.append('Weighted Avg')
            metrics = classification_report['weighted avg']
            row = [
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                f"{metrics['support']}"
            ]
            rows.append(row)
        
        # Add accuracy row
        row_labels.append('Accuracy')
        rows.append([f"{accuracy:.3f}", "", "", ""])
        
        # Shorten overly long row labels to prevent overflow
        def _shorten(lbl, maxlen=18):
            return (lbl[:maxlen-1] + '…') if isinstance(lbl, str) and len(lbl) > maxlen else lbl

        display_row_labels = [_shorten(l) for l in row_labels]

        # Create table
        table = ax.table(
            cellText=rows,
            rowLabels=display_row_labels,
            colLabels=columns,
            loc='center',
            cellLoc='center',
            colColours=[header_color] * len(columns)
        )
        
        # Style table
        # Dynamic font sizing and scaling based on number of rows
        n_rows = len(rows)
        table.auto_set_font_size(False)
        # Smaller font for many classes; keep within [7, 11]
        base_font = 11 if publication_ready else 10
        cell_font = max(7, min(base_font, int(base_font - max(0, n_rows - 6) * 0.4)))
        table.set_fontsize(cell_font)
        # Adjust vertical scale to fit more rows; narrower height for many rows
        v_scale = 1.2 if n_rows <= 6 else max(0.7, 1.6 - 0.08 * n_rows)
        table.scale(1.05, v_scale)
        
        # Add title
        title = f"Classification Metrics - {model_name}" if model_name else "Classification Metrics"
        ax.set_title(title, fontsize=12, color=main_color, pad=10)
        
        # Style cells
        cells = table.get_celld()
        for (row, col), cell in cells.items():
            cell.set_edgecolor(grid_color)
            # Header row is row 0 in matplotlib tables
            if row == 0:
                cell.set_text_props(weight='bold', color=header_text_color)
            else:
                cell.set_text_props(color=main_color)
            # Left-align row labels (col == -1)
            if col == -1:
                cell._loc = 'left'
                cell.PAD = 0.02
        
        # Give extra room on the left for row labels and tighten
        try:
            fig.subplots_adjust(left=0.25, right=0.98, top=0.88, bottom=0.05)
        except Exception:
            pass
        try:
            fig.tight_layout()
        except Exception:
            pass
    
    def plot_metrics_comparison(self, fig, ax, model_names: List[str],
                               accuracy: List[float], precision: List[float],
                               recall: List[float], f1: List[float],
                               title: str = "Model Performance Comparison",
                               publication_ready: bool = True):
        """
        Plot a bar chart comparing model metrics.
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axis
            model_names: List of model names
            accuracy: List of accuracy scores
            precision: List of precision scores
            recall: List of recall scores
            f1: List of F1 scores
            title: Plot title
            publication_ready: Whether to use publication-ready styling
        """
        # Configure colors based on publication readiness
        if publication_ready:
            colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
            main_color = "black"
            grid_alpha = 0.3
        else:
            colors = ['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0']
            main_color = "white"
            grid_alpha = 0.2
        
        # Set width of each bar group
        n_models = len(model_names)
        bar_width = 0.15 if n_models <= 3 else 0.1
        
        # Set positions of bars on X-axis
        r = np.arange(n_models)
        
        # Create bars
        ax.bar(r - bar_width*1.5, accuracy, width=bar_width, label='Accuracy', color=colors[0])
        ax.bar(r - bar_width*0.5, precision, width=bar_width, label='Precision', color=colors[1])
        ax.bar(r + bar_width*0.5, recall, width=bar_width, label='Recall', color=colors[2])
        ax.bar(r + bar_width*1.5, f1, width=bar_width, label='F1 Score', color=colors[3])
        
        # Add labels and title
        ax.set_xlabel('Model', fontsize=10, color=main_color)
        ax.set_ylabel('Score', fontsize=10, color=main_color)
        ax.set_title(title, fontsize=12, color=main_color)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=8)
        
        # Customize X-axis
        ax.set_xticks(r)
        ax.set_xticklabels(model_names, fontsize=8, rotation=45, ha='right', color=main_color)
        
        # Add grid and set Y-axis range
        ax.grid(axis='y', alpha=grid_alpha)
        ax.set_ylim([0, 1.05])
        
        # Add value labels on top of bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom',
                        fontsize=6, color=main_color, rotation=90)
        
        # Apply value labels to each group
        bars = ax.patches
        n_metrics = 4  # accuracy, precision, recall, f1
        for i in range(n_metrics):
            add_labels(bars[i*n_models:(i+1)*n_models])
    
    def plot_training_history(self, fig, ax, history: Dict, 
                             title: str = "Training History",
                             publication_ready: bool = True):
        """
        Plot training history (loss and accuracy).
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axis
            history: Dictionary with training history (loss, accuracy, val_loss, val_accuracy)
            title: Plot title
            publication_ready: Whether to use publication-ready styling
        """
        # Configure colors based on publication readiness
        if publication_ready:
            colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
            main_color = "black"
            grid_alpha = 0.3
        else:
            colors = ['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0']
            main_color = "white"
            grid_alpha = 0.2
        
        # Create twin axis for loss and accuracy
        ax2 = ax.twinx()
        
        # Plot training loss and accuracy
        epochs = range(1, len(history['loss']) + 1)
        train_loss_line, = ax.plot(epochs, history['loss'], 'o-', color=colors[0], label='Training Loss')
        train_acc_line, = ax2.plot(epochs, history['accuracy'], 's-', color=colors[1], label='Training Accuracy')
        
        # Plot validation loss and accuracy if available
        if 'val_loss' in history and 'val_accuracy' in history:
            val_loss_line, = ax.plot(epochs, history['val_loss'], 'o--', color=colors[2], label='Validation Loss')
            val_acc_line, = ax2.plot(epochs, history['val_accuracy'], 's--', color=colors[3], label='Validation Accuracy')
            lines = [train_loss_line, train_acc_line, val_loss_line, val_acc_line]
        else:
            lines = [train_loss_line, train_acc_line]
            # Add labels and title
            ax.set_xlabel('Epoch', fontsize=10, color=main_color)
            ax.set_ylabel('Loss', fontsize=10, color=main_color)
            ax2.set_ylabel('Accuracy', fontsize=10, color=main_color)
            ax.set_title(title, fontsize=12, color=main_color)
            
            # Set y-axis limits
            ax.set_ylim([0, max(history['loss']) * 1.1])
            ax2.set_ylim([0, 1.05])
            
            # Add grid
            ax.grid(alpha=grid_alpha)
            
            # Add legend
            ax.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=8)
    
    def plot_training_history_comparison(self, fig, ax, histories: Dict, 
                                        title: str = "Training History Comparison",
                                        publication_ready: bool = True):
        """
        Plot training history comparison between multiple models.
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axis
            histories: Dictionary mapping model names to training history dictionaries
            title: Plot title
            publication_ready: Whether to use publication-ready styling
        """
        # Configure colors based on publication readiness
        if publication_ready:
            colors = self.default_colors
            main_color = "black"
            grid_alpha = 0.3
        else:
            colors = None  # Use matplotlib defaults
            main_color = "white"
            grid_alpha = 0.2
        
        # Decide what to plot based on what's available
        plot_val = all('val_accuracy' in hist for hist in histories.values())
        
        # Set up markers and line styles
        markers = self.default_markers
        
        # Plot accuracy and validation accuracy for each model
        lines = []
        for i, (model_name, history) in enumerate(histories.items()):
            color = colors[i % len(colors)] if colors else None
            marker = markers[i % len(markers)]
            
            epochs = range(1, len(history['accuracy']) + 1)
            train_line, = ax.plot(epochs, history['accuracy'], marker=marker, 
                                  linestyle='-', color=color, 
                                  label=f'{model_name} (Train)')
            lines.append(train_line)
            
            if plot_val and 'val_accuracy' in history:
                val_line, = ax.plot(epochs, history['val_accuracy'], marker=marker, 
                                    linestyle='--', color=color, 
                                    label=f'{model_name} (Val)')
                lines.append(val_line)
        
        # Add labels and title
        ax.set_xlabel('Epoch', fontsize=10, color=main_color)
        ax.set_ylabel('Accuracy', fontsize=10, color=main_color)
        ax.set_title(title, fontsize=12, color=main_color)
        
        # Set y-axis limits
        ax.set_ylim([0, 1.05])
        
        # Add grid
        ax.grid(alpha=grid_alpha)
        
        # Add legend
        ax.legend(loc='lower right', fontsize=8)
        
        # Add xticks at each epoch
        if len(epochs) <= 20:
            ax.set_xticks(epochs)
        
        # Add a horizontal line at accuracy = 1.0
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    def plot_loss_history_comparison(self, fig, ax, histories: Dict,
                                     title: str = "Loss Curves Comparison",
                                     publication_ready: bool = True):
        """Plot training/validation loss across multiple models for comparison."""
        if publication_ready:
            main_color = "black"
            grid_alpha = 0.3
        else:
            main_color = "white"
            grid_alpha = 0.2

        # Determine max epochs to set x-axis
        max_epochs = 0
        for hist in histories.values():
            max_epochs = max(max_epochs, len(hist.get('loss', [])))
        epochs = range(1, max_epochs + 1) if max_epochs else []

        # Plot each model's loss
        for name, hist in histories.items():
            tr = hist.get('loss', [])
            vl = hist.get('val_loss', [])
            if tr:
                ax.plot(range(1, len(tr)+1), tr, label=f"{name} (train)", linestyle='-', marker='o')
            if vl:
                ax.plot(range(1, len(vl)+1), vl, label=f"{name} (val)", linestyle='--', marker='s')

        ax.set_xlabel('Epoch', fontsize=10, color=main_color)
        ax.set_ylabel('Loss', fontsize=10, color=main_color)
        ax.set_title(title, fontsize=12, color=main_color)
        if max_epochs and max_epochs <= 20:
            ax.set_xticks(list(epochs))
        ax.grid(alpha=grid_alpha)
        ax.legend(loc='best', fontsize=8)

    def plot_loss_curves(self, fig, ax, history: Dict,
                         title: str = "Training and Validation Loss",
                         publication_ready: bool = True):
        """Plot training and validation loss curves."""
        if publication_ready:
            main_color = "black"
            grid_alpha = 0.3
        else:
            main_color = "white"
            grid_alpha = 0.2
        losses = history.get('loss', []) or history.get('train_loss', [])
        val_losses = history.get('val_loss', [])
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, 'o-', label='Train Loss')
        if val_losses:
            ax.plot(epochs, val_losses, 's--', label='Val Loss')
        ax.set_xlabel('Epoch', fontsize=10, color=main_color)
        ax.set_ylabel('Loss', fontsize=10, color=main_color)
        ax.set_title(title, fontsize=12, color=main_color)
        ax.grid(alpha=grid_alpha)
        ax.legend(loc='best', fontsize=8)
        if len(epochs) <= 20:
            ax.set_xticks(epochs)

    def plot_word_cloud(self, fig, ax, texts: List[str],
                        title: str = "Word Cloud",
                        publication_ready: bool = True,
                        max_words: int = 200):
        """Plot a word cloud from a list of texts."""
        try:
            from wordcloud import WordCloud
        except Exception as e:
            ax.text(0.5, 0.5, "Install 'wordcloud' to view\nword cloud visualization",
                    ha='center', va='center', fontsize=12, color='red')
            ax.set_axis_off()
            return
        text_blob = " ".join(texts) if texts else ""
        if not text_blob.strip():
            ax.text(0.5, 0.5, "No text available for word cloud",
                    ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            return
        bg = 'white' if publication_ready else 'black'
        wc = WordCloud(width=800, height=400, background_color=bg,
                       max_words=max_words, collocations=False).generate(text_blob)
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(title, fontsize=12, color=('black' if publication_ready else 'white'))
        ax.axis('off')
