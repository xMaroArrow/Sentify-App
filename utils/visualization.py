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
        
        # Plot the confusion matrix
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        
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
        thresh = confusion_matrix.max() * text_color_threshold
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if confusion_matrix[i, j] > thresh else "black",
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
            if i in precision and i in recall:
                color = colors[i % len(colors)] if colors else None
                ax.plot(recall[i], precision[i], color=color, lw=2,
                        label=f'{class_name} (AP = {average_precision[i]:.2f})')
        
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
            if i in fpr and i in tpr:
                color = colors[i % len(colors)] if colors else None
                ax.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
        
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
        
        # Turn off axis
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
        
        # Create table
        table = ax.table(
            cellText=rows,
            rowLabels=row_labels,
            colLabels=columns,
            loc='center',
            cellLoc='center',
            colColours=[header_color] * len(columns)
        )
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Add title
        title = f"Classification Metrics - {model_name}" if model_name else "Classification Metrics"
        ax.set_title(title, fontsize=12, color=main_color, pad=20)
        
        # Style cells
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(grid_color)
            if row == 0:  # Header
                cell.set_text_props(weight='bold', color=header_text_color)
            else:
                cell.set_text_props(color=main_color)
    
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