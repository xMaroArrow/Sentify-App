# evaluation.py
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ModelEvaluator:
    def __init__(self, model_service):
        self.model_service = model_service
        
    def evaluate_on_dataset(self, texts, true_labels):
        """Evaluate model on a labeled dataset."""
        pred_labels = []
        
        for text in texts:
            sentiment = self.model_service.analyze_sentiment(text)
            pred_label = max(sentiment.items(), key=lambda x: x[1])[0]
            pred_labels.append(pred_label)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        cm = confusion_matrix(true_labels, pred_labels, labels=["Positive", "Neutral", "Negative"])
        report = classification_report(true_labels, pred_labels, labels=["Positive", "Neutral", "Negative"])
        
        return {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "classification_report": report,
            "predictions": pred_labels
        }
    
    def plot_confusion_matrix(self, true_labels, pred_labels, master):
        """Create a confusion matrix visualization."""
        plt.rcParams.update({
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        })
        
        cm = confusion_matrix(true_labels, pred_labels, labels=["Positive", "Neutral", "Negative"])
        
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#2B2B2B")
        ax.set_facecolor("#2B2B2B")
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Confusion Matrix', color="white")
        ax.set_xlabel('Predicted label', color="white")
        ax.set_ylabel('True label', color="white")
        
        # Add labels
        classes = ["Positive", "Neutral", "Negative"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        
        # Add values in cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] < cm.max() / 2 else "black")
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.draw()
        
        return canvas