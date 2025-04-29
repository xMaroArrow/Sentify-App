"""
Sentiment Analysis Model Implementation
--------------------------------------
This module implements a custom BiLSTM with Attention model for sentiment analysis
and compares it with RoBERTa transfer learning on the Sentiment140 dataset.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.layers import Attention, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import time
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
CONFIG = {
    'data_path': 'data/sentiment140_dataset.csv',
    'sample_size': 100000,  # Reduce for faster development
    'max_words': 15000,
    'max_seq_length': 50,
    'embedding_dim': 100,
    'batch_size': 64,
    'epochs': 10,
    'early_stopping_patience': 3,
    'glove_path': 'glove/glove.6B.100d.txt',
    'output_dir': 'experiment_results'
}

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)

def preprocess_text(text):
    """
    Clean and normalize text for sentiment analysis.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove user mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags symbols but keep the hashtag text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_sentiment140_data(file_path, sample_size=None):
    """
    Load and preprocess the Sentiment140 dataset.
    """
    print(f"Loading data from {file_path}")
    
    # Define column names
    columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    # Load data
    df = pd.read_csv(file_path, encoding='latin-1', names=columns)
    
    # Map sentiment values (0=negative, 2=neutral, 4=positive) to (0, 1, 2)
    sentiment_map = {0: 0, 2: 1, 4: 2}
    df['sentiment'] = df['target'].map(sentiment_map)
    
    # Take a sample if specified
    if sample_size and sample_size < len(df):
        # Stratified sampling to maintain class distribution
        df = df.groupby('sentiment', group_keys=False).apply(
            lambda x: x.sample(int(sample_size * len(x) / len(df)), random_state=42)
        )
    
    print(f"Loaded {len(df)} tweets")
    
    # Preprocess text
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    return df

def prepare_data_for_models(df):
    """
    Prepare data for both custom models and RoBERTa.
    """
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['sentiment'])
    
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Custom model data preparation
    tokenizer = Tokenizer(num_words=CONFIG['max_words'], oov_token='<OOV>')
    tokenizer.fit_on_texts(train_df['processed_text'])
    
    # Convert texts to sequences
    train_sequences = tokenizer.texts_to_sequences(train_df['processed_text'])
    val_sequences = tokenizer.texts_to_sequences(val_df['processed_text'])
    test_sequences = tokenizer.texts_to_sequences(test_df['processed_text'])
    
    # Pad sequences
    X_train_pad = pad_sequences(train_sequences, maxlen=CONFIG['max_seq_length'], padding='post')
    X_val_pad = pad_sequences(val_sequences, maxlen=CONFIG['max_seq_length'], padding='post')
    X_test_pad = pad_sequences(test_sequences, maxlen=CONFIG['max_seq_length'], padding='post')
    
    # Get labels
    y_train = train_df['sentiment'].values
    y_val = val_df['sentiment'].values
    y_test = test_df['sentiment'].values
    
    # Load GloVe embeddings if available
    embedding_matrix = None
    if os.path.exists(CONFIG['glove_path']):
        print("Loading GloVe embeddings...")
        embeddings_index = {}
        with open(CONFIG['glove_path'], encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        print("Creating embedding matrix...")
        embedding_matrix = np.zeros((CONFIG['max_words'], CONFIG['embedding_dim']))
        for word, i in tokenizer.word_index.items():
            if i >= CONFIG['max_words']:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    
    # Store test texts for error analysis
    test_texts = test_df['processed_text'].values
    
    return {
        'custom': {
            'X_train': X_train_pad,
            'y_train': y_train,
            'X_val': X_val_pad,
            'y_val': y_val,
            'X_test': X_test_pad,
            'y_test': y_test,
            'tokenizer': tokenizer,
            'embedding_matrix': embedding_matrix
        },
        'roberta': {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df
        },
        'test_texts': test_texts
    }

def create_bilstm_attention_model(vocab_size, embedding_dim=100, input_length=50, embedding_matrix=None):
    """
    Create a Bidirectional LSTM model with attention mechanism.
    """
    # Input layer
    input_layer = Input(shape=(input_length,))
    
    # Embedding layer
    if embedding_matrix is not None:
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=input_length,
            trainable=False
        )(input_layer)
    else:
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=input_length
        )(input_layer)
    
    # Bidirectional LSTM layer
    bilstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedding)
    
    # Attention mechanism
    attention = tf.keras.layers.Attention()([bilstm, bilstm])
    
    # Pooling layers
    avg_pool = GlobalAveragePooling1D()(attention)
    max_pool = GlobalMaxPooling1D()(attention)
    
    # Concatenate pooling outputs
    concat = Concatenate()([avg_pool, max_pool])
    
    # Dense layers
    dense = Dense(128, activation='relu')(concat)
    dropout = Dropout(0.5)(dense)
    output = Dense(3, activation='softmax')(dropout)
    
    # Create and compile model
    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def train_custom_model(model, data, model_name):
    """
    Train a custom model with early stopping and checkpointing.
    """
    # Create checkpoint directory
    checkpoint_dir = os.path.join(CONFIG['output_dir'], 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Measure training time
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save final model
    model.save(os.path.join(CONFIG['output_dir'], f"{model_name}_final.h5"))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], f"{model_name}_training_history.png"))
    plt.close()
    
    return {
        'history': history.history,
        'training_time': training_time
    }

def setup_roberta_model():
    """
    Set up RoBERTa model for fine-tuning.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment",
        num_labels=3
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    return model, tokenizer

def prepare_roberta_data(data, tokenizer, max_length=128):
    """
    Prepare data for RoBERTa model.
    """
    # We'll use a smaller subset for training RoBERTa to keep it manageable
    train_df = data['train_df'].sample(min(50000, len(data['train_df'])), random_state=42)
    val_df = data['val_df'].sample(min(5000, len(data['val_df'])), random_state=42)
    test_df = data['test_df']
    
    # Tokenize texts
    train_encodings = tokenizer(
        train_df['processed_text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='tf'
    )
    
    val_encodings = tokenizer(
        val_df['processed_text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='tf'
    )
    
    test_encodings = tokenizer(
        test_df['processed_text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='tf'
    )
    
    # Get labels
    y_train = train_df['sentiment'].values
    y_val = val_df['sentiment'].values
    y_test = test_df['sentiment'].values
    
    return {
        'train_encodings': train_encodings,
        'val_encodings': val_encodings,
        'test_encodings': test_encodings,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

def train_roberta_model(model, data):
    """
    Fine-tune RoBERTa model.
    """
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
    
    # Measure training time
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        data['train_encodings'],
        data['y_train'],
        validation_data=(data['val_encodings'], data['y_val']),
        batch_size=16,  # Smaller batch size for RoBERTa
        epochs=3,       # Fewer epochs for RoBERTa
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"RoBERTa training completed in {training_time:.2f} seconds")
    
    # Save model
    model.save_pretrained(os.path.join(CONFIG['output_dir'], 'roberta_model'))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('RoBERTa Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('RoBERTa Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], "roberta_training_history.png"))
    plt.close()
    
    return {
        'history': history.history,
        'training_time': training_time
    }

def evaluate_model(model, X_test, y_test, model_name, is_roberta=False):
    """
    Evaluate model performance and generate metrics.
    """
    print(f"Evaluating {model_name}...")
    
    # Measure inference time
    start_time = time.time()
    
    if is_roberta:
        logits = model(X_test).logits
        y_pred_proba = tf.nn.softmax(logits, axis=1).numpy()
    else:
        y_pred_proba = model.predict(X_test)
    
    inference_time = time.time() - start_time
    inference_speed = len(y_test) / inference_time  # samples per second
    
    # Get class predictions
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Inference time: {inference_time:.2f}s for {len(y_test)} samples")
    print(f"Inference speed: {inference_speed:.2f} samples/second")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], f"{model_name}_confusion_matrix.png"))
    plt.close()
    
    # Store results
    results = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_avg": report['weighted avg']['precision'],
        "recall_avg": report['weighted avg']['recall'],
        "f1_avg": report['weighted avg']['f1-score'],
        "class_report": report,
        "confusion_matrix": cm.tolist(),
        "inference_time": inference_time,
        "inference_speed": inference_speed
    }
    
    # Save results to JSON
    with open(os.path.join(CONFIG['output_dir'], f"{model_name}_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def compare_models(results):
    """
    Generate comparison visualizations between models.
    """
    model_names = [r['model_name'] for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]
    f1_scores = [r['f1_avg'] * 100 for r in results]
    inference_speeds = [r['inference_speed'] for r in results]
    
    # Accuracy and F1 comparison
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
    plt.bar(x + width/2, f1_scores, width, label='F1 Score', color='salmon')
    
    plt.xlabel('Models')
    plt.ylabel('Score (%)')
    plt.title('Performance Comparison')
    plt.xticks(x, model_names)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], "model_performance_comparison.png"))
    plt.close()
    
    # Inference speed comparison
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, inference_speeds, color='lightgreen')
    plt.xlabel('Models')
    plt.ylabel('Samples per second')
    plt.title('Inference Speed Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], "inference_speed_comparison.png"))
    plt.close()
    
    # Create performance by class comparison
    class_metrics = []
    for result in results:
        for sentiment in ["Negative", "Neutral", "Positive"]:
            class_metrics.append({
                'Model': result['model_name'],
                'Class': sentiment,
                'F1 Score': result['class_report'][sentiment]['f1-score'] * 100
            })
    
    df = pd.DataFrame(class_metrics)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='F1 Score', hue='Model', data=df)
    plt.title('F1 Score by Sentiment Class')
    plt.ylabel('F1 Score (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], "class_performance_comparison.png"))
    plt.close()
    
    # Create summary table
    summary = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Accuracy (%)': f"{r['accuracy']*100:.2f}",
            'F1 Score (%)': f"{r['f1_avg']*100:.2f}",
            'Precision (%)': f"{r['precision_avg']*100:.2f}",
            'Recall (%)': f"{r['recall_avg']*100:.2f}",
            'Inference Speed': f"{r['inference_speed']:.2f}"
        } for r in results
    ])
    
    summary.to_csv(os.path.join(CONFIG['output_dir'], "model_comparison_summary.csv"), index=False)
    
    # Print summary table
    print("\nModel Comparison Summary:")
    print(summary)

def run_experiment():
    """
    Run the full experiment.
    """
    print("Starting sentiment analysis experiment...")
    
    # Load and preprocess data
    df = load_sentiment140_data(CONFIG['data_path'], sample_size=CONFIG['sample_size'])
    
    # Prepare data for models
    data = prepare_data_for_models(df)
    
    # Create, train and evaluate BiLSTM+Attention model
    print("\n=== Training BiLSTM+Attention Model ===")
    bilstm_model = create_bilstm_attention_model(
        vocab_size=CONFIG['max_words'],
        embedding_dim=CONFIG['embedding_dim'],
        input_length=CONFIG['max_seq_length'],
        embedding_matrix=data['custom']['embedding_matrix']
    )
    
    bilstm_model.summary()
    
    bilstm_results = train_custom_model(
        bilstm_model,
        data['custom'],
        'bilstm_attention'
    )
    
    bilstm_eval = evaluate_model(
        bilstm_model,
        data['custom']['X_test'],
        data['custom']['y_test'],
        'BiLSTM_Attention'
    )
    
    # Create, train and evaluate RoBERTa model
    print("\n=== Training RoBERTa Model ===")
    roberta_model, roberta_tokenizer = setup_roberta_model()
    
    roberta_data = prepare_roberta_data(
        data['roberta'],
        roberta_tokenizer
    )
    
    roberta_results = train_roberta_model(
        roberta_model,
        roberta_data
    )
    
    roberta_eval = evaluate_model(
        roberta_model,
        roberta_data['test_encodings'],
        roberta_data['y_test'],
        'RoBERTa',
        is_roberta=True
    )
    
    # Compare models
    compare_models([bilstm_eval, roberta_eval])
    
    print(f"\nExperiment completed. Results saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    run_experiment()