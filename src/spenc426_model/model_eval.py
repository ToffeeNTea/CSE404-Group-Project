import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def evaluate_model(model_path="cache/logistic_regression_model.pkl", 
                  features_path="cache/image_features.npy",
                  labels_path="cache/image_labels.npy",
                  test_size=0.2,
                  random_state=22):
    """
    Evaluate the trained model with precision, recall, accuracy, and F1 score.
    
    Args:
        model_path: Path to the saved model
        features_path: Path to the image features
        labels_path: Path to the image labels
        test_size: Fraction of data to be used for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load data
    X = np.load(features_path)
    y = np.load(labels_path)
    
    # Load model
    model = joblib.load(model_path)
    
    # Split into train and test sets (using same parameters as in training)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print results
    print(f"Model Performance Metrics:")
    print(f"-------------------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create and save confusion matrix visualization
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    # Create 'results' directory if it doesn't exist
    if not os.path.exists('metrics'):
        os.makedirs('metrics')
        
    plt.savefig('metrics/confusion_matrix.png')
    print("Confusion matrix saved to 'metrics/confusion_matrix.png'")
    
    # Return metrics as dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics

evaluate_model()