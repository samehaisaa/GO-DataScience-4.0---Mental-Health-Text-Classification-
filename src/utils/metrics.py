from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for model predictions.
    
    Args:
        eval_pred: Tuple containing predictions and labels
        
    Returns:
        Dictionary containing computed metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        predictions, 
        average='weighted'
    )
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
