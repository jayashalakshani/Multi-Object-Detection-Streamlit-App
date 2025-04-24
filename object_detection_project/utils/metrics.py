# utils/metrics.py

import numpy as np
from sklearn.metrics import average_precision_score  # Example metric

def calculate_map(true_labels, predicted_scores):
    """
    Calculates mean average precision (mAP).  Placeholder.  Replace with
    actual object detection mAP calculation using bounding boxes.
    """
    try:
        # Ensure labels and scores are appropriate shape, deal with any -1 values
        true_labels = np.array(true_labels).ravel()
        predicted_scores = np.array(predicted_scores).ravel()

        # Check if labels contains more than one class
        if len(np.unique(true_labels)) > 1:
            # Calculate average precision
            average_precision = average_precision_score(true_labels, predicted_scores)
        else:
            # If only one class present, return a default value
            average_precision = 0.0  # Or another suitable value
            print("Warning: Only one class present in true labels. mAP set to 0.")
        return average_precision
    except Exception as e:
        print(f"Error calculating mAP: {e}")
        return 0.0 # Or another suitable value

def calculate_inference_time(start_time, end_time):
    return end_time - start_time