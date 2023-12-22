import numpy as np

def r2_score(y_true, y_pred):
    """
    Compute the R-squared score.

    Parameters:
        - y_true: True values.
        - y_pred: Predicted values.

    Returns:
        - r2: R-squared score.
    """
    residual = ((y_true - y_pred)**2).sum()
    total = ((y_true - y_true.mean())**2).sum()
    r2 = 1 - residual/total 
    return r2

def mean_squared_error(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE).

    Parameters:
        - y_true: True values.
        - y_pred: Predicted values.

    Returns:
        - mse: Mean Squared Error.
    """
    mse = np.mean((y_true - y_pred)**2)
    return mse

def mean_absolute_error(y_true, y_pred): 
    """
    Compute the Mean Absolute Error (MAE).

    Parameters:
        - y_true: True values.
        - y_pred: Predicted values.

    Returns:
        - mae: Mean Absolute Error.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def f1_score(y_true, y_pred):
    """
    Compute the F1 Score.

    Parameters:
        - y_true: True values.
        - y_pred: Predicted values.

    Returns:
        - f1: F1 Score.
    """
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def accuracy(y_true, y_pred):
    """
    Compute the Accuracy.

    Parameters:
        - y_true: True values.
        - y_pred: Predicted values.

    Returns:
        - accuracy: Accuracy.
    """
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)

    accuracy = correct_predictions / total_samples
    return accuracy

# Additional metric functions

def precision(y_true, y_pred):
    """
    Compute Precision.

    Parameters:
        - y_true: True values.
        - y_pred: Predicted values.

    Returns:
        - precision: Precision.
    """
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))

    precision = true_positive / (true_positive + false_positive)
    return precision

def recall(y_true, y_pred):
    """
    Compute Recall.

    Parameters:
        - y_true: True values.
        - y_pred: Predicted values.

    Returns:
        - recall: Recall.
    """
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    recall = true_positive / (true_positive + false_negative)
    return recall
