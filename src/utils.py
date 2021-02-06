import numpy as np

##### METRICS #####
def recall(y_true, y_pred):
    '''
    Computes the recall, a metric for multi-label classification of 
    how many relevant items are selected. Only computes a batch-wise average of recall.

    Arguments:
        y_true  -- numpy vector of true classes
        y_pred  -- numpy vector of predicted classes

    Returns: numpy value of recall
    '''
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + np.finfo(float).eps)
    return recall

def precision(y_true, y_pred):
    """
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant. Only computes a batch-wise average of precision.

    Arguments:
    y_true  -- numpy vector of true classes
    y_pred  -- numpy vector of predicted classes

    Returns: numpy value of precision
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + np.finfo(float).eps)
    return precision

def f1_score(y_true, y_pred):
    """
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant. Only computes a batch-wise average of precision.

    Arguments:
    y_true  -- tensor of true classes
    y_pred  -- tensor of predicted classes

    Returns: numpy value of precision
    """ 
    true_np = y_true.to("cpu").numpy()
    pred_np = y_pred.to("cpu").numpy()
    precision_score = precision(true_np, pred_np)
    recall_score = recall(true_np, pred_np)
    return 2*((precision_score * recall_score) / (precision_score + recall_score + np.finfo(float).eps))