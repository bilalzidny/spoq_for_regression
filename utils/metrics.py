import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, confusion_matrix, classification_report


def compute_rss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

def compute_mse(w,X,y):
    y_pred = X @ w
    return mean_squared_error(y,y_pred)

def compute_mae(w,X,y):
    y_pred = X @ w
    return mean_absolute_error(y, y_pred)

def compute_mape(w,X,y):
    y_pred = X @ w
    return mean_absolute_percentage_error(y, y_pred)

def compute_relative_sse(w, X, y):
    y_pred = X @ w
    return np.linalg.norm(y_pred - y) / np.linalg.norm(y)

def compute_aic(y_true, y_pred, k):
    n = len(y_true)
    rss = compute_rss(y_true, y_pred)
    return n * np.log(rss / n) + 2 * k

def compute_bic(y_true, y_pred, k):
    n = len(y_true)
    rss = compute_rss(y_true, y_pred)
    return n * np.log(rss / n) + k * np.log(n)

def binarize_weights(w, threshold=1e-6):
    return (np.abs(w) > threshold).astype(int)

def compare_sparsity(w_ref, w_test, label_ref="Reference", label_test="Test"):
    y_true = binarize_weights(w_ref[1:])
    y_pred = binarize_weights(w_test[1:])
    
    cm = confusion_matrix(y_true, y_pred)
    
    return cm

def jaccard_similarity(w1, w2, threshold=1e-6):

    # to not take the biais into account : 
    weights_1, weights_2 = w1[1:], w2[1:]
    b1 = binarize_weights(weights_1, threshold)
    b2 = binarize_weights(weights_2, threshold)
    intersection = np.sum(b1 & b2)
    union = np.sum(b1 | b2)
    return intersection / union if union != 0 else 1.0  

def hamming_distance(w1, w2, threshold=1e-6):
    # to not take the biais into account : 
    weights_1, weights_2 = w1[1:], w2[1:]
    b1 = binarize_weights(weights_1, threshold)
    b2 = binarize_weights(weights_2, threshold)
    return np.mean(b1 != b2)

def euclidian_distance(w1, w2):
    return np.linalg.norm(w1 - w2, ord=2)

def relative_euclidian_distance(w1, w2):
    if np.array_equal(w1, w2):
        return 0
    else :
        return euclidian_distance(w1,w2)/np.linalg.norm(w2, ord=2)