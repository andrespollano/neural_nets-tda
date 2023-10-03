import pandas as pd
import numpy as np
from numpy.linalg import inv, norm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, roc_curve

def compute_mahalanobis_params(val_id_features, val_classes):
    uc_list = []
    val_classes = pd.DataFrame(val_classes, columns=['class'])
    class_labels = val_classes['class'].unique()
    
    for c in class_labels:
        class_data = val_id_features[val_classes['class'] == c]
        uc_list.append(np.mean(class_data, axis=0))
    
    uc = np.stack(uc_list)
    sigma = np.cov(val_id_features, rowvar=False)
    regularization_term = 1e-6
    sigma_inv = inv(sigma + np.eye(sigma.shape[0]) * regularization_term)
    
    return uc, sigma_inv

def compute_mahalanobis_score(x, uc, sigma_inv):
    scores = []
    for c in uc:
        diff = x - c
        score = diff.T @ sigma_inv @ diff
        scores.append(score)
    return min(scores)

def compute_knn_score(x, val_id_features_normalized):
    nn = NearestNeighbors(n_neighbors=1).fit(val_id_features_normalized.values)
    distance, _ = nn.kneighbors([x], 1)
    return -distance[0][0]



def prepare_ood(val_id_features, val_classes):

    uc, sigma_inv = compute_mahalanobis_params(val_id_features, val_classes)
    val_id_features_normalized = val_id_features.apply(lambda x: x/np.linalg.norm(x, 2), axis=1)
    
    return uc, sigma_inv, val_id_features_normalized

def compute_ood_scores(features_df, uc, sigma_inv, val_id_features_normalized):
    maha_scores = []
    knn_scores = []
    
    for _, row in features_df.iterrows():
        x = row.values
        maha_scores.append(compute_mahalanobis_score(x, uc, sigma_inv))
        knn_scores.append(compute_knn_score(x/np.linalg.norm(x, 2), val_id_features_normalized))
    
    return maha_scores, knn_scores

def get_thresholds(test_id_features, uc, sigma_inv, val_id_features_normalized, id_percentile=0.95):
    maha_scores, knn_scores = compute_ood_scores(test_id_features, uc, sigma_inv, val_id_features_normalized)
    maha_threshold = np.quantile(maha_scores, id_percentile)
    knn_threshold = np.quantile(knn_scores, id_percentile)
    return maha_threshold, knn_threshold

def evaluate_ood(id_scores, ood_scores, threshold):
    # 1 is OOD, 0 is ID
    y_true = [0]* len(id_scores) + [1] * len(ood_scores)
    combined_scores = list(id_scores) + list(ood_scores)
    y_pred = [1 if score >= threshold else 0 for score in combined_scores]

    # Compute FPR at TPR=95%
    fpr, tpr, thresholds = roc_curve(y_true, combined_scores, pos_label=1)
    fpr95_indices = np.where(tpr >= 0.95)[0]
    if fpr95_indices.size > 0:
        fpr95_index = fpr95_indices[-1]
        fpr95 = fpr[fpr95_index]
    else:
        fpr95 = "N/A"

    # Compute AUC-ROC
    auc = roc_auc_score(y_true, combined_scores)

    # Compute OOD Classification Accuracy
    y_true = [1] * len(ood_scores)
    y_pred = [1 if score >= threshold else 0 for score in ood_scores]
    accuracy = sum([1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)]) / len(y_true)

    return fpr95, auc, accuracy


