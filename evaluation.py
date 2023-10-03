import os
import pandas as pd
from ood_detector import prepare_ood, compute_ood_scores, get_thresholds, evaluate_ood
from utils import load_from_pkl, load_from_pt_to_df
from configs import Config

exp_files = {
    "graph_type" : "undir",
    "symmetry" : "max",
    "id_name" : 'news-category',
    "id_val_exp_date" : '2023-Sep-13_14:02',
    "id_test_exp_date" :  '2023-Sep-13_14:02',
    "ood_names" : ['imdb', 'cnn_dailymail', 'news-category'],
    "ood_exp_date" : '2023-Sep-14_15:38'
}


def create_result_df(scores):
    """
        Organizes the scores into the specified DataFrame format.
    """
    df = pd.DataFrame(index=['tda', 'cls', 'combined'])
    for feature_type, score in scores.items():
        for distance, metrics in score.items():
            for metric, value in metrics.items():
                df.loc[feature_type, f"{distance}_{metric}"] = value
    return df

def evaluate_experiment(exp_files):
    default_config = Config()
    results_dir = os.path.join(default_config.base.root_dir, 'outputs_remote')

    # Load ID Val labels
    id_val_labels = load_from_pkl(default_config.base.DATA_DIR, 'id_val_labels.pkl')

    # Load ID topological features
    top_features_dir = os.path.join(results_dir, 'topological_features')
    id_val_tda = load_from_pkl(os.path.join(top_features_dir, exp_files["id_val_exp_date"]), f'{exp_files["id_name"]}_val.pkl')
    id_test_tda = load_from_pkl(os.path.join(top_features_dir, exp_files["id_test_exp_date"]), f'{exp_files["id_name"]}_test.pkl')

    # Load CLS embeddings
    cls_embeddings_dir = os.path.join(results_dir, 'sentence_embeddings_cls')
    id_val_cls = load_from_pt_to_df(os.path.join(cls_embeddings_dir, exp_files["id_val_exp_date"]), f'{exp_files["id_name"]}_val.pt')
    id_test_cls = load_from_pt_to_df(os.path.join(cls_embeddings_dir, exp_files["id_test_exp_date"]), f'{exp_files["id_name"]}_test.pt')

    # Combine features
    id_val_combined = pd.concat([id_val_cls, id_val_tda], axis=1)
    id_test_combined = pd.concat([id_test_cls, id_test_tda], axis=1)    

    # Prepare OOD detector parameters
    uc_tda, sigma_inv_tda, val_id_tda_normalized = prepare_ood(id_val_tda, id_val_labels)
    uc_cls, sigma_inv_cls, val_id_cls_normalized = prepare_ood(id_val_cls, id_val_labels)
    uc_combined, sigma_inv_combined, val_id_combined_normalized = prepare_ood(id_val_combined, id_val_labels)

    # Compute thresholds
    maha_threshold_tda, knn_threshold_tda = get_thresholds(id_test_tda, uc_tda, sigma_inv_tda, val_id_tda_normalized)
    maha_threshold_cls, knn_threshold_cls  = get_thresholds(id_test_cls, uc_cls, sigma_inv_cls, val_id_cls_normalized)
    maha_threshold_combined, knn_threshold_combined = get_thresholds(id_test_combined, uc_combined, sigma_inv_combined, val_id_combined_normalized)

    # Compute OOD scores for val set
    id_maha_scores_tda, id_knn_scores_tda = compute_ood_scores(id_val_tda, uc_tda, sigma_inv_tda, val_id_tda_normalized)
    id_maha_scores_cls, id_knn_scores_cls = compute_ood_scores(id_val_cls, uc_cls, sigma_inv_cls, val_id_cls_normalized)
    id_maha_scores_combined, id_knn_scores_combined = compute_ood_scores(id_val_combined, uc_combined, sigma_inv_combined, val_id_combined_normalized)

    # Compute OOD scores
    for ood_name in exp_files["ood_names"]:
        # Load features
        ood_tda = load_from_pkl(os.path.join(top_features_dir, exp_files["ood_exp_date"]), f'{ood_name}_ood_test.pkl')
        ood_cls = load_from_pt_to_df(os.path.join(cls_embeddings_dir, exp_files["ood_exp_date"]), f'{ood_name}_ood_test.pt')
        ood_combined = pd.concat([ood_cls, ood_tda], axis=1)

        # Compute OOD scores
        maha_scores_tda, knn_scores_tda = compute_ood_scores(ood_tda, uc_tda, sigma_inv_tda, val_id_tda_normalized)
        maha_scores_cls, knn_scores_cls = compute_ood_scores(ood_cls, uc_cls, sigma_inv_cls, val_id_cls_normalized)
        maha_scores_combined, knn_scores_combined = compute_ood_scores(ood_combined, uc_combined, sigma_inv_combined, val_id_combined_normalized)

        # Evaluate OOD scores -> Maha Distance
        fpr95_tda, auc_tda, accuracy_tda = evaluate_ood(id_maha_scores_tda, maha_scores_tda, maha_threshold_tda)
        fpr95_cls, auc_cls, accuracy_cls = evaluate_ood(id_maha_scores_cls, maha_scores_cls, maha_threshold_cls)
        fpr95_combined, auc_combined, accuracy_combined = evaluate_ood(id_maha_scores_combined, maha_scores_combined, maha_threshold_combined)

        # Save Results
        maha_scores = {
            'tda': {'maha': {'fpr95': fpr95_tda, 'auc': auc_tda, 'accuracy': accuracy_tda}},
            'cls': {'maha': {'fpr95': fpr95_cls, 'auc': auc_cls, 'accuracy': accuracy_cls}},
            'combined': {'maha': {'fpr95': fpr95_combined, 'auc': auc_combined, 'accuracy': accuracy_combined}}
        }
        df_maha = create_result_df(maha_scores)

        # Evaluate OOD scores -> KNN
        fpr95_tda, auc_tda, accuracy_tda = evaluate_ood(id_knn_scores_tda, knn_scores_tda, knn_threshold_tda)
        fpr95_cls, auc_cls, accuracy_cls = evaluate_ood(id_knn_scores_cls, knn_scores_cls, knn_threshold_cls)
        fpr95_combined, auc_combined, accuracy_combined = evaluate_ood(id_knn_scores_combined, knn_scores_combined, knn_threshold_combined)

        # Save Results
        knn_scores = {
            'tda': {'knn': {'fpr95': fpr95_tda, 'auc': auc_tda, 'accuracy': accuracy_tda}},
            'cls': {'knn': {'fpr95': fpr95_cls, 'auc': auc_cls, 'accuracy': accuracy_cls}},
            'combined': {'knn': {'fpr95': fpr95_combined, 'auc': auc_combined, 'accuracy': accuracy_combined}}
        }
        df_knn = create_result_df(knn_scores)

        # Combine Maha and KNN results
        df_combined = pd.concat([df_maha, df_knn], axis=1)

        # Define saving directory and filename
        save_dir = os.path.join(results_dir, 'summary', f'{exp_files["graph_type"]}_{exp_files["symmetry"]}')
        os.makedirs(save_dir, exist_ok=True) # If directory exists, do nothing
        save_path = os.path.join(save_dir, f'{ood_name}.pkl')

        # Save the result
        df_combined.to_pickle(save_path)

if __name__ == "__main__":
    evaluate_experiment(exp_files)