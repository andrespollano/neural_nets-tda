import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from scipy.spatial import distance
import numpy as np
from ood_detector import prepare_ood, compute_ood_scores, get_thresholds, evaluate_ood
from utils import load_from_pkl, load_from_pt_to_df, compute_min_mahalanobis
from configs import Config
from data_loader import DatasetLoader

exp_files = {
    "graph_type" : "undir",
    "symmetry" : "max",
    "id_name" : 'news-category',
    "ood_names" : ['imdb', 'cnn_dailymail', 'news-category'],
}

seed = 42
id_dataset = {
            "name": "news-category",
            "train_size": 30000,
            "val_size": 1000,
            "test_size": 1000,
            "labels_subset": ["POLITICS", "ENTERTAINMENT"]
        }
ood_datasets = [
            {
                "name": "imdb",
                "train_size": 0,
                "val_size": 0,
                "test_size": 1000
            },
            {
                "name": "cnn_dailymail",
                "train_size": 0,
                "val_size": 0,
                "test_size": 1000
            },
            {
                "name": "news-category",
                "train_size": 0,
                "val_size": 0,
                "test_size": 1000,
                "labels_subset": ["BUSINESS"]
            }]


root_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(os.path.dirname(root_dir), 'final_files')
data_dir = os.path.join(results_dir, 'data_labels')
mode = 'fine-tuned' #pre-trained or fine-tuned

sentence_embeddings_dir = os.path.join(results_dir, mode, 'sentence_embeddings')
top_features_dir = os.path.join(results_dir, mode, 'topological_features')

id_dataset_loader = DatasetLoader(id_dataset, seed=seed)

# Load ID Val labels
id_val_labels = load_from_pkl(data_dir, 'id_val_labels.pkl')

# Load ID topological features
top_features_dir = os.path.join(results_dir, mode, 'topological_features')
id_val_tda = load_from_pkl(top_features_dir, f'{exp_files["id_name"]}_val.pkl')
id_test_tda = load_from_pkl(top_features_dir, f'{exp_files["id_name"]}_test.pkl')

# Load CLS embeddings
id_val_cls = load_from_pt_to_df(sentence_embeddings_dir, f'{exp_files["id_name"]}_val.pt')
id_test_cls = load_from_pt_to_df(sentence_embeddings_dir, f'{exp_files["id_name"]}_test.pt')

# Combine features
id_val_combined = pd.concat([id_val_cls, id_val_tda], axis=1)
id_test_combined = pd.concat([id_test_cls, id_test_tda], axis=1) 

results = []

# Compute OOD scores
for dataset_index, ood_name in enumerate(exp_files['ood_names']):

    ood_dataset_loader = DatasetLoader(ood_datasets[dataset_index], seed=seed)

    # Load features
    ood_tda = load_from_pkl(top_features_dir, f'{ood_name}_ood_test.pkl')
    ood_cls = load_from_pt_to_df(sentence_embeddings_dir, f'{ood_name}_ood_test.pt')
    ood_combined = pd.concat([ood_cls, ood_tda], axis=1)

    feature_sets = ['TDA', 'CLS', 'Combined']
    id_val_features = [id_val_tda, id_val_cls, id_val_combined]
    id_test_features = [id_test_tda, id_test_cls, id_test_combined]
    ood_features = [ood_tda, ood_cls, ood_combined]

    feature_results = []
    for idx in range(len(feature_sets)):
        #print(feature_sets[idx])

        # Normalize embeddings
        scaler = StandardScaler().fit(id_val_features[idx])
        test_data_normalized = scaler.transform(id_test_features[idx])
        val_data_normalized = scaler.transform(id_val_features[idx])
        ood_data_normalized = scaler.transform(ood_features[idx])

        # Apply dimensionality reduction
        dim_reduction = False
        if dim_reduction:
            n_components = 3
            dim_reductor = PCA(n_components=n_components)
            #dim_reductor = UMAP(n_components=n_components)

            val_data_reduced = dim_reductor.fit_transform(val_data_normalized)
            test_data_reduced = dim_reductor.transform(test_data_normalized)
            ood_data_reduced = dim_reductor.transform(ood_data_normalized)
        else:
            test_data_reduced = test_data_normalized
            val_data_reduced = val_data_normalized
            ood_data_reduced = ood_data_normalized

        # Compute class centroids for Mahalanobis distance
        unique_labels = np.unique(id_val_labels)
        class_centroids = {}

        for c in unique_labels:
            class_indices = [i for i, label in enumerate(id_val_labels) if label == c]
            class_samples = val_data_reduced[class_indices]
            centroid = np.mean(class_samples, axis=0)
            class_centroids[c] = centroid

        cov_matrix = np.cov(val_data_reduced, rowvar=False)
        cov_inv = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6) # Add small epsilon to the diagonal for regularisation

        # Fit KNN to top_features_val and Evaluate the performance using KNN-based OOD detection
        K = 5
        knn = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(val_data_reduced)

        distances_test, _ = knn.kneighbors(test_data_reduced)
        mean_distance_to_kth_neighbour_test = distances_test[:, -1]
        threshold = np.percentile(mean_distance_to_kth_neighbour_test, 95)

        distances_ood, _ = knn.kneighbors(ood_data_reduced)
        mean_distance_to_kth_neighbour_ood = distances_ood[:, -1]

        """
        if feature_sets[idx] == 'CLS':

            most_confident_ood_indices = np.argsort(mean_distance_to_kth_neighbour_ood)[-5:]
            least_confident_ood_indices = np.argsort(mean_distance_to_kth_neighbour_ood)[:5]

            most_confident_oods = [ood_dataset_loader.test_dataset[i] for i in most_confident_ood_indices]
            least_confident_oods = [ood_dataset_loader.test_dataset[i] for i in least_confident_ood_indices]

            
            # Get the nearest neighbor indices in val_data_reduced for the least confident OODs
            _, nearest_neighbour_indices_array = knn.kneighbors(ood_data_reduced[least_confident_ood_indices])
            nearest_neighbour_indices = nearest_neighbour_indices_array[:, 0]  # Get the first nearest neighbor for each sample

            print(ood_name)
            print("Most confident OODs:")
            for instance in most_confident_oods:
                print()
                print(instance)

            print("Least confident OODs:")
            for idx, instance in enumerate(least_confident_oods):
                print()
                print(instance)
                print('-> Nearest Neighbour:')
                print(nearest_neighbour_indices[idx])
                print(id_dataset_loader.val_dataset[nearest_neighbour_indices[idx]])
            """

        y_scores_knn = np.concatenate([-mean_distance_to_kth_neighbour_test, -mean_distance_to_kth_neighbour_ood])
        y_true = np.concatenate([np.ones(len(test_data_reduced)), np.zeros(len(ood_data_reduced))]) # 1 is ID, 0 is OOD

        threshold_95_tpr_knn = np.percentile(y_scores_knn[y_true==1], 5)
        FPR95_knn = np.mean(y_scores_knn[y_true==0] > threshold_95_tpr_knn)
        AUROC_knn = roc_auc_score(y_true, y_scores_knn)

        knn_results = [AUROC_knn, FPR95_knn]

        # Evaluate the performance using Mahalanobis-based OOD detection
        mahalanobis_test = [compute_min_mahalanobis(x, class_centroids, cov_inv) for x in test_data_reduced]
        mahalanobis_ood = [compute_min_mahalanobis(x, class_centroids, cov_inv) for x in ood_data_reduced]

        # Find the indices for the 5 highest and 5 lowest Mahalanobis distances for the OOD data
        """
        if feature_sets[idx] == 'TDA':

            most_confident_ood_indices = np.argsort(mahalanobis_ood)[-5:]
            least_confident_ood_indices = np.argsort(mahalanobis_ood)[:5]

            most_confident_oods = [ood_dataset_loader.test_dataset[i] for i in most_confident_ood_indices]
            least_confident_oods = [ood_dataset_loader.test_dataset[i] for i in least_confident_ood_indices]

            print(ood_name)
            print("Most confident OODs:")
            for instance in most_confident_oods:
                print()
                print(instance)

            print("Least confident OODs:")
            for instance in least_confident_oods:
                print()
                print(instance)
        """

        y_scores_maha = np.concatenate([-np.array(mahalanobis_test), -np.array(mahalanobis_ood)]) # Take negative of the Mahalanobis distances, lower distance means in-distribution
        threshold_95_tpr_maha = np.percentile(y_scores_maha[y_true==1], 5)
        FPR95_maha = np.mean(y_scores_maha[y_true==0] > threshold_95_tpr_maha) # > because lower distance means in-distribution
        AUROC_maha = roc_auc_score(y_true, y_scores_maha) 

        maha_results = [AUROC_maha, FPR95_maha]

        feature_results.append(knn_results + maha_results)
    
    results.append((ood_name, feature_results))

# Print formatted results
for ood_name, feature_results in results:
    print(f"\n{ood_name}")
    print("-" * 60)
    print(f"{'':<10} | {'KNN_AUROC':<10} | {'KNN_FPR95':<10} | {'MAHA_AUROC':<10} | {'MAHA_FPR95':<10}")
    print("-" * 60)
    for idx, res in enumerate(feature_results):
        print(f"{feature_sets[idx]:<10} | {res[0]:<10.3f} | {res[1]:<10.3f} | {res[2]:<10.3f} | {res[3]:<10.3f}")