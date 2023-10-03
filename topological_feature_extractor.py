from gtda.homology import VietorisRipsPersistence, FlagserPersistence
from gtda.diagrams import Amplitude, PersistenceEntropy
from sklearn.pipeline import make_pipeline, make_union
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from configs import TDAConfig


def get_distance_matrix(attention_matrix, tda_config: TDAConfig):      
    """
    Computes the distance matrix from the given attention matrix based on the provided configuration.

    Parameters:
    - attention_matrix (numpy.ndarray): The attention matrix to be converted into a distance matrix.
    - tda_config (TDAConfig): Configuration object that contains settings for the graph type and symmetry.

    Returns:
    - numpy.ndarray: The computed distance matrix.
    """

    if tda_config.graph_type == 'directed':
        distance_matrix = 1 - attention_matrix
        # Convert 0s to inf
        distance_matrix[distance_matrix == 0] = np.inf
        np.fill_diagonal(distance_matrix, 0)

    else:
        if tda_config.symmetry=='max':
            distance_matrix = 1 - np.maximum(attention_matrix, attention_matrix.T)
        elif tda_config.symmetry=='mean':
            distance_matrix = 1 - (attention_matrix + attention_matrix.T) / 2
        
        distance_matrix[distance_matrix == 0] = np.inf
        np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix

def get_dataset_diagrams(attentions, tda_config: TDAConfig):
    """
    Generate persistence diagrams for each sentence in the dataset based on their attention matrices.

    Parameters:
    - attentions (list of torch.Tensor): List of attention matrices from the model.
    - tda_config (TDAConfig): Configuration object with settings for persistence diagram computation.

    Returns:
    - list of list of numpy.ndarray: List of persistence diagrams for each sentence.
    """
    dataset_diagrams = []
    n_sentences = int(attentions[0].shape[0])
    
    # Get sentence diagrams
    for i in range(n_sentences):
        sentence_matrices = []

        # Loop over layers
        for layer in attentions:
            n_heads = layer.shape[1] # should be 12 for standard BERT base
            # Loop over heads in current layer
            for j in range(n_heads): 
                attention_matrix = layer[i][j].cpu().numpy()
                distance_matrix = get_distance_matrix(attention_matrix, tda_config)
                sentence_matrices.append(distance_matrix)


        # Define persistence type
        if tda_config.graph_type == 'directed':
            persistence = FlagserPersistence(homology_dimensions=tda_config.homology_dims)
        else:
            persistence = VietorisRipsPersistence(metric='precomputed',
                                                homology_dimensions=tda_config.homology_dims,
                                                collapse_edges=True)
        sentence_diagrams = persistence.fit_transform(sentence_matrices)
    
        dataset_diagrams.append(sentence_diagrams)
    
    return dataset_diagrams

def compute_sentence_features(args):
    """Helper function to compute features for a set of sentence diagrams."""
    sentence_diagrams, metrics, feature_union = args
    return feature_union.fit_transform(sentence_diagrams).flatten()


def get_dataset_features(dataset_diagrams, tda_config: TDAConfig):
    """
    Extracts topological features for each sentence's persistence diagrams.

    Parameters:
    - dataset_diagrams (list of list of numpy.ndarray): List of persistence diagrams for each sentence.
    - tda_config (TDAConfig): Configuration object with settings for amplitude metrics computation.

    Returns:
    - pandas DataFrame: DataFrame containing topological feature vectors for each sentence.
    """
    metrics = [
        {"metric": metric}
        for metric in tda_config.amplitude_metrics
    ]
    
    feature_union = make_union(
        *[PersistenceEntropy(nan_fill_value=-1)]
        + [Amplitude(**metric, n_jobs=-1) for metric in metrics]
    )
    
    all_features = []
    for sentence_diagrams in dataset_diagrams:
        # Compute features for this set of diagrams
        features = feature_union.fit_transform(sentence_diagrams).flatten()
        all_features.append(features)

    # Parallel processing
    """
    with Pool(processes=cpu_count()) as pool:
        args = [(sentence_diagrams, metrics, feature_union) for sentence_diagrams in dataset_diagrams]
        all_features = pool.map(compute_sentence_features, args)
    """

    # Return feature vector for each sentence
    # Generate dataframe with topological features
    columns = get_feature_names(tda_config)
    topological_features = pd.DataFrame(all_features, columns=columns)

    return topological_features



def get_feature_names(tda_config: TDAConfig):
    
    """
    Generate the feature names based on layer, head, dimension, and metric.

    Parameters:
    - tda_config (TDAConfig): Configuration object with settings related to homology dimensions and amplitude metrics.

    Returns:
    - list of str: List of feature names.
    """
  
    feature_names = []
    num_layers = 12
    num_heads = 12

    for layer_num in range(num_layers):
        for head_num in range(num_heads):
            # Names of the persistence entropy features
            persistence_entropy_feature_names = [f'{layer_num}_{head_num}_persistence_entropy_dim_{dim}' for dim in tda_config.homology_dims]

            # Names of the amplitude features
            amplitude_feature_names = [f'{layer_num}_{head_num}_amplitude_{metric}_dim_{dim}' for metric in tda_config.amplitude_metrics for dim in tda_config.homology_dims]

            # Combining all feature names
            feature_names += persistence_entropy_feature_names + amplitude_feature_names

    return feature_names