import os
import torch
import pickle
import pandas as pd
from configs import BaseConfig
from scipy.spatial import distance


def save_model(model, tokenizer, model_path):
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

def load_model(model_path):
    model = torch.load(os.path.join(model_path, 'pytorch_model.bin'))
    tokenizer = torch.load(os.path.join(model_path, 'tokenizer_config.json'))
    return model, tokenizer

def save_model_outputs(attentions, cls_embeddings, base_config : BaseConfig, file_name):
    # If directory does not exist, create it
    if not os.path.exists(base_config.attentions_dir):
        os.makedirs(base_config.attentions_dir)
    if not os.path.exists(base_config.sent_embeddings_cls_dir):
        os.makedirs(base_config.sent_embeddings_cls_dir)

    attentions_path = os.path.join(base_config.attentions_dir, file_name + '.pt')
    cls_embeddings_path = os.path.join(base_config.sent_embeddings_cls_dir, file_name + '.pt')

    torch.save(attentions, attentions_path)
    torch.save(cls_embeddings, cls_embeddings_path)

def save_cls_embeddings(cls_embeddings, base_config : BaseConfig, file_name):
    # If directory does not exist, create it
    if not os.path.exists(base_config.sent_embeddings_cls_dir):
        os.makedirs(base_config.sent_embeddings_cls_dir)

    cls_embeddings_path = os.path.join(base_config.sent_embeddings_cls_dir, file_name + '.pt')

    torch.save(cls_embeddings, cls_embeddings_path)

def load_model_outputs(base_config : BaseConfig, file_name):
    attentions_path = os.path.join(base_config.attentions_dir, file_name + '.pt')
    cls_embeddings_path = os.path.join(base_config.sent_embeddings_cls_dir, file_name + '.pt')

    # Check if file exists
    if not os.path.exists(attentions_path) or not os.path.exists(cls_embeddings_path):
        raise Exception(f"File {attentions_path} or {cls_embeddings_path} does not exist") 

    attentions = torch.load(attentions_path)
    cls_embeddings = torch.load(cls_embeddings_path)

    return attentions, cls_embeddings

def save_diagrams(diagrams, base_config : BaseConfig, file_name):
    # If directory does not exist, create it
    if not os.path.exists(base_config.diagrams_dir):
        os.makedirs(base_config.diagrams_dir)

    diagrams_path = os.path.join(base_config.diagrams_dir, file_name + '.pkl')

    with open(diagrams_path, 'wb') as f:
        pickle.dump(diagrams, f)


def load_diagrams(base_config : BaseConfig, file_name):
    diagrams_path = os.path.join(base_config.diagrams_dir, file_name + '.pkl')

    # Check if file exists
    if not os.path.exists(diagrams_path):
        raise Exception(f"File {diagrams_path} does not exist") 

    with open(diagrams_path, 'rb') as f:
        diagrams = pickle.load(f)

    return diagrams

def save_features(features, base_config : BaseConfig, file_name):
    # If directory does not exist, create it
    if not os.path.exists(base_config.topological_features_dir):
        os.makedirs(base_config.topological_features_dir)

    features_path = os.path.join(base_config.topological_features_dir, file_name + '.pkl')

    with open(features_path, 'wb') as f:
        pickle.dump(features, f)

def load_features(base_config : BaseConfig, file_name):
    features_path = os.path.join(base_config.topological_features_dir, file_name + '.pkl')

    # Check if file exists
    if not os.path.exists(features_path):
        raise Exception(f"File {features_path} does not exist") 

    with open(features_path, 'rb') as f:
        features = pickle.load(f)

    return features

def save_to_pkl(obj, dir_path, file_name):
    # If directory does not exist, create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = os.path.join(dir_path, file_name)

    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_from_pkl(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)

    # Check if file exists
    if not os.path.exists(file_path):
        raise Exception(f"File {file_path} does not exist") 

    with open(file_path, 'rb') as f:
        obj = pickle.load(f)

    return obj

def load_from_pt_to_df(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)

    # Check if file exists
    if not os.path.exists(file_path):
        raise Exception(f"File {file_path} does not exist") 
    
    tensor = torch.load(file_path)

    array = tensor.cpu().numpy()

    df = pd.DataFrame(array)

    return df


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


def compute_min_mahalanobis(x, centroids, cov_inv):
    distances = []
    for _, centroid in centroids.items():      
        dist = distance.mahalanobis(x, centroid, cov_inv)
        distances.append(dist)
    return min(distances)