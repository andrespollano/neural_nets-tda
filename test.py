import json
import os
import argparse
from datetime import datetime
from configs import Config, DatasetConfig, TrainingConfig, TDAConfig
from data_loader import DatasetLoader
from model import ModelManager
from utils import save_model, save_model_outputs, load_model_outputs, save_diagrams, save_features, load_features, save_to_pkl, save_cls_embeddings
from topological_feature_extractor import get_dataset_diagrams, get_dataset_features
from model_utils import get_attention_weights, get_cls_embeddings


def test(config: Config):

    print("Loading data")
    # Load ID dataset
    id_dataset_loader = DatasetLoader(config.dataset.id_dataset, seed=config.base.seed)
    
    # Load model & tokenizer
    model_manager = ModelManager(config.model, config.training)

    start = datetime.now()
    id_val_cls_embeddings = get_cls_embeddings(model_manager.model, model_manager.tokenizer, id_dataset_loader.val_dataset, max_seq_length=config.model.max_seq_length)
    print(f"Saving ID validation model outputs. Duration: {datetime.now() - start}")
    save_cls_embeddings(id_val_cls_embeddings, config.base, file_name = f"{config.dataset.id_dataset['name']}_val")

    start = datetime.now()
    id_test_cls_embeddings = get_cls_embeddings(model_manager.model, model_manager.tokenizer, id_dataset_loader.test_dataset, max_seq_length=config.model.max_seq_length)
    print(f"Saving ID test model outputs. Duration: {datetime.now() - start}")
    save_cls_embeddings(id_test_cls_embeddings, config.base, file_name = f"{config.dataset.id_dataset['name']}_test")

    for ood_dataset_config in config.dataset.ood_datasets:

        #Load and process data
        ood_dataset_loader = DatasetLoader(ood_dataset_config, seed=config.base.seed)

        # Extract attention weights, embeddings and softmax probs
        start = datetime.now()
        ood_test_cls_embeddings = get_cls_embeddings(model_manager.model, model_manager.tokenizer, ood_dataset_loader.test_dataset, max_seq_length=config.model.max_seq_length)
        print(f"Saving {ood_dataset_config['name']} model outputs. Duration: {datetime.now() - start}")
        save_cls_embeddings(ood_test_cls_embeddings, config.base, file_name = f"{ood_dataset_config['name']}_ood_test")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load a configuration file in json format')
    parser.add_argument('--config', type=str, default=None, help='Config file name. If not provided, default config will be used')

    args = parser.parse_args()

    config = Config(config_file=args.config)

    test(config)