import argparse
import json
import os
from datetime import datetime
from configs import Config, DatasetConfig, TrainingConfig, TDAConfig
from data_loader import DatasetLoader
from model import ModelManager
from utils import save_model, save_model_outputs, load_model_outputs, load_diagrams, save_diagrams, save_features
from model_utils import get_model_outputs
from topological_feature_extractor import get_dataset_diagrams, get_dataset_features
#from ood_detector import mahalanobis_distance_to_centroids, distance_to_nearest_neighbour
#from evaluator import calculate_auroc, calculate_fpr95, generate_summary

def run_experiment(config):

    # Load ID dataset
    id_dataset_loader = DatasetLoader(config.dataset.id_dataset, seed=config.base.seed)
    
    # Load model & tokenizer
    model_manager = ModelManager(config.model, config.training)
    if config.training.do_train:
            model_manager.train(id_dataset_loader)
            print(f"Saving model to {config.model.model_path}")
            save_model(model_manager.model, model_manager.tokenizer, config.model.model_path)

    # Perform TDA on ID dataset
    if config.tda.do_id_tda:        

        # Extract attention weights and embeddings
        start = datetime.now()
        if config.base.load_model_outputs_from_dir:
            id_val_attentions, id_val_cls_embeddings = load_model_outputs(config, f"{config.dataset.id_dataset['name']}_val")
            id_test_attentions, id_test_cls_embeddings = load_model_outputs(config, f"{config.dataset.id_dataset['name']}_test")
        else:
            start = datetime.now()
            id_val_attentions, id_val_cls_embeddings = get_model_outputs(model_manager.model, model_manager.tokenizer, id_dataset_loader.val_dataset, max_seq_length=config.model.max_seq_length)
            print(f"Saving ID validation model outputs. Duration: {datetime.now() - start}")
            save_model_outputs(id_val_attentions, id_val_cls_embeddings, config.base, file_name = f"{config.dataset.id_dataset['name']}_val")

            start = datetime.now()
            id_test_attentions, id_test_cls_embeddings = get_model_outputs(model_manager.model, model_manager.tokenizer, id_dataset_loader.test_dataset, max_seq_length=config.model.max_seq_length)
            print(f"Saving ID test model outputs. Duration: {datetime.now() - start}")
            save_model_outputs(id_test_attentions, id_test_cls_embeddings, config.base, file_name = f"{config.dataset.id_dataset['name']}_test")


        # Generate persistence diagrams
        if config.base.load_diagrams_from_dir:
            id_val_diagrams = load_diagrams(config, f"{config.dataset.id_dataset['name']}_val")
            id_test_diagrams = load_diagrams(config, f"{config.dataset.id_dataset['name']}_test")
        else:
            start = datetime.now()
            id_val_diagrams = get_dataset_diagrams(id_val_attentions, config.tda)
            del id_val_attentions, id_val_cls_embeddings
            print(f"Saving ID validation diagrams. Duration: {datetime.now() - start}")
            save_diagrams(id_val_diagrams, config.base, f"{config.dataset.id_dataset['name']}_val")

            start = datetime.now()
            id_test_diagrams = get_dataset_diagrams(id_test_attentions, config.tda)
            del id_test_attentions, id_test_cls_embeddings
            print(f"Saving ID test diagrams. Duration: {datetime.now() - start}")
            save_diagrams(id_test_diagrams, config.base, f"{config.dataset.id_dataset['name']}_test")

        # Extract topological features
        start = datetime.now()
        id_val_topological_features = get_dataset_features(id_val_diagrams, config.tda)
        del id_val_diagrams
        print(f"Saving ID validation topological features. Duration: {datetime.now() - start}")
        save_features(id_val_topological_features, config.base, f"{config.dataset.id_dataset['name']}_val")

        start = datetime.now()
        id_test_topological_features = get_dataset_features(id_test_diagrams, config.tda)
        del id_test_diagrams
        print(f"Saving ID test topological features. Duration: {datetime.now() - start}")
        save_features(id_test_topological_features, config.base, f"{config.dataset.id_dataset['name']}_test")

        del id_dataset_loader


    # Perform TDA on OOD datasets
    if config.dataset.ood_datasets is not None:
        for ood_dataset_config in config.dataset.ood_datasets:

            #Load and process data
            ood_dataset_loader = DatasetLoader(ood_dataset_config, seed=config.base.seed)

            # Extract attention weights and embeddings
            if config.base.load_model_outputs_from_dir:
                try:
                    ood_test_attentions, ood_test_cls_embeddings = load_model_outputs(config, f"{ood_dataset_config['name']}_ood_test")
                except Exception as e:
                    print(f'Exception: {e}')
                    print('Loading model outputs from directory failed. Generating model outputs from scratch')
                    start = datetime.now()
                    ood_test_attentions, ood_test_cls_embeddings = get_model_outputs(model_manager.model, model_manager.tokenizer, ood_dataset_loader.test_dataset, max_seq_length=config.model.max_seq_length)
                    print(f"Saving {ood_dataset_config['name']} model outputs. Duration: {datetime.now() - start}")
                    save_model_outputs(ood_test_attentions, ood_test_cls_embeddings, config.base, file_name = f"{ood_dataset_config['name']}_ood_test")
            else:
                start = datetime.now()
                ood_test_attentions, ood_test_cls_embeddings = get_model_outputs(model_manager.model, model_manager.tokenizer, ood_dataset_loader.test_dataset, max_seq_length=config.model.max_seq_length)
                print(f"Saving {ood_dataset_config['name']} model outputs. Duration: {datetime.now() - start}")
                save_model_outputs(ood_test_attentions, ood_test_cls_embeddings, config.base, file_name = f"{ood_dataset_config['name']}_ood_test")

            # Generate persistence diagrams
            if config.base.load_diagrams_from_dir:
                try:
                    ood_test_attentions, ood_test_cls_embeddings = load_diagrams(config, f"{ood_dataset_config['name']}_ood_test")
                except Exception as e:
                    print(f'Exception: {e}')
                    print('Loading diagrams from directory failed. Generating diagrams instead')
                    start = datetime.now()
                    ood_test_diagrams = get_dataset_diagrams(ood_test_attentions, config.tda)
                    print(f"Saving {ood_dataset_config['name']} diagrams. Duration: {datetime.now() - start}")
                    save_diagrams(ood_test_diagrams, config.base, f"{ood_dataset_config['name']}_ood_test")


            else:
                start = datetime.now()
                ood_test_diagrams = get_dataset_diagrams(ood_test_attentions, config.tda)
                print(f"Saving {ood_dataset_config['name']} diagrams. Duration: {datetime.now() - start}")
                save_diagrams(ood_test_diagrams, config.base, f"{ood_dataset_config['name']}_ood_test")

            del ood_test_attentions, ood_test_cls_embeddings

            # Extract topological features
            start = datetime.now()
            ood_test_topological_features = get_dataset_features(ood_test_diagrams, config.tda)
            print(f"Saving {ood_dataset_config['name']} test topological features. Duration: {datetime.now() - start}")
            save_features(ood_test_topological_features, config.base, f"{ood_dataset_config['name']}_ood_test")

            del ood_test_diagrams



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load a configuration file in json format')
    parser.add_argument('--config', type=str, default=None, help='Config file name. If not provided, default config will be used')

    args = parser.parse_args()

    config = Config(config_file=args.config)
    run_experiment(config)