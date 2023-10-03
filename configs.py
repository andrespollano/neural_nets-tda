import os
import json
import torch
from datetime import datetime


class BaseConfig(object):
    def __init__(self):
        # Experiment configs
        self.experiment_date = datetime.now().strftime("%Y-%b-%d_%H:%M")
        self.experiment_loaddir_name = self.experiment_date 
        self.set_seed = True
        self.seed = 42

        self.load_model_outputs_from_dir = False
        self.load_diagrams_from_dir = False

        # Set paths
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.MODEL_DIR = os.path.join(os.path.dirname(self.root_dir), 'models') 
        self.DATA_DIR = os.path.join(os.path.dirname(self.root_dir), 'data') # One dir above root_dir
        self.RESULTS_DIR = os.path.join(os.path.dirname(self.root_dir), 'output') # One dir above root_dir

        self.attentions_dir = os.path.join(self.RESULTS_DIR, 'attentions', self.experiment_date)
        self.diagrams_dir = os.path.join(self.RESULTS_DIR, 'diagrams', self.experiment_date)
        self.sent_embeddings_cls_dir = os.path.join(self.RESULTS_DIR, 'sentence_embeddings_cls', self.experiment_date)
        self.topological_features_dir = os.path.join(self.RESULTS_DIR, 'topological_features', self.experiment_date)
        self.ood_scores_dir = os.path.join(self.RESULTS_DIR, 'ood_scores', self.experiment_date)


    def update_config(self, config_dict):
        """Update config with a dictionary"""
        for key, value in config_dict.items():
            if not hasattr(self, key):
                raise ValueError(f'{key} is not a valid config key')
            setattr(self, key, value)
        

class ModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = 'bert-base-uncased'
        self.load_from_dir = True
        self.load_model_date = ""
        self.model_path = os.path.join(self.MODEL_DIR, 'finetuned_models')
        self.max_seq_length = 128
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()


class DatasetConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # Dataset configs
        self.id_dataset = {
            "name": "news-category",
            "train_size": 30000,
            "val_size": 1000,
            "test_size": 1000,
            "labels_subset": ["POLITICS", "ENTERTAINMENT"]
        }
        self.ood_datasets = [
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
            }
        ]

class TrainingConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # Training configs
        self.do_train = False
        self.num_train_epochs = 5
        self.batch_size = 32
        self.learning_rate = 1e-5
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.weight_decay = 0.01

class TDAConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.do_id_tda = True
        self.graph_type = 'undirected' # 'directed' or 'undirected'
        self.symmetry = 'max' # 'max' or 'mean' for undirected graphs
        self.homology_dims = [0, 1, 2, 3]
        self.amplitude_metrics = ["bottleneck", "wasserstein"]


class Config:
    def __init__(self, config_file=None):
        super().__init__()
        self.base = BaseConfig()
        self.model = ModelConfig()
        self.dataset = DatasetConfig()
        self.training = TrainingConfig()
        self.tda = TDAConfig()
        self.model.model_path = os.path.join(self.base.MODEL_DIR, 'fine_tuned_models', f'{self.dataset.id_dataset["name"]}_{self.base.experiment_date}')

        config_file_path = os.path.join(self.base.root_dir, 'config_files', f'{config_file}')
        if os.path.exists(config_file_path):
            self.update_global_config(json.load(open(config_file_path)))
            print('Loaded config from file')
        
        if self.model.load_from_dir:
            self.model.model_path = os.path.join(self.base.MODEL_DIR, 'fine_tuned_models', f'{self.dataset.id_dataset["name"]}_{self.model.load_model_date}')

    def update_global_config(self, config_dict):
        """Update config with a dictionary"""
        for key, value in config_dict.items():
            if not hasattr(self, key):
                raise ValueError(f'{key} is not a valid config key')
            # If value is another dictionary, assume nested configuration and recurse
            if isinstance(value, dict) and hasattr(getattr(self, key), 'update_config'):
                getattr(self, key).update_config(value)
            else:
                setattr(self, key, value)

        
        if self.base.load_model_outputs_from_dir:
            self.attentions_dir = os.path.join(self.base.RESULTS_DIR, 'attentions', self.base.experiment_loaddir_name)
            self.sent_embeddings_cls_dir = os.path.join(self.base.RESULTS_DIR, 'sentence_embeddings_cls', self.base.experiment_loaddir_name)

        if self.base.load_diagrams_from_dir:
            self.diagrams_dir = os.path.join(self.base.RESULTS_DIR, 'diagrams', self.base.experiment_loaddir_name)