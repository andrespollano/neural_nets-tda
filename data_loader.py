from datasets import load_dataset, Dataset, concatenate_datasets, get_dataset_split_names
from datasets import Dataset, Features, Value
import numpy as np

# Splits to download from HuggingFace
download_splits = {
    'news-category': ['train'],
    'imdb': ['test'],
    'cnn_dailymail': ['test']
}

huggingface_datasets = {
    'news-category': 'heegyu/news-category-dataset',
    'imdb': 'imdb',
    'cnn_dailymail': 'cnn_dailymail'
}

class DatasetLoader():
    def __init__(self, dataset_config, seed=42):
        self.dataset_name = dataset_config["name"]
        self.seed = seed

        # Load specified classes only
        if "labels_subset" in dataset_config:
            self.labels_subset = dataset_config["labels_subset"]
        else:
            self.labels_subset = None

        self.test_size = dataset_config["test_size"]
        self.test_dataset = None
        self.test_labels = None

        self.train_size = dataset_config["train_size"]
        self.train_dataset = None
        self.train_labels = None

        self.val_size = dataset_config["val_size"]
        self.val_dataset = None
        self.val_labels = None

        np.random.seed(self.seed)

        self.load_data()

    def load_data(self):
        
        # Load full dataset
        loaded_splits = []
        splits = download_splits[self.dataset_name]
        for split in splits:
            try:
                # Attempt to load the dataset split without specifying a version.
                loaded_splits.append(load_dataset(huggingface_datasets[self.dataset_name], split=split))
            except ValueError as e:
                # If an error occurs and the dataset is cnn_dailymail, attempt to load with version '3.0.0'.
                if "Config name is missing" in str(e) and self.dataset_name == 'cnn_dailymail':
                    loaded_splits.append(load_dataset(huggingface_datasets[self.dataset_name], '3.0.0', split=split))
                else:
                    # If the error is something else, raise the original exception.
                    raise e
        
        dataset = concatenate_datasets(loaded_splits)

        # Setect dataset features to load
        if self.dataset_name == 'news-category':
            dataset = dataset.map(function=self.filter_features)
        elif self.dataset_name == 'imdb':
            dataset = dataset.cast_column("label", Value(dtype='string', id=None))
            dataset = dataset.map(function=self.filter_features)
        elif self.dataset_name == 'cnn_dailymail':
            dataset = dataset.map(function=self.filter_features)
            dataset = dataset.remove_columns(["article"])

        # Filter labels
        if self.labels_subset is not None:
            # Separate instances by label
            label_to_instances = {}
            for label in self.labels_subset:
                label_to_instances[label] = dataset.filter(lambda instance: instance['label'] == label)
            
            # Balance classes
            min_count = min([len(label_instances) for label_instances in label_to_instances.values()])
            balanced_instances = []
            for label_instances in label_to_instances.values():
                sampled_instances = np.random.choice(label_instances, min_count, replace=False).tolist()
                balanced_instances.extend(sampled_instances)
            
            # Convert list of dictionaries to Dataset
            dataset = Dataset.from_dict({key: [d[key] for d in balanced_instances] for key in balanced_instances[0]})
            
        

        # Split into train and test
        split_data = dataset.train_test_split(test_size=self.test_size, seed=self.seed)
        self.test_dataset = [instance['text'] for instance in split_data["test"]]
        self.test_labels = [instance['label'] for instance in split_data["test"]]
        if self.val_size >= 1:
            train_split_data = split_data["train"].train_test_split(test_size=self.val_size, seed=self.seed)
            if self.train_size > 0:
                self.train_dataset = [instance['text'] for instance in train_split_data["train"]][:self.train_size]
                self.train_labels = [instance['label'] for instance in train_split_data["train"]][:self.train_size]
            self.val_dataset = [instance['text'] for instance in train_split_data["test"]]
            self.val_labels = [instance['label'] for instance in train_split_data["test"]]



    def filter_features(self, instance):
        """Filter only desired text feature, and label"""
        if self.dataset_name == 'news-category':
            instance['text'] = instance['headline'] + ". " + instance['short_description']
            instance['label'] = instance['category']
            return {'text': instance['text'], 'label': instance['label']}
        
        elif self.dataset_name == 'imdb':
            instance['label'] = "OOD"
            return {'text': instance['text'], 'label': instance['label']}
        
        elif self.dataset_name == 'cnn_dailymail':
            instance['text'] = instance['highlights']
            instance['label'] = "OOD"
            return {'text': instance['text'], 'label': instance['label']}



""" Example usage 
id_dataset = {
            "name": "news-category",
            "test_size": 1000,
            "labels_subset": ["POLITICS", "ENTERTAINMENT"]
            }

id_dataset_loader = DatasetLoader(id_dataset)

"""