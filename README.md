# Topological Data Analysis for OOD Detection using BERT

This repository contains the code for our research paper "<Paper Name>" (TODO: Add paper name and link)

---

## Getting Started

Create and activate a new virtual environment for the project and install all the necessary packages using the provided requirements file.
1. 'conda create -n tda_ood python==3.9'
2. 'conda activate tda_ood'
3. 'pip install -r requirements.txt'

Clone this directory, and set up two new directories:
1. Model directory - this should contain a subdirectory: 'fine_tuned_models'
2. Output directory - this is where results and intermediate files will be stored

Note: The locations of both directories should be specified in config.py. If not, they will be auto-created in the parent directory of the repository root.

Both directories can be located anywhere but should be specified in 'config.py'. If not specified, the directories will be created one directory above the root.
if you wish to use the fine-tuned model used in the paper, download and extract the contents from Figshare <TODO: Insert link> to the 'fine_tuned_models' subdirectory.

## Running experiments

### Configuration: 
Adjust your experimental parameters in config.py or use a specific config JSON file located in the config_files directory.

### Default Run
To run experiments with default configurations from 'config.py', execute:
'python run_tda.py'

### Custom run configuration
To specify a particular configuration from the config_files directory:
'python run_tda.py --config= <confif_filename>.json'

The JSON file should contain the custom experiment configurations with the following keys:
- "base": directory paths and basic configurations, for example:
    - 'MODEL_DIR' : Path to the Model directory
    - 'RESULTS_DIR' : Path to the Output directory
    - 'load_model_outputs_from_dir': If true, persistence diagrams will be generated from saved model outputs (attentions and sentence embeddings) from the 'experiment_loaddir_name' subdirectory
    - 'load_diagrams_from_dir: If true, topological features will be calculated from the saved diagrams from the 'experiment_loaddir_name' subdirectory
- "model": model specs
    - 'load_from_dir': If true, fine-tuned model parameters will be loaded. Directory can be specified in 'model_path'
- "training": fine-tuning model configurations
    - 'do_train': If true, BERT model will be fine-tuned with in-distribution data with specified hyper-parameters (e.g. 'num_train_epochs', 'batch_size', etc.)
- "dataset": dictionary (for 'id_dataset) or list of dictionaries (for ood_datasets) with configurations to load datasets from HuggingFace. Specifications include:
    - 'name': name of the HuggingFace dataset
    - number of samples to load as 'train_size', 'val_size' and 'test_size'
    - 'labels_subset' (Optional):  list of labels to load
- "tda": configurations to run Topological Data Analysis, for example:
    - 'do_id_tda': If true, topological features will be calculated for the in-distribution dataset
    - 'graph_type': 'undirected' or 'directed'
    - 'symmetry': 'mean' or 'max' attention for undirected graphs
    - 'homology_dims': list of dimensions (e.g. [0,1,2])
    - 'amplitude_metrics': list of amplitude metrics supported by gta-tda library

 Run the following command to run the experiments with the default configurations in configs.py
 'python run_tda.py'

 The values in configs.py can be updated using a json file in the 'config_files' directory with the following command:
 'python run_tda.py --config= <confif_filename>.json'

 ## Citation
TODO: Add citations once paper is uploaded

