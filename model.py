import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from data_loader import DatasetLoader
from configs import TrainingConfig, ModelConfig

class ModelManager:
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        
        
        if self.model_config.load_from_dir:
            #self.model = BertForSequenceClassification.from_pretrained(self.model_config.model_path)
            self.model = BertModel.from_pretrained(self.model_config.model_path)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_config.model_path)
        else:
            self.model = BertModel.from_pretrained(self.model_config.model_name, output_attentions=True)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_config.model_name)
            
        self.model.to(self.model_config.device)
        if self.training_config.do_train:
            self.optimizer = AdamW(self.model.parameters(),
                                   lr=self.training_config.learning_rate,
                                   eps=self.training_config.adam_epsilon)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                             num_warmup_steps=self.training_config.warmup_steps,
                                                             num_training_steps=self.training_config.num_train_epochs)
    
    def train(self, dataset_loader: DatasetLoader):
        if not self.training_config.do_train:
            print("Training is turned off")
            return

        # Tokenize data
        train_encodings = self.tokenizer(dataset_loader.train_dataset, truncation=True, padding=True, max_length=self.model_config.max_seq_length, return_tensors="pt")
        val_encodings = self.tokenizer(dataset_loader.val_dataset, truncation=True, padding=True, max_length=self.model_config.max_seq_length, return_tensors="pt")

        # Convert tokenized data to DataLoader
        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(dataset_loader.train_labels))
        val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(dataset_loader.val_labels))
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.training_config.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=self.training_config.batch_size)

        # Define optimizer and criterion
        criterion = CrossEntropyLoss()

        # Training loop
        for epoch in range(self.training_config.num_train_epochs):
            self.model.train()
            for batch in train_loader:
                batch = tuple(b.to(self.device) for b in batch)
                inputs, masks, labels = batch
                outputs = self.model(inputs, attention_mask=masks, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Evaluation on validation set
            self.model.eval()
            total_loss = 0
            for batch in val_loader:
                batch = tuple(b.to(self.device) for b in batch)
                inputs, masks, labels = batch
                with torch.no_grad():
                    outputs = self.model(inputs, attention_mask=masks, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
            print(f"Validation Loss after epoch {epoch+1}: {total_loss/len(val_loader)}")
        
        # Save the model
        self.save_model()

    def evaluate(self, dataset_loader: DatasetLoader):
        test_encodings = self.tokenizer(dataset_loader.test_dataset, truncation=True, padding=True, max_length=self.model_config.max_seq_length, return_tensors="pt")
        test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(dataset_loader.test_labels))
        test_loader = DataLoader(test_dataset, batch_size=self.training_config.batch_size)
        
        self.model.eval()
        total_loss = 0
        for batch in test_loader:
            batch = tuple(b.to(self.device) for b in batch)
            inputs, masks, labels = batch
            with torch.no_grad():
                outputs = self.model(inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
        return total_loss/len(test_loader)
    
    def save_model(self):
        self.model.save_pretrained(self.model_config.model_path)
        self.tokenizer.save_pretrained(self.model_config.model_path)


""" Example usage
if __name__ == "__main__":
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    model_manager = ModelManager(model_config, training_config)
    id_dataset_config = {
        "name": "news-category",
        "test_size": 1000,
        "labels_subset": ["POLITICS", "ENTERTAINMENT"]
    }
    dataset_loader = DatasetLoader(id_dataset_config)
    model_manager.train(dataset_loader)

"""
