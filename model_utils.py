import torch

def get_model_outputs(model, tokenizer, sentences, max_seq_length=128):
    inputs = tokenizer(sentences, return_tensors="pt", padding='max_length', truncation=True, max_length=max_seq_length)

    # Feed sentences to BERT model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the attention weights and CLS embeddings from the output
    attentions = outputs.attentions # This will be a tuple with 12 elements for BERT base (each torch.Size([n_instances, 12, 128, 128]))
    cls_embeddings = outputs.last_hidden_state[:, 0, :] # This will be a torch.Size([n_instances, 768])

    return attentions, cls_embeddings


def get_cls_embeddings(model, tokenizer, sentences, max_seq_length=128):
    inputs = tokenizer(sentences, return_tensors="pt", padding='max_length', truncation=True, max_length=max_seq_length)

    # Feed sentences to BERT model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the attention weights from the output
    cls_embeddings = outputs.last_hidden_state[:, 0, :] # This will be a torch.Size([n_instances, 768])

    return cls_embeddings
