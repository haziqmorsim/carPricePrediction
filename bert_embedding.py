from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample description
desc = "Full service history, accident-free, great mileage."

# Tokenize
inputs = tokenizer(desc, return_tensors='pt', truncation=True, padding=True, max_length=64)
with torch.no_grad():
    outputs = model(**inputs)

# Get [CLS] token as sentence embedding
bert_embedding = outputs.last_hidden_state[:, 0, :].numpy()