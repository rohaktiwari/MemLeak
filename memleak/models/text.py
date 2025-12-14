import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class TextClassifier(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2):
        super(TextClassifier, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)
