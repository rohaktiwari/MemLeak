import torch
import torch.nn as nn
from tqdm import tqdm
from .defenses.training import configure_optimizer, get_loss_criterion, privacy_training_wrapper

class Trainer:
    def __init__(self, model, train_loader, test_loader=None, device='cuda', config=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config or {}
        
        # Unpack config
        self.lr = self.config.get('learning_rate', 0.01)
        self.epochs = self.config.get('epochs', 10)
        self.weight_decay = self.config.get('weight_decay', 0.0)
        self.label_smoothing = self.config.get('label_smoothing', 0.0)
        self.target_epsilon = self.config.get('dp_epsilon', None)
        
        # Setup optimizer and loss
        self.optimizer = configure_optimizer(self.model, lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = get_loss_criterion(label_smoothing=self.label_smoothing)

        # Apply DP if configured
        if self.target_epsilon:
            self.model, self.optimizer, self.privacy_engine = privacy_training_wrapper(
                self.model, self.optimizer, self.train_loader, self.epochs, self.target_epsilon
            )
        else:
            self.privacy_engine = None

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch in pbar:
                if isinstance(batch, dict): # HF dataset
                    inputs = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask').to(self.device)
                    targets = batch['label'].to(self.device)
                    outputs = self.model(input_ids=inputs, attention_mask=attention_mask).logits
                else: # Generic torch dataset
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                    outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({'loss': total_loss/total, 'acc': 100.*correct/total})
            
            # Optional: Validate after epoch
            if self.test_loader:
                self.evaluate(self.test_loader)
        
        return self.model

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    inputs = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask').to(self.device)
                    targets = batch['label'].to(self.device)
                    outputs = self.model(inputs, attention_mask=attention_mask).logits
                else:
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                    outputs = self.model(inputs)
                    
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        print(f"Validation Acc: {acc:.2f}%")
        return acc

    def predict(self, loader):
        """Returns softmax probabilities and labels"""
        self.model.eval()
        probs_list = []
        targets_list = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    inputs = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask').to(self.device)
                    targets = batch['label'].to(self.device)
                    outputs = self.model(inputs, attention_mask=attention_mask).logits
                else:
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                    outputs = self.model(inputs)
                
                probs = torch.softmax(outputs, dim=1)
                probs_list.append(probs)
                targets_list.append(targets)
                
        return torch.cat(probs_list), torch.cat(targets_list)
