import torch
import torch.nn as nn
from opacus import PrivacyEngine

def configure_optimizer(model, lr=0.01, weight_decay=0.0):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

def get_loss_criterion(label_smoothing=0.0):
    if label_smoothing > 0:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss()

def privacy_training_wrapper(model, optimizer, data_loader, epochs, target_epsilon, max_grad_norm=1.0):
    """
    Wraps model and optimizer with Opacus for DP-SGD.
    """
    if target_epsilon is None:
        return model, optimizer, None

    privacy_engine = PrivacyEngine()
    
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=1e-5,
        max_grad_norm=max_grad_norm,
    )
    
    return model, optimizer, privacy_engine
