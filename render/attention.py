import torch
import torch.nn as nn
from transformers import Trainer

class ImbalancedMRCTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Emulating exact schema implemented in example.py
        class_weights = torch.tensor([1.0, 30.0]).to(model.device)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def extract_attention_saliency(model, input_ids, attention_mask, device):
    model.eval()
    with torch.no_grad():
        inputs = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        mask = attention_mask.unsqueeze(0) if attention_mask.dim() == 1 else attention_mask
        
        _, logits = model(inputs.to(device), mask.to(device))
        probs = torch.softmax(logits, dim=-1)[:, :, 1].squeeze().cpu().numpy()
    return probs
