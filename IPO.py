import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class IPOTrainer:
    def __init__(self, model, ref_model, eta=0.1, lr=1e-5):
        self.model = model
        self.ref_model = ref_model
        self.eta = eta
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    def loss(self, batch):
        policy_chosen_logps = self._get_logps(batch['chosen_input_ids'])
        policy_rejected_logps = self._get_logps(batch['rejected_input_ids'])

        with torch.no_grad():
            ref_chosen_logps = self._get_logps(batch['chosen_input_ids'], self.ref_model)
            ref_rejected_logps = self._get_logps(batch['rejected_input_ids'], self.ref_model)

        log_ratio_diff = (
            (policy_chosen_logps - ref_chosen_logps) - 
            (policy_rejected_logps - ref_rejected_logps)
        )
        loss = (log_ratio_diff - 1/(2 * self.eta)) ** 2

        return loss

    def _get_logps(self, input_ids, model=None):
        model = self.model or model
        outputs = model(input_ids, model)
        logps = -outputs.loss() * input_ids.size(1)
        return logps
    
    def train_step(self, batch):
        self.optimizer.zero_grad()
        loss = self.loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

ipo_trainer = IPOTrainer(model, ref_model, eta=0.1)