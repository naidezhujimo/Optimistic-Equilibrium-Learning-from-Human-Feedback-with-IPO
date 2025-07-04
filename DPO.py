import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class DPOTrainer:
    def __init__(self, model, ref_model, beta=0.1, lr=1e-5):
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def loss(self, batch):
        # batch: {input_ids, chosen_input_ids, rejected_input_ids}
        # 获取策略模型对数概率
        policy_chosen_logps = self._get_logps(batch['chosen_input_ids'])
        policy_rejected_logps = self._get_logps(batch['rejected_input_ids'])

        # 获取参考模型对数概率
        with torch.no_grad():
            ref_chosen_logps = self._get_logps(batch["chosen_input_ids"], self.ref_model)
            ref_rejected_logps = self._get_logps(batch["rejected_input_ids"], self.ref_model)
        
        # 计算对数比值
        logits = self.beta * (
            (policy_chosen_logps - ref_chosen_logps) -
            (policy_rejected_logps - ref_rejected_logps)
        )
        loss = -F.logsigmoid(logits)
        return loss

    def _get_logps(self, input_ids, model=None):
        model = model or self.model
        outputs = model(input_ids, labels=input_ids)
        logps = -outputs.loss * input_ids.size(1)  # 计算序列平均对数概率
        return logps
    
    def train_step(self, batch):
        self.optimizer.zero_grad()
        loss = self.loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()


model = AutoModelForCausalLM.from_pretrained("gpt2")
ref_model = AutoModelForCausalLM.from_pretrained("gpt2")  # 通常与初始模型相同
trainer = DPOTrainer(model, ref_model, beta=0.1)

for batch in dataloader:
    loss = trainer.train_step(batch)
    
