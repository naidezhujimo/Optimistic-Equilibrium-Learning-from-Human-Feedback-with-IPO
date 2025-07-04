import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm


class PreferenceModel(nn.Module):
    # 偏好模型（用于预测响应偏好）
    def __init__(self, model_name='roberta_base'):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, prompts, responses_a, responses_b):
        # 构建输入文本：[CONTEXT]{prompt}[RESPONSE_A]{response_a}[RESPONSE_B]{response_b}
        texts = [
            f"[CONTEXT]{p}[RESPONSE_A]{a}[RESPONSE_B]{b}" 
            for p, a, b in zip(prompts, responses_a, responses_b)
        ]
        input = self.tokenizer(
            texts,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 预测偏好概率 P(a > b)
        logits = self.model(**inputs).logits
        return torch.sigmoid(logits).squeeze(-1)
    
    def predict_preference(self, prompt, response_a, response_b):
        self.eval()
        with torch.no_grad():
            prob = self([prompt], [response_a], [response_b]).item()
        return prob
    
class PreferenceDataset(Dataset):
    # 偏好数据集
    def __init__(self, prompts, responses_a, responses_b, preferences):
        self.prompts = prompts
        self.responses_a = responses_a
        self.responses_b = responses_b
        self.preferences = preferences  # 1: a > b, 0: b > a
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "response_a": self.responses_a[idx],
            "response_b": self.responses_b[idx],
            "preference": self.preferences[idx]
        }
    
# 偏好预言机
class PreferenceOracle:
    def __init__(self, ground_truth_model=None):
        self.ground_truth = ground_truth_model
    
    def query(self, prompt, response_a, response_b):
        if self.ground_truth:
            # 如果有模型，使用其预测
            return 1 if self.ground_truth.predict_preference(
                prompt, response_a, response_b) > 0.5 else 0
        else:
            # 更长的响应被偏好
            return 1 if len(response_a) > len(response_b) else 0


class OnlineELHFIPOTrainer:
    def __init__(self, base_model, pref_model, oracle, tokenizer, eta=0.1, n_candidates=4, lr=1e-5, device='cuda'):
        self.policy = base_model.to(device)  # 主代理策略
        self.ref_model = base_model.to(device)  # 参考模型
        self.pref_model = pref_model.to(device)  # 偏好模型
        self.oracle = oracle   # 偏好预言机
        self.tokenizer = tokenizer
        self.eta = eta
        self.n_candidates = n_candidates
        self.lr = lr
        self.device = device

        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
    def rejection_sampling(self, prompt):
        # 拒绝采样生成增强器响应
        # 1. 主代理生成候选响应
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        candidates = self.policy.generate(
            **inputs,
            max_new_tokens=128,
            num_return_sequences=self.n_candidates,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        candidates = [self.tokenizer.decode(
            o, skip_special_tokens=True) for o in candidates]
        
        # 2. 偏好模型锦标赛排序
        if len(candidates) < 2:
            return candidates[0] if candidates else ""
        
        # 计算每个候选的得分
        scores = np.zeros(len(candidates))
        for i in range(len(candidates)):
            for j in range(len(candidates)):
                if i != j:
                    prob = self.pref_model.predict_preference(
                        prompt, candidates[i], candidates[j]
                    )
                    scores[i] += prob

        # 3. 选择最优响应
        best_ids = np.argmax(scores)
        return candidates[best_ids]
    
    def train_pref_model(self, dataset, epochs=1, batch_size=8):
        # 训练偏好模型
        self.pref_model.train()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adamw(self.pref_model.parameters(), lr=self.lr)

        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()

                # 获取预测概率
                probs = self.pref_model(
                    batch["prompt"],
                    batch["response_a"],
                    batch["response_b"]
                )

                # 计算二元交叉熵损失
                targets = torch.tensor(
                    batch['preference'], dtype=torch.float32
                ).to(self.device)
                loss = F.binary_cross_entropy(probs, targets)

                loss.backward()
                optimizer.step()

    def train_policy_with_ipo(self, dataset, epochs=1, batch_size=4):
        self.policy.train()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.lr * 0.1)

        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                
                inputs_chosen = self.tokenizer(
                    batch["chosen_response"], 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(self.device)
                
                inputs_rejected = self.tokenizer(
                    batch["rejected_response"], 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(self.device)
                
                # 获取策略模型对数概率
                outputs_chosen = self.policy(**inputs_chosen, labels=inputs_chosen['input_ids'])
                policy_chosen_logps = -outputs_chosen.loss * inputs_chosen['input_ids'].size(1)

                outputs_rejected = self.policy(**inputs_rejected, labels=inputs_rejected["input_ids"])
                policy_rejected_logps = -outputs_rejected.loss * inputs_rejected["input_ids"].size(1)
        
                # 获取参考模型对数概率
                with torch.no_grad():
                    ref_outputs_chosen = self.ref_model(
                        **inputs_chosen, labels=inputs_chosen["input_ids"])
                    ref_chosen_logps = -ref_outputs_chosen.loss * inputs_chosen["input_ids"].size(1)
                    
                    ref_outputs_rejected = self.ref_model(
                        **inputs_rejected, labels=inputs_rejected["input_ids"])
                    ref_rejected_logps = -ref_outputs_rejected.loss * inputs_rejected["input_ids"].size(1)

                # 计算IPO损失
                log_ratio_diff = (
                    (policy_chosen_logps - policy_rejected_logps) -
                    (ref_chosen_logps - ref_rejected_logps)
                )
                loss = (log_ratio_diff - 1/(2 * self.eta)) ** 2
                loss = loss.mean()

                loss.backward()
                optimizer.step()

    def online_iteration(self, prompt_batch):
        # 收集新数据
        new_pref_data = {
            "prompts": [],
            "responses_a": [],
            "responses_b": [],
            "preferences": []
        }
        
        new_ipo_data = {
            "prompts": [],
            "chosen_response": [],
            "rejected_response": []
        }
        
        for prompt in prompt_batch:
            # 主代理生成响应
            main_response = self._generate_response(prompt)
            
            # 增强器通过拒绝采样生成响应
            booster_response = self.rejection_sampling(prompt)
            
            # 查询偏好预言机
            preference = self.oracle.query(prompt, main_response, booster_response)
            
            # 存储偏好数据
            new_pref_data["prompts"].append(prompt)
            new_pref_data["responses_a"].append(main_response)
            new_pref_data["responses_b"].append(booster_response)
            new_pref_data["preferences"].append(preference)
            
            # 存储IPO训练数据
            new_ipo_data["prompts"].append(prompt)
            if preference == 1:  # main_response 被偏好
                new_ipo_data["chosen_response"].append(main_response)
                new_ipo_data["rejected_response"].append(booster_response)
            else:  # booster_response 被偏好
                new_ipo_data["chosen_response"].append(booster_response)
                new_ipo_data["rejected_response"].append(main_response)
        
        # 更新偏好模型
        pref_dataset = PreferenceDataset(
            new_pref_data["prompts"],
            new_pref_data["responses_a"],
            new_pref_data["responses_b"],
            new_pref_data["preferences"]
        )
        self.train_pref_model(pref_dataset, epochs=1)
        
        # 更新策略模型
        ipo_dataset = PreferenceDataset(
            new_ipo_data["prompts"],
            new_ipo_data["chosen_response"],
            new_ipo_data["rejected_response"],
            [1] * len(new_ipo_data["prompts"])  # 偏好标签均为1（chosen > rejected）
        )
        self.train_policy_with_ipo(ipo_dataset, epochs=1)

    def _generate_response(self, prompt, max_length=128):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.policy.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 基础语言模型（主代理）
    base_lm = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # 偏好模型（初始化为随机权重）
    pref_model = PreferenceModel("roberta-base")
    
    # 偏好预言机（使用真实模型或模拟器）
    oracle = PreferenceOracle()
    
    trainer = OnlineELHFIPOTrainer(
        base_model=base_lm,
        pref_model=pref_model,
        oracle=oracle,
        tokenizer=tokenizer,
        eta=0.1,
        n_candidates=4,
        device=device
    )
    
    prompts = [
        "Explain the concept of quantum entanglement.",
        "Describe the process of photosynthesis.",
        "What are the main causes of climate change?",
        "How does blockchain technology work?",
        "Discuss the impacts of artificial intelligence on society."
    ]
    
    n_iterations = 5
    batch_size = 2
    
    for iter in range(n_iterations):
        print(f"\n=== 开始迭代 {iter+1}/{n_iterations} ===")
        
        # 随机选择一批提示
        batch_indices = np.random.choice(len(prompts), batch_size, replace=False)
        prompt_batch = [prompts[i] for i in batch_indices]
        
        # 执行在线迭代
        trainer.online_iteration(prompt_batch)
        
        test_prompt = "Explain the theory of relativity."
        print("\n测试提示:", test_prompt)
        print("主代理响应:", trainer._generate_response(test_prompt))
    
    print("\n训练完成!")
