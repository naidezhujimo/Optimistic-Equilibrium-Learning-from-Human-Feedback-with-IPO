# Optimistic-Equilibrium-Learning-from-Human-Feedback-with-IPO
传统RLHF框架依赖Bradley_Terry（BT）模型，其假设偏好传递性（若A>B且B>C，则A>C）和标量奖励函数存在。但在现实环境中人类偏好存在非传递性（原论文中实验证据表明70%准确率是瓶颈），并且BT模型无法捕捉群体偏好的复杂性（如多用户偏好聚合）。

由此引出偏好预言机：其偏好信息直接由伯努利分布生成，无需假设奖励函数或BT模型，允许非传递偏好：

$y \sim \text{Ber}(\mathbb{P}(a^1 \succ a^2)|x,a^1,a^2)$

在 Identity Policy Optimization（IPO）方法 中通过将偏好学习建模为两个策略的极小极大博弈，通过KL散度约束策略接近初始模型 \pi_0 解决了 Direct Preference Optimization（DPO）方法的过度优化问题，即直接最大化偏好对数似然，但易过拟合噪声偏好信号，导致策略偏离真实偏好（称为奖励过度优化）。（小声BB：当然了 GRPO 通过KL正则限制策略更新显然更简单）而 IPO 的问题是静态偏好模型假设和缺乏主动探索，易导致奖励黑客或分布偏移失效。《Online Iterative Reinforcement Learning from Human Feedback with General Preference Model》论文中提出Online OELHF-IPO方法，通过在线迭代更新偏好模型和 PELHF（悲观均衡学习）与 OELHF（乐观均衡学习）解决了上述问题，

目标函数为 Minimax目标：

$\max_{\pi^1 \in \Pi} \min_{\pi^2 \in \Pi} \mathbb{E}_{x \sim d_0} \mathbb{E}_{a^1 \sim \pi^1, a^2 \sim \pi^2} \left[ P^*(x, a^1, a^2) - \frac{1}{\eta} \left( D_{KL}(\pi^1(\cdot|x) \| \pi_0(\cdot|x)) - D_{KL}(\pi^2(\cdot|x) \| \pi_0(\cdot|x)) \right) \right]$

- Max玩家（主代理 $\pi^1$）最大化期望偏好得分 $P^*$ 并最小化与初始模型 $\pi_0$ 的KL散度。
- Min玩家（增强器 $\pi^2$）最小化期望偏好得分，同时约束自身接近 $\pi_0$ 。
- $\eta$ **控制正则化强度**（实验最优值 $\eta=0.1$）。

**迭代更新步骤**

输入：历史数据集 $D_{1:t-1} = \{(x_i, a_i^1, a_i^2, y_i)\}_{i=1}^{(t-1)m}$

1.偏好模型MLE估计：

$\hat{P}_t = \arg\min_{P \in \mathcal{P}} \sum_{i=1}^{(t-1)m} \left[ y_i \log P(x_i, a_i^1, a_i^2) + (1 - y_i) \log(1 - P(x_i, a_i^1, a_i^2)) \right]$

2.纳什策略求解（通过自我博弈IPO近似）：

$\hat{\pi}^1_t = \arg\max_{\pi^1} \min_{\pi^2} \mathbb{E}_{x \sim d_0} \left[ \hat{P}_t(x, \pi^1, \pi^2) - \frac{1}{\eta} \left( D_{KL}(\pi^1 \| \pi_0) - D_{KL}(\pi^2 \| \pi_0) \right) \right]$

实际优化损失（自我博弈IPO损失）：

$\mathcal{L}_{\text{IPO}} = \mathbb{E}_{\substack{x \sim d_0 \\ a^+ \sim \pi_1 \\ a^- \sim \pi_2}} \left[ \left( \log \frac{\pi^1(a^+|x) \pi_0(a^-|x)}{\pi^1(a^-|x) \pi_0(a^+|x)} - \frac{1}{2} \right)^2 \right]$

**增强器更新**

目标：选择最大化不确定性的 $\pi_t^2$ 促进探索：

$\hat{\pi}^2_t = \arg\max_{\pi^2 \in \Pi} \sup_{P \in \mathcal{P}} \sqrt{\frac{1}{m} \sum_{j=1}^{m} \left( P(x_j, \hat{\pi}^1_t, \pi^2) - \hat{P}_t(x_j, \hat{\pi}^1_t, \pi^2) \right)^2}$

实际实现：通过拒绝采样高效生成 $\hat{\pi}_t^2$：

1. 从$\hat{\pi}_t^1$采样 $n$ 个响应 $\{a^{(k)}\}_{k=1}^n$
2. 用偏好模型进行锦标赛排名，选择排名最高的响应作为 $\hat{\pi}_t^2$ 的输出

对每个批次 $t$：

$\begin{equation*}
\text{For } i = 1 \text{ to } m: 
\begin{cases}
x_i \sim d_0 \\
a_i^1 \sim \hat{\pi}_t^1(\cdot | x_i) \\
a_i^2 \sim \hat{\pi}_t^2(\cdot | x_i) \\
y_i \sim \text{Ber}(P^*(x_i, a_i^1, a_i^2))
\end{cases}
\end{equation*}$

更新数据集：$D_t = \{(x_i, a_i^1, a_i^2, y_i)\}_{i=1}^m, \quad D_{1:t} = D_{1:t-1} \cup D_t$
