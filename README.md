# grpo-minimal

最小可运行的 **GRPO** 实验，学习链的最后一个环节：

```
Prompt -> 采样 G 条 response -> 规则 reward -> 组内相对 advantage -> GRPO 更新 Policy
```

对比 PPO-RLHF，GRPO **不需要**：

```
× Reward Model
× Critic
× GAE
× chosen / rejected 偏好对
```

但和 DPO 不同，GRPO **是在线的**：训练时必须让 Policy 自己 rollout。

**两个模型角色**（和 DPO 一样）：

| 模型 | 来源 | 是否训练 |
|---|---|---|
| Policy | Base + SFT adapter（merge）再加 GRPO LoRA | 训练 |
| Reference | 关掉 GRPO LoRA 现算（`beta=0.05` 时启用） | 冻结 |

## 数据

只需要 `prompt` + `target`，**不需要 chosen/rejected**：

```json
{"prompt": [{"role":"system","content":"..."},{"role":"user","content":"北京今天天气怎么样？"}],
 "target": "{\"tool\":\"weather\",\"arguments\":{\"city\":\"北京\"}}"}
```

`prompt` 给 Policy 生成，`target` 只给 reward function 判分。本项目的 train/test 由 `reward-model-minimal` 的数据转换而来（`target = chosen`，删掉 `rejected`），保证四个项目用同一批 prompt。

## 安装与运行

`GRPOTrainer` 是 TRL 1.14 的稳定 API，**不需要 PPO 那个 venv**，用主环境即可。

```powershell
python evaluate.py --base                # GRPO 之前的 SFT 策略（基线）
python train.py                          # GRPO 训练，约 7.5 分钟（240 步）
python evaluate.py --adapter ./output    # GRPO 之后
```

版本（跑通时实测）：`trl 1.14.0` / `transformers 5.14.1` / `peft 0.19.1` / `torch 2.13.0.dev+cu132`。

## Reward function：本项目最重要的新组件

0~4 分级规则 reward：

| 得分 | 条件 |
|---|---|
| +1 | 能解析出合法 JSON |
| +1 | `tool` 正确 |
| +1 | `arguments` 的 key 集合正确 |
| +1 | `arguments` 完全正确 |

**为什么不做成 0/1？** 因为你的模型已经有能力，group 里很容易出现全 0 或全 1，此时组内标准差 = 0，GRPO 的组内相对优势就是 0，**拿不到任何学习信号**。分级 reward 让 `[4,3,1,0]` 这种有梯度的组更容易出现。

训练脚本开头的 `show_group()` 会先采 4 条并打印各自的 reward。实测第一个 prompt 就撞上了这个问题：

```
[训练前的 group] 北京今天天气怎么样？
  response 1: reward=4  '{"tool":"weather","arguments":{"city":"北京"}}'
  response 2: reward=4  '{"tool":"weather","arguments":{"city":"北京"}}'
  response 3: reward=4  '{"tool":"weather","arguments":{"city":"北京"}}'
  response 4: reward=4  '{"tool":"weather","arguments":{"city":"北京"}}'
  -> mean=4.00 std=0.00（std=0 时组内没有谁比谁好，GRPO 拿不到相对信号）
```

## 结果

| 指标 | SFT（GRPO 前） | GRPO 后 |
|---|---|---|
| JSON Valid | 90.0% | **100.0%** |
| Tool Accuracy | 70.0% | **90.0%** |
| Arguments Accuracy | 45.0% | **70.0%** |
| Exact Match | 45.0% | **70.0%** |

**四项全部上升，Exact Match 45% → 70%，是四个项目里提升最大的。**

训练过程的三个关键指标（第 21 节要求记录的）：

| 指标 | 训练初期 | 训练后期 |
|---|---|---|
| `reward` 均值 | ~2.4 | ~3.0~3.5 |
| `reward_std` | 0.5~0.9 | 0.4~1.0（始终非零） |
| `kl` | 0.0004 | 0.03~0.09 |
| `frac_reward_zero_std` | 0~0.2 | 0.2~0.8 |

`reward_std` 全程非零，说明组内一直有可比较的差异；`kl` 缓慢上升但没有爆掉。`frac_reward_zero_std` 后期升到 0.8 是预期现象：策略变好后越来越多的组「全对」，组内方差自然消失——这也解释了为什么 GRPO 需要持续的新 prompt。

## 四个项目的完整对照

相同 SFT 起点、相同 prompt 集合、相同 LoRA（`q_proj`/`v_proj`, r=16）：

| 方法 | 在线生成 | Reward Model | Critic | Preference pair | Verifier | Exact Match |
|---|---:|---:|---:|---:|---:|---:|
| SFT | ✗ | ✗ | ✗ | ✗ | ✗ | 45.0% |
| PPO-RLHF | ✓ | ✓ | ✓ | RM 阶段 | ✗ | 30.0% |
| DPO | ✗ | ✗ | ✗ | ✓ | ✗ | 50.0% |
| **GRPO** | ✓ | ✗ | ✗ | ✗ | ✓ | **70.0%** |

| 指标 | SFT | PPO-RLHF | DPO | GRPO |
|---|---|---|---|---|
| JSON Valid | 90.0% | 95.0% | 90.0% | **100.0%** |
| Tool Accuracy | 70.0% | 55.0% | 75.0% | **90.0%** |
| Arguments Accuracy | 45.0% | 40.0% | 50.0% | **70.0%** |
| Exact Match | 45.0% | 30.0% | 50.0% | **70.0%** |

三者的核心区别：

```
PPO : Reward -> Critic + GAE -> A_t          （多一个最容易坏的环节）
DPO : chosen/rejected -> DPO Loss            （离线，不生成）
GRPO: 多个 rollout rewards -> group-relative A -> PPO-style 更新
```

**为什么 GRPO 在这里赢得最多？** 三个原因叠加：

1. **监督信号直接对齐最终指标**：verifier 判的就是 Exact Match 那套标准，没有 proxy（PPO 用的是 RM 打分，RM 自己就有偏差）
2. **不需要 Critic**：PPO 的失败点被整个绕开（`ppo-rlhf-minimal` 的诊断：Critic 学不动 → advantage 变噪声 → 策略漂移）
3. **仍然在线探索**：比 DPO 多了 rollout，能走出偏好数据没覆盖的区域

**诚实说明**：这个对比是教学实验，四种方法使用的监督信息并不等价（SFT 用标准答案、DPO 用偏好对、GRPO 用 verifier target），所以它不是严格 benchmark。GRPO 的优势很大程度来自「verifier 恰好等于评测标准」这一设定。

## 核心代码位置

| 步骤 | 位置 |
|---|---|
| 抽取 JSON（与 `evaluate.py` 同一套解析） | `train.py:29` |
| **0~4 分级 rule reward** | `train.py:57` |
| 训练前 group 采样与打印 | `train.py:82` |
| Policy = Base + SFT adapter（merge） | `train.py:109` |
| GRPO LoRA 配置 | `train.py:112` |
| GRPOConfig（`G=4` / `temperature=1.0` / `epsilon=0.2` / `beta=0.05`） | `train.py:118` |
| GRPOTrainer：`reward_funcs=tool_reward` | `train.py:144` |
| rollout + 组内归一化 + clipped loss + backward | `train.py:152` |
| 四项任务指标 | `evaluate.py:112` |

rollout、组内 advantage、clipped GRPO loss 都在 TRL 内部，没有自己实现。

## 关于代码量

`train.py` **158 行**，比原计划的 80~110 行多。多出来的部分：

| 多出的内容 | 行数 | 说明 |
|---|---:|---|
| `extract_json` | 15 | 故意与 `evaluate.py` 用同一套解析，保证**训练信号 = 评测指标**；否则 reward 和指标会各算各的 |
| `show_group` | 13 | 第 22 节要求的训练前 group 采样打印，也是本 README 里那个 `std=0.00` 案例的来源 |
| 中文注释 | ~10 | GRPOConfig 里每个关键参数的说明 |

去掉这两个函数，核心逻辑正好在 120 行以内。如果要求严格合规，删掉 `show_group` 即可（TRL 也提供 `log_completions=True` 作为替代）。

## 关于 TRL 版本的一个细节

`GRPOConfig` 在 TRL 1.14 里 `loss_type` 默认是 **`'dapo'`**（DAPO 论文的长度偏差修正），不是原始 GRPO。二者差别只在**token loss 的归一化方式**，核心机制（组内相对优势 + ratio + clipping）完全相同。另：当 `per_device_train_batch_size==1` 时，`loss_type='bnpo'` 按 TRL 文档等价于原始 GRPO loss。本项目沿用默认值。
