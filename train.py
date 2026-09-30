"""最小 GRPO：Base + SFT adapter + GRPO LoRA，用 TRL GRPOTrainer。

Prompt -> 采样 G 条 response -> 规则 reward -> 组内相对 advantage -> 更新 Policy
不需要 Reward Model / Critic / GAE，但需要在线 rollout。运行：python train.py
"""

import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
BASE = "Qwen/Qwen3-0.6B"
SFT_ADAPTER = HERE.parent / "sft_tool_call" / "output"
DATA = HERE / "train.jsonl"
OUTPUT = HERE / "output"
CHAT_KWARGS = {"enable_thinking": False}
MAX_NEW_TOKENS = 64


def extract_json(text):
    """抽出第一个完整的 {...}；和 evaluate.py 用同一套解析，保证训练信号与评测指标一致。"""
    text = re.sub(r"<think>.*?</think>", "", text.replace("<|im_end|>", ""), flags=re.S)
    for m in re.finditer(r"\{", text):
        depth, in_str, esc = 0, False, False
        for i in range(m.start(), len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[m.start():i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def tool_reward(completions, target, **kwargs):
    """0~4 分级规则 reward：能解析 JSON +1、tool 对 +1、arguments 的 key 集合对 +1、arguments 全对 +1。

    分级而不是 0/1 的原因：组内如果全是 0 或全是 1，组内标准差为 0，
    GRPO 的相对优势就是 0，拿不到任何学习信号。
    """
    rewards = []
    for completion, gold_text in zip(completions, target):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        pred, gold = extract_json(text), json.loads(gold_text)
        if pred is None:
            rewards.append(0.0)
            continue
        r = 1.0
        if pred.get("tool") == gold.get("tool"):
            r += 1.0
        pred_args, gold_args = pred.get("arguments", {}), gold.get("arguments", {})
        if isinstance(pred_args, dict) and set(pred_args) == set(gold_args):
            r += 1.0
        if pred_args == gold_args:
            r += 1.0
        rewards.append(r)
    return rewards


def show_group(tok, policy, prompt_msgs, target, g):
    """训练前先看一眼：同一个 prompt 采 G 条，reward 各是多少。GRPO 就是拿这个做组内比较。"""
    text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, **CHAT_KWARGS)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(policy.device)
    with torch.no_grad():
        out = policy.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=1.0,
                              num_return_sequences=g, pad_token_id=tok.pad_token_id)
    n = inputs["input_ids"].shape[1]
    comps = [[{"role": "assistant", "content": tok.decode(o[n:], skip_special_tokens=True)}] for o in out]
    rs = tool_reward(comps, [target] * g)
    print(f"[训练前的 group] {prompt_msgs[1]['content']}")
    for i, (c, r) in enumerate(zip(comps, rs), 1):
        print(f"  response {i}: reward={r:.0f}  {c[0]['content']!r}")
    m = sum(rs) / g
    print(f"  -> mean={m:.2f} std={(sum((r - m) ** 2 for r in rs) / g) ** 0.5:.2f}"
          f"（std=0 时组内没有谁比谁好，GRPO 拿不到相对信号）")


def main():
    # ---- 1. 数据：只有 prompt + target，没有 chosen/rejected ----
    ds = load_dataset("json", data_files=str(DATA), split="train")
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token

    # ---- 2. Policy = Base + SFT adapter（merge），GRPO 阶段再加 LoRA ----
    policy = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16), str(SFT_ADAPTER)
    ).merge_and_unload()

    # ---- 3. GRPO LoRA（和 SFT / PPO / DPO 完全一致）----
    grpo_lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )

    # ---- 4. 训练配置 ----
    args = GRPOConfig(
        output_dir=str(OUTPUT),
        num_train_epochs=3,
        per_device_train_batch_size=1,   # x grad_accum 4 = 有效 batch 4
        gradient_accumulation_steps=4,   # 正好等于 num_generations，即一个组 = 一个 prompt 的 4 条
        learning_rate=1e-5,
        num_generations=4,               # G
        max_completion_length=MAX_NEW_TOKENS,
        temperature=1.0,                 # 必须让同一 prompt 产生不同回答，否则组内 std=0
        epsilon=0.2,                     # PPO-style clipping
        beta=0.05,                       # reference KL 系数（TRL 默认 0，这里显式开启）
        num_iterations=1,
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        chat_template_kwargs=CHAT_KWARGS,
    )
    print("[GRPO] G=%d temperature=%.1f epsilon=%.2f beta=%.2f | 有效 batch size = %d"
          % (args.num_generations, args.temperature, args.epsilon, args.beta,
             args.per_device_train_batch_size * args.gradient_accumulation_steps))
    print("[模型] Policy = Base + SFT adapter(merged) + GRPO LoRA <- 训练")
    print("       Reference = 关掉 GRPO LoRA 现算 SFT 模型 <- 冻结 | 没有 Reward Model / Critic / GAE")
    show_group(tok, policy, ds[0]["prompt"], ds[0]["target"], args.num_generations)

    # ---- 5. GRPOTrainer ----
    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=tool_reward,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=grpo_lora,
    )
    trainer.train()
    trainer.save_model(str(OUTPUT))
    print(f"\nGRPO adapter 已保存到 {OUTPUT}")


if __name__ == "__main__":
    main()
