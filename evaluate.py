"""评测 GRPO：SFT 策略（GRPO 前）vs SFT+GRPO。

四项任务指标，生成用 do_sample=False 保证 SFT/PPO/DPO/GRPO 横向可比：
  JSON Valid / Tool Accuracy / Arguments Accuracy / Exact Match

用法：
  python evaluate.py --base                # GRPO 之前的 SFT 策略（基线）
  python evaluate.py --adapter ./output    # GRPO 之后
  python evaluate.py                       # 不传参数时自动检测 ./output
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
BASE = "Qwen/Qwen3-0.6B"
SFT_ADAPTER = HERE.parent / "sft_tool_call" / "output"
DATA = HERE / "test.jsonl"
DEFAULT_ADAPTER = HERE / "output"

CHAT_KWARGS = {"enable_thinking": False}
MAX_NEW_TOKENS = 64
DEMO = "帮我查一下杭州明天天气"


def extract_json(text):
    """从模型输出里抽出第一个完整的 {...} 对象；失败返回 None。"""
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


def load_policy(grpo_adapter):
    """Base + SFT adapter（merge） [+ GRPO adapter]。"""
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    policy = PeftModel.from_pretrained(base, str(SFT_ADAPTER)).merge_and_unload()
    if grpo_adapter:
        policy = PeftModel.from_pretrained(policy, str(grpo_adapter))
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token
    return tok, policy.to("cuda").eval()


def generate(tok, policy, prompt_msgs):
    text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, **CHAT_KWARGS)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(policy.device)
    with torch.no_grad():
        out = policy.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="只测 GRPO 之前的 SFT 策略")
    ap.add_argument("--adapter", default=None, help="GRPO LoRA adapter 目录")
    a = ap.parse_args()

    adapter = None if a.base else (a.adapter or (str(DEFAULT_ADAPTER) if DEFAULT_ADAPTER.exists() else None))
    label = f"SFT + GRPO({adapter})" if adapter else "SFT（GRPO 之前）"
    print(f"===== 评测对象：{label} =====")

    tok, policy = load_policy(adapter)
    rows = [json.loads(x) for x in DATA.read_text(encoding="utf-8").splitlines() if x.strip()]

    stats = {"json_valid": 0, "tool": 0, "args": 0, "exact": 0}
    failures = []
    for row in rows:
        target = json.loads(row["target"])
        raw = generate(tok, policy, row["prompt"])
        pred = extract_json(raw)
        if pred is not None:
            stats["json_valid"] += 1
            stats["tool"] += pred.get("tool") == target["tool"]
            stats["args"] += pred.get("arguments") == target["arguments"]
            stats["exact"] += pred == target
        if pred != target:
            failures.append((row["prompt"], target, raw, pred))

    n = len(rows)
    print(f"\nTotal: {n}")
    print(f"JSON Valid Rate    : {stats['json_valid'] / n:6.1%}")
    print(f"Tool Accuracy      : {stats['tool'] / n:6.1%}")
    print(f"Arguments Accuracy : {stats['args'] / n:6.1%}")
    print(f"Exact Match        : {stats['exact'] / n:6.1%}")

    print(f"\n===== 固定样例：{DEMO} =====")
    demo_msgs = [rows[0]["prompt"][0], {"role": "user", "content": DEMO}]
    print(f"  生成 : {generate(tok, policy, demo_msgs)!r}")

    if failures:
        print(f"\n===== 未完全匹配的样例（共 {len(failures)} 条，展示前 8 条）=====")
        for prompt_msgs, tgt, raw, pred in failures[:8]:
            print(f"\nInput      : {prompt_msgs[1]['content']}")
            print(f"Target     : {json.dumps(tgt, ensure_ascii=False)}")
            print(f"Prediction : {json.dumps(pred, ensure_ascii=False) if pred else raw!r}")


if __name__ == "__main__":
    main()
