
import subprocess, sys
PINS = {"transformers": "5.18.0", "peft": "0.21.1", "trl": "1.14.2", "accelerate": "1.15.0", "datasets": "5.1.0"}
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + [f"{k}=={v}" for k, v in PINS.items()])

from pathlib import Path
import os, json, time

CONFIG = {
    "base_model": "Qwen/Qwen3.5-0.8B",   # 小さな指示チューニング済みモデル（Apache-2.0）。量子化なしで T4 に載る大きさ
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",  # いつ実行しても同じ重みを使うために固定
    "seed": 42,
    "num_train_epochs": 20,  # 学習データ（14 件）を何周学習するか。1 周 2 ステップなので合計 40 ステップ（T4 で 4〜5 分）
    "warmup_steps": 6,
    "lr": 2e-4,
    "lora_r": 16,            # LoRA の行列の次元（小さいほど学習するパラメータが少ない）
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 96,
    "gen": {"do_sample": False},   # 評価は貪欲生成で固定（前後比較の条件を揃える）
    "work_dir": "/content/qlora_lab",                          # 結果ファイルの保存先（左のファイル一覧から見られる）
    "drive_dir": "/content/drive/MyDrive/aie2026/qlora_lab",  # USE_DRIVE=True のときの保存先
}
WORK = Path(CONFIG["work_dir"]); WORK.mkdir(parents=True, exist_ok=True)

USE_DRIVE = False  # True にすると Google Drive をマウントして adapter と出力を保存する
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    Path(CONFIG["drive_dir"]).mkdir(parents=True, exist_ok=True)
print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
FAQ = {
    "deadline": "課題の提出期限は 9 月 30 日 23:59（日本時間）です。",
    "destination": "提出先は学習ポータルの「課題」欄です。",
    "format": "提出形式は PDF 1 ファイルで、ファイル名は 学籍番号_課題番号.pdf です。",
    "office_hours": "質問は毎週火曜 17:00〜18:00 のオフィスアワーで受け付けます。",
    "late": "期限後の提出は、期限から 48 時間以内なら減点のうえ受け付けます。",
}
FAQ_TEXT = "\n".join(f"- {v}" for v in FAQ.values())

# 根拠あり: (FAQ キー, 質問テンプレート群, 正解の answer に含まれるべき語)
GROUNDED = {
    "deadline": (["課題の提出期限はいつですか", "いつまでに出せばいいですか", "締切を教えてください", "提出の期限は？"], ["9 月 30 日", "23:59"]),
    "destination": (["提出先はどこですか", "どこにアップロードしますか", "課題はどこから出しますか", "提出場所は？"], ["学習ポータル", "課題"]),
    "format": (["提出形式を教えてください", "ファイル名の決まりはありますか", "何のファイル形式で出しますか", "PDF 以外でも出せますか"], ["PDF"]),
    "office_hours": (["質問はいつできますか", "オフィスアワーはいつですか", "先生に質問する時間はありますか", "相談できる時間帯は？"], ["火曜", "17:00"]),
    "late": (["遅れて出したらどうなりますか", "期限を過ぎた場合は受け付けますか", "遅延提出のルールは？", "締切後の扱いを教えてください"], ["48 時間", "減点"]),
}
# 根拠なし: FAQ に無いので needs_review=true が正解
UNGROUNDED = [
    "再提出はできますか", "欠席した回の扱いを教えてください", "他の受講者の点数は見られますか",
    "成績はいつ出ますか", "提出期限は延長できますか", "グループでの提出は可能ですか",
    "課題の配点はどうなっていますか", "講義の録画は公開されますか",
]

# test 専用の追加質問（train / valid には入れない）
TEST_EXTRA_GROUNDED = {
    "deadline": ["提出期限の日付と時刻を教えてください", "課題の締切は何月何日の何時ですか"],
    "destination": ["課題の提出はどこで行いますか", "提出物を出す先を教えてください"],
    "format": ["提出するファイルの形式は何ですか", "提出ファイルは何形式で、いくつ出しますか"],
    "office_hours": ["オフィスアワーの曜日と時間を教えてください", "質問を受け付けてもらえるのは何曜日の何時からですか"],
    "late": ["提出が締切に間に合わなかったらどうなりますか", "期限後に提出した課題は受理されますか"],
}
TEST_EXTRA_UNGROUNDED = [
    "課題の文字数に制限はありますか", "提出した課題へのフィードバックはもらえますか",
    "参考文献の書き方に指定はありますか", "オフィスアワーは事前予約が必要ですか",
]
UNGROUNDED_ANSWER = "この FAQ には記載がありません。運営に確認が必要です。"

def make_example(question: str, answer: str, needs_review: bool) -> dict:
    target = json.dumps({"answer": answer, "needs_review": needs_review}, ensure_ascii=False)
    return {"question": question, "faq": FAQ_TEXT, "target": target, "needs_review": needs_review}

def split_templates(templates):
    # テンプレートの先頭 2 つを train、3 つ目を valid、4 つ目を test に固定（言い換えを跨がせない）
    return templates[:2], templates[2:3], templates[3:4]

train, valid, test, gold = [], [], [], {}
for key, (templates, must_include) in GROUNDED.items():
    tr, va, te = split_templates(templates)
    for q in tr: train.append(make_example(q, FAQ[key], False))
    for q in va: valid.append(make_example(q, FAQ[key], False))
    for q in te + TEST_EXTRA_GROUNDED[key]:
        test.append(make_example(q, FAQ[key], False)); gold[q] = {"needs_review": False, "must_include": must_include}
for i, q in enumerate(UNGROUNDED):
    ex = make_example(q, UNGROUNDED_ANSWER, True)
    bucket = [train, train, valid, test][i % 4]   # 根拠なしの質問は 4 問ごとに train 2 / valid 1 / test 1 に振る
    bucket.append(ex)
    if bucket is test: gold[q] = {"needs_review": True, "must_include": ["確認"]}
for q in TEST_EXTRA_UNGROUNDED:
    test.append(make_example(q, UNGROUNDED_ANSWER, True)); gold[q] = {"needs_review": True, "must_include": ["確認"]}

# train は 14 件（根拠あり 10、根拠なし 4）。同じ例を何回学習するかは、データを複製せず、セクション 4 の num_train_epochs で決める
all_questions = [r["question"] for r in train + valid + test]
assert len(set(all_questions)) == len(all_questions), "質問が train / valid / test のどこかで重複しています。テンプレートを見直してください"
print(f"train={len(train)} valid={len(valid)} test={len(test)} gold={len(gold)}")
print("学習例の 1 件 | 質問:", train[0]["question"], "| 望ましい出力:", train[0]["target"])
_ = (WORK / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
_ = (WORK / "test.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in test), encoding="utf-8")
_ = (WORK / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"])
if tok.pad_token is None: tok.pad_token = tok.eos_token
# 量子化なし。重みは 32bit のまま保持する（dtype を省くと配布時の精度で読み込まれるので明示する）
model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"], dtype=torch.float32).to(DEVICE)
model.config.use_cache = True
BASE_MEMORY_GB = round(model.get_memory_footprint() / 1e9, 2)
print("loaded:", CONFIG["base_model"], "| device:", DEVICE, "| 重みのメモリ(GB):", BASE_MEMORY_GB)

# 形式はプロンプトで指示している。モデルがこれを守れるかが学習前の確認点
SYSTEM = "あなたは講座 FAQ の回答係です。与えられた FAQ だけを根拠に、必ず JSON 1 行で答えてください。形式: {\"answer\": 文字列, \"needs_review\": 真偽値}。FAQ に記載が無いことは answer に『確認が必要』と書き、needs_review を true にします。"

def build_messages(example):
    user = f"FAQ:\n{example['faq']}\n\n質問: {example['question']}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

def generate(model, example):
    # enable_thinking=False: 思考ブロックを空にして回答だけを生成させる（モデルを替えても条件が揃うように明示）
    prompt = tok.apply_chat_template(build_messages(example), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"], pad_token_id=tok.pad_token_id, **CONFIG["gen"])
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
import re
import pandas as pd
pd.set_option("display.max_colwidth", None)   # 出力の原文を省略せずに表示する

def check_format(text: str):
    '''JSON として読め、必須キーと型が合うか。失敗理由も返す'''
    try:
        obj = json.loads(text)
    except Exception as e:
        return False, f"json_error: {type(e).__name__}", None
    if not isinstance(obj, dict): return False, "not_object", None
    if set(obj) - {"answer", "needs_review"}: return False, "extra_keys", obj
    if "answer" not in obj or "needs_review" not in obj: return False, "missing_key", obj
    if not isinstance(obj["answer"], str) or not isinstance(obj["needs_review"], bool): return False, "wrong_type", obj
    return True, "", obj

def check_content(obj, gold_entry):
    '''内容の合否: needs_review が正解と一致し、必須語句を answer に含む'''
    if obj is None: return False
    if obj.get("needs_review") != gold_entry["needs_review"]: return False
    return all(w in obj.get("answer", "") for w in gold_entry["must_include"])

def mark(ok: bool) -> str:
    return "○" if ok else "×"

def evaluate(model, label: str):
    '''test の 21 問を生成して採点し、要約と質問ごとの結果を返す。原文は eval_<label>.json に保存する'''
    rows = []
    t0 = time.time()
    for ex in test:
        raw = generate(model, ex)
        ok_fmt, reason, obj = check_format(raw)
        # 形式が壊れていても JSON らしい部分が読めるなら、内容だけは別に採点する
        loose = obj
        if not ok_fmt:
            m = re.search(r"\{.*\}", raw, re.S)
            try: loose = json.loads(m.group(0)) if m else None
            except Exception: loose = None
        ok_content = check_content(loose if isinstance(loose, dict) else None, gold[ex["question"]])
        rows.append({"question": ex["question"], "raw": raw, "format_ok": ok_fmt, "format_reason": reason,
                     "content_ok": ok_content, "both_ok": ok_fmt and ok_content, "gold_needs_review": gold[ex["question"]]["needs_review"]})
    n = len(rows)
    summary = {"label": label, "n": n,
               "format_rate": sum(r["format_ok"] for r in rows)/n,
               "content_rate": sum(r["content_ok"] for r in rows)/n,
               "both_rate": sum(r["both_ok"] for r in rows)/n,
               "seconds": round(time.time()-t0, 1)}
    (WORK / f"eval_{label}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary, rows



import gc
del model; gc.collect(); torch.cuda.empty_cache()
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

SEED = int(os.environ.get("SWEEP_SEED", "42"))
EPOCHS = [int(x) for x in os.environ.get("SWEEP_EPOCHS", "1,2,3,4,5,7,10,15,20").split(",")]
CONFIG["seed"] = SEED

def to_chat(example):
    return {"messages": build_messages(example) + [{"role": "assistant", "content": example["target"]}]}
train_ds = Dataset.from_list([to_chat(e) for e in train])
valid_ds = Dataset.from_list([to_chat(e) for e in valid])

def load_base():
    m = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"], dtype=torch.float32).to(DEVICE)
    m.config.use_cache = False
    return m

out_path = WORK / f"sweep_seed{SEED}.jsonl"
for ep in EPOCHS:
    torch.manual_seed(SEED)
    base = load_base()
    lora = LoraConfig(r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    sft = SFTConfig(output_dir=str(WORK / f"ckpt_{SEED}_{ep}"), num_train_epochs=ep, per_device_train_batch_size=4, gradient_accumulation_steps=2,
                    learning_rate=CONFIG["lr"], lr_scheduler_type="cosine", warmup_steps=CONFIG["warmup_steps"], logging_steps=1, save_strategy="no",
                    max_length=CONFIG["max_seq_len"], packing=False, report_to="none", seed=SEED, disable_tqdm=True)
    trainer = SFTTrainer(model=base, args=sft, train_dataset=train_ds, eval_dataset=valid_ds, peft_config=lora, processing_class=tok)
    t0 = time.time()
    trainer.train()
    train_seconds = round(time.time() - t0, 1)
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    eval_loss = trainer.evaluate()["eval_loss"]
    tuned = trainer.model
    tuned.config.use_cache = True
    tuned.eval()
    summary, rows = evaluate(tuned, f"sweep_s{SEED}_e{ep}")
    rec = {"seed": SEED, "epochs": ep, "steps": trainer.state.global_step, "train_seconds": train_seconds,
           "final_loss": losses[-1], "losses": losses, "eval_loss": eval_loss,
           "format_rate": summary["format_rate"], "content_rate": summary["content_rate"], "both_rate": summary["both_rate"],
           "failed": [{"question": r["question"], "raw": r["raw"], "gold_needs_review": r["gold_needs_review"],
                       "format_ok": r["format_ok"], "content_ok": r["content_ok"]} for r in rows if not r["both_ok"]]}
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("SWEEP", json.dumps({k: rec[k] for k in ["seed", "epochs", "steps", "train_seconds", "final_loss", "eval_loss", "format_rate", "content_rate"]}), flush=True)
    del trainer, tuned, base
    gc.collect(); torch.cuda.empty_cache()
print("SWEEP_DONE", flush=True)
