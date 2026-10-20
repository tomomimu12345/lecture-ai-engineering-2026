"""Run a small model/question comparison and record latency and token usage."""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

from google import genai
from google.genai import errors

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.config import MODEL, api_error_message, get_api_key

QUESTIONS = {
    "general": "LoRAとは何ですか？",
    "lecture-specific": "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
}
CSV_FIELDS = [
    "timestamp",
    "model",
    "question_type",
    "question",
    "repeat",
    "elapsed_seconds",
    "prompt_tokens",
    "candidate_tokens",
    "total_tokens",
    "status",
    "error",
    "answer",
]


def _measure(client, model: str, question_type: str, repeat: int) -> dict:
    question = QUESTIONS[question_type]
    started = perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=question)
    except errors.APIError as error:
        return {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model": model,
            "question_type": question_type,
            "question": question,
            "repeat": repeat,
            "elapsed_seconds": f"{perf_counter() - started:.3f}",
            "prompt_tokens": "",
            "candidate_tokens": "",
            "total_tokens": "",
            "status": str(error.code),
            "error": api_error_message(error.code, error.message),
            "answer": "",
        }

    elapsed = perf_counter() - started
    usage = response.usage_metadata
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": model,
        "question_type": question_type,
        "question": question,
        "repeat": repeat,
        "elapsed_seconds": f"{elapsed:.3f}",
        "prompt_tokens": usage.prompt_token_count if usage else "",
        "candidate_tokens": usage.candidates_token_count if usage else "",
        "total_tokens": usage.total_token_count if usage else "",
        "status": "ok",
        "error": "",
        "answer": response.text or "",
    }


def _write_report(path: Path, rows: list[dict], planned_calls: int) -> None:
    lines = [
        "# 第1回 発展課題: モデル・質問別の計測",
        "",
        f"- 実測リクエスト数: {len(rows)} / 予定: {planned_calls}",
        "",
        "| モデル | 質問 | 成功数 | 平均応答時間（秒） | 平均入力token | 平均出力token | エラー |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    for model in dict.fromkeys(row["model"] for row in rows):
        for question_type in QUESTIONS:
            group = [
                row
                for row in rows
                if row["model"] == model and row["question_type"] == question_type
            ]
            if not group:
                continue
            successful = [row for row in group if row["status"] == "ok"]
            if successful:
                latency = f"{mean(float(row['elapsed_seconds']) for row in successful):.3f}"
                input_tokens = _average_field(successful, "prompt_tokens")
                output_tokens = _average_field(successful, "candidate_tokens")
            else:
                latency = input_tokens = output_tokens = "-"
            errors_seen = ", ".join(row["status"] for row in group if row["status"] != "ok") or "なし"
            lines.append(
                f"| {model} | {question_type} | {len(successful)}/{len(group)} | "
                f"{latency} | {input_tokens} | {output_tokens} | {errors_seen} |"
            )

    rate_limit_hit = any(row["status"] == "429" for row in rows)
    lines.extend(
        [
            "",
            "## レート制限について",
            "",
            f"- この実験でHTTP 429を{ '検出しました' if rate_limit_hit else '検出しませんでした' }。",
            "- 429は実行時点でプロジェクトまたはモデルの利用上限に達したことを示します。無料枠の上限はモデル・地域・時期などで異なり、この少数回の実験から上限値は推定できません。",
            "- 上限を調べるために連続リクエストを増やすのではなく、429が出たら時間を置き、AI Studioの利用状況とモデル別の最新クォータを確認してください。",
            "",
            "応答時間はネットワークや混雑の影響も受けます。少ない反復数では参考値として扱ってください。",
            "生ログと回答本文は `metrics.csv` にあります。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _average_field(rows: list[dict], field: str) -> str:
    values = [float(row[field]) for row in rows if row[field] != ""]
    return f"{mean(values):.1f}" if values else "-"


def main() -> None:
    default_models = list(
        dict.fromkeys(
            [MODEL, *os.getenv("AIE_COMPARE_MODELS", "gemini-3.5-flash").split(",")]
        )
    )
    parser = argparse.ArgumentParser(description="少数リクエストで応答時間とtoken数を比較する")
    parser.add_argument("--models", nargs="+", default=default_models)
    parser.add_argument(
        "--question-types",
        nargs="+",
        choices=tuple(QUESTIONS),
        default=list(QUESTIONS),
    )
    parser.add_argument("--repeats", type=int, default=1, help="各組み合わせの反復数（1〜3）")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results"),
    )
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず実験条件だけ表示する")
    args = parser.parse_args()

    if not 1 <= args.repeats <= 3:
        parser.error("--repeats は1〜3で指定してください")
    planned_calls = len(args.models) * len(args.question_types) * args.repeats
    if planned_calls > 12:
        parser.error(f"リクエスト数が上限12を超えています: {planned_calls}")

    if args.dry_run:
        print(f"予定リクエスト数: {planned_calls}")
        for model in args.models:
            for question_type in args.question_types:
                print(f"{model} / {question_type} x {args.repeats}")
        return

    client = genai.Client(api_key=get_api_key())
    rows = []
    stop_after_rate_limit = False
    for model in args.models:
        for question_type in args.question_types:
            for repeat in range(1, args.repeats + 1):
                row = _measure(client, model, question_type, repeat)
                rows.append(row)
                print(
                    f"{model} / {question_type} / {repeat}: "
                    f"status={row['status']}, {row['elapsed_seconds']}s"
                )
                if row["status"] == "429":
                    stop_after_rate_limit = True
                    break
            if stop_after_rate_limit:
                break
        if stop_after_rate_limit:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = args.output_dir / "metrics.csv"
    with metrics_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report_file = args.output_dir / "report.md"
    _write_report(report_file, rows, planned_calls)
    print(f"計測結果: {metrics_file}")
    print(f"集計レポート: {report_file}")


if __name__ == "__main__":
    main()