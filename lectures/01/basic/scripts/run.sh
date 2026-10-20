#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
basic_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$script_dir/../../../.." && pwd)"
cd "$repo_root"

temperatures=(0.0 0.4 0.8 1.2)
results_dir="$basic_dir/results"
questions=(
    'lecture-specific|AIE2026の講座の宿題にある「3レベル」とは何ですか？'
    'general|LoRAとは何ですか？'
)
model="$(uv run python -c 'from app.config import MODEL; print(MODEL)')"

for question_case in "${questions[@]}"; do
    IFS='|' read -r question_type question <<< "$question_case"
    question_results_dir="$results_dir/$question_type"
    results_file="$question_results_dir/results.md"
    analysis_file="$question_results_dir/analysis.md"
    mkdir -p "$question_results_dir"

    {
        printf '# 第1回 基礎演習の実行結果\n\n'
        printf -- '- 質問種別: %s\n' "$question_type"
        printf -- '- 実行日時: %s\n' "$(date -Is)"
        printf -- '- モデル: `%s`\n' "$model"
        printf -- '- 質問: %s\n' "$question"
        printf -- '- 条件: temperature 4段階（0.4刻み）、各3回\n\n'
    } > "$results_file"

    printf '\n[%s] 実行結果を %s に保存します\n' "$question_type" "$results_file"

    for temperature in "${temperatures[@]}"; do
        for run in 1 2 3; do
            printf '## temperature=%s / run=%s\n\n````text\n' "$temperature" "$run" \
                | tee -a "$results_file"
            uv run python -m app.ask "$question" --temperature "$temperature" 2>&1 \
                | tee -a "$results_file"
            printf '\n````\n\n' | tee -a "$results_file"
        done
    done
    uv run python "$script_dir/analyze.py" "$results_file" "$analysis_file"
done