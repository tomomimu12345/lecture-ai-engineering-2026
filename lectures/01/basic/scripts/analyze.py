"""Summarize response variation using character n-gram similarity."""

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from math import sqrt
from pathlib import Path
from statistics import mean

RUN_PATTERN = re.compile(
    r"^## temperature=(?P<temperature>[\d.]+) / run=(?P<run>\d+)\n\n````text\n"
    r"(?P<output>.*?)\n````(?=\n|$)",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class Run:
    temperature: float
    number: int
    answer: str


def _answer_from_output(output: str) -> str:
    lines = output.splitlines()
    question_index = next(
        (index for index, line in enumerate(lines) if line.startswith("[質問]")),
        -1,
    )
    answer_lines = lines[question_index + 1 :] if question_index >= 0 else lines
    answer_lines = [
        line for line in answer_lines if not line.startswith("[トークン数]")
    ]
    return "\n".join(answer_lines).strip()


def parse_results(text: str) -> list[Run]:
    runs = [
        Run(
            temperature=float(match["temperature"]),
            number=int(match["run"]),
            answer=_answer_from_output(match["output"]),
        )
        for match in RUN_PATTERN.finditer(text)
    ]
    runs = [run for run in runs if run.answer]
    if not runs:
        raise ValueError("実行結果が見つかりません。先に run.sh を実行してください。")
    return runs


def _character_bigrams(text: str) -> Counter[str]:
    normalized = "".join(text.casefold().split())
    if len(normalized) < 2:
        return Counter([normalized]) if normalized else Counter()
    return Counter(normalized[index : index + 2] for index in range(len(normalized) - 1))


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    left_norm = sqrt(sum(count * count for count in left.values()))
    right_norm = sqrt(sum(count * count for count in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    dot_product = sum(count * right[bigram] for bigram, count in left.items())
    return dot_product / (left_norm * right_norm)


def _mean_similarity(pairs: list[tuple[Run, Run]]) -> float:
    if not pairs:
        return 1.0
    return mean(
        _cosine_similarity(_character_bigrams(left.answer), _character_bigrams(right.answer))
        for left, right in pairs
    )


def render_analysis(runs: list[Run], source_name: str) -> str:
    grouped: dict[float, list[Run]] = defaultdict(list)
    for run in runs:
        grouped[run.temperature].append(run)
    if 0.0 not in grouped:
        raise ValueError("temperature=0.0 の回答がありません。")

    baseline = grouped[0.0]
    lines = [
        "# 第1回 基礎演習の解析",
        "",
        f"- 元データ: `{source_name}`",
        f"- 解析した回答数: {len(runs)}",
        "",
        "| temperature | 回答数 | 異なる回答数 | 平均文字数 | 同温度内類似度 | 0.0との距離 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    baseline_vectors = [_character_bigrams(run.answer) for run in baseline]
    for temperature in sorted(grouped):
        temperature_runs = grouped[temperature]
        vectors = [_character_bigrams(run.answer) for run in temperature_runs]
        within_pairs = list(combinations(temperature_runs, 2))
        within_similarity = _mean_similarity(within_pairs)
        if temperature == 0.0:
            baseline_similarity = within_similarity
        else:
            baseline_similarity = mean(
                _cosine_similarity(vector, baseline_vector)
                for vector, baseline_vector in product(vectors, baseline_vectors)
            )
        normalized_answers = {" ".join(run.answer.casefold().split()) for run in temperature_runs}
        average_length = mean(len(run.answer) for run in temperature_runs)
        lines.append(
            f"| {temperature:g} | {len(temperature_runs)} | {len(normalized_answers)} | "
            f"{average_length:.1f} | {within_similarity:.1%} | "
            f"{1 - baseline_similarity:.1%} |"
        )

    lines.extend(
        [
            "",
            "類似度は空白を除いた文字bigramのコサイン類似度です。",
            "「異なる回答数」は空白だけを正規化した完全一致で数えています。",
            "これらは表現の変化を測る指標で、意味の正しさや意味的な近さは判定しません。",
            "回答本文は同じフォルダの `results.md` を確認し、用語集と照らし合わせてください。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="temperature別の回答差を集計する")
    results_dir = Path(__file__).parent.parent / "results" / "lecture-specific"
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=results_dir / "results.md",
    )
    parser.add_argument(
        "analysis",
        nargs="?",
        type=Path,
        default=results_dir / "analysis.md",
    )
    args = parser.parse_args()

    runs = parse_results(args.results.read_text(encoding="utf-8"))
    source_name = f"{args.results.parent.name}/{args.results.name}"
    args.analysis.write_text(
        render_analysis(runs, source_name), encoding="utf-8"
    )
    print(f"解析結果を {args.analysis} に保存しました")


if __name__ == "__main__":
    main()