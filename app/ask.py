"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 空の状態から会話を始める
    uv run python -m app.ask "LoRAとは何ですか？" --chat  # 質問から会話を始める

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, temperature: float | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    config = types.GenerateContentConfig(temperature=temperature)
    response = client.models.generate_content(
        model=MODEL, contents=question, config=config
    )

    _print_usage(response)

    return response.text


def _print_usage(response: types.GenerateContentResponse) -> None:
    if response.usage_metadata is not None:
        usage = response.usage_metadata
        print(f"[トークン数] 入力: {usage.prompt_token_count}, 出力: {usage.candidates_token_count}")


def chat(question: str | None = None, temperature: float | None = None) -> None:
    client = genai.Client(api_key=get_api_key())
    config = types.GenerateContentConfig(temperature=temperature)
    conversation = client.chats.create(model=MODEL, config=config)

    print(f"[model] {MODEL}")
    if temperature is not None:
        print(f"[temperature] {temperature}")
    print("会話を開始します。終了するには /exit を入力してください。")

    while True:
        if question is None:
            try:
                current_question = input("あなた> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n会話を終了します。")
                break
        else:
            current_question = question.strip()
            print(f"あなた> {current_question}")
            question = None

        if current_question.lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        if not current_question:
            continue

        try:
            response = conversation.send_message(current_question, config=config)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            continue

        print(response.text)
        _print_usage(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って対話する")
    parser.add_argument("--temperature", type=float, help="生成時のtemperature（例: 0.7）")
    args = parser.parse_args()

    try:
        if args.chat:
            chat(question=args.question, temperature=args.temperature)
        else:
            question = args.question or DEFAULT_QUESTION
            print(f"[model] {MODEL}")
            if args.temperature is not None:
                print(f"[temperature] {args.temperature}")
            print(f"[質問] {question}\n")
            print(ask(question, temperature=args.temperature))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
