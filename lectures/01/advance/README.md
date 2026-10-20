# 第1回 発展課題

モデルと質問の組み合わせごとに、応答時間とtoken数を比較します。

## 実行

既定では `gemini-3.5-flash-lite` と `gemini-3.5-flash` に2種類の質問を1回ずつ送り、合計4リクエストで計測します。まずはAPIを呼ばない確認ができます。

```bash
uv run python lectures/01/advance/run.py --dry-run
uv run python lectures/01/advance/run.py
```

モデル、質問、反復数を絞る場合：

```bash
uv run python lectures/01/advance/run.py \
  --models gemini-3.5-flash-lite \
  --question-types general \
  --repeats 1
```

反復数は1〜3、全体の上限は12リクエストです。`AIE_COMPARE_MODELS` にカンマ区切りのモデル名を設定すると、既定の比較モデルを変更できます。APIキーは通常どおり `GEMINI_API_KEY` から読み込みます。

## 出力と考察

- `results/metrics.csv`: 質問、モデル、応答時間、入力・出力token数、状態コード、回答本文
- `results/report.md`: モデル・質問別の集計とレート制限についての考察欄

429が返った場合はそこで追加リクエストを止めます。無料枠のクォータはモデルや時期などで変わるため、この少数回の計測だけで上限値は決められません。429の有無を記録し、AI Studioの利用状況と最新のモデル別クォータを確認してください。応答時間は通信状況にも左右されるので、少ない反復数では参考値として扱います。