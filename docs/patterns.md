# Patterns Deep Dive — 本番で遭遇するハマりどころと対策

本ドキュメントは `src/` 配下のコードを補完する「**なぜそうしたのか**」の説明集です。実コードは最小限に抑えているので、背景と判断理由はここに集約しています。

---

## 1. なぜ `tool_choice` を強制するのか

**素朴な実装**:
```python
# tool を提供するだけ。Claude が使うかは任意。
response = client.messages.create(
    model="claude-sonnet-4-5",
    tools=[{"name": "extract_person", ...}],
    messages=[{"role": "user", "content": text}],
)
```

**問題**: Claude はツールを使わず、「山田太郎さん、30歳、エンジニアです。」のようにテキストで返すことがある。その頻度は入力が曖昧 (カンマ区切りのリストなど) になるほど上がる。

**解決**: `tool_choice={"type": "tool", "name": "extract_person"}` で **ツール呼び出しを強制**。Claude は必ず `tool_use` ブロックを返すようになる。

この一行で、naive 実装の ~95% のテキスト返却失敗が消える。

---

## 2. retry で `ValidationError` を Claude に送り返す理由

**素朴な実装**:
```python
for _ in range(3):
    try:
        return extract(...)
    except ValidationError:
        continue  # just retry — Claude が前回のミスを知らない
```

**問題**: Claude は毎回同じ入力を見ているので、同じエラーを繰り返す可能性が高い。特に enum の typo や境界値 (age=200 等) は**同じ誤答を 3 回返す**ことが珍しくない。

**解決**: `ValidationError` の内容を自然言語で要約して Claude に伝える:

```python
messages.append({"role": "assistant", "content": previous_response.content})
messages.append({"role": "user", "content": (
    "前回のツール呼び出しはスキーマ違反で失敗しました:\n"
    f"- severity: 'critical' は許可されていません。許可値: low, medium, high\n"
    "\nこの違反を修正して、もう一度ツールを呼び出してください。"
)})
```

Claude は**前回の出力 + 具体的な誤り**を見るので、狙って直しにいく。体感では成功率が 40% → 85% 程度に上がる。

---

## 3. `max_tokens` で切れた時、「部分復旧」vs「増量して全部再生成」

**素朴な実装**: `max_tokens` を最初から大きくする (8192 など)。

**問題**:
- コストが単調に増える (大半の入力は 500 tokens で収まるのに)
- `max_tokens` を増やしてもそれを超えた時に結局切れる
- 切れた時に既出力を全部捨てて再生成するのは二重コスト

**解決 (本リポの partial.py)**:
1. 通常サイズ (例: 2048) で投げる
2. `stop_reason == "max_tokens"` なら、既に出力された部分を Claude に提示して続きを要求

```python
messages.append({"role": "user", "content": (
    "前回のツール呼び出しは完了前に打ち切られました。"
    f"既に入力済みのフィールド (同じ値で再出力): {partial}"
    "残りのフィールドを含めて完全な呼び出しをやり直してください。"
)})
```

Claude は partial を見ているので、同じフィールドを再生成せず残り部分に集中する。コストは小さく済む。

---

## 4. streaming で partial model を yield する時の 3 つの罠

罠 1: **`json.loads` が例外を上げる頻度が高い**
→ 解決: `_try_parse` で `JSONDecodeError` を握りつぶしつつ、末尾補完 (`}`, `"]}`) を試みる。

罠 2: **`partial_json` の蓄積が `content_block_delta` イベントでしか来ない**
→ 解決: 他イベント (start/stop/message_delta) を全て無視する分岐を明示。

罠 3: **同じパース状態を何度も yield してしまう**
→ 解決: `last_emitted` を保持して、差分があるときだけ yield。

これらを押さえないと、UI が高頻度で再レンダリングして滑らかさを失う (特に React は state 更新回数に敏感)。

---

## 5. 複雑スキーマで頻出する 4 つの失敗と設計対策

### 失敗 A: Enum の値がサイレントに別の値にすり替わる

**再現**: `enum = ["draft", "in_review", "approved"]` に対して Claude が `"in-review"` (ハイフン区切り) を返す。Pydantic 側は `ValidationError`。

**対策**: Literal で明示しつつ、`Field(description="...")` にも「許可される文字列を exact に」と明記する:

```python
status: Literal["draft", "in_review", "approved"] = Field(
    description="Current status. Must be exactly one of: draft, in_review, approved."
)
```

### 失敗 B: Union 型が判別できない

**再現**: `Union[TextAttachment, UrlAttachment]` で、Claude が両方の中間みたいな JSON (`{"content": "...", "url": "..."}`) を返す。

**対策**: discriminator field を必須化して、Pydantic の `Annotated[Union[...], Field(discriminator="kind")]` で分岐を明示する。`kind` は Literal なので Claude は選択肢を知っている。

### 失敗 C: Optional フィールドが**欠落**する (null ではなく)

**再現**: `assignee: str | None` だけだと、Claude が**キー自体を省略**することがある (「該当なしなのでフィールドを出さない」判断)。Pydantic では「field required」エラー。

**対策**: `= None` を必ず付ける + description に「**If unknown, emit null (do not omit)**」と明記。

### 失敗 D: `list[Tag]` の Tag 内でスキーマ違反するとリスト全体が失敗

**再現**: 10 要素のうち 1 つが不正で、リスト全体が `ValidationError`。

**対策**: retry パターンで自動修復 (本リポの `retry.py`)。ただしリストの 1 要素だけを指摘するエラーメッセージになるので、Claude は該当要素だけ直して残りは温存する。

---

## まとめ

| 優先度 | 使う場面 | パターン |
|---|---|---|
| 必須 | Claude API で Structured Output を取るとき全般 | **basic (tool_choice forced)** |
| 推奨 | 本番運用 (出力品質に SLA があるとき) | **retry** |
| 長文抽出 | 法務文書・議事録など | **partial** |
| UI 体感速度が必要 | チャット UI / ドラフト生成 | **streaming** |
| 複雑スキーマ | ネスト / Union / Enum を含む | **nested** (設計パターン) |

何か詰まったら [Issues](https://github.com/itoshota9029/sterriar-sample-claude-json-structured-output/issues) にぜひ。
