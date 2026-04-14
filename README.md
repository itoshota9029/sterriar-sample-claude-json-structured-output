# Claude Structured Output — Production Patterns

[日本語](#japanese) | [English](#english)

5 battle-tested patterns for getting **reliably parsed JSON** out of the Anthropic Claude API, with full type safety via Pydantic, CI-verified tests, and no mocking surprises at runtime.

Built and maintained by [SterriaR LLC](https://sterriar.com/) from our daily AI video-pipeline operations.

![CI](https://github.com/itoshota9029/sterriar-sample-claude-json-structured-output/actions/workflows/ci.yml/badge.svg)

---

<a id="japanese"></a>

## Claude API で JSON を安全に返させる5つのパターン 🇯🇵

Claude から「確実にパースできる JSON」を取り出すための、**本番運用で検証した 5 パターン**。Pydantic で型安全、モック応答ベースのテスト付き、CI 緑。

### このリポジトリで解決する課題

Anthropic 公式ドキュメントには「Tool Use で JSON が取れる」とは書いてあるものの、実務で本当にハマる以下のケースのベストプラクティスはほぼ未掲載です。

| 本番で遭遇する課題 | 本リポの対応パターン |
|---|---|
| 「Claude がたまに JSON じゃなくて文章で返してくる」 | **1. basic** — `tool_choice` で Tool Use を強制 |
| 「スキーマ違反 (例: enum 外の値) で ValidationError」 | **2. retry** — Claude 自身に違反を伝えて再試行 |
| 「`max_tokens` 超過で JSON が途中で切れる」 | **3. partial** — 部分復旧 + 継続指示 |
| 「streaming で UI を早く更新したい」 | **4. streaming** — 途中 JSON を watch して逐次 yield |
| 「enum / Union / ネスト / Literal を含む複雑スキーマで安定しない」 | **5. nested** — 安定化するスキーマ設計の工夫 |

### クイックスタート

```bash
git clone https://github.com/itoshota9029/sterriar-sample-claude-json-structured-output.git
cd sterriar-sample-claude-json-structured-output
pip install -e ".[dev]"

# テストだけならこれで OK (mock 応答なので API キー不要)
pytest
```

### 各パターンの要点

#### 1. **basic** (`src/basic.py`)
最小構成で安全に JSON を取る。鍵は `tool_choice={"type": "tool", "name": "..."}` による **Tool Use の強制**。

```python
from pydantic import BaseModel
from src.basic import extract

class Person(BaseModel):
    name: str
    age: int
    role: str

person = extract(client, "山田太郎(30歳)はエンジニアです", Person, tool_name="extract_person")
# Person(name='山田太郎', age=30, role='engineer')
```

#### 2. **retry** (`src/retry.py`)
スキーマ違反時に **Claude 自身に違反箇所を伝えて再生成**。単純な `try/except` ループより成功率が明確に上がる。

```python
from src.retry import extract_with_retry

# スキーマ違反があっても最大 3 回まで Claude に修正させる
result = extract_with_retry(client, text, Person, max_retries=3)
```

#### 3. **partial** (`src/partial.py`)
`stop_reason == "max_tokens"` で JSON が途中切断された時、**既に確定した部分を抽出して残りを継続要求**。

#### 4. **streaming** (`src/streaming.py`)
Claude の streaming 応答を消費しつつ、パース可能になった段階で partial object を yield。UI 体感速度を上げたい時に。

#### 5. **nested** (`src/nested.py`)
Enum / Union / Literal / ネスト Pydantic モデルを含む複雑スキーマを**壊れずに受け取る**ための設計指針集。

### 開発 / テスト

```bash
# Ruff (lint + format check)
ruff check src tests

# Pytest (mock 応答ベース・API キー不要)
pytest
```

全テストは `anthropic` クライアントをモック化し、Claude 実 API を呼びません。CI は GitHub Actions で自動検証。

### 採用技術

- **Python 3.11+**
- [`anthropic`](https://pypi.org/project/anthropic/) 公式 SDK >= 0.40
- [`pydantic`](https://pydantic.dev/) v2
- `pytest` / `ruff` / `mypy`

### ライセンス

[MIT License](LICENSE) — 商用・非商用を問わず自由に利用・改変・再配布可能です。

### このリポの位置付け

SterriaR LLC が自社 AI 動画パイプラインを毎日運用する中で蓄積した、Claude Structured Output の**ハマりどころと対策**をオープンソースで共有したものです。業務コードそのものではなく、そこから**抽出した汎用パターン**を提供しています。

本番で使うときに「もう一歩踏み込んだ実装例」を探している方に役立てば。

---

<a id="english"></a>

## Claude API Structured Output — 5 Production Patterns 🌏

Five **battle-tested patterns** for reliably extracting JSON from the Anthropic Claude API, fully type-checked with Pydantic, backed by mock-based tests, and CI-verified.

### Problems this repo solves

Anthropic's official docs tell you Tool Use can return JSON. They don't cover the failure modes you actually hit in production.

| Real-world failure mode | Pattern |
|---|---|
| "Claude sometimes returns free text instead of JSON" | **1. basic** — force Tool Use via `tool_choice` |
| "ValidationError when model emits out-of-enum values" | **2. retry** — send the violation back to Claude and retry |
| "JSON gets truncated at `max_tokens`" | **3. partial** — extract confirmed prefix, request continuation |
| "Want to stream partial JSON to UI" | **4. streaming** — watch during streaming, yield progressive objects |
| "Complex schemas with enum / Union / nested Pydantic models are unstable" | **5. nested** — schema design techniques that stabilize output |

### Quickstart

```bash
git clone https://github.com/itoshota9029/sterriar-sample-claude-json-structured-output.git
cd sterriar-sample-claude-json-structured-output
pip install -e ".[dev]"
pytest
```

### Pattern overview

#### 1. **basic** (`src/basic.py`)
Minimal end-to-end setup with `tool_choice={"type": "tool", "name": "..."}` to **force Tool Use**, preventing prose fallback.

#### 2. **retry** (`src/retry.py`)
On schema violation, **hand the error back to Claude** and let it correct itself. Dramatically better than blind `try/except` loops.

#### 3. **partial** (`src/partial.py`)
When `stop_reason == "max_tokens"`, extract the confirmed JSON prefix and ask for a continuation — preserves progress.

#### 4. **streaming** (`src/streaming.py`)
Consumes the streaming API, yields partial `model` objects as soon as each field becomes parseable. For low-latency UI.

#### 5. **nested** (`src/nested.py`)
Design patterns for complex schemas (Enum / Union / Literal / nested models) that survive real-world LLM outputs.

### Development

```bash
ruff check src tests
pytest
```

All tests mock the `anthropic` client — no live API calls, safe for CI.

### Stack

- **Python 3.11+**
- [`anthropic`](https://pypi.org/project/anthropic/) >= 0.40
- [`pydantic`](https://pydantic.dev/) v2
- `pytest` / `ruff` / `mypy`

### License

[MIT License](LICENSE) — free for commercial and non-commercial use.

### About this repo

Extracted from [SterriaR LLC](https://sterriar.com/)'s daily AI video-pipeline operations. Not business code itself, but the **generic patterns we converged on** after repeatedly hitting these failure modes in production.

If you're past the docs and looking for "one step deeper", this is for you.
