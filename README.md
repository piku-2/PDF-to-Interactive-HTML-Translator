# 📄💥 DeepL PDF.ver 💥📄

論文とかレポートとか技術文書を放り込んだら、**1枚のHTML**になって返ってくるWebアプリだよ！！！
ネット無しで開けるし、単語にマウス乗せたら意味がピコーンって出るし、原文と日本語を行ったり来たりできるし、マジで脳が助かる😇✨✨✨

難しい仕様書は `SPEC.md` に置いてある。🥺👉👈

---

## 🎀 これで何が起きるの？？

1. 📤 英語のPDFをブラウザにドロップ！！
2. 📖 中身をMarkdownっぽくバラして、見出し・文・表に分ける！！
3. 🤖 25ブロックずつAIに投げて、日本語と「この文での意味」を回収！！
4. 🪄 単語ホバーを安全に埋め込んで、CSSもJSも全部入りのHTMLを1ファイルで書き出す！！
5. 💾 ダウンロードすれば、飛行機の中でもカフェでも読めるね！！

---

## 🚀 起動

Python 3.10 以上。ターミナルをプロジェクトのフォルダで開いて、これをコピペ。📋

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` を開いて、**どれか1個でいいから** APIキーを入れる。キーは `.env` に書いて。チャットに貼るな。リポジトリにコミットするな。

```bash
python main.py
```

ブラウザでここを開く。👉 http://127.0.0.1:8000

初回は NLTK の文分割データ（punkt）を取りに行くことがある。ちょっと待って。画面が黙ってても裏で準備してる。コンソールを見ろ。👀

---

## 🤖 モデル

`.env` の `LLM_PROVIDER_PRIORITY` が順番。カンマ区切り。空白は捨てる。大文字は小文字にする。空ならこの並び。

`anthropic,gemini,openai,grok,deepseek,kimi,glm,openrouter,nvidia_nim`

**キーが入ってる子だけ**が候補。一番左がメイン。右は控え。

| 名前 | 環境変数 | モデル | 呼び方 |
| --- | --- | --- | --- |
| anthropic | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` | Anthropic SDK。ツール呼び出し＋プロンプトキャッシュ。`max_retries=5` |
| gemini | `GEMINI_API_KEY` | `gemini-2.5-flash-lite` | google-genai。JSONスキーマで同じ形 |
| openai | `OPENAI_API_KEY` | `gpt-4o-mini` | OpenAI SDK。`response_format` は json_object |
| grok | `GROK_API_KEY` が先。無ければ `XAI_API_KEY` | `grok-2-latest` | OpenAI互換。`https://api.x.ai/v1` |
| deepseek | `DEEPSEEK_API_KEY` | `deepseek-chat` | OpenAI互換。`https://api.deepseek.com` |
| kimi | `KIMI_API_KEY` | `moonshot-v1-8k` | OpenAI互換。`https://api.moonshot.cn/v1` |
| glm | `GLM_API_KEY` | `glm-4-flash` | OpenAI互換。`https://open.bigmodel.cn/api/paas/v4/` |
| openrouter | `OPENROUTER_API_KEY` | `anthropic/claude-3.5-haiku` | OpenAI互換。`https://openrouter.ai/api/v1` |
| nvidia_nim | `NVIDIA_NIM_API_KEY` | `meta/llama-3.3-70b-instruct` | OpenAI互換。`https://integrate.api.nvidia.com/v1` |

空欄、`changeme`、`your-api-key` みたいな仮の文字、末尾が `...` の値は「入れてない」扱い。見栄を張るな。

### 🫠 落ちたら次の子へバトン

429（レート制限）、500番台、タイムアウト、認証エラー、接続エラー、壊れたJSON。これが来たらコンソールに警告を出して、次に優先度が高い子で同じバッチを続ける。

出力が長すぎて途中で切れたときだけは、プロバイダーを替えない。バッチを半分に割って、同じ子にもう一回頼む。

そのジョブの中で死んだ子は、残りバッチでも飛ばす。次のジョブではまた最初からチャンスをあげる。

全員死んだらエラーログを吐いて、ジョブは `failed`。画面に理由が出る。APIキー本体は画面に出さない。

サーバー起動時と、翻訳が始まるときに、コンソールへこう出る。

- 今キーがあるモデルの順番
- メインで使うモデル
- 今試してるモデル
- フォールバックでどのモデルからどのモデルに切り替えたか

429 と 500 番台が来たら、同じモデルで粘らず、ログを出して即座に次のモデルへバトンタッチ。タイムアウトや認証エラーや壊れた json も次の子。同時に走らせるのは5本まで（`asyncio.Semaphore(5)`）。1バッチは25ブロック。

どのモデルが答えても中身の形は同じ。`items[]` の中に `id` と `translated` と `words[{word, context_meaning}]`。OpenAI互換のモデルには `response_format={"type":"json_object"}` を付けて、プロンプトに `json` って書いてある。形が崩れたら次のモデル。

---

## 🖱️ 画面の遊び方

アップロード画面は「ファイルを選択」か、PDFをドロップ。ボタンは「翻訳を開始」。進捗バーが伸びる。終わったら「HTMLをダウンロード」。

ダウンロードしたファイルの中でできること。🎉

- 📖 **単語モード** (`mode-word-hover`): 英語だけ。単語に乗せると、その文での意味がピコーン
- 💬 **1文モード** (`mode-sentence-hover`): 英語だけ。文に乗せると、その文の日本語がすぐ上にふわっと出る。単語ホバーも同時に生きてる
- 🇯🇵 **全文モード** (`mode-full-ja`): 日本語だけを、論文っぽい紙面で流し読み。原文は隠す
- 🌗 **テーマ**: ライト / ダーク / システム。選んだものはブラウザに覚えさせる
- 🔤 **文字サイズ**: 大きくしたり小さくしたり。CSS変数 `--base-font-size`
- 🧭 **目次**: 見出しから自動。押すとそこまでスーッとスクロール
- 📊 **表**: Markdownの表を本物の表にして、ヘッダーに色、はみ出したら横スクロール
- 💻 **コードと引用**: コードは囲いのまま。引用は1文ずつホバーできる
- 🧠 **単語**: 1文につき3個以上、目安は8個くらい。短い文は取れるだけ

単語ホバーは、先にHTMLエスケープしてから、長い語を先に、単語境界 `\b` で置換する。短い語が長い熟語の中にネストしてタグが壊れる、みたいな事故は潰してある。

ファイル名は `元の名前-ja.html`。日本語のファイル名もそのまま。

---

## 📮 API（ポーリング）

でかいPDFを同期で抱えない。投げたら番号が返って、あとから進捗を聞く。

| メソッド | パス | なにが返る |
| --- | --- | --- |
| POST | `/api/translate` | すぐ `job_id`（UUID）。中身は裏で進む |
| GET | `/api/status/{job_id}` | `queued` / `parsing` / `translating` / `completed` / `failed` と `progress` 0〜100 |
| GET | `/api/download/{job_id}` | 完成したHTML。未完成なら 409 |

ついでに知っとく数字。

- PDF以外、または先頭がPDFじゃないものは 400
- 100MBを超えたら 413
- 同時に走れるジョブは 4 本。超えたら 429
- ジョブはメモリの中だけ。サーバーを落としたら消える。完成品は6時間でお片付け
- ダウンロード前に再起動したら、もう一回翻訳。悲しいけどそういう子

---

## 🧱 中身の地図

```
main.py                     入口。起動ログでプロバイダー順を叫ぶ
app/parser.py               PDF → ブロック（pymupdf4llm + nltk）
app/translator.py           25個ずつ、同時5本、失敗した子を覚える
app/providers.py            優先順位、クライアント、フォールバック
app/prompts.py              みんな共通の翻訳ルールとJSONの形
app/annotator.py            安全な単語ホバー
app/html_builder.py         単一HTMLを組み立て
app/templates/document.html.j2
app/templates/index.html
.env.example                キーの見本。本物は .env
```

流れはこれ。短くて偉い。

```
PDF
 → Markdown
 → 見出し / 文 / 表
 → 優先度いちばん高いLLM
 → ダメなら次のLLM
 → ホバー付きHTML 1枚
```

---

## ✅ テスト

キーが要る試験はしない。通信もしない。フェイクで優先順位とフォールバックを叩く。

```bash
.venv/bin/python -m unittest discover -s tests -v
```

---

## ⚠️ やらかし防止

- 🔑 キーは `.env` だけ。READMEに書くんじゃない
- 🌐 生成HTMLは外に通信しない。翻訳してる最中のサーバーは各社のAPIに通信する
- 📚 英語の論文・技術文書・レポート向け。漫画の吹き出しとかスキャン画像の中の文字は、テキストとして取れてないと訳せない
- 💥 全員のAPIが同時に死んだら、そのジョブは失敗。順番を変えるか、生きてるキーを足して、もう一回

以上。PDFを投げろ。日本語が返ってくる。🌍➡️🇯🇵✨🎉🥳
