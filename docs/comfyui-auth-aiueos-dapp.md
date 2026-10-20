# app-comfyui（認可層）— itonami（dapp）上の分散型設計

状態: 第 1 スライス実装（2026-10-20）。正本はこの文書。
追跡: Linear JUN-128（親 JUN-116）。

## 位置づけ

app-comfyui は画像生成ゲートウェイ（ComfyUI front、OpenAI 兼容
/v1/images/* + XRPC）。認可は **4 層**（Murakumo pattern）、生成は
ワークフログラフ（9 モード）。

## このスライス: 認可判定コア

| 層 | 実体 | 状態 |
|---|---|---|
| 判定コア guest | `aiueos/comfyui_auth.cljk` — auth-admission（**4 層**: sk_live は **scope=comfyui:generate 必須** / break-glass は緊急 / mkc は **1h TTL** / internal x-kotodama-verified）/ workflow-mode-ok（**9 モードの閉じた集合**: txt2img / img2img / animatediff / svd / wan5b_i2v / sbv2 / xtts / musicgen / stable_audio） | **実装済み・実測** |
| 記述子 | `aiueos/comfyui-auth.edn` | **実装済み** |
| parity | `adapter/tests/comfyui/auth_parity_test.cljk` — 正本（AGENTS.md の Auth layers + workflows.py）の認可層・scope・TTL・集合と一致 | **実装済み・実測** |

実測: `amu test aiueos/comfyui_auth.cljk` → **15/15**（jvm-kir / js / wasm）。
parity → 2 tests / 15 assertions / 0 failures。

## 正本の正確な規則（guest はこれに追従 —— AGENTS.md の明文）

- **認可層（target, Murakumo pattern）**:
  1. `sk_live_*`（HYPERDRIVE `vertex_api_key`、**scope=`comfyui:generate`**）——
     API 消費者。scope 無しは通さない。
  2. `COMFYUI_API_KEY`（**break-glass、ADR-0023**）—— 緊急時。
  3. `mkc_*` HMAC ephemeral（`COMFYUI_CHAT_SECRET`、匿名ブラウザプレビュー、
     **1h TTL**）—— TTL 超過は拒否。
  4. `x-kotodama-verified`（dispatcher 内部信頼）。
- **ワークフロー 9 モード**（`workflows.py` のビルダー群）。

## 未実装境界

- 認可の実体（キー検証・HMAC・scope 照合）は host（CF Worker / adapter）。
  guest は層と規則の判定のみ（Phase 1 scaffold のため実装は未着）。
- workflow の node graph 判定（KSampler パラメータ、node class の pin）。
- mon（DNA）化、aiueos session 載せ（fleet 共通）。
