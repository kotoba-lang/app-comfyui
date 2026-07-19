# etzhayyim-project-comfyui — Image Generation Gateway (ComfyUI)

## Overview

ComfyUI-based image generation service. CF Worker at `comfyui.etzhayyim.com` fronts a local or remote ComfyUI instance, exposing OpenAI-compat `/v1/images/{generations,edits}` + XRPC `com.etzhayyim.apps.comfyui.*` for txt2img / img2img / (planned) inpaint / ControlNet / LoRA.

Identity: `did:web:comfyui.etzhayyim.com` / nanoid `c0mfyu1x` / handle `comfyui.etzhayyim.com`.

Default checkpoint: **Animagine XL 4.0** (cagliostrolab/animagine-xl-4.0, SDXL-based anime-specialist).

## Topology (target)

```
Client (OpenAI SDK / curl / CLI)
  → comfyui.etzhayyim.com (CF Worker, auth + proxy) [Phase 2]
    - POST /v1/images/generations   → requireAuth → UPSTREAM_URL
    - POST /v1/images/edits          → requireAuth → UPSTREAM_URL
    - POST /xrpc/com.etzhayyim.apps.comfyui.* → XRPC dispatch
    - GET  /v1/models, /_app/meta, /health
  → CF Tunnel comfyui-etzhayyim (cloudflared) [Phase 3]
    ingress catch-all → http://127.0.0.1:8001
  → adapter (~/.local/animagine/server.py, Starlette)
    translates OpenAI /v1/images/* → ComfyUI workflow graph
  → ComfyUI :8188 (~/.local/comfyui/ComfyUI/)
    workflow graph submission → KSampler → MPS
  → PNG returned as b64_json (OpenAI Images API shape)
```

**Auth layers (target, Murakumo pattern)**:
- `sk_live_*` (HYPERDRIVE `vertex_api_key`, scope=`comfyui:generate`) — API consumers
- `COMFYUI_API_KEY` (break-glass, ADR-0023) — emergency
- `mkc_*` HMAC ephemeral (`COMFYUI_CHAT_SECRET`) — browser preview UI (anonymous, 1h TTL)
- `x-kotodama-verified` (internal) — dispatcher-side trust

## Current state (2026-04-22)

**Phase 1 scaffold only.** Project identity + docs exist. No CF Worker, no tunnel, no DNS. Local adapter + ComfyUI run on MacBook Air for development.

### Local dev quick-start

See `~/.local/animagine/` (adapter + LiteLLM) and `~/.local/comfyui/ComfyUI/` (model runtime). These are **not** in the monorepo — personal dev environment on the MacBook Air.

```bash
# Boot all three
cd ~/.local/comfyui/ComfyUI && PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python main.py --port 8188 --listen 127.0.0.1 --disable-auto-launch > /tmp/comfyui.log 2>&1 &
cd ~/.local/animagine && uv run python server.py > /tmp/animagine-server.log 2>&1 &
cd ~/.local/animagine && uv run litellm --config litellm_config.yaml --port 4000 --host 127.0.0.1 > /tmp/litellm-proxy.log 2>&1 &

# Stop
lsof -ti:8001,4000,8188 | xargs kill
```

### Local API surface (verified 2026-04-22)

| Endpoint | Port | Auth | Verified | Latency (MPS M4) |
|---|---|---|---|---|
| `POST /v1/images/generations` | 4000 (LiteLLM) / 8001 (direct) | Bearer (LiteLLM) / none (direct) | ✅ | 169s @ 832x1216, 25 steps |
| `POST /v1/images/edits` | 8001 (direct only) | none | ✅ | 161s @ 832x1216, 25 steps, strength 0.55 |
| `GET /v1/models` | both | — | ✅ | — |
| `GET /health` | 8001 | — | ✅ | — |

LiteLLM proxy does not forward `/v1/images/edits` (v1.52 pass-through gap). img2img must hit adapter `:8001` directly.

## Phase roadmap

### Phase 1 — Scaffold ✅ (2026-04-22)
- `60-apps/etzhayyim-project-comfyui/` dir with CLAUDE.md, PROJECT.jsonld, kotodama.jsonld, deps.toml
- Root `deps.toml` `[[projects]]` + `[[legacy_nanoids]]` entries
- No Worker, no DNS, no tunnel. Identity fixed: `did:web:comfyui.etzhayyim.com` / `c0mfyu1x`.

### Phase 2 — CF Worker gateway (planned)
- `infra/cloudflare/workers/comfyui/` (Hono + TypeScript)
- Auth layer (Murakumo-equivalent: `sk_live_*` HYPERDRIVE + `mkc_*` HMAC + `COMFYUI_API_KEY` break-glass)
- OpenAI-compat routes (`/v1/images/generations`, `/v1/images/edits`, `/v1/models`, `/_app/meta`, `/health`)
- XRPC routes (`com.etzhayyim.apps.comfyui.*`)
- `wrangler.jsonc` with `UPSTREAM_URL` env binding (defaults to tunnel)
- Lexicon JSONs in `00-contracts/lexicons/com/etzhayyim/apps/comfyui/`

### Phase 3 — Tunnel + DNS + Deploy (requires shared-infra approval)
- cloudflared tunnel `comfyui-etzhayyim` on MacBook Air
  - ingress catch-all → `http://127.0.0.1:8001` (adapter)
  - connector runs as launchd service for persistence
- CF DNS: `comfyui.etzhayyim.com` CNAME → `<tunnel-id>.cfargotunnel.com`
- `etzhayyim deploy comfyui` publishes Worker
- Post-deploy smoke: `curl https://comfyui.etzhayyim.com/v1/models` → 200 OK

## NSID namespace (planned)

```
com.etzhayyim.apps.comfyui.generateImage    # txt2img (lexicon: prompt, size, steps, cfg, seed, negative)
com.etzhayyim.apps.comfyui.editImage         # img2img (lexicon: image blob, prompt, strength, ...)
com.etzhayyim.apps.comfyui.inpaintImage      # mask-based inpaint (future)
com.etzhayyim.apps.comfyui.runWorkflow       # raw workflow graph submission (advanced / agent use)
com.etzhayyim.apps.comfyui.listCheckpoints   # enumerate available models
```

Lexicon JSONs go in `00-contracts/lexicons/com/etzhayyim/apps/comfyui/<method>.json` (created in Phase 2 alongside Worker).

## Dependencies

- **ComfyUI** (github.com/comfyanonymous/ComfyUI) — workflow-graph inference engine
- **Animagine XL 4.0** (cagliostrolab/animagine-xl-4.0, ~6.5GB safetensors) — default checkpoint
- **CF Tunnel** (`cloudflared`) — MacBook Air ↔ `comfyui.etzhayyim.com` connectivity (Phase 3)
- **@etzhayyim/xrpc**, **@etzhayyim/kotodama-host-sdk** — when Worker is added (Phase 2)

## Relationship to other projects

- **Murakumo** (`projects/etzhayyim-project-murakumo/`) — server-side text inference (Mac Mini fleet + LiteLLM + Ollama). ComfyUI reuses Murakumo's auth/proxy pattern but is image-specialized, MacBook Air for now.
- **Animeka** / **Mangaka** — future consumers (image gen for character sheets, storyboard panels). They call `comfyui.etzhayyim.com/xrpc/com.etzhayyim.apps.comfyui.generateImage` once Phase 2 lands.
- **Ameno** (`projects/etzhayyim-project-ameno/`) — browser WebGPU image gen (per-actor LoRA merge). Complementary: Ameno = browser-local, ComfyUI = server-side with ControlNet/IPAdapter.

## Non-goals (for now)

- Multi-user scheduling / queue management — single MacBook Air, serial execution
- Remote model management UI — use ComfyUI's own UI at `http://127.0.0.1:8188` locally
- Public anonymous generation — all access gated by Murakumo-style auth
- Video generation (AnimateDiff, SVD) — explicit future scope
