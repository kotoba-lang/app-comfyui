# comfyui Ansible — Vultr L40S Pattern 1

Idempotent provision of a single Vultr L40S 48GB Ubuntu 22.04 instance into the
`comfyui.etzhayyim.com` upstream (per ADR-0050).

## Prerequisites

- Ubuntu 22.04 LTS instance (Vultr L40S, 100 GiB block storage mounted at `/data`).
- SSH access as `ubuntu` (or any sudo-capable user).
- Ansible 2.15+ on the operator machine (`uv tool install ansible`).
- Secrets in macOS Keychain (per root CLAUDE.md local-secret-storage rule):
  - `etzhayyim.hf / HF_TOKEN` — gated checkpoint downloads
  - `etzhayyim.cf / COMFYUI_TUNNEL_TOKEN` — `cloudflared tunnel token comfyui-etzhayyim` output

## Usage

```bash
cd 60-apps/etzhayyim-project-comfyui/ansible

# 1. set host (Terraform output or manual)
export COMFYUI_HOST=<instance ip>
export COMFYUI_SSH_KEY=~/.ssh/id_ed25519

# 2. run
ansible-playbook site.yml \
  -e hf_token="$(security find-generic-password -s etzhayyim.hf -a HF_TOKEN -w)" \
  -e cloudflared_tunnel_token="$(security find-generic-password -s etzhayyim.cf -a COMFYUI_TUNNEL_TOKEN -w)"
```

First run takes ~1-2 hours (driver install → reboot → CUDA → ComfyUI → model DL
~100-150 GiB). Later runs converge drifted state in 1-2 min.

## Roles

| Role | Responsibility |
|---|---|
| `common` | Ubuntu packages, `comfy` user, UFW (SSH only), `uv` toolchain |
| `nvidia` | NVIDIA 545+ driver, CUDA 12.4 toolkit, reboot on driver change |
| `comfyui` | git clone ComfyUI, CUDA torch, 8 custom node plugins, systemd |
| `models` | HF model downloads into `/data/comfyui/models/{checkpoints,controlnet,...}` |
| `adapter` | sparse-checkout repo `60-apps/.../adapter`, `uv sync`, systemd |
| `llm-backend` | llama.cpp CUDA build + Qwen 2.5 7B Q4_K_M GGUF, systemd (:8010) |
| `cloudflared` | cloudflared install + tunnel token + systemd |

## Destroy-safe layout

Persistent state lives on `/data` (Vultr block storage volume). Destroying the
instance and re-attaching the same volume to a fresh OS disk preserves:

- `/data/comfyui/models/*` (100-150 GiB model cache)
- `/data/comfyui/.venv` (Python deps)
- `/data/adapter/` (repo checkout + venv)
- `/data/llm-backend/` (llama.cpp build + Qwen GGUF)
- `/etc/cloudflared/tunnel-token` (regenerate if rotated)

On re-provision, all roles detect existing artifacts and skip downloads/builds.

## Ports

All bound to `127.0.0.1` — external reach via `cloudflared` only.

| Port | Service |
|---|---|
| 22 | SSH (UFW allow) |
| 8001 | adapter (OpenAI→ComfyUI) |
| 8010 | llm-backend (Qwen 7B OpenAI-compat) |
| 8188 | ComfyUI |

## Verification

```bash
ansible comfyui-l40s -m shell -a 'systemctl is-active comfyui adapter llm-backend cloudflared'
curl -sf https://comfyui.etzhayyim.com/health | jq
```

## Relationship

- ADR-0050 (decision rationale, model catalog, VRAM budget)
- `50-infra/cloudflare/workers/comfyui/` (CF Worker gateway, Phase 2/A2)
- `60-apps/etzhayyim-project-comfyui/adapter/` (canonical Starlette source)
- `50-infra/terraform/comfyui-l40s/` (infra provision — A5, planned)
