"""Murakumo fleet ComfyUI router — jacob:8200.

Capability-aware round-robin: inspects the workflow JSON of each /prompt
request, identifies the model referenced (CheckpointLoaderSimple /
DiffusersLoader / WanVideo*), and dispatches only to minis declared in
`fleet-models.json` as having that model.

Topology (CF Worker = edge thin proxy only; control plane = jacob):
    client → comfyui.etzhayyim.com (CF Worker auth)
        → CF Tunnel → jacob:8200 (this router)
            → <eligible-mini>.murakumo.lan:8188 (vanilla ComfyUI worker)

Declaration SSoT: 60-apps/etzhayyim-project-murakumo/fleet-models.json
Managed by: `etzhayyim murakumo models {declare,list,apply}`.
"""
from __future__ import annotations

import asyncio
import collections
import itertools
import json
import os
import time
import urllib.parse
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

# ── Fleet + capability config (declaration-driven) ──────────────────────────

DEFAULT_FLEET_MODELS = (
    Path(__file__).resolve().parents[2]
    / "etzhayyim-project-murakumo"
    / "fleet-models.json"
)
FLEET_MODELS_PATH = Path(os.environ.get("FLEET_MODELS_PATH", str(DEFAULT_FLEET_MODELS)))


def _load_declaration() -> dict:
    if FLEET_MODELS_PATH.exists():
        return json.loads(FLEET_MODELS_PATH.read_text())
    return {"fleet": [], "models": {}}


DECL = _load_declaration()
COMFY_PORT = int(os.environ.get("COMFY_WORKER_PORT", "8188"))
FLEET = [
    h.strip() for h in os.environ.get("COMFY_FLEET", ",".join(DECL.get("fleet", []))).split(",")
    if h.strip()
]
COMFY_URLS = [f"http://{h}.murakumo.lan:{COMFY_PORT}" for h in FLEET]


def _mini_url(name: str) -> str:
    return f"http://{name}.murakumo.lan:{COMFY_PORT}"


# Build capability index: workflow signature → eligible mini URLs
CHECKPOINT_INDEX: dict[str, list[str]] = {}    # filename → minis (comfyui_checkpoint)
DIFFUSERS_INDEX: dict[str, list[str]] = {}     # repo or sub-string → minis (comfyui_diffusers)
WAN_MINIS: list[str] = []                       # any wan model

for _name, _m in DECL.get("models", {}).items():
    _kind = _m.get("kind")
    _targets = [_mini_url(h) for h in _m.get("target_minis", []) if h in FLEET]
    if not _targets:
        continue
    if _kind == "comfyui_checkpoint":
        CHECKPOINT_INDEX[_m.get("filename", "")] = _targets
    elif _kind == "comfyui_diffusers":
        repo = _m.get("diffusers_repo", "")
        CHECKPOINT_INDEX[repo] = _targets  # match by repo string in DiffusersLoader path
        DIFFUSERS_INDEX[repo.replace("/", "--")] = _targets
    elif _kind == "comfyui_wan":
        for t in _targets:
            if t not in WAN_MINIS:
                WAN_MINIS.append(t)

LISTEN_PORT = int(os.environ.get("FLEET_ROUTER_PORT", "8200"))
TTL_S = int(os.environ.get("FLEET_TTL_S", "3600"))
POLL_TIMEOUT_S = int(os.environ.get("FLEET_POLL_TIMEOUT_S", "600"))

# ── State ───────────────────────────────────────────────────────────────────

_rr_counters: dict[str, itertools.count] = collections.defaultdict(itertools.count)
_prompt_routes: dict[str, tuple[str, float]] = {}   # prompt_id  -> (upstream_url, ts)
_filename_routes: dict[str, tuple[str, float]] = {}  # filename   -> (upstream_url, ts)


def _gc():
    now = time.time()
    for table in (_prompt_routes, _filename_routes):
        stale = [k for k, (_, ts) in table.items() if now - ts > TTL_S]
        for k in stale:
            del table[k]


def _eligible_minis_for(workflow: dict) -> tuple[list[str], str | None]:
    """Inspect workflow JSON; return (eligible_minis, matched_model_signature).
    Falls back to entire fleet (round-robin) when no checkpoint reference is
    found (e.g. /object_info probe, control-plane calls)."""
    prompt = workflow.get("prompt") if isinstance(workflow, dict) else None
    if not isinstance(prompt, dict):
        return COMFY_URLS, None
    for _, node in prompt.items():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        ins = node.get("inputs", {}) or {}
        if ct == "CheckpointLoaderSimple":
            fn = ins.get("ckpt_name", "")
            if fn in CHECKPOINT_INDEX:
                return CHECKPOINT_INDEX[fn], fn
        elif ct == "DiffusersLoader":
            mp = ins.get("model_path", "")
            for repo, urls in CHECKPOINT_INDEX.items():
                if "/" in repo and (repo in mp or repo.replace("/", "--") in mp):
                    return urls, repo
        elif ct.startswith("WanVideo") or ct.startswith("Wan_"):
            return WAN_MINIS or COMFY_URLS, "wan"
    return COMFY_URLS, None


def _pick_round_robin(pool: list[str], key: str) -> str:
    if not pool:
        raise RuntimeError(f"no eligible minis for {key}")
    counter = _rr_counters[key or "_default"]
    idx = next(counter) % len(pool)
    return pool[idx]


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=POLL_TIMEOUT_S, write=30, pool=5)
    )


# ── /prompt — dispatch + record ─────────────────────────────────────────────


async def post_prompt(request: Request) -> Response:
    body = await request.body()
    try:
        workflow = json.loads(body) if body else {}
    except json.JSONDecodeError:
        workflow = {}
    pool, sig = _eligible_minis_for(workflow)
    if not pool:
        return JSONResponse({"error": f"no fleet mini hosts model required by workflow (signature={sig})"}, 503)
    upstream = _pick_round_robin(pool, sig or "_default")
    async with _client() as cx:
        try:
            r = await cx.post(f"{upstream}/prompt", content=body,
                              headers={"content-type": "application/json"})
        except httpx.RequestError as e:
            return JSONResponse({"error": f"upstream {upstream} unreachable: {e}"}, 502)
    if r.status_code == 200:
        try:
            data = r.json()
            pid = data.get("prompt_id")
            if pid:
                _prompt_routes[pid] = (upstream, time.time())
                data["_fleet_upstream"] = upstream
                data["_fleet_model_signature"] = sig
                data["_fleet_pool_size"] = len(pool)
                _gc()
                return JSONResponse(data, r.status_code)
        except Exception:
            pass
    return Response(r.content, r.status_code,
                    headers={"content-type": r.headers.get("content-type", "application/json")})


# ── /history/{prompt_id} — proxy to assigned mini ───────────────────────────


async def get_history(request: Request) -> Response:
    pid = request.path_params.get("prompt_id", "")
    route = _prompt_routes.get(pid)
    if not route:
        return JSONResponse({"error": f"unknown prompt_id {pid}; not routed by this fleet router"}, 404)
    upstream, _ = route
    async with _client() as cx:
        r = await cx.get(f"{upstream}/history/{pid}")
    # opportunistically record filename→mini for /view routing
    if r.status_code == 200:
        try:
            data = r.json()
            for _pid, info in data.items():
                outputs = info.get("outputs", {}) or {}
                for _node_id, node_out in outputs.items():
                    for kind in ("images", "gifs"):
                        for item in node_out.get(kind, []) or []:
                            fn = item.get("filename")
                            if fn:
                                _filename_routes[fn] = (upstream, time.time())
        except Exception:
            pass
    return Response(r.content, r.status_code,
                    headers={"content-type": r.headers.get("content-type", "application/json")})


# ── /view — look up by filename, then fan-out fallback ──────────────────────


async def get_view(request: Request) -> Response:
    params = dict(request.query_params)
    fn = params.get("filename", "")
    candidates: list[str] = []
    route = _filename_routes.get(fn)
    if route:
        candidates.append(route[0])
    candidates.extend(u for u in COMFY_URLS if u not in candidates)
    query = urllib.parse.urlencode(params)
    async with _client() as cx:
        for upstream in candidates:
            try:
                r = await cx.get(f"{upstream}/view?{query}")
            except httpx.RequestError:
                continue
            if r.status_code == 200:
                _filename_routes[fn] = (upstream, time.time())
                return Response(r.content, 200,
                                headers={"content-type": r.headers.get("content-type", "image/png")})
    return JSONResponse({"error": f"file {fn} not found on any fleet mini"}, 404)


# ── /queue — aggregate ──────────────────────────────────────────────────────


async def get_queue(request: Request) -> Response:
    async def one(upstream: str):
        try:
            async with _client() as cx:
                r = await cx.get(f"{upstream}/queue", timeout=3)
            return upstream, r.json()
        except Exception as e:
            return upstream, {"error": str(e)}
    results = await asyncio.gather(*[one(u) for u in COMFY_URLS])
    agg = {"running": 0, "pending": 0, "by_upstream": {}}
    for upstream, q in results:
        if "error" in q:
            agg["by_upstream"][upstream] = q
            continue
        running = len(q.get("queue_running", []) or [])
        pending = len(q.get("queue_pending", []) or [])
        agg["running"] += running
        agg["pending"] += pending
        agg["by_upstream"][upstream] = {"running": running, "pending": pending}
    return JSONResponse(agg)


# ── /system_stats — aggregate ───────────────────────────────────────────────


async def get_system_stats(request: Request) -> Response:
    async def one(upstream: str):
        try:
            async with _client() as cx:
                r = await cx.get(f"{upstream}/system_stats", timeout=3)
            return upstream, r.json(), r.status_code
        except Exception as e:
            return upstream, {"error": str(e)}, 0
    results = await asyncio.gather(*[one(u) for u in COMFY_URLS])
    healthy = sum(1 for _, _, code in results if code == 200)
    return JSONResponse({
        "fleet_size": len(COMFY_URLS),
        "healthy": healthy,
        "minis": {u: data for u, data, _ in results},
    })


# ── /health — router itself ─────────────────────────────────────────────────


async def get_health(request: Request) -> Response:
    return JSONResponse({
        "ok": True,
        "fleet": FLEET,
        "fleet_size": len(COMFY_URLS),
        "declaration_path": str(FLEET_MODELS_PATH),
        "capability_index": {
            "checkpoint": {k: [u.split("//")[-1] for u in v] for k, v in CHECKPOINT_INDEX.items()},
            "wan_minis": [u.split("//")[-1] for u in WAN_MINIS],
        },
        "tracked_prompts": len(_prompt_routes),
        "tracked_filenames": len(_filename_routes),
    })


# ── Generic passthrough for everything else ─────────────────────────────────


async def passthrough(request: Request) -> Response:
    """For misc endpoints (/object_info, /embeddings, /extensions, etc.) — fan
    out to one mini (first) since they return uniform fleet-wide info."""
    path = request.url.path
    query = request.url.query
    suffix = f"{path}?{query}" if query else path
    upstream = COMFY_URLS[0]
    async with _client() as cx:
        try:
            r = await cx.request(request.method, f"{upstream}{suffix}",
                                  content=await request.body(),
                                  headers={k: v for k, v in request.headers.items()
                                           if k.lower() not in ("host", "content-length")})
        except httpx.RequestError as e:
            return JSONResponse({"error": f"upstream {upstream}: {e}"}, 502)
    return Response(r.content, r.status_code,
                    headers={"content-type": r.headers.get("content-type", "application/json")})


# ── App ─────────────────────────────────────────────────────────────────────

app = Starlette(routes=[
    Route("/prompt", post_prompt, methods=["POST"]),
    Route("/history/{prompt_id}", get_history, methods=["GET"]),
    Route("/view", get_view, methods=["GET"]),
    Route("/queue", get_queue, methods=["GET"]),
    Route("/system_stats", get_system_stats, methods=["GET"]),
    Route("/health", get_health, methods=["GET"]),
    Route("/{rest:path}", passthrough,
          methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]),
])


if __name__ == "__main__":
    import uvicorn
    print(f"fleet router listening :{LISTEN_PORT}, upstreams = {COMFY_URLS}")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
