"""OpenAI API → ComfyUI adapter — multi-modal (image + video + audio + LLM).

Translates OpenAI-compat endpoints into ComfyUI workflow graphs, submits to
/prompt, polls /history, fetches /view, and returns bytes as b64 JSON.

Canonical source for `comfyui.etzhayyim.com` upstream. Per ADR-0050 this process
runs on a Vultr L40S 48GB CUDA node (Phase B). On MacBook Air MPS (dev)
image endpoints work; video/audio require ComfyUI custom nodes that may
not be installed — those endpoints will surface ComfyUI 400 errors until
the Ansible role (Phase A4) provisions them.

Endpoints:
    POST /v1/images/generations    # txt2img
    POST /v1/images/edits           # img2img
    POST /v1/videos/generations     # animatediff / svd / wan5b
    POST /v1/audio/speech           # sbv2 / xtts (OpenAI-compat TTS shape)
    POST /v1/audio/music            # musicgen / stable-audio
    POST /v1/chat/completions       # LLM passthrough (Qwen 7B on L40S, LLM_BACKEND_URL)
    GET  /v1/models
    GET  /health
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import random
import time
import uuid

import httpx
from PIL import Image
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

import workflows as W

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
LLM_BACKEND_URL = os.environ.get("LLM_BACKEND_URL", "")  # OpenAI-compat (vllm/llama.cpp/ollama)
LLM_BACKEND_KEY = os.environ.get("LLM_BACKEND_KEY", "")

IMAGE_CHECKPOINT = os.environ.get("COMFY_CHECKPOINT", "animagine-xl-4.0.safetensors")
ANIMATEDIFF_MOTION = os.environ.get("ANIMATEDIFF_MOTION_MODULE", "mm_sdxl_v10.ckpt")
SVD_CHECKPOINT = os.environ.get("SVD_CHECKPOINT", "svd_xt_1_1.safetensors")
WAN5B_MODEL = os.environ.get("WAN5B_MODEL", "wan-5b.safetensors")
MUSICGEN_MODEL = os.environ.get("MUSICGEN_MODEL", "facebook/musicgen-medium")
STABLE_AUDIO_MODEL = os.environ.get("STABLE_AUDIO_MODEL", "stable-audio-open-1.0.safetensors")
SBV2_DEFAULT_MODEL = os.environ.get("SBV2_DEFAULT_MODEL", "sbv2_jp_default.safetensors")
XTTS_DEFAULT_MODEL = os.environ.get("XTTS_DEFAULT_MODEL", "xtts-v2")

PORT = int(os.environ.get("ANIMAGINE_PORT", "8001"))
POLL_TIMEOUT_S = int(os.environ.get("COMFY_POLL_TIMEOUT_S", "600"))
CLIENT_ID = str(uuid.uuid4())


# ── Helpers ─────────────────────────────────────────────────────────────────


def _rand_seed() -> int:
    return random.randint(0, 2**32 - 1)


def _parse_size(size: str, default_w: int = 1024, default_h: int = 1024) -> tuple[int, int]:
    if not size:
        return default_w, default_h
    try:
        w_str, h_str = size.split("x", 1)
        return int(w_str), int(h_str)
    except Exception:
        return default_w, default_h


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=POLL_TIMEOUT_S, write=30, pool=5)
    )


async def _submit(client: httpx.AsyncClient, graph: dict) -> str:
    r = await client.post(
        f"{COMFY_URL}/prompt", json={"prompt": graph, "client_id": CLIENT_ID}
    )
    r.raise_for_status()
    return r.json()["prompt_id"]


async def _wait_for(client: httpx.AsyncClient, prompt_id: str) -> dict:
    """Poll /history until done; return full outputs dict keyed by node id."""
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        r = await client.get(f"{COMFY_URL}/history/{prompt_id}")
        if r.status_code == 200:
            data = r.json()
            if prompt_id in data:
                outs = data[prompt_id].get("outputs", {})
                if outs:
                    return outs
        await asyncio.sleep(1.0)
    raise TimeoutError(f"ComfyUI timed out after {POLL_TIMEOUT_S}s")


def _collect_media(outputs: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Walk outputs; return (images, videos, audios) lists of metadata dicts."""
    images, videos, audios = [], [], []
    for node_out in outputs.values():
        images.extend(node_out.get("images", []))
        videos.extend(node_out.get("gifs", []))  # VHS_VideoCombine emits under 'gifs'
        videos.extend(node_out.get("videos", []))  # some nodes use 'videos'
        audios.extend(node_out.get("audio", []))
        audios.extend(node_out.get("audios", []))
    return images, videos, audios


async def _fetch_view(client: httpx.AsyncClient, meta: dict) -> bytes:
    r = await client.get(
        f"{COMFY_URL}/view",
        params={
            "filename": meta["filename"],
            "subfolder": meta.get("subfolder", ""),
            "type": meta.get("type", "output"),
        },
    )
    r.raise_for_status()
    return r.content


async def _upload_image(
    client: httpx.AsyncClient, image_bytes: bytes, basename: str
) -> str:
    r = await client.post(
        f"{COMFY_URL}/upload/image",
        data={"overwrite": "true"},
        files={"image": (basename, image_bytes, "image/png")},
    )
    r.raise_for_status()
    j = r.json()
    subfolder = j.get("subfolder", "")
    name = j.get("name", basename)
    return f"{subfolder}/{name}" if subfolder else name


def _ext_from_filename(name: str, fallback: str) -> str:
    if "." in name:
        return name.rsplit(".", 1)[1].lower()
    return fallback


# ── Image endpoints ─────────────────────────────────────────────────────────


async def images_generations(request: Request) -> JSONResponse:
    body = await request.json()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"error": {"message": "prompt is required"}}, status_code=400)

    width, height = _parse_size(body.get("size", ""))
    n = max(1, int(body.get("n") or 1))
    steps = int(body.get("steps") or body.get("num_inference_steps") or 28)
    cfg = float(body.get("guidance_scale") or 6.5)
    negative = body.get("negative_prompt", W.DEFAULT_NEGATIVE_IMAGE)
    seed_raw = body.get("seed")
    base_seed = int(seed_raw) if seed_raw is not None else _rand_seed()

    data: list[dict] = []
    async with _client() as client:
        for i in range(n):
            graph = W.txt2img(
                IMAGE_CHECKPOINT, prompt, negative, width, height, steps, cfg, base_seed + i
            )
            prompt_id = await _submit(client, graph)
            imgs, _, _ = _collect_media(await _wait_for(client, prompt_id))
            for meta in imgs:
                png = await _fetch_view(client, meta)
                data.append({"b64_json": base64.b64encode(png).decode(), "format": "png"})
    return JSONResponse({"created": int(time.time()), "data": data})


async def images_edits(request: Request) -> JSONResponse:
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("image")
        if upload is None:
            return JSONResponse({"error": {"message": "image is required"}}, status_code=400)
        init_bytes = await upload.read() if hasattr(upload, "read") else bytes(upload)
        prompt = (form.get("prompt") or "").strip()
        n = max(1, int(form.get("n") or 1))
        size = form.get("size") or ""
        strength = float(form.get("strength") or 0.7)
        steps = int(form.get("steps") or 28)
        cfg = float(form.get("guidance_scale") or 6.5)
        negative = form.get("negative_prompt") or W.DEFAULT_NEGATIVE_IMAGE
        seed_raw = form.get("seed")
        seed = int(seed_raw) if seed_raw not in (None, "", "null") else None
    else:
        body = await request.json()
        image_field = body.get("image", "")
        if not image_field:
            return JSONResponse(
                {"error": {"message": "image (base64) is required"}}, status_code=400
            )
        try:
            payload = (
                image_field.split(",", 1)[1]
                if image_field.startswith("data:")
                else image_field
            )
            init_bytes = base64.b64decode(payload)
        except Exception as exc:
            return JSONResponse({"error": {"message": f"invalid image: {exc}"}}, status_code=400)
        prompt = (body.get("prompt") or "").strip()
        n = max(1, int(body.get("n") or 1))
        size = body.get("size") or ""
        strength = float(body.get("strength") or 0.7)
        steps = int(body.get("steps") or 28)
        cfg = float(body.get("guidance_scale") or 6.5)
        negative = body.get("negative_prompt", W.DEFAULT_NEGATIVE_IMAGE)
        seed_raw = body.get("seed")
        seed = int(seed_raw) if seed_raw is not None else None

    if not prompt:
        return JSONResponse({"error": {"message": "prompt is required"}}, status_code=400)

    if size:
        width, height = _parse_size(size)
        resized = Image.open(io.BytesIO(init_bytes)).convert("RGB").resize(
            (width, height), Image.LANCZOS
        )
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        init_bytes = buf.getvalue()

    base_seed = seed if seed is not None else _rand_seed()
    data: list[dict] = []
    async with _client() as client:
        comfy_name = await _upload_image(
            client, init_bytes, f"adapter_init_{uuid.uuid4().hex}.png"
        )
        for i in range(n):
            graph = W.img2img(
                IMAGE_CHECKPOINT,
                comfy_name,
                prompt,
                negative,
                strength,
                steps,
                cfg,
                base_seed + i,
            )
            prompt_id = await _submit(client, graph)
            imgs, _, _ = _collect_media(await _wait_for(client, prompt_id))
            for meta in imgs:
                png = await _fetch_view(client, meta)
                data.append({"b64_json": base64.b64encode(png).decode(), "format": "png"})
    return JSONResponse({"created": int(time.time()), "data": data})


# ── Video endpoint ──────────────────────────────────────────────────────────


async def videos_generations(request: Request) -> JSONResponse:
    """Body: {model: animatediff|svd|wan5b, mode?: text2video|image2video,
    prompt, image?, duration_s?, fps?, size?, steps?, cfg?, seed?}.
    Returns {data: [{b64_json, format: 'mp4', duration_s, fps, model}]}.
    """
    body = await request.json()
    model = body.get("model", "animatediff")
    prompt = (body.get("prompt") or "").strip()
    negative = body.get("negative_prompt", W.DEFAULT_NEGATIVE_VIDEO)
    duration_s = float(body.get("duration_s") or 2.0)
    fps = int(body.get("fps") or 8)
    frames = max(2, int(round(duration_s * fps)))
    steps = int(body.get("steps") or 20)
    cfg = float(body.get("cfg") or body.get("guidance_scale") or 7.0)
    seed_raw = body.get("seed")
    seed = int(seed_raw) if seed_raw is not None else _rand_seed()

    # Optional init image for image-to-video
    init_b64 = body.get("image") or ""
    init_bytes: bytes | None = None
    if init_b64:
        try:
            payload = init_b64.split(",", 1)[1] if init_b64.startswith("data:") else init_b64
            init_bytes = base64.b64decode(payload)
        except Exception as exc:
            return JSONResponse({"error": {"message": f"invalid image: {exc}"}}, status_code=400)

    # Build graph per model
    async with _client() as client:
        if model == "animatediff":
            if not prompt:
                return JSONResponse({"error": {"message": "prompt required for animatediff"}}, status_code=400)
            width, height = _parse_size(body.get("size", "1024x576"))
            graph = W.animatediff(
                IMAGE_CHECKPOINT,
                ANIMATEDIFF_MOTION,
                prompt,
                negative,
                width,
                height,
                frames,
                fps,
                steps,
                cfg,
                seed,
            )
        elif model == "svd":
            if init_bytes is None:
                return JSONResponse({"error": {"message": "image required for svd"}}, status_code=400)
            width, height = _parse_size(body.get("size", "1024x576"))
            comfy_name = await _upload_image(
                client, init_bytes, f"adapter_svd_{uuid.uuid4().hex}.png"
            )
            motion_bucket = int(body.get("motion_bucket_id") or 127)
            augment = float(body.get("augmentation_level") or 0.0)
            graph = W.svd(
                SVD_CHECKPOINT,
                comfy_name,
                width,
                height,
                min(frames, 25),
                fps,
                motion_bucket,
                augment,
                steps,
                cfg,
                seed,
            )
        elif model == "wan5b":
            if init_bytes is None:
                return JSONResponse({"error": {"message": "image required for wan5b"}}, status_code=400)
            comfy_name = await _upload_image(
                client, init_bytes, f"adapter_wan_{uuid.uuid4().hex}.png"
            )
            graph = W.wan5b_i2v(
                WAN5B_MODEL, comfy_name, prompt, negative, frames, fps, steps, cfg, seed
            )
        else:
            return JSONResponse(
                {"error": {"message": f"unknown video model: {model}"}}, status_code=400
            )

        prompt_id = await _submit(client, graph)
        _, vids, _ = _collect_media(await _wait_for(client, prompt_id))
        if not vids:
            return JSONResponse(
                {"error": {"message": "no video output — check ComfyUI node availability"}},
                status_code=502,
            )
        data = []
        for meta in vids:
            blob = await _fetch_view(client, meta)
            fmt = _ext_from_filename(meta.get("filename", ""), "mp4")
            data.append(
                {
                    "b64_json": base64.b64encode(blob).decode(),
                    "format": fmt,
                    "duration_s": duration_s,
                    "fps": fps,
                    "model": model,
                }
            )
    return JSONResponse({"created": int(time.time()), "data": data})


# ── Audio: speech (TTS) ──────────────────────────────────────────────────────


async def audio_speech(request: Request) -> JSONResponse | StreamingResponse:
    """OpenAI-compat TTS body: {input, voice?, model?, language?, response_format?}.
    model = 'sbv2' (default, JP anime) | 'xtts' (multilingual).
    response_format = 'json' (default, b64) | 'wav' (binary stream).
    """
    body = await request.json()
    text = (body.get("input") or "").strip()
    if not text:
        return JSONResponse({"error": {"message": "input is required"}}, status_code=400)

    model = body.get("model", "sbv2")
    voice = body.get("voice") or None
    language = body.get("language") or body.get("lang") or "ja"
    response_format = body.get("response_format", "json")

    async with _client() as client:
        if model == "sbv2":
            model_name = voice or SBV2_DEFAULT_MODEL
            graph = W.sbv2(
                model_name,
                text,
                language,
                int(body.get("speaker_id") or 0),
                float(body.get("length_scale") or 1.0),
                float(body.get("sdp_ratio") or 0.2),
                float(body.get("noise_scale") or 0.667),
            )
        elif model == "xtts":
            graph = W.xtts(
                XTTS_DEFAULT_MODEL,
                text,
                language,
                voice,  # speaker_wav filename (must be pre-uploaded)
            )
        else:
            return JSONResponse(
                {"error": {"message": f"unknown TTS model: {model}"}}, status_code=400
            )

        prompt_id = await _submit(client, graph)
        _, _, audios = _collect_media(await _wait_for(client, prompt_id))
        if not audios:
            return JSONResponse(
                {"error": {"message": "no audio output — check ComfyUI TTS node"}},
                status_code=502,
            )
        meta = audios[0]
        blob = await _fetch_view(client, meta)
        fmt = _ext_from_filename(meta.get("filename", ""), "wav")

        if response_format in ("wav", "mp3", "binary"):
            mime = {"wav": "audio/wav", "mp3": "audio/mpeg"}.get(fmt, "audio/wav")
            return StreamingResponse(iter([blob]), media_type=mime)

        return JSONResponse(
            {
                "created": int(time.time()),
                "data": [
                    {"b64_json": base64.b64encode(blob).decode(), "format": fmt, "model": model}
                ],
            }
        )


# ── Audio: music / SFX ──────────────────────────────────────────────────────


async def audio_music(request: Request) -> JSONResponse:
    """Body: {prompt, duration_s?, model? ('musicgen'|'stable-audio'), cfg?, seed?}."""
    body = await request.json()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"error": {"message": "prompt is required"}}, status_code=400)

    model = body.get("model", "musicgen")
    duration_s = float(body.get("duration_s") or 15.0)
    cfg = float(body.get("cfg") or body.get("cfg_coef") or 3.0)
    seed_raw = body.get("seed")
    seed = int(seed_raw) if seed_raw is not None else _rand_seed()

    async with _client() as client:
        if model == "musicgen":
            graph = W.musicgen(MUSICGEN_MODEL, prompt, duration_s, cfg, seed)
        elif model == "stable-audio":
            steps = int(body.get("steps") or 50)
            graph = W.stable_audio(
                STABLE_AUDIO_MODEL, prompt, duration_s, steps, float(body.get("cfg_scale") or 7.0), seed
            )
        else:
            return JSONResponse(
                {"error": {"message": f"unknown music model: {model}"}}, status_code=400
            )

        prompt_id = await _submit(client, graph)
        _, _, audios = _collect_media(await _wait_for(client, prompt_id))
        if not audios:
            return JSONResponse(
                {"error": {"message": "no audio output — check ComfyUI music node"}},
                status_code=502,
            )
        meta = audios[0]
        blob = await _fetch_view(client, meta)
        fmt = _ext_from_filename(meta.get("filename", ""), "wav")
    return JSONResponse(
        {
            "created": int(time.time()),
            "data": [
                {
                    "b64_json": base64.b64encode(blob).decode(),
                    "format": fmt,
                    "duration_s": duration_s,
                    "model": model,
                }
            ],
        }
    )


# ── LLM passthrough (mid-tier Qwen 7B on L40S) ──────────────────────────────


async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    """OpenAI-compat /v1/chat/completions passthrough. Upstream =
    LLM_BACKEND_URL env var (vllm / llama.cpp / ollama on same L40S).
    503 when backend not configured (MacBook dev)."""
    if not LLM_BACKEND_URL:
        return JSONResponse(
            {
                "error": {
                    "message": "LLM backend not configured (LLM_BACKEND_URL empty)",
                    "hint": "Qwen 7B hosted only on L40S per ADR-0050; use Murakumo for dev",
                }
            },
            status_code=503,
        )
    body_bytes = await request.body()
    headers = {"Content-Type": "application/json"}
    if LLM_BACKEND_KEY:
        headers["Authorization"] = f"Bearer {LLM_BACKEND_KEY}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(read=120, connect=5, write=30, pool=5)) as c:
        r = await c.post(
            f"{LLM_BACKEND_URL}/v1/chat/completions", content=body_bytes, headers=headers
        )
    return JSONResponse(r.json(), status_code=r.status_code)


# ── Meta ────────────────────────────────────────────────────────────────────


async def models(_: Request) -> JSONResponse:
    data = [
        {"id": "animagine-xl", "object": "model", "owned_by": "local-comfy", "modality": "image"},
        {"id": "animatediff", "object": "model", "owned_by": "local-comfy", "modality": "video"},
        {"id": "svd", "object": "model", "owned_by": "local-comfy", "modality": "video"},
        {"id": "wan5b", "object": "model", "owned_by": "local-comfy", "modality": "video"},
        {"id": "sbv2", "object": "model", "owned_by": "local-comfy", "modality": "audio-tts"},
        {"id": "xtts", "object": "model", "owned_by": "local-comfy", "modality": "audio-tts"},
        {"id": "musicgen", "object": "model", "owned_by": "local-comfy", "modality": "audio-music"},
        {"id": "stable-audio", "object": "model", "owned_by": "local-comfy", "modality": "audio-sfx"},
    ]
    if LLM_BACKEND_URL:
        data.append({"id": "qwen2.5-7b", "object": "model", "owned_by": "local-llm", "modality": "chat"})
    return JSONResponse({"object": "list", "data": data})


async def health(_: Request) -> JSONResponse:
    status: dict = {
        "status": "ok",
        "backend": "comfyui",
        "comfy_url": COMFY_URL,
        "image_checkpoint": IMAGE_CHECKPOINT,
        "comfy_reachable": False,
        "llm_backend_configured": bool(LLM_BACKEND_URL),
    }
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(f"{COMFY_URL}/system_stats")
            status["comfy_reachable"] = r.status_code == 200
    except Exception:
        pass
    return JSONResponse(status)


app = Starlette(
    routes=[
        Route("/v1/images/generations", images_generations, methods=["POST"]),
        Route("/v1/images/edits", images_edits, methods=["POST"]),
        Route("/v1/videos/generations", videos_generations, methods=["POST"]),
        Route("/v1/audio/speech", audio_speech, methods=["POST"]),
        Route("/v1/audio/music", audio_music, methods=["POST"]),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/models", models, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
    ]
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT)
