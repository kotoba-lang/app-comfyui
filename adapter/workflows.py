"""ComfyUI workflow graph templates.

Each function returns a dict describing a ComfyUI prompt graph (JSON shape
accepted by POST /prompt). The node class names assume the following custom
nodes are installed on the ComfyUI instance (Ansible role A4 is responsible):

  - Image:
      core ComfyUI (CheckpointLoaderSimple, KSampler, CLIPTextEncode,
      VAEEncode, VAEDecode, EmptyLatentImage, LoadImage, SaveImage)

  - Video:
      ComfyUI-AnimateDiff-Evolved (ADE_LoadAnimateDiffModel,
        ADE_ApplyAnimateDiffModel)
      ComfyUI-VideoHelperSuite (VHS_VideoCombine)
      ComfyUI (ImageOnlyCheckpointLoader, SVD_img2vid_Conditioning,
        VideoLinearCFGGuidance — bundled in recent releases)
      ComfyUI-WAN  (WanVideoModelLoader, WanVideoTextEncode,
        WanVideoSampler — or equivalent names once pinned)

  - Audio:
      ComfyUI-SBV2 (SBV2_LoadModel, SBV2_Synthesize, SaveAudio)
      ComfyUI-XTTSv2 (XTTS_LoadModel, XTTS_Synthesize)
      ComfyUI-MusicGen (MusicGenLoader, MusicGenGenerate)
      ComfyUI-StableAudio (StableAudioLoader, StableAudioGenerate)

Node class names are stabilized once the Ansible role pins versions. Until
then these templates are "best-effort" based on the upstream projects'
current node registrations and may need adjustment after Phase B B2.
"""
from __future__ import annotations

DEFAULT_NEGATIVE_IMAGE = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, low quality, "
    "normal quality, jpeg artifacts, signature, watermark, username, blurry"
)

DEFAULT_NEGATIVE_VIDEO = (
    "lowres, blurry, jittery, flicker, text, watermark, distorted anatomy, "
    "deformed, worst quality, low quality"
)


# ── Image: txt2img / img2img ─────────────────────────────────────────────────


def txt2img(
    checkpoint: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
) -> dict:
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["4", 1]}},
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "adapter", "images": ["8", 0]},
        },
    }


def img2img(
    checkpoint: str,
    init_filename: str,
    prompt: str,
    negative: str,
    strength: float,
    steps: int,
    cfg: float,
    seed: int,
) -> dict:
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "10": {"class_type": "LoadImage", "inputs": {"image": init_filename}},
        "11": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["4", 2]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["4", 1]}},
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": strength,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["11", 0],
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "adapter-i2i", "images": ["8", 0]},
        },
    }


# ── Video: AnimateDiff / SVD / WAN 5B ────────────────────────────────────────


def animatediff(
    checkpoint: str,
    motion_module: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    frames: int,
    fps: int,
    steps: int,
    cfg: float,
    seed: int,
) -> dict:
    """AnimateDiff SDXL motion (2-4s clips). Checkpoint = same SDXL base
    (e.g. animagine-xl-4.0.safetensors). Motion module e.g. mm_sdxl_v10.ckpt."""
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "20": {
            "class_type": "ADE_LoadAnimateDiffModel",
            "inputs": {"model_name": motion_module},
        },
        "21": {
            "class_type": "ADE_ApplyAnimateDiffModel",
            "inputs": {"model": ["4", 0], "motion_model": ["20", 0]},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": frames},
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["4", 1]}},
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["21", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["8", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "adapter-anim",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": False,
                "pingpong": False,
            },
        },
    }


def svd(
    checkpoint: str,
    init_filename: str,
    width: int,
    height: int,
    frames: int,
    fps: int,
    motion_bucket_id: int,
    augmentation_level: float,
    steps: int,
    cfg: float,
    seed: int,
) -> dict:
    """Stable Video Diffusion 1.1 XT — image-to-video. Checkpoint =
    svd_xt_1_1.safetensors. 25 frames @ 1024×576 is standard."""
    return {
        "4": {
            "class_type": "ImageOnlyCheckpointLoader",
            "inputs": {"ckpt_name": checkpoint},
        },
        "10": {"class_type": "LoadImage", "inputs": {"image": init_filename}},
        "11": {
            "class_type": "SVD_img2vid_Conditioning",
            "inputs": {
                "clip_vision": ["4", 1],
                "init_image": ["10", 0],
                "vae": ["4", 2],
                "width": width,
                "height": height,
                "video_frames": frames,
                "motion_bucket_id": motion_bucket_id,
                "fps": fps,
                "augmentation_level": augmentation_level,
            },
        },
        "12": {
            "class_type": "VideoLinearCFGGuidance",
            "inputs": {"model": ["4", 0], "min_cfg": 1.0},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["12", 0],
                "positive": ["11", 0],
                "negative": ["11", 1],
                "latent_image": ["11", 2],
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["8", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "adapter-svd",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": False,
                "pingpong": False,
            },
        },
    }


def wan5b_i2v(
    model_name: str,
    init_filename: str,
    prompt: str,
    negative: str,
    frames: int,
    fps: int,
    steps: int,
    cfg: float,
    seed: int,
) -> dict:
    """WAN 5B image-to-video (per main commit e217cc72f06, os project).
    Node names follow the public ComfyUI-WAN plugin as of 2026-04; confirm
    after Ansible plugin pin."""
    return {
        "20": {
            "class_type": "WanVideoModelLoader",
            "inputs": {"model_name": model_name, "precision": "bf16"},
        },
        "21": {
            "class_type": "WanVideoTextEncode",
            "inputs": {"text_pos": prompt, "text_neg": negative, "model": ["20", 0]},
        },
        "10": {"class_type": "LoadImage", "inputs": {"image": init_filename}},
        "22": {
            "class_type": "WanVideoSampler",
            "inputs": {
                "model": ["20", 0],
                "text_embeds": ["21", 0],
                "image": ["10", 0],
                "num_frames": frames,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
            },
        },
        "9": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["22", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "adapter-wan",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": False,
                "pingpong": False,
            },
        },
    }


# ── Audio: TTS (Style-Bert-VITS2 / XTTS v2) ─────────────────────────────────


def sbv2(
    model_name: str,
    text: str,
    language: str,
    speaker_id: int,
    length_scale: float,
    sdp_ratio: float,
    noise_scale: float,
) -> dict:
    """Style-Bert-VITS2 — JP anime voice, primary TTS."""
    return {
        "20": {"class_type": "SBV2_LoadModel", "inputs": {"model_name": model_name}},
        "21": {
            "class_type": "SBV2_Synthesize",
            "inputs": {
                "model": ["20", 0],
                "text": text,
                "language": language,
                "speaker_id": speaker_id,
                "length_scale": length_scale,
                "sdp_ratio": sdp_ratio,
                "noise_scale": noise_scale,
            },
        },
        "22": {
            "class_type": "SaveAudio",
            "inputs": {
                "audio": ["21", 0],
                "filename_prefix": "adapter-tts-sbv2",
                "format": "wav",
            },
        },
    }


def xtts(
    model_name: str,
    text: str,
    language: str,
    speaker_wav: str | None,
) -> dict:
    """XTTS v2 — multilingual voice cloning TTS, fallback when SBV2 unused."""
    inputs: dict = {
        "model": ["20", 0],
        "text": text,
        "language": language,
    }
    if speaker_wav:
        inputs["speaker_wav"] = speaker_wav
    return {
        "20": {"class_type": "XTTS_LoadModel", "inputs": {"model_name": model_name}},
        "21": {"class_type": "XTTS_Synthesize", "inputs": inputs},
        "22": {
            "class_type": "SaveAudio",
            "inputs": {
                "audio": ["21", 0],
                "filename_prefix": "adapter-tts-xtts",
                "format": "wav",
            },
        },
    }


# ── Audio: Music (MusicGen / Stable Audio) ──────────────────────────────────


def musicgen(
    model_name: str,
    prompt: str,
    duration_s: float,
    cfg: float,
    seed: int,
) -> dict:
    """MusicGen medium — BGM short loops (10-30s)."""
    return {
        "20": {"class_type": "MusicGenLoader", "inputs": {"model_name": model_name}},
        "21": {
            "class_type": "MusicGenGenerate",
            "inputs": {
                "model": ["20", 0],
                "prompt": prompt,
                "duration": duration_s,
                "cfg_coef": cfg,
                "seed": seed,
            },
        },
        "22": {
            "class_type": "SaveAudio",
            "inputs": {
                "audio": ["21", 0],
                "filename_prefix": "adapter-music",
                "format": "wav",
            },
        },
    }


def stable_audio(
    model_name: str,
    prompt: str,
    duration_s: float,
    steps: int,
    cfg: float,
    seed: int,
) -> dict:
    """Stable Audio Open — SFX + short BGM, higher-fidelity than MusicGen."""
    return {
        "20": {"class_type": "StableAudioLoader", "inputs": {"model_name": model_name}},
        "21": {
            "class_type": "StableAudioGenerate",
            "inputs": {
                "model": ["20", 0],
                "prompt": prompt,
                "duration_seconds": duration_s,
                "steps": steps,
                "cfg_scale": cfg,
                "seed": seed,
            },
        },
        "22": {
            "class_type": "SaveAudio",
            "inputs": {
                "audio": ["21", 0],
                "filename_prefix": "adapter-sfx",
                "format": "wav",
            },
        },
    }
