#!/usr/bin/env python3
"""comfyui adapter — workflow graph-integrity tests (coverage loop iter 9).

workflows.py builds ComfyUI prompt graphs (9 builders, 449 LoC) submitted to
POST /prompt. The load-bearing invariant is GRAPH INTEGRITY: every node-input
reference `[node_id, slot]` must point to a node that exists in the same
graph — a dangling reference makes ComfyUI reject the whole prompt. These
tests check that generically across all builders, plus per-builder terminal
node + parameter placement.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q
"""
import json
import pathlib
import sys

ADAPTER_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ADAPTER_DIR))

import workflows as wf  # noqa: E402


def _refs(graph: dict):
    """Yield every [node_id, slot] reference found in any node's inputs."""
    for node in graph.values():
        for v in node.get("inputs", {}).values():
            # a wire is [str_node_id, int_slot]
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int):
                yield v


def assert_graph_integrity(graph: dict):
    assert isinstance(graph, dict) and graph, "graph must be a non-empty dict"
    for nid, node in graph.items():
        assert isinstance(nid, str)
        assert "class_type" in node and node["class_type"], f"node {nid} missing class_type"
        assert isinstance(node.get("inputs", {}), dict), f"node {nid} inputs must be a dict"
    ids = set(graph.keys())
    for ref in _refs(graph):
        assert ref[0] in ids, f"dangling reference {ref} → no node {ref[0]!r}"
    # the graph must be JSON-serializable (it is POSTed as JSON to /prompt)
    json.dumps(graph)


# ── one parametrized integrity sweep over every builder ──────────────────────

ALL_GRAPHS = {
    "txt2img": wf.txt2img("ckpt.safetensors", "a cat", "bad", 832, 1216, 25, 7.0, 42),
    "img2img": wf.img2img("ckpt.safetensors", "init.png", "a cat", "bad", 0.55, 25, 7.0, 42),
    "animatediff": wf.animatediff("ckpt", "mm.ckpt", "p", "n", 512, 512, 16, 8, 20, 7.0, 1),
    "svd": wf.svd("svd.safetensors", "init.png", 1024, 576, 14, 6, 127, 0.0, 20, 2.5, 3),
    "wan5b_i2v": wf.wan5b_i2v("wan5b", "init.png", "p", "n", 49, 16, 30, 6.0, 7),
    "sbv2": wf.sbv2("jvnv", "こんにちは", "JP", 0, 1.0, 0.2, 0.6),
    "xtts": wf.xtts("xtts_v2", "hello", "en", "spk.wav"),
    "musicgen": wf.musicgen("medium", "lofi", 15.0, 3.0, 9),
    "stable_audio": wf.stable_audio("sa-open", "rain sfx", 8.0, 50, 7.0, 11),
}


def test_every_builder_produces_an_integral_graph():
    for name, graph in ALL_GRAPHS.items():
        assert_graph_integrity(graph), name


def test_each_graph_has_exactly_one_terminal_save_node():
    terminals = {"SaveImage", "VHS_VideoCombine", "SaveAudio"}
    for name, graph in ALL_GRAPHS.items():
        saves = [n for n in graph.values() if n["class_type"] in terminals]
        assert len(saves) == 1, f"{name}: expected one terminal node, got {len(saves)}"


# ── per-builder parameter placement ──────────────────────────────────────────

def test_txt2img_params_land_in_the_right_nodes():
    g = wf.txt2img("animagine.safetensors", "1girl", "lowres", 832, 1216, 25, 7.5, 12345)
    assert g["4"]["inputs"]["ckpt_name"] == "animagine.safetensors"
    assert g["5"]["inputs"] == {"width": 832, "height": 1216, "batch_size": 1}
    assert g["6"]["inputs"]["text"] == "1girl"          # positive
    assert g["7"]["inputs"]["text"] == "lowres"         # negative
    ks = g["3"]["inputs"]
    assert (ks["seed"], ks["steps"], ks["cfg"], ks["denoise"]) == (12345, 25, 7.5, 1.0)


def test_img2img_strength_is_the_ksampler_denoise():
    g = wf.img2img("c", "photo.png", "p", "n", 0.42, 30, 6.0, 1)
    assert g["10"]["inputs"]["image"] == "photo.png"    # LoadImage
    assert g["3"]["inputs"]["denoise"] == 0.42          # strength → denoise
    # latent comes from VAEEncode (node 11), not an empty latent
    assert g["3"]["inputs"]["latent_image"] == ["11", 0]


def test_video_builders_carry_frame_and_fps_params():
    ad = wf.animatediff("c", "mm", "p", "n", 768, 768, 24, 12, 20, 7.0, 5)
    assert ad["5"]["inputs"]["batch_size"] == 24        # frames → latent batch
    wan = wf.wan5b_i2v("m", "i.png", "p", "n", 49, 16, 30, 6.0, 7)
    assert wan["22"]["inputs"]["num_frames"] == 49
    assert wan["9"]["inputs"]["frame_rate"] == 16


def test_xtts_omits_speaker_wav_when_absent():
    with_wav = wf.xtts("m", "hi", "en", "ref.wav")
    without = wf.xtts("m", "hi", "en", None)
    assert with_wav["21"]["inputs"]["speaker_wav"] == "ref.wav"
    assert "speaker_wav" not in without["21"]["inputs"]


def test_audio_builders_route_duration_and_seed():
    mg = wf.musicgen("medium", "jazz", 20.0, 3.0, 99)
    assert mg["21"]["inputs"]["duration"] == 20.0
    assert mg["21"]["inputs"]["seed"] == 99
    sa = wf.stable_audio("open", "thunder", 5.0, 60, 7.0, 3)
    assert sa["21"]["inputs"]["duration_seconds"] == 5.0
    assert sa["21"]["inputs"]["steps"] == 60
