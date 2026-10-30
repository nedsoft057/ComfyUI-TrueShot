#!/usr/bin/env python3
"""Headless TrueShot: one product photo -> pixel-true commerce shot pack.

  python run_trueshot.py --comfy-dir ComfyUI --image product.jpg \
      --product "a bottle of LUMA vitamin C serum" \
      --brief "on a marble bathroom counter with eucalyptus|on wet black stones with splashes"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid


def post(server, path, payload):
    req = urllib.request.Request(f"http://{server}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"ComfyUI rejected the prompt ({e.code}):\n{e.read().decode(errors='replace')}")


def get(server, path):
    with urllib.request.urlopen(f"http://{server}{path}") as r:
        return json.loads(r.read())


def wait(server, pid, watch_dir, label="TrueShot"):
    t0, last = time.time(), None
    while True:
        h = get(server, f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                for m in st.get("messages", []):
                    if m[0] == "execution_error":
                        d = m[1]
                        sys.exit(f"[{label}] node {d.get('node_id')} ({d.get('node_type')}) failed:\n"
                                 f"{d.get('exception_message')}\n{''.join(d.get('traceback', [])[-8:])}")
                sys.exit(f"[{label}] failed: {st}")
            print(f"[{label}] done in {time.time() - t0:.0f}s", flush=True)
            return h[pid]
        n = len(os.listdir(watch_dir)) if os.path.isdir(watch_dir) else 0
        if n != last or int(time.time() - t0) % 60 < 5:
            print(f"[{label}] running... {time.time() - t0:.0f}s | {n} files written", flush=True)
            last = n
        time.sleep(5)


def build(a, run):
    P = f"TrueShot/{run}"
    g = {
        "1": {"class_type": "LoadImage", "inputs": {"image": os.path.basename(a.image)}},
        "2": {"class_type": "TrueShotCutout", "inputs": {"image": ["1", 0], "method": a.cutout}},
        "3": {"class_type": "TrueShotMarketplace", "inputs": {"product": ["2", 0], "matte": ["2", 1],
                                                               "size": 1200, "fill": 0.93}},
        "4": {"class_type": "TrueShotPlan", "inputs": {
            "product": ["2", 0], "matte": ["2", 1], "product_name": a.product, "brief": a.brief.replace("|", "\n"),
            "formats": a.formats, "variants": a.variants, "seed": a.seed}},
        # FLUX.2 Klein reference edit, one run per planned shot (list mapping)
        "10": {"class_type": "UNETLoader", "inputs": {"unet_name": a.klein, "weight_dtype": "default"}},
        "11": {"class_type": "CLIPLoader", "inputs": {"clip_name": a.klein_te, "type": "flux2", "device": "default"}},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": a.flux_vae}},
        "13": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": ["4", 2]}},
        "14": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": ""}},
        "16": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 1], "vae": ["12", 0]}},
        "17": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["13", 0], "latent": ["16", 0]}},
        "18": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["14", 0], "latent": ["16", 0]}},
        "19": {"class_type": "CFGGuider", "inputs": {"model": ["10", 0], "positive": ["17", 0],
                                                       "negative": ["18", 0], "cfg": a.klein_cfg}},
        "20": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "21": {"class_type": "Flux2Scheduler", "inputs": {"steps": a.klein_steps, "width": ["4", 3], "height": ["4", 4]}},
        "22": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["4", 3], "height": ["4", 4], "batch_size": 1}},
        "23": {"class_type": "RandomNoise", "inputs": {"noise_seed": ["4", 5]}},
        "24": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["23", 0], "guider": ["19", 0], "sampler": ["20", 0], "sigmas": ["21", 0], "latent_image": ["22", 0]}},
        "25": {"class_type": "VAEDecode", "inputs": {"samples": ["24", 0], "vae": ["12", 0]}},
        # pixel-true restore + outputs
        "30": {"class_type": "TrueShotRestore", "inputs": {"plan": ["4", 0], "product": ["2", 0], "matte": ["2", 1],
                                                            "generated": ["25", 0], "harmonise": a.harmonise,
                                                            "edge_softness": 1.5}},
        "31": {"class_type": "SaveImage", "inputs": {"images": ["30", 0], "filename_prefix": f"{P}/shot"}},
        "32": {"class_type": "SaveImage", "inputs": {"images": ["30", 1], "filename_prefix": f"{P}/compare"}},
        "33": {"class_type": "SaveImage", "inputs": {"images": ["30", 2], "filename_prefix": f"{P}/ai_only"}},
        "34": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": f"{P}/marketplace"}},
        "35": {"class_type": "SaveImage", "inputs": {"images": ["2", 2], "filename_prefix": f"{P}/cutout"}},
        "36": {"class_type": "TrueShotContactSheet", "inputs": {"images": ["30", 0], "columns": 3}},
        "37": {"class_type": "SaveImage", "inputs": {"images": ["36", 0], "filename_prefix": f"{P}/sheet"}},
        "38": {"class_type": "PreviewAny", "inputs": {"source": ["30", 3]}},
        "39": {"class_type": "PreviewAny", "inputs": {"source": ["4", 6]}},
        "40": {"class_type": "PreviewAny", "inputs": {"source": ["3", 1]}},
    }
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="127.0.0.1:8188")
    ap.add_argument("--comfy-dir", default="ComfyUI")
    ap.add_argument("--image", required=True)
    ap.add_argument("--product", default="the product in the reference image")
    ap.add_argument("--brief", default="on a marble bathroom counter with eucalyptus|on wet black stones with water "
                                       "splashes|on a sunlit beach towel next to sunglasses")
    ap.add_argument("--formats", default="lifestyle,lifestyle,lifestyle,story,banner")
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cutout", default="auto", choices=["auto", "rembg", "grabcut"])
    ap.add_argument("--harmonise", type=float, default=0.55)
    ap.add_argument("--klein", default="flux-2-klein.safetensors")
    ap.add_argument("--klein-te", default="qwen_3_4b.safetensors")
    ap.add_argument("--flux-vae", default="flux2-vae.safetensors")
    ap.add_argument("--klein-steps", type=int, default=4)
    ap.add_argument("--klein-cfg", type=float, default=1.0)
    ap.add_argument("--dump", default="", help="Write the API prompt JSON here and exit.")
    a = ap.parse_args()
    run = time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:4]
    g = build(a, run)
    if a.dump:
        json.dump(g, open(a.dump, "w"), indent=1)
        return
    inp = os.path.join(a.comfy_dir, "input")
    os.makedirs(inp, exist_ok=True)
    dst = os.path.join(inp, os.path.basename(a.image))
    if os.path.abspath(a.image) != os.path.abspath(dst):
        shutil.copy(a.image, dst)
    print(f"TrueShot run {run}", flush=True)
    r = post(a.server, "/prompt", {"prompt": g, "client_id": run})
    h = wait(a.server, r["prompt_id"], os.path.join(a.comfy_dir, "output", "TrueShot", run))
    for out in h.get("outputs", {}).values():
        if "text" in out:
            print("\n".join(map(str, out["text"])), flush=True)
    print("OUTPUT DIR:", os.path.join(a.comfy_dir, "output", "TrueShot", run))


if __name__ == "__main__":
    main()
