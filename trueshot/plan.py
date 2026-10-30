"""Shot-pack planner: one product photo -> a set of commerce-ready formats."""
from __future__ import annotations

import cv2
import numpy as np

from . import core

# name, width, height, product height fraction, centre-x, base-y, framing note
FORMATS = {
    "lifestyle": (768, 768, 0.56, 0.50, 0.86, "centered hero composition"),
    "story": (576, 1024, 0.42, 0.50, 0.78, "vertical 9:16 social story composition, open space above the product"),
    "banner": (1024, 576, 0.70, 0.70, 0.90, "wide 16:9 web banner composition, product on the right, empty space on the left for text"),
    "closeup": (768, 768, 0.80, 0.50, 0.95, "tight close-up, shallow depth of field, product fills the frame"),
}

LIGHT = [
    "soft morning window light, gentle shadows",
    "golden hour sunlight, warm rim light",
    "clean studio softbox lighting, subtle reflections",
    "moody dramatic side light, deep shadows",
]


def _scenes(brief):
    s = [x.strip() for x in brief.replace("|", "\n").splitlines()]
    return [x for x in s if x] or ["on a marble bathroom counter with plants"]


def make_plan(prod, alpha, product, brief, formats="lifestyle,lifestyle,lifestyle,story,banner",
              variants=1, seed=0):
    scenes = _scenes(brief)
    fmts = [f.strip() for f in formats.split(",") if f.strip()]
    shots = []
    for i, f in enumerate(fmts):
        if f not in FORMATS:
            raise ValueError(f"TrueShot: unknown format '{f}' (use {', '.join(FORMATS)})")
        W, H, hf, cx, by, note = FORMATS[f]
        scene = scenes[i % len(scenes)]
        light = LIGHT[i % len(LIGHT)]
        staged, a, M = core.stage(prod, alpha, W, H, hf, cx, by)
        # regenerate everything except the product (slightly grown so the
        # latent keeps a clean product; real pixels are pasted back anyway)
        keep = cv2.dilate((a > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
        regen = 1.0 - cv2.GaussianBlur(keep, (0, 0), 2.0)
        prompt = (f"Professional commercial product photograph of {product}, standing {scene}. {light}. {note}. "
                  f"The product is exactly the one in the reference image: identical shape, proportions, colors, "
                  f"label, logo and printed text, in the same position and size, standing naturally on the surface with a "
                  f"realistic contact shadow and reflections. Photorealistic advertising "
                  f"photography, sharp focus, high detail, 85mm lens.")
        for v in range(max(1, variants)):
            shots.append({"shot": i, "variant": v, "format": f, "scene": scene, "light": light,
                          "width": W, "height": H, "prompt": prompt, "seed": seed * 1000 + i * 97 + v * 13 + 5,
                          "M": M.tolist(), "_staged": staged, "_regen": regen.astype(np.float32)})
    return {"product": product, "brief": brief, "variants": max(1, variants), "shots": shots}


def pick_best(plan, reports):
    """Best variant per shot: aligned OK, then highest pre-restore fidelity
    (= the generator kept the product most faithfully, so the seams are
    cleanest), tie-break on feature inliers."""
    best = {}
    for idx, (s, r) in enumerate(zip(plan["shots"], reports)):
        key = (r.get("ok", False), r.get("method") == "features", r.get("fidelity_before", 0), r.get("inliers", 0))
        if s["shot"] not in best or key > best[s["shot"]][0]:
            best[s["shot"]] = (key, idx)
    return [best[k][1] for k in sorted(best)]


def stats(reports):
    ok = [r for r in reports if r.get("ok")]
    if not ok:
        return "no shots restored"
    b = np.mean([r["fidelity_before"] for r in ok])
    a = np.mean([r["fidelity_after"] for r in ok])
    return f"{len(ok)}/{len(reports)} shots restored | label fidelity (SSIM vs real product) {b:.2f} -> {a:.2f}"
