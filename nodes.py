"""ComfyUI nodes for TrueShot."""
from __future__ import annotations

import json

import numpy as np
import torch

from .trueshot import core
from .trueshot.plan import make_plan, pick_best, stats, FORMATS

CAT = "TrueShot"
PLAN = "TRUESHOT_PLAN"


def _np(t):
    if isinstance(t, torch.Tensor):
        t = t.detach().float().cpu().numpy()
    t = np.asarray(t, dtype=np.float32)
    return t[0] if t.ndim == 4 else t


def _m(t):
    a = _np(t)
    while a.ndim > 2:
        a = a[0]
    return a


def _t(img):
    return torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None]


def _first(x):
    return x[0] if isinstance(x, (list, tuple)) else x


class TrueShotCutout:
    """Phone photo -> tight product crop + matte."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                             "method": (["auto", "rembg", "grabcut"], {"default": "auto"})}}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("product", "matte", "preview")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, method):
        img = _np(image)[..., :3]
        a = core.cutout(img, method)
        prod, pa = core.crop_product(img, a)
        prev = prod * pa[..., None] + (1 - pa[..., None]) * 0.12
        return (_t(prod), torch.from_numpy(pa)[None], _t(prev))


class TrueShotMarketplace:
    """White-background listing photo built from 100% real product pixels."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"product": ("IMAGE",), "matte": ("MASK",),
                             "size": ("INT", {"default": 1200, "min": 256, "max": 4096, "step": 8}),
                             "fill": ("FLOAT", {"default": 0.93, "min": 0.3, "max": 0.98, "step": 0.01})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("listing", "compliance")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, product, matte, size, fill):
        img = core.marketplace(_np(product)[..., :3], _m(matte), size, fill)
        c = core.marketplace_check(img)
        rep = (f"marketplace main image: {'PASS' if c['pass'] else 'FAIL'} | white bg {c['white_background']} | "
               f">=1000px {c['resolution']} | product fill {c['fill']:.0%} (>=85%: {c['product_fill']}) | "
               f"not cropped {c['not_cropped']} | product pixels: 100% real")
        print("[TrueShot] " + rep)
        return (_t(img), rep)


class TrueShotPlan:
    """Plan the shot pack. Outputs per-shot LISTS (staged reference, prompt,
    size, seed) so the generator downstream runs once per shot."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "product": ("IMAGE",), "matte": ("MASK",),
            "product_name": ("STRING", {"default": "a bottle of vitamin C serum"}),
            "brief": ("STRING", {"default": "on a marble bathroom counter with eucalyptus\n"
                                            "on wet black stones with water splashes\n"
                                            "on a sunlit beach towel next to sunglasses", "multiline": True}),
            "formats": ("STRING", {"default": "lifestyle,lifestyle,lifestyle,story,banner",
                                   "tooltip": f"Comma list from: {', '.join(FORMATS)}"}),
            "variants": ("INT", {"default": 1, "min": 1, "max": 4,
                                 "tooltip": "Generations per shot; Restore keeps the most faithful one."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
        }}

    RETURN_TYPES = (PLAN, "IMAGE", "STRING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("plan", "staged", "prompts", "widths", "heights", "seeds", "plan_text")
    OUTPUT_IS_LIST = (False, True, True, True, True, True, False)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, product, matte, product_name, brief, formats, variants, seed):
        p = make_plan(_np(product)[..., :3], _m(matte), product_name, brief, formats, variants, seed)
        sh = p["shots"]
        staged = [_t(s.pop("_staged")) for s in sh]
        txt = "\n".join(f"#{s['shot']}.{s['variant']} {s['format']:9s} {s['width']}x{s['height']}  {s['scene']}"
                        for s in sh)
        return (p, staged, [s["prompt"] for s in sh], [s["width"] for s in sh], [s["height"] for s in sh],
                [s["seed"] for s in sh], txt)


class TrueShotRestore:
    """Generated shots -> pixel-true shots: locate the product, paste the real
    pixels back, harmonise to the scene light, keep the best variant."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan": (PLAN,), "product": ("IMAGE",), "matte": ("MASK",),
                             "generated": ("IMAGE",),
                             "harmonise": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
                             "edge_softness": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 6.0, "step": 0.1})}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("shots", "comparisons", "generated_best", "report")
    OUTPUT_IS_LIST = (True, True, True, False)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, plan, product, matte, generated, harmonise, edge_softness):
        p = _first(plan)
        prod, pa = _np(_first(product))[..., :3], _m(_first(matte))
        h_amt, e_px = float(_first(harmonise)), float(_first(edge_softness))
        outs, reps, gens = [], [], []
        for s, g in zip(p["shots"], generated):
            g = _np(g)[..., :3]
            o, _, r = core.restore(prod, pa, g, prior_M=np.array(s["M"]), harmonise_amount=h_amt, edge_px=e_px)
            r.update(shot=s["shot"], variant=s["variant"], format=s["format"])
            outs.append(o), reps.append(r), gens.append(g)
        keep = pick_best(p, reps)
        shots = [_t(outs[i]) for i in keep]
        comps = [_t(core.compare_card(prod, gens[i], outs[i], reps[i]["box"],
                                      scores=(reps[i]["truth_ai"], reps[i]["truth"]) if "truth" in reps[i] else None))
                 if reps[i].get("ok") else _t(gens[i]) for i in keep]
        best_gen = [_t(gens[i]) for i in keep]
        lines = [stats([reps[i] for i in keep])]
        for i in keep:
            r = reps[i]
            t_ai, t_ts = r.get("truth_ai") or {}, r.get("truth") or {}
            lines.append(f"shot {r['shot']} ({r['format']}) v{r['variant']}: {r.get('verdict', 'REVIEW')} | "
                         f"truth AI-only {t_ai.get('total', '-')} -> TrueShot {t_ts.get('total', '-')} "
                         f"[label {t_ai.get('label', '-')}->{t_ts.get('label', '-')}, "
                         f"shape {t_ai.get('shape', '-')}->{t_ts.get('shape', '-')}, "
                         f"colour {t_ai.get('colour', '-')}->{t_ts.get('colour', '-')}] "
                         f"geometry {t_ts.get('geometry', 'unverified')} ({r.get('method')}, {r.get('inliers', 0)} inliers)"
                         f"{', ghost removed ' + str(round(100 * r['ghost'])) + '%' if r.get('ghost') else ''}")
        rep = "\n".join(lines)
        print("[TrueShot]\n" + rep)
        return (shots, comps, best_gen, rep)


class TrueShotContactSheet:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "columns": ("INT", {"default": 3, "min": 1, "max": 8})},
                "optional": {"labels": ("STRING", {"default": "", "multiline": True})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, images, columns, labels=None):
        ims = [_np(i)[..., :3] for i in images]
        names = None
        if labels and _first(labels):
            names = [x for x in _first(labels).splitlines() if x.strip()]
            names = (names + [""] * len(ims))[:len(ims)]
        return (_t(core.contact_sheet(ims, names, _first(columns))),)


NODE_CLASS_MAPPINGS = {
    "TrueShotCutout": TrueShotCutout,
    "TrueShotMarketplace": TrueShotMarketplace,
    "TrueShotPlan": TrueShotPlan,
    "TrueShotRestore": TrueShotRestore,
    "TrueShotContactSheet": TrueShotContactSheet,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TrueShotCutout": "TrueShot · Cutout Product",
    "TrueShotMarketplace": "TrueShot · Marketplace (white bg)",
    "TrueShotPlan": "TrueShot · Plan Shot Pack",
    "TrueShotRestore": "TrueShot · Pixel-True Restore",
    "TrueShotContactSheet": "TrueShot · Contact Sheet",
}
