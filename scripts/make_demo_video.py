#!/usr/bin/env python3
"""Render the TrueShot demo video from a real run folder.

  python scripts/make_demo_video.py RUN_DIR out.mp4 [--photo phone.png] [--song song.wav]
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

W, H, FPS = 1920, 1080, 30
BG = (11, 12, 16)
FG = (240, 240, 236)
MUTED = (150, 152, 160)
GREEN = (86, 230, 150)
RED = (255, 98, 98)
ACCENT = (255, 168, 64)
FONT_DIRS = ["/usr/share/fonts/opentype/inter", "/usr/share/fonts/truetype/google-fonts"]


def font(weight="Bold", size=48):
    for d, name in ((FONT_DIRS[0], f"Inter-{weight}.otf"), (FONT_DIRS[1], f"Poppins-{weight}.ttf")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def load(p):
    return Image.open(p).convert("RGB")


def fit(im, w, h):
    s = min(w / im.width, h / im.height)
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)


def paste_card(canvas, im, cx, cy, alpha=1.0, radius=18, shadow=True):
    x, y = int(cx - im.width / 2), int(cy - im.height / 2)
    m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, im.width - 1, im.height - 1], radius, fill=int(255 * alpha))
    if shadow and alpha > 0.05:
        sh = Image.new("L", (im.width + 80, im.height + 80), 0)
        ImageDraw.Draw(sh).rounded_rectangle([40, 52, im.width + 40, im.height + 52], radius, fill=int(140 * alpha))
        sh = sh.filter(ImageFilter.GaussianBlur(22))
        canvas.paste((0, 0, 0), (x - 40, y - 40), sh)
    canvas.paste(im, (x, y), m)


def text(d, xy, s, f, fill=FG, anchor="la", alpha=1.0):
    if alpha <= 0:
        return
    c = tuple(int(BG[i] + (fill[i] - BG[i]) * alpha) for i in range(3))
    d.text(xy, s, font=f, fill=c, anchor=anchor)


class Demo:
    def __init__(self, run, photo):
        g = lambda pat: sorted(glob.glob(os.path.join(run, pat)))  # noqa: E731
        self.shots = [load(p) for p in g("shot_*.png")]
        self.ai = [load(p) for p in g("ai_only_*.png")]
        self.locked = [load(p) for p in g("locked_raw_*.png")]
        self.compare = [load(p) for p in g("compare_*.png")]
        self.market = load(g("marketplace_*.png")[0])
        self.cutout = load(g("cutout_*.png")[0])
        self.photo = load(photo) if photo else self.cutout
        self.staged, self.mask = self._stage(photo)
        self.report = open(os.path.join(run, "report.txt")).read().splitlines() if os.path.exists(
            os.path.join(run, "report.txt")) else []
        self.f_title = font("Bold", 120)
        self.f_h1 = font("Bold", 64)
        self.f_h2 = font("SemiBold", 40)
        self.f_body = font("Medium", 32)
        self.f_small = font("Medium", 26)
        self.f_mono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24)

    def _stage(self, photo):
        try:
            from trueshot import core
            from trueshot.plan import make_plan
            img = np.asarray(self.photo).astype(np.float32) / 255
            a = core.cutout(img, "grabcut")
            prod, pa = core.crop_product(img, a)
            p = make_plan(prod, pa, "x", "a", "lifestyle")
            s = p["shots"][0]
            st = Image.fromarray(core.to_u8(s["_staged"]))
            mk = Image.fromarray((s["_regen"] * 255).astype(np.uint8)).convert("RGB")
            return st, mk
        except Exception as e:  # pragma: no cover
            print("stage preview failed:", e)
            return self.cutout, self.cutout

    # ---------------------------------------------------------------- scenes
    def s_title(self, c, d, t, T):
        a = ease(t / 0.8) * (1 - ease((t - T + 0.5) / 0.5))
        text(d, (W / 2, H / 2 - 70), "TrueShot", self.f_title, FG, "mm", a)
        text(d, (W / 2, H / 2 + 50), "Your product. Better presented. Never misrepresented.", self.f_h2, ACCENT, "mm",
             ease((t - 0.6) / 0.8) * a)
        text(d, (W / 2, H / 2 + 120), "open source  ·  open weights  ·  ComfyUI node pack  ·  #OpenSourceHiggsfield",
             self.f_small, MUTED, "mm", ease((t - 1.2) / 0.8) * a)

    def s_problem(self, c, d, t, T):
        a = ease(t / 0.5) * (1 - ease((t - T + 0.4) / 0.4))
        text(d, (120, 120), "The problem", self.f_small, ACCENT, "la", a)
        text(d, (120, 165), "AI redraws your product.", self.f_h1, FG, "la", a)
        text(d, (120, 250), "Labels melt. Shapes drift. Colours change.", self.f_h2, MUTED, "la", ease((t - 0.5) / 0.6) * a)
        text(d, (120, 310), "For a seller, that image misrepresents what ships.", self.f_h2, MUTED, "la",
             ease((t - 1.0) / 0.6) * a)
        ai = fit(self.ai[1 if len(self.ai) > 1 else 0], 620, 620)
        paste_card(c, ai, 1450, 560, a)
        # zoom on the AI label: the left half of the compare card
        cmp = self.compare[3 if len(self.compare) > 3 else 0]
        left = cmp.crop((0, 56, cmp.width // 2 - 6, cmp.height - 40))
        z = fit(left, 420, 520)
        za = ease((t - 1.6) / 0.6) * a
        if za > 0:
            paste_card(c, z, 700, 690, za)
            text(d, (700, 690 + z.height / 2 + 34), "AI-only: \"VITAMIN C\" → gibberish", self.f_body, RED, "mm", za)

    def s_input(self, c, d, t, T):
        a = ease(t / 0.5) * (1 - ease((t - T + 0.4) / 0.4))
        text(d, (W / 2, 120), "One phone photo in", self.f_h1, FG, "mm", a)
        ph = fit(self.photo, 560, 700)
        paste_card(c, ph, 620, 590, a)
        k = ease((t - 0.8) / 0.8) * a
        cut = self.cutout.copy()
        cut = fit(cut, 420, 700)
        if k > 0:
            paste_card(c, cut, 1300, 590, k)
            text(d, (960, 590), "→", self.f_title, ACCENT, "mm", k)
            text(d, (1300, 590 + cut.height / 2 + 40), "cutout (IS-Net / GrabCut)", self.f_body, MUTED, "mm", k)
        text(d, (620, 590 + ph.height / 2 + 40), "bedroom light, messy table", self.f_body, MUTED, "mm", a)

    def s_how(self, c, d, t, T):
        a = ease(t / 0.4) * (1 - ease((t - T + 0.4) / 0.4))
        text(d, (120, 90), "How it works", self.f_small, ACCENT, "la", a)
        text(d, (120, 130), "Lock the real product. Generate everything else.", self.f_h1, FG, "la", a)
        steps = [
            (self.staged, "1  Stage", "real product placed exactly\nwhere the shot needs it"),
            (self.mask, "2  Regen mask", "white = generate,\nblack = product, locked"),
            (self.locked[0] if self.locked else self.shots[0], "3  FLUX.2 Klein", "masked sampling paints\nthe scene around it"),
            (self.shots[0], "4  Pixel-true restore", "real pixels pasted back,\nshadow + ghost removal"),
        ]
        for i, (im, h, body) in enumerate(steps):
            k = ease((t - 0.4 - i * 0.9) / 0.6) * a
            if k <= 0:
                continue
            x = 300 + i * 440
            card = fit(im, 380, 380)
            paste_card(c, card, x, 520 + (1 - k) * 40, k)
            text(d, (x, 750), h, self.f_h2, GREEN if i == 3 else FG, "mm", k)
            for j, line in enumerate(body.split("\n")):
                text(d, (x, 800 + j * 36), line, self.f_small, MUTED, "mm", k)
            if i < 3:
                text(d, (x + 220, 520), "→", self.f_h1, ACCENT, "mm", k)
        text(d, (W / 2, 990), "geometry is exact by construction: nothing to guess, nothing to re-detect", self.f_body,
             ACCENT, "mm", ease((t - 4.4) / 0.6) * a)

    def s_compare(self, c, d, t, T):
        a = ease(t / 0.4) * (1 - ease((t - T + 0.4) / 0.4))
        text(d, (W / 2, 95), "Product Truth Score", self.f_h1, FG, "mm", a)
        text(d, (W / 2, 160), "label (SSIM) · shape (silhouette IoU) · colour (chroma ΔE) · geometry", self.f_body, MUTED,
             "mm", a)
        picks = self.compare[:3] if len(self.compare) >= 3 else self.compare
        for i, cm in enumerate(picks):
            k = ease((t - 0.3 - i * 0.5) / 0.6) * a
            im = fit(cm, 560, 760)
            paste_card(c, im, 340 + i * 620, 610, k)

    def s_pack(self, c, d, t, T):
        a = ease(t / 0.4) * (1 - ease((t - T + 0.4) / 0.4))
        text(d, (120, 90), "One run → a full shot pack", self.f_h1, FG, "la", a)
        text(d, (120, 170), "lifestyle 1:1  ·  story 9:16  ·  banner 16:9  ·  marketplace main image", self.f_body,
             MUTED, "la", a)
        order = self.shots
        pos = [(330, 470, 400), (760, 470, 400), (1190, 470, 400), (330, 860, 300), (980, 860, 330)]
        for i, (im, (x, y, s)) in enumerate(zip(order, pos)):
            k = ease((t - 0.2 - i * 0.25) / 0.5) * a
            if i == 4:
                card = fit(im, 640, 340)
                x = 860
            else:
                card = fit(im, s, s if i != 3 else 360)
            paste_card(c, card, x, y + (1 - k) * 30, k)
        k = ease((t - 1.6) / 0.6) * a
        if k > 0:
            m = fit(self.market, 360, 360)
            paste_card(c, m, 1640, 470, k)
            checks = ["pure white background", "≥ 1000 px", "product fills ≥ 85%", "not cropped",
                      "100% real product pixels"]
            text(d, (1640, 690), "Marketplace check: PASS", self.f_h2, GREEN, "mm", k)
            for j, ch in enumerate(checks):
                text(d, (1460, 740 + j * 40), "✓  " + ch, self.f_small, FG, "la", ease((t - 2.0 - j * 0.15) / 0.4) * a)

    def s_report(self, c, d, t, T):
        a = ease(t / 0.4) * (1 - ease((t - T + 0.4) / 0.4))
        text(d, (120, 110), "Every shot is scored before export", self.f_h1, FG, "la", a)
        lines = self.report or ["(report)"]
        n = int(min(len(lines), max(0, (t - 0.4) * 3.0)))
        for i, line in enumerate(lines[:n]):
            col = GREEN if "PASS" in line else (RED if "REVIEW" in line else FG)
            text(d, (120, 230 + i * 52), line[:118], self.f_mono, col, "la", a)
        text(d, (120, 930), "PASS only when geometry is verified and truth ≥ 85, otherwise REVIEW, never a silent fake.",
             self.f_body, ACCENT, "la", ease((t - 2.5) / 0.6) * a)

    def s_outro(self, c, d, t, T):
        a = ease(t / 0.6)
        text(d, (W / 2, H / 2 - 120), "TrueShot", self.f_title, FG, "mm", a)
        text(d, (W / 2, H / 2 + 0), "github.com/nedsoft057/ComfyUI-TrueShot", self.f_h2, ACCENT, "mm", a)
        text(d, (W / 2, H / 2 + 70), "FLUX.2 Klein 4B · open weights only · runs free on a Kaggle T4 in one cell",
             self.f_body, MUTED, "mm", ease((t - 0.5) / 0.6))
        text(d, (W / 2, H / 2 + 130), "Apache-2.0  ·  #OpenSourceHiggsfield", self.f_body, MUTED, "mm",
             ease((t - 0.9) / 0.6))

    def render(self, out, song=None):
        scenes = [(self.s_title, 4.0), (self.s_problem, 7.0), (self.s_input, 5.0), (self.s_how, 9.0),
                  (self.s_compare, 7.0), (self.s_pack, 7.5), (self.s_report, 7.0), (self.s_outro, 5.0)]
        total = sum(T for _, T in scenes)
        tmp = out + ".video.mp4"
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                              "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                              "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
        for fn, T in scenes:
            for k in range(int(T * FPS)):
                c = Image.new("RGB", (W, H), BG)
                d = ImageDraw.Draw(c)
                fn(c, d, k / FPS, T)
                p.stdin.write(c.tobytes())
        p.stdin.close()
        p.wait()
        if song:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-stream_loop", "-1", "-i", song,
                            "-filter_complex", f"[1:a]volume=0.35,afade=t=out:st={total - 2}:d=2[a]",
                            "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                            "-t", str(total), "-movflags", "+faststart", out], check=True)
            os.remove(tmp)
        else:
            os.replace(tmp, out)
        print("wrote", out, f"{total:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--photo", default="")
    ap.add_argument("--song", default="")
    a = ap.parse_args()
    Demo(a.run, a.photo).render(a.out, a.song or None)
