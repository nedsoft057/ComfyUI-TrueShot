#!/usr/bin/env python3
"""TrueShot promo cut: beat-synced, motion-designed, built from real outputs."""
import glob, os, subprocess, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1920, 1080, 30
RUN, PHOTO, MUSIC, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
STOCK = sys.argv[5:]
MUSIC_START = 40.0           # drop of the track lands at t=7.6s
BEAT = 60 / 126.3
BAR = 4 * BEAT
INTER = "/usr/share/fonts/opentype/inter"
F = lambda w, s: ImageFont.truetype(f"{INTER}/Inter-{w}.otf", s)  # noqa: E731
SERIF = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf", s) if os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf") else F("Italic", s)  # noqa
MONO = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", s)  # noqa
GREEN, RED, INK, PAPER = (52, 211, 120), (245, 86, 86), (18, 18, 20), (244, 244, 242)

for cand in ["/usr/share/fonts/truetype/google-fonts/PlayfairDisplay-Italic.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"]:
    if os.path.exists(cand):
        SERIF = lambda s, c=cand: ImageFont.truetype(c, s)  # noqa
        break

def ease(x): x = min(1, max(0, x)); return 1 - (1 - x) ** 3
def eio(x): x = min(1, max(0, x)); return x * x * (3 - 2 * x)
def L(p): return Image.open(p).convert("RGB")
def g(pat): return sorted(glob.glob(os.path.join(RUN, pat)))

shots = [L(p) for p in g("shot_*.png")]
ai = [L(p) for p in g("ai_only_*.png")]
locked = [L(p) for p in g("locked_raw_*.png")]
cmp = [L(p) for p in g("compare_*.png")]
market = L(g("marketplace_*.png")[0])
cut = Image.open(g("cutout_*.png")[0]).convert("RGB")
photo = L(PHOTO)
extra = []
for d in sys.argv[1:2]:
    pass

def fit(im, w, h, cover=False):
    s = (max if cover else min)(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    if cover:
        x, y = (im.width - w) // 2, (im.height - h) // 2
        im = im.crop((x, y, x + w, y + h))
    return im

def rr_mask(size, r):
    m = Image.new("L", size, 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], r, fill=255); return m

def card(c, im, cx, cy, a=1.0, r=22, sh=True, scale=1.0):
    if a <= 0.01: return
    if scale != 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    x, y = int(cx - im.width / 2), int(cy - im.height / 2)
    if sh:
        s = Image.new("L", (im.width + 120, im.height + 120), 0)
        ImageDraw.Draw(s).rounded_rectangle([60, 80, im.width + 60, im.height + 80], r, fill=int(120 * a))
        c.paste((0, 0, 0), (x - 60, y - 60), s.filter(ImageFilter.GaussianBlur(30)))
    m = rr_mask(im.size, r)
    if a < 1: m = m.point(lambda v: int(v * a))
    c.paste(im, (x, y), m)

def vignette(dark=True):
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    if dark:
        v = np.clip(34 - d * 60, 6, 40)
        return Image.fromarray(np.dstack([v, v, v * 1.02]).astype(np.uint8))
    v = np.clip(250 - d * 70, 200, 250)
    return Image.fromarray(np.dstack([v, v, v - 2]).astype(np.uint8))
BG_D, BG_L = vignette(True), vignette(False)
# dark grid bg
GRID = BG_D.copy(); gd = ImageDraw.Draw(GRID)
for x in range(0, W, 120): gd.line([x, 0, x, H], fill=(28, 28, 32))
for y in range(0, H, 120): gd.line([0, y, W, y], fill=(28, 28, 32))

def T(d, xy, s, f, fill, anchor="la", a=1.0, bg=(10, 10, 12)):
    if a <= 0: return
    col = tuple(int(bg[i] + (fill[i] - bg[i]) * a) for i in range(3))
    d.text(xy, s, font=f, fill=col, anchor=anchor)

# ------------------------------------------------------------- stock frames
def stock_frames(path, n, start=1.0):
    p = subprocess.run(["ffmpeg", "-v", "quiet", "-ss", str(start), "-i", path, "-frames:v", str(n), "-vf",
                        f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS}",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8)
    k = len(a) // (W * H * 3)
    fr = [Image.fromarray(a[i * W * H * 3:(i + 1) * W * H * 3].reshape(H, W, 3)) for i in range(k)]
    while len(fr) < n: fr.append(fr[-1] if fr else Image.new("RGB", (W, H)))
    return fr

hook = []
segs = [(STOCK[0], BAR * 0.75, 2.0), (STOCK[1], BAR * 0.5, 1.0), (STOCK[2], BAR * 0.5, 2.0), (STOCK[3], BAR * 0.25, 1.0)]
for pth, dur, st in segs:
    hook += stock_frames(pth, int(round(dur * FPS)), st)

# ------------------------------------------------------------------ scenes
def s_hook(t, i):
    fr = hook[min(i, len(hook) - 1)].copy()
    fr = Image.blend(fr, Image.new("RGB", (W, H), (0, 0, 0)), 0.35)
    d = ImageDraw.Draw(fr)
    words = ["Every", "seller", "needs", "product", "photos."]
    n = min(len(words), int(t / (BEAT * 0.75)) + 1)
    d.text((W / 2, H / 2), " ".join(words[:n]), font=F("Bold", 96), fill=(255, 255, 255), anchor="mm")
    return fr

def s_type(t):
    c = BG_D.copy(); d = ImageDraw.Draw(c)
    a = "AI product photos"
    b = " that don't lie."
    full = a + b
    n = int(min(len(full), t * 16))
    s = full[:n]
    fa, fb = F("Medium", 92), SERIF(98)
    sa = s[:len(a)]; sb = s[len(a):]
    wa = d.textlength(a, font=fa); wb = d.textlength(b, font=fb)
    x0 = W / 2 - (wa + wb) / 2
    d.text((x0, H / 2), sa, font=fa, fill=(235, 235, 235), anchor="lm")
    xb = x0 + d.textlength(sa, font=fa)
    d.text((xb, H / 2 + 4), sb, font=fb, fill=GREEN, anchor="lm")
    xc = xb + d.textlength(sb, font=fb) + 6
    if int(t * 2.4) % 2 == 0 or n < len(full):
        d.rectangle([xc, H / 2 - 50, xc + 5, H / 2 + 50], fill=GREEN)
    return c

def prompt_card(text_shown, sent=0.0):
    cw, ch = 1100, 420
    cc = Image.new("RGB", (cw, ch), (255, 255, 255)); d = ImageDraw.Draw(cc)
    d.rounded_rectangle([30, 30, 470, 120], 20, fill=(242, 242, 240))
    th = fit(photo, 70, 70, cover=True); cc.paste(th, (42, 40), rr_mask(th.size, 12))
    d.text((126, 58), "otentika_phone_photo.jpg", font=F("Medium", 26), fill=INK)
    d.text((126, 90), "JPG · 1 product", font=F("Regular", 20), fill=(130, 130, 130))
    d.text((40, 190), text_shown, font=F("Regular", 44), fill=INK)
    tw = d.textlength(text_shown, font=F("Regular", 44))
    d.rectangle([44 + tw, 165, 48 + tw, 218], fill=GREEN)
    d.rounded_rectangle([30, 320, 420, 390], 35, fill=(242, 242, 240))
    d.ellipse([56, 344, 78, 366], fill=GREEN)
    d.text((94, 355), "ComfyUI · FLUX.2 Klein", font=F("Medium", 28), fill=INK, anchor="lm")
    r = 44 + 8 * math.sin(sent * math.pi) if sent > 0 else 44
    d.ellipse([cw - 80 - r, 355 - r, cw - 80 + r, 355 + r], fill=INK)
    d.text((cw - 80, 352), "↑", font=F("Bold", 44), fill=(255, 255, 255), anchor="mm")
    return cc

def s_prompt(t):
    c = BG_D.copy()
    msg = "put it on a vanity with soft towels and flowers"
    n = int(min(len(msg), max(0, t - 0.4) * 22))
    k = ease(t / 0.5)
    pc = prompt_card(msg[:n], sent=max(0, (t - 2.9) / 0.4) if t > 2.9 else 0)
    card(c, pc, W / 2, H / 2 + (1 - k) * 80, k, r=36)
    return c

def wipe(a_img, b_img, x):
    a = np.asarray(a_img); b = np.asarray(b_img)
    xi = int(x * a.shape[1]); out = a.copy(); out[:, xi:] = b[:, xi:]
    im = Image.fromarray(out); d = ImageDraw.Draw(im)
    d.line([xi, 0, xi, im.height], fill=(255, 255, 255), width=4)
    return im

def s_reveal(t):
    c = BG_D.copy(); d = ImageDraw.Draw(c)
    i = 1
    A = fit(ai[i], 860, 860); B = fit(shots[i], 860, 860)
    x = 0.5 + 0.42 * math.sin(min(1, t / (BAR * 2)) * math.pi * 2 - math.pi / 2) * -1
    x = 1 - eio(t / (BAR * 1.2)) if t < BAR * 1.2 else 0.5 + 0.5 * math.cos((t - BAR * 1.2) * 2.2)
    im = wipe(B, A, min(1, max(0, x)))
    k = ease(t / 0.4)
    card(c, im, W / 2, H / 2 + 30, k, r=28)
    T(d, (W / 2 - 430, 110), "TrueShot", F("Bold", 40), GREEN, "la", k)
    T(d, (W / 2 + 430, 110), "AI only", F("Bold", 40), RED, "ra", k)
    return c

def s_zoom(t):
    c = BG_D.copy(); d = ImageDraw.Draw(c)
    cm = cmp[0]
    w2 = cm.width // 2
    left = cm.crop((0, 56, w2 - 10, cm.height - 40)); right = cm.crop((w2 + 16, 56, cm.width, cm.height - 40))
    k1, k2 = ease(t / 0.4), ease((t - BEAT * 2) / 0.4)
    card(c, fit(left, 640, 820), 560, 560, k1, r=24)
    card(c, fit(right, 640, 820), 1360, 560, k2, r=24)
    T(d, (560, 105), "AI only", F("Bold", 44), RED, "mm", k1)
    T(d, (1360, 105), "TrueShot", F("Bold", 44), GREEN, "mm", k2)
    # counters
    kk = ease((t - BEAT * 4) / 1.2)
    rows = [("LABEL", 57, 99), ("COLOUR", 71, 93), ("TRUTH SCORE", 43, 96)]
    for j, (nm, a0, a1) in enumerate(rows):
        v = int(a0 + (a1 - a0) * kk)
        y = 330 + j * 120
        T(d, (960, y), nm, MONO(26), (170, 170, 175), "mm", k2)
        T(d, (960, y + 44), f"{a0} → {v}", F("Bold", 40), (255, 255, 255), "mm", k2)
    T(d, (960, 720), "GEOMETRY", MONO(26), (170, 170, 175), "mm", kk)
    T(d, (960, 764), "locked", SERIF(44), GREEN, "mm", kk)
    return c

def s_pipeline(t):
    c = BG_L.copy(); d = ImageDraw.Draw(c)
    T(d, (120, 110), "Lock the real product.", F("Bold", 64), INK, "la", ease(t / 0.4), bg=(245, 245, 243))
    T(d, (120, 190), "Generate everything else.", SERIF(64), (30, 140, 80), "la", ease((t - 0.3) / 0.4), bg=(245, 245, 243))
    steps = [(photo, "phone photo"), (cut, "cutout"), (locked[1], "masked Klein"), (shots[1], "pixel-true restore")]
    for j, (im, lab) in enumerate(steps):
        k = ease((t - 0.3 - j * BEAT) / 0.35)
        x = 300 + j * 440
        card(c, fit(im, 370, 470, cover=True), x, 600 + (1 - k) * 60, k, r=20)
        T(d, (x, 880), f"0{j + 1}  {lab}", MONO(28), INK, "mm", k, bg=(245, 245, 243))
        if j < 3:
            T(d, (x + 220, 600), "→", F("Bold", 56), (30, 160, 90), "mm", k, bg=(245, 245, 243))
    return c

def s_pack(t):
    c = GRID.copy(); d = ImageDraw.Draw(c)
    T(d, (120, 80), "SHOT PACK", F("Bold", 72), (255, 255, 255), "la", ease(t / 0.3))
    T(d, (560, 92), "· one photo, every format", SERIF(60), (200, 200, 200), "la", ease((t - 0.2) / 0.4))
    order = [shots[0], shots[1], shots[2], shots[3], shots[4], market]
    pos = [(300, 520, 380, -6), (680, 560, 380, -2), (1060, 520, 380, 2), (1440, 560, 320, 5), (760, 880, 520, 0), (1640, 860, 300, 3)]
    for j, (im, (x, y, s, rot)) in enumerate(zip(order, pos)):
        k = ease((t - 0.2 - j * BEAT * 0.5) / 0.35)
        if k <= 0: continue
        fi = fit(im, s, s if j != 4 else 300)
        fi = fi.convert("RGBA").rotate(rot, expand=True, resample=Image.BICUBIC)
        sh = Image.new("RGBA", fi.size, (0, 0, 0, 0))
        c.paste(fi, (int(x - fi.width / 2), int(y - fi.height / 2 + (1 - k) * 80)), fi)
    return c

def s_market(t):
    c = BG_L.copy(); d = ImageDraw.Draw(c)
    card(c, fit(market, 620, 620), 520, 560, ease(t / 0.4), r=24)
    bg = (245, 245, 243)
    T(d, (960, 250), "Marketplace check", F("Bold", 60), INK, "la", ease(t / 0.4), bg=bg)
    T(d, (960, 330), "PASS", SERIF(80), (30, 160, 90), "la", ease((t - 0.3) / 0.4), bg=bg)
    items = ["pure white background", "≥ 1000 px", "product fills ≥ 85%", "not cropped", "100% real product pixels"]
    for j, s in enumerate(items):
        k = ease((t - 0.6 - j * BEAT * 0.5) / 0.3)
        T(d, (960, 470 + j * 74), "✓  " + s, F("Medium", 40), INK, "la", k, bg=bg)
    return c

ALL = shots + ai + cmp + [market, cut] + locked
def s_mosaic(t):
    c = GRID.copy()
    z = 1.35 - 0.35 * eio(t / (BAR * 2))
    tile = 300
    cols, rows = 9, 5
    big = Image.new("RGB", (cols * (tile + 16), rows * (tile + 16)), (10, 10, 12))
    for r in range(rows):
        for q in range(cols):
            im = ALL[(r * cols + q * 3) % len(ALL)]
            big.paste(fit(im, tile, tile, cover=True), (q * (tile + 16) + 8, r * (tile + 16) + 8))
    bw, bh = int(big.width * z * 0.68), int(big.height * z * 0.68)
    bb = big.resize((bw, bh), Image.LANCZOS)
    c.paste(bb, (int(W / 2 - bw / 2), int(H / 2 - bh / 2)))
    return c

def s_end(t):
    c = BG_D.copy(); d = ImageDraw.Draw(c)
    if t < BAR * 1.0:
        k = ease(t / 0.4)
        d.rounded_rectangle([460, 300, 1460, 520], 28, fill=(22, 22, 26), outline=(50, 50, 56), width=2)
        T(d, (520, 360), "nedsoft057 / ComfyUI-TrueShot", F("Medium", 40), (120, 170, 255), "la", k)
        T(d, (520, 430), "Apache-2.0 · Public", MONO(26), (160, 160, 165), "la", k)
        chips = ["ComfyUI", "FLUX.2 Klein", "open weights", "Kaggle T4"]
        x = 470
        for j, ch in enumerate(chips):
            kj = ease((t - 0.3 - j * 0.12) / 0.3)
            wch = d.textlength(ch, font=F("Medium", 30)) + 70
            col = tuple(int(10 + (v - 10) * kj) for v in (40, 40, 46))
            d.rounded_rectangle([x, 600, x + wch, 660], 30, fill=col)
            d.ellipse([x + 20, 620, x + 40, 640], fill=GREEN if kj > 0.5 else (30, 30, 30))
            T(d, (x + 52, 630), ch, F("Medium", 30), (235, 235, 235), "lm", kj)
            x += wch + 20
        T(d, (W / 2, 800), "Open source.  Open weights.  Never", F("Medium", 48), (235, 235, 235), "mm", ease((t - 0.8) / 0.4))
        T(d, (W / 2 + 520, 802), "misrepresented.", SERIF(52), GREEN, "lm", ease((t - 1.1) / 0.4))
    else:
        tt = t - BAR
        k = ease(tt / 0.5)
        d.ellipse([W / 2 - 330, H / 2 - 120, W / 2 - 230, H / 2 - 20], outline=GREEN, width=8)
        d.line([W / 2 - 300, H / 2 - 70, W / 2 - 285, H / 2 - 52, W / 2 - 255, H / 2 - 92], fill=GREEN, width=8)
        T(d, (W / 2 - 200, H / 2 - 70), "TrueShot", F("Bold", 110), (245, 245, 245), "lm", k)
        T(d, (W / 2, H / 2 + 70), "Your product. Better presented.", F("Medium", 44), (200, 200, 200), "mm", ease((tt - 0.3) / 0.4))
        T(d, (W / 2, H / 2 + 135), "Never misrepresented.", SERIF(50), GREEN, "mm", ease((tt - 0.6) / 0.4))
        d.rounded_rectangle([W / 2 - 230, H / 2 + 205, W / 2 + 230, H / 2 + 265], 30, fill=(30, 150, 85))
        T(d, (W / 2, H / 2 + 235), "#OpenSourceHiggsfield", F("Bold", 30), (255, 255, 255), "mm", ease((tt - 0.9) / 0.4))
    return c

SCENES = [("hook", 2 * BAR), ("type", 2 * BAR), ("prompt", 2 * BAR), ("reveal", 2 * BAR), ("zoom", 2 * BAR),
          ("pipeline", 2 * BAR), ("pack", 2 * BAR), ("market", 2 * BAR), ("mosaic", 2 * BAR), ("end", 2.6 * BAR)]
FN = {"type": s_type, "prompt": s_prompt, "reveal": s_reveal, "zoom": s_zoom, "pipeline": s_pipeline,
      "pack": s_pack, "market": s_market, "mosaic": s_mosaic, "end": s_end}

def render():
    total = sum(d for _, d in SCENES)
    tmp = OUT + ".v.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
                          "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    f0 = 0
    for name, dur in SCENES:
        a = int(round(f0)); b = int(round(f0 + dur * FPS)); f0 += dur * FPS
        for i in range(b - a):
            t = i / FPS
            fr = s_hook(t, i) if name == "hook" else FN[name](t)
            # quick flash on scene cut
            if i < 3:
                fr = Image.blend(fr, Image.new("RGB", (W, H), (255, 255, 255)), 0.25 * (3 - i) / 3)
            p.stdin.write(fr.tobytes())
    p.stdin.close(); p.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-ss", str(MUSIC_START), "-i", MUSIC,
                    "-filter_complex", f"[1:a]afade=t=in:st=0:d=0.3,afade=t=out:st={total - 1.5}:d=1.5[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", str(total),
                    "-movflags", "+faststart", OUT], check=True)
    os.remove(tmp)
    print("wrote", OUT, round(total, 1), "s")

if __name__ == "__main__":
    if os.environ.get("PREVIEW"):
        for name, dur in SCENES:
            t = dur * 0.7
            fr = s_hook(t, int(t * FPS)) if name == "hook" else FN[name](t)
            fr.resize((640, 360)).save(f"/tmp/pv_{name}.jpg")
    else:
        render()
