#!/usr/bin/env python3
"""TrueShot launch video v2 - structure modelled on the Open Design launch reel."""
import glob, os, subprocess, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1920, 1080, 30
RUN, PHOTO, MUSIC, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
CLIPS = sys.argv[5:]
MUSIC_START = 40.0
BEAT = 60 / 126.3
BAR = 4 * BEAT
IN = "/usr/share/fonts/opentype/inter"
F = lambda w, s: ImageFont.truetype(f"{IN}/Inter-{w}.otf", s)  # noqa
MONO = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", s)  # noqa
SERIF = None
for c in ["/usr/share/fonts/truetype/google-fonts/PlayfairDisplay-Italic.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"]:
    if os.path.exists(c):
        SERIF = (lambda s, c=c: ImageFont.truetype(c, s)); break
GREEN, INK, GREY, MUTE = (34, 197, 94), (17, 17, 19), (242, 242, 240), (120, 120, 126)


def ease(x): x = min(1, max(0, x)); return 1 - (1 - x) ** 3
def eio(x): x = min(1, max(0, x)); return x * x * (3 - 2 * x)
def L(p): return Image.open(p).convert("RGB")
def g(p): return sorted(glob.glob(os.path.join(RUN, p)))


_pol = RUN.rstrip("/") + "_polished"
shots = [L(p) for p in sorted(glob.glob(os.path.join(_pol, "shot_*.png")))] if os.path.isdir(_pol) else [L(p) for p in g("shot_*.png")]
import json as _json
BOXES = _json.load(open(os.path.join(_pol, "boxes.json"))) if os.path.isdir(_pol) else None
ai = [L(p) for p in g("ai_only_*.png")]
cmp = [L(p) for p in g("compare_*.png")]
market = L(g("marketplace_*.png")[0])
cut = L(g("cutout_*.png")[0])
photo = L(PHOTO)


def fit(im, w, h, cover=False):
    s = (max if cover else min)(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    if cover:
        x, y = (im.width - w) // 2, (im.height - h) // 2
        im = im.crop((x, y, x + w, y + h))
    return im


def rmask(size, r):
    m = Image.new("L", size, 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], r, fill=255)
    return m


def paste(c, im, cx, cy, a=1.0, r=24, shadow=True):
    if a <= 0.01: return
    x, y = int(cx - im.width / 2), int(cy - im.height / 2)
    if shadow:
        s = Image.new("L", (im.width + 140, im.height + 140), 0)
        ImageDraw.Draw(s).rounded_rectangle([70, 95, im.width + 70, im.height + 95], r, fill=int(110 * a))
        c.paste((0, 0, 0), (x - 70, y - 70), s.filter(ImageFilter.GaussianBlur(34)))
    m = rmask(im.size, r)
    if a < 1: m = m.point(lambda v: int(v * a))
    c.paste(im, (x, y), m)


def persp(im, tilt, a=1.0):
    """Fake 3D: rotate card around Y axis by `tilt` (-1..1)."""
    w, h = im.size
    k = 0.16 * tilt
    dst = [(0, h * max(0, k)), (w, h * max(0, -k)), (w, h * (1 - max(0, -k))), (0, h * (1 - max(0, k)))]
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    A = []; B = []
    for (x, y), (u, v) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]; B += [u, v]
    coef = np.linalg.solve(np.array(A, float), np.array(B, float))
    rgba = im.convert("RGBA")
    m = rmask(im.size, 22)
    rgba.putalpha(m)
    return rgba.transform((w, h), Image.PERSPECTIVE, coef, Image.BICUBIC)


def bg_dark():
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    v = np.clip(40 - d * 70, 5, 44)
    return Image.fromarray(np.dstack([v, v, v]).astype(np.uint8))


def bg_light():
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - W * 0.45) / W) ** 2 + ((yy - H * 0.35) / H) ** 2)
    v = np.clip(252 - d * 60, 214, 252)
    return Image.fromarray(np.dstack([v, v, v - 1]).astype(np.uint8))


BD, BL = bg_dark(), bg_light()
GRID = BD.copy(); _g = ImageDraw.Draw(GRID)
for x in range(0, W, 150): _g.line([x, 0, x, H], fill=(30, 30, 34), width=2)
for y in range(0, H, 150): _g.line([0, y, W, y], fill=(30, 30, 34), width=2)


def T(d, xy, s, f, fill, anchor="la", a=1.0, bg=(12, 12, 14)):
    if a <= 0: return
    d.text(xy, s, font=f, fill=tuple(int(bg[i] + (fill[i] - bg[i]) * a) for i in range(3)), anchor=anchor)


# ---------------------------------------------------------------- clips
def frames(path, n, w, h, start=0.5):
    p = subprocess.run(["ffmpeg", "-v", "quiet", "-ss", str(start), "-i", path, "-frames:v", str(n), "-vf",
                        f"fps={FPS},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8); k = len(a) // (w * h * 3)
    fr = [Image.fromarray(a[i * w * h * 3:(i + 1) * w * h * 3].reshape(h, w, 3)) for i in range(k)]
    while len(fr) < n: fr.append(fr[-1] if fr else Image.new("RGB", (w, h)))
    return fr


NCL = int(BAR * 2 * FPS) + 5
VC = [frames(p, NCL, 420, 700, 1.0) for p in CLIPS[:5]]          # vertical cards
HOOK = frames(CLIPS[5] if len(CLIPS) > 5 else CLIPS[0], int(BAR * FPS) + 3, W, H, 1.5)


# ---------------------------------------------------------------- UI bits
def chip(text, w=None, dot=True, fs=30, h=64):
    f = F("Medium", fs)
    tw = int(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(text, font=f))
    w = w or tw + (84 if dot else 48)
    im = Image.new("RGB", (w, h), GREY); d = ImageDraw.Draw(im)
    x = 24
    if dot:
        d.ellipse([24, h / 2 - 10, 44, h / 2 + 10], fill=GREEN); x = 60
    d.text((x, h / 2), text, font=f, fill=INK, anchor="lm")
    return im


def prompt(text, n, att_img, att_name, att_meta, agent, w=1180, h=470, cursor=True):
    im = Image.new("RGB", (w, h), (255, 255, 255)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([34, 34, 34 + 520, 34 + 92], 22, fill=GREY)
    th = fit(att_img, 70, 70, cover=True); im.paste(th, (46, 45), rmask(th.size, 14))
    d.text((132, 62), att_name, font=F("Medium", 28), fill=INK, anchor="lm")
    d.text((132, 98), att_meta, font=F("Regular", 21), fill=MUTE, anchor="lm")
    s = text[:n]
    d.text((44, 220), s, font=F("Regular", 50), fill=INK, anchor="lm")
    if cursor:
        x = 48 + d.textlength(s, font=F("Regular", 50))
        d.rectangle([x, 190, x + 5, 250], fill=GREEN)
    c = chip(agent); im.paste(c, (34, h - 34 - c.height), rmask(c.size, 32))
    d.ellipse([w - 34 - 96, h - 34 - 96, w - 34, h - 34], fill=INK)
    d.text((w - 34 - 48, h - 34 - 50), "↑", font=F("Bold", 46), fill=(255, 255, 255), anchor="mm")
    return im


def label_row(title, body_fn, w=560, h=150):
    im = Image.new("RGB", (w, h), (255, 255, 255)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], 22, outline=(230, 230, 228), width=2)
    d.text((28, 30), title, font=MONO(22), fill=(140, 140, 146))
    body_fn(im, d)
    return im


def memory_rows():
    def r_label(im, d):
        d.text((28, 98), "O'TENTIKA · MAXITONE", font=F("Bold", 36), fill=(200, 30, 140), anchor="lm")
    def r_col(im, d):
        for i, c in enumerate([(236, 226, 205), (205, 40, 140), (60, 40, 40)]):
            d.rounded_rectangle([28 + i * 64, 72, 80 + i * 64, 124], 10, fill=c)
        d.text((240, 98), "Cream · Magenta", font=F("Medium", 30), fill=INK, anchor="lm")
    def r_shape(im, d):
        t = fit(cut, 60, 80); im.paste(t, (28, 60))
        d.text((110, 98), "Bottle · 300 ml", font=F("Medium", 30), fill=INK, anchor="lm")
    def r_photo(im, d):
        t = fit(photo, 60, 80, cover=True); im.paste(t, (28, 60), rmask(t.size, 10))
        d.text((110, 98), "phone photo · real pixels", font=F("Medium", 30), fill=INK, anchor="lm")
    return [label_row("LABEL", r_label), label_row("COLOUR", r_col), label_row("SHAPE", r_shape),
            label_row("SOURCE", r_photo)]


ROWS = memory_rows()
PHOTO_CHIP = ("otentika_phone_photo.jpg", "JPG · 1 product")


# ---------------------------------------------------------------- scenes
def s_type(t):
    c = BD.copy(); d = ImageDraw.Draw(c)
    a, b = "Product photos", " that don't lie."
    full = a + b; n = int(min(len(full), t * 13))
    fa, fb = F("Medium", 110), SERIF(116)
    sa, sb = full[:n][:len(a)], full[:n][len(a):]
    x0 = W / 2 - (d.textlength(a, font=fa) + d.textlength(b, font=fb)) / 2 if n > len(a) else W / 2 - d.textlength(sa, font=fa) / 2
    d.text((x0, H / 2), sa, font=fa, fill=(236, 236, 236), anchor="lm")
    xb = x0 + d.textlength(sa, font=fa)
    d.text((xb, H / 2 + 4), sb, font=fb, fill=GREEN, anchor="lm")
    xc = xb + d.textlength(sb, font=fb) + 8
    if n < len(full) or int(t * 2.5) % 2 == 0:
        d.rectangle([xc, H / 2 - 62, xc + 6, H / 2 + 62], fill=GREEN)
    return c


def s_ui(t):
    """Big light app window: prompt card + product memory panel, slow push-in."""
    c = BD.copy()
    z = 1.18 - 0.12 * eio(t / (BAR * 2))
    win = Image.new("RGB", (1800, 1000), (236, 236, 234))
    wd = ImageDraw.Draw(win)
    msg = "a shot pack for our new body lotion"
    pc = prompt(msg, int(max(0, t - 0.3) * 20), photo, *PHOTO_CHIP, "TrueShot · ComfyUI")
    win.paste(pc, (60, 470), rmask(pc.size, 40))
    wd.rounded_rectangle([1290, 40, 1770, 960], 30, fill=(250, 250, 249))
    wd.text((1320, 80), "PRODUCT MEMORY", font=MONO(22), fill=(140, 140, 146))
    for i, r in enumerate(ROWS):
        k = ease((t - 0.8 - i * BEAT) / 0.4)
        if k <= 0: continue
        rr = r.resize((440, 120))
        win.paste(rr, (1310, int(130 + i * 140 + (1 - k) * 30)), rmask(rr.size, 20))
    ww, wh = int(1800 * z), int(1000 * z)
    wim = win.resize((ww, wh), Image.LANCZOS)
    paste(c, wim, W / 2 - 60 * (z - 1) * 6, H / 2 + 40, ease(t / 0.4), r=44)
    return c


def s_wire(t):
    """Memory rows wired with green lines into the live preview (shot)."""
    c = BL.copy(); d = ImageDraw.Draw(c)
    bgc = (246, 246, 244)
    T(d, (90, 70), "TrueShot", F("Bold", 34), INK, "la", 1, bg=bgc)
    T(d, (90, 112), "product memory · locked", MONO(20), MUTE, "la", 1, bg=bgc)
    ys = [230, 400, 570, 740]
    for i, (r, y) in enumerate(zip(ROWS, ys)):
        rr = r.resize((470, 126)); c.paste(rr, (90, y), rmask(rr.size, 20))
    which = 1 if t < BAR * 1.1 else (0 if t < BAR * 1.6 else 4)
    pv = fit(shots[which], 980, 820, cover=False)
    px, py = 1290, 560
    T(d, (px - pv.width / 2, py - pv.height / 2 - 34), "preview · 768×768" if which != 4 else "preview · story 9:16",
      MONO(20), MUTE, "la", 1, bg=bgc)
    k = ease((t - 0.2) / 0.8)
    for i, y in enumerate(ys):
        kk = ease((t - 0.2 - i * 0.12) / 0.6)
        x0, y0 = 560, y + 63
        x1, y1 = px - pv.width / 2 + 20, py - pv.height / 2 + 120 + i * 160
        pts = [(x0 + (x1 - x0) * u * kk, y0 + (y1 - y0) * eio(u) * kk) for u in np.linspace(0, 1, 30)]
        d.line(pts, fill=GREEN, width=4)
        d.ellipse([x0 - 7, y0 - 7, x0 + 7, y0 + 7], fill=GREEN)
    paste(c, pv, px, py, k, r=28)
    if which != 1:
        ch = chip("copy layer · " + ("marble" if which == 0 else "story 9:16"), dot=False, fs=24, h=50)
        paste(c, ch, px + pv.width / 2 - ch.width / 2 - 10, py - pv.height / 2 - 34, 1, r=25, shadow=False)
    return c


def s_prompt2(t, msg, att, name, meta, agent="TrueShot · ComfyUI"):
    c = BD.copy()
    k = ease(t / 0.4)
    pc = prompt(msg, int(max(0, t - 0.2) * 24), att, name, meta, agent, w=1100, h=420)
    paste(c, pc, W / 2, H / 2 + (1 - k) * 60, k, r=40)
    return c


def tightc(im, box, pad=0.10, wpad=0.55):
    x0, y0, x1, y1 = box; h = y1 - y0; w = x1 - x0
    return im.crop((max(0, int(x0 - w * wpad)), max(0, int(y0 - h * pad)), min(im.width, int(x1 + w * wpad)),
                    min(im.height, int(y1 + h * pad))))


def ai_box(im):
    """Locate the AI-drawn bottle: brightest tall blob near the centre."""
    a = np.asarray(im.convert("L")).astype(np.float32)
    m = (a > np.percentile(a, 80)).astype(np.uint8)
    import cv2 as _cv
    m = _cv.morphologyEx(m, _cv.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, l, st, cen = _cv.connectedComponentsWithStats(m, 8)
    best, bs = None, -1
    for k in range(1, n):
        x, y, w, h, ar = st[k]
        if h < im.height * 0.25: continue
        sc = ar - abs(cen[k][0] - im.width / 2) * 20
        if sc > bs: best, bs = (x, y, x + w, y + h), sc
    return best or (int(im.width * .35), int(im.height * .15), int(im.width * .65), int(im.height * .9))


def s_explode(t):
    """Your photo -> AI only -> TrueShot, same bottle size, honest crops."""
    yy, xx = np.mgrid[0:H, 0:W]
    v = np.clip(246 - np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 80, 205, 246)
    c = Image.fromarray(np.dstack([v, v - 3, v - 8]).astype(np.uint8)); d = ImageDraw.Draw(c)
    bgc = (240, 236, 229)
    i = 0
    ts = tightc(shots[i], BOXES[i]) if BOXES else shots[i]
    aim = ai[i]
    x0, y0, x1, y1 = BOXES[i]; cxm = (x0 + x1) / 2; hh = (y1 - y0) * 1.35; ww = (x1 - x0) * 1.35
    aic = tightc(aim, (int(cxm - ww / 2), int(y1 - hh), int(cxm + ww / 2), int(min(aim.height, y1 + (y1 - y0) * 0.05))))
    ph = fit(photo, 900, 1200)
    pb = ai_box(ph); phc = tightc(ph, pb, 0.06, 0.45)
    panels = [(phc, "Your photo", (70, 70, 76), "real label"), (aic, "AI only", (215, 55, 55), "MAXITONE → garbled"),
              (ts, "TrueShot", (20, 150, 70), "MAXITONE · exact")]
    for j, (im, title, col, sub) in enumerate(panels):
        k = ease((t - j * BEAT * 1.5) / 0.45)
        if k <= 0: continue
        pim = fit(im, 420, 600)
        x = 400 + j * 560
        paste(c, pim, x, 470 + (1 - k) * 50, k, r=22)
        T(d, (x, 110), title, F("Bold", 44), col, "mm", k, bg=bgc)
        # magnifier on the label band (MAXITONE area), same relative region in each crop
        kz = ease((t - j * BEAT * 1.5 - 0.5) / 0.4)
        if kz > 0:
            w0, h0 = im.size
            band = im.crop((int(w0 * 0.30), int(h0 * 0.50), int(w0 * 0.70), int(h0 * 0.68)))
            z = band.resize((440, int(440 * band.height / max(1, band.width))), Image.LANCZOS)
            z = z.rotate(90, expand=True) if z.height < z.width * 0.6 and False else z
            zz = fit(z, 460, 200)
            paste(c, zz, x, 880, kz, r=16)
            d.rounded_rectangle([x - zz.width / 2 - 4, 880 - zz.height / 2 - 4, x + zz.width / 2 + 4,
                                 880 + zz.height / 2 + 4], 18, outline=col, width=4)
        T(d, (x, 1010), sub, MONO(26), (60, 60, 64), "mm", k, bg=bgc)
    kk = ease((t - BEAT * 5) / 0.4)
    if kk > 0:
        d.rounded_rectangle([W / 2 + 520 - 190, 170, W / 2 + 520 + 190, 230], 30, fill=(22, 163, 74))
        T(d, (W / 2 + 520, 200), "truth 43 → 96", F("Bold", 30), (255, 255, 255), "mm", kk, bg=(22, 163, 74))
    return c


def s_sellers(t, i0):
    """3D-tilted cards playing the stock clips + shots, serif-italic header."""
    c = BD.copy(); d = ImageDraw.Draw(c)
    hk = ease(t / 0.4)
    d.rounded_rectangle([60, 46, 520, 130], 8, fill=(70, 90, 200))
    T(d, (80, 88), "SELLERS", SERIF(78), (255, 255, 255), "lm", hk)
    T(d, (550, 88), "· one photo, every format", SERIF(62), (240, 240, 240), "lm", hk)
    T(d, (80, 150), "● vendors · launch brands · catalogues · second-hand", MONO(20), (150, 150, 155), "la", hk)
    pan = eio(t / (BAR * 2)) * 260
    items = [("clip", 0), ("shot", 1), ("clip", 1), ("shot", 4), ("clip", 2)]
    xs = [330, 760, 1160, 1560, 1960]
    for j, ((kind, idx), x) in enumerate(zip(items, xs)):
        k = ease((t - j * BEAT * 0.5) / 0.4)
        if k <= 0: continue
        if kind == "clip":
            fr = VC[idx % len(VC)][min(i0, NCL - 1)]
        else:
            fr = fit(shots[idx], 420, 700, cover=True)
        cx = x - pan
        tilt = (cx - W / 2) / (W / 2) * 0.9
        sc = 0.92 if j % 2 else 1.0
        im = persp(fr.resize((int(420 * sc), int(700 * sc))), tilt)
        tag = ("RAW · phone" if kind == "clip" else "TRUESHOT · shot")
        c.paste(im, (int(cx - im.width / 2), int(600 - im.height / 2 + (1 - k) * 60)), im)
        T(d, (cx - im.width / 2 + 16, 600 - im.height / 2 - 26), tag, MONO(18), (180, 180, 185), "la", k)
    return c


def s_grid(t):
    c = GRID.copy()
    k1, k2 = ease(t / 0.4), ease((t - BAR * 0.9) / 0.4)
    p1 = prompt("a WhatsApp status for today", int(max(0, t - 0.2) * 26), shots[4],
                "shot_story.png", "PNG · 576×1024", "TrueShot · ComfyUI", w=860, h=330, cursor=t < BAR)
    paste(c, p1, 560, 330 + (1 - k1) * 40, k1, r=30)
    st = fit(shots[4], 300, 520)
    paste(c, st, 1180, 330, k1, r=20)
    if k2 > 0:
        p2 = prompt("a marketplace main image", int(max(0, t - BAR * 0.9) * 26), market,
                    "listing_check.json", "white bg · ≥85% fill · 1200px", "TrueShot · ComfyUI", w=860, h=330)
        paste(c, p2, 1300, 760 + (1 - k2) * 40, k2, r=30)
        mk = fit(market, 360, 360)
        paste(c, mk, 640, 780, k2, r=20)
        d = ImageDraw.Draw(c)
        T(d, (640, 990), "✓ PASS · 100% real pixels", F("Bold", 30), GREEN, "mm", k2)
    return c


ALLM = shots + ai + [market] + [fit(v[10], 420, 700) for v in VC]
def s_mosaic(t, i0):
    c = BD.copy()
    tile, gap, cols, rows = 230, 14, 12, 6
    big = Image.new("RGB", (cols * (tile + gap), rows * (tile + gap)), (12, 12, 14))
    for r in range(rows):
        for q in range(cols):
            idx = (r * 5 + q * 7) % (len(ALLM) + len(VC))
            if idx >= len(ALLM):
                im = VC[idx - len(ALLM)][min(i0, NCL - 1)]
            else:
                im = ALLM[idx]
            big.paste(fit(im, tile, tile, cover=True), (q * (tile + gap), r * (tile + gap)))
    z = 1.5 - 0.5 * eio(t / (BAR * 1.6))
    bw, bh = int(big.width * z * 0.62), int(big.height * z * 0.62)
    c.paste(big.resize((bw, bh), Image.LANCZOS), (int(W / 2 - bw / 2), int(H / 2 - bh / 2)))
    return c


def logo(d, cx, cy, r, col=GREEN):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=max(4, r // 7))
    d.line([cx - r * 0.45, cy, cx - r * 0.1, cy + r * 0.35, cx + r * 0.5, cy - r * 0.35], fill=col, width=max(4, r // 7))


def s_repo(t):
    c = BD.copy(); d = ImageDraw.Draw(c)
    k = ease(t / 0.4)
    d.rounded_rectangle([520, 250, 1400, 470], 26, fill=(20, 20, 23), outline=(48, 48, 54), width=2)
    T(d, (570, 300), "nedsoft057 / ComfyUI-TrueShot", F("Medium", 36), (110, 160, 255), "la", k)
    T(d, (1350, 300), "Public", MONO(20), (150, 150, 150), "ra", k)
    T(d, (570, 400), "Apache-2.0 · 5 nodes · one-cell Kaggle run", F("Medium", 34), (230, 230, 230), "lm", k)
    logo(d, W / 2, 590, 44)
    chips = ["ComfyUI", "FLUX.2 Klein", "open weights", "Kaggle T4", "OpenCV", "rembg"]
    allw = 0; ims = [chip(s, fs=24, h=52) for s in chips]
    x = W / 2 - (sum(i.width for i in ims) + 16 * (len(ims) - 1)) / 2
    for j, im in enumerate(ims):
        kj = ease((t - 0.3 - j * 0.1) / 0.3)
        dark = Image.new("RGB", im.size, (30, 30, 34)); dd = ImageDraw.Draw(dark)
        dd.ellipse([24, im.height / 2 - 9, 42, im.height / 2 + 9], fill=GREEN)
        dd.text((60, im.height / 2), chips[j], font=F("Medium", 24), fill=(230, 230, 230), anchor="lm")
        if kj > 0: c.paste(dark, (int(x), 690), rmask(dark.size, 26))
        x += im.width + 16
    k2 = ease((t - 0.9) / 0.4)
    T(d, (W / 2 - 200, 850), "Open", F("Medium", 52), (240, 240, 240), "rm", k2)
    T(d, (W / 2 - 185, 852), "source.", SERIF(56), (240, 240, 240), "lm", k2)
    T(d, (W / 2 + 60, 850), "Open weights.", F("Medium", 52), (240, 240, 240), "lm", ease((t - 1.2) / 0.4))
    return c


def s_end(t):
    c = BD.copy(); d = ImageDraw.Draw(c)
    if t < BEAT * 2:
        T(d, (140, 640), "That's", F("Medium", 80), (220, 220, 220), "la", ease(t / 0.3))
        return c
    tt = t - BEAT * 2
    k = ease(tt / 0.5)
    logo(d, W / 2 - 300, H / 2 - 70, 58)
    T(d, (W / 2 - 215, H / 2 - 70), "TrueShot", F("Bold", 120), (245, 245, 245), "lm", k)
    T(d, (W / 2 - 330, H / 2 + 70), "The open-source product studio that", F("Medium", 40), (210, 210, 210), "lm",
      ease((tt - 0.3) / 0.4))
    T(d, (W / 2 + 395, H / 2 + 72), "never lies.", SERIF(44), GREEN, "lm", ease((tt - 0.5) / 0.4))
    kk = ease((tt - 0.8) / 0.4)
    if kk > 0:
        d.rounded_rectangle([W / 2 - 220, H / 2 + 150, W / 2 + 220, H / 2 + 206], 28, fill=(22, 163, 74))
        T(d, (W / 2, H / 2 + 178), "#OpenSourceHiggsfield", F("Bold", 28), (255, 255, 255), "mm", kk, bg=(22, 163, 74))
    return c


def s_hook(t, i):
    fr = HOOK[min(i, len(HOOK) - 1)]
    fr = Image.blend(fr, Image.new("RGB", (W, H)), 0.6); d = ImageDraw.Draw(fr)
    T(d, (W / 2, H / 2 - 30), "One phone photo.", F("Bold", 96), (255, 255, 255), "mm", ease(t / 0.3), bg=(0, 0, 0))
    T(d, (W / 2, H / 2 + 70), "Every listing, story and ad.", SERIF(70), GREEN, "mm", ease((t - BEAT) / 0.3), bg=(0, 0, 0))
    return fr


SC = [
    ("type", 2 * BAR, lambda t, i: s_type(t)),
    ("ui", 2 * BAR, lambda t, i: s_ui(t)),
    ("wire", 2 * BAR, lambda t, i: s_wire(t)),
    ("p2", 1 * BAR, lambda t, i: s_prompt2(t, "show me what the AI changed", cut, "label_check", "SSIM · IoU · ΔE")),
    ("explode", 2 * BAR, lambda t, i: s_explode(t)),
    ("p3", 1 * BAR, lambda t, i: s_prompt2(t, "turn it into content for every seller", photo, *PHOTO_CHIP)),
    ("sellers", 2 * BAR, lambda t, i: s_sellers(t, i)),
    ("hook", 1 * BAR, lambda t, i: s_hook(t, i)),
    ("grid", 2 * BAR, lambda t, i: s_grid(t)),
    ("mosaic", 1.5 * BAR, lambda t, i: s_mosaic(t, i)),
    ("repo", 2 * BAR, lambda t, i: s_repo(t)),
    ("end", 2.5 * BAR, lambda t, i: s_end(t)),
]


def render():
    total = sum(d for _, d, _ in SC)
    tmp = OUT + ".v.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                          "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    f0 = 0.0
    for name, dur, fn in SC:
        a = int(round(f0)); b = int(round(f0 + dur * FPS)); f0 += dur * FPS
        for i in range(b - a):
            p.stdin.write(fn(i / FPS, i).tobytes())
    p.stdin.close(); p.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-ss", str(MUSIC_START), "-i", MUSIC, "-filter_complex",
                    f"[1:a]afade=t=in:st=0:d=0.2,afade=t=out:st={total - 1.6}:d=1.6[a]", "-map", "0:v", "-map", "[a]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", str(total), "-movflags", "+faststart", OUT],
                   check=True)
    os.remove(tmp)
    print("wrote", OUT, round(total, 1))


if __name__ == "__main__":
    if os.environ.get("PREVIEW"):
        ims = []
        for name, dur, fn in SC:
            t = dur * 0.75
            ims.append(fn(t, int(t * FPS)).resize((640, 360)))
        gimg = Image.new("RGB", (1280, 360 * ((len(ims) + 1) // 2)))
        for k, im in enumerate(ims): gimg.paste(im, ((k % 2) * 640, (k // 2) * 360))
        gimg.save("/tmp/pv2.jpg")
    else:
        render()
