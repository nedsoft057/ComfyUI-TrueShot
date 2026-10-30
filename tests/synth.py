"""Synthetic 'phone photo' of a labelled product + a simulated AI render that
garbles the label, for testing without a GPU."""
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
def font(s):
    try: return ImageFont.truetype(F, s)
    except Exception: return ImageFont.load_default()

def product_rgba(w=300, h=620, text=("LUMA", "VITAMIN C", "SERUM 30ml"), seed=0):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([w*0.38, 0, w*0.62, h*0.16], 12, fill=(30, 30, 34, 255))           # cap
    d.rounded_rectangle([w*0.05, h*0.14, w*0.95, h], 40, fill=(236, 160, 60, 255))          # body
    d.rectangle([w*0.1, h*0.36, w*0.9, h*0.82], fill=(250, 246, 238, 255))                   # label
    d.text((w*0.5, h*0.45), text[0], fill=(20, 20, 30, 255), font=font(64), anchor="mm")
    d.text((w*0.5, h*0.58), text[1], fill=(200, 80, 30, 255), font=font(30), anchor="mm")
    d.text((w*0.5, h*0.68), text[2], fill=(60, 60, 70, 255), font=font(24), anchor="mm")
    for i in range(6):
        d.line([w*0.15, h*(0.73+i*0.013), w*0.85, h*(0.73+i*0.013)], fill=(120,120,130,255), width=2)
    a = np.asarray(im).astype(np.float32) / 255
    # glossy highlight
    xx = np.linspace(0, 1, w)[None, :]
    a[..., :3] = np.clip(a[..., :3] + 0.18 * np.exp(-((xx - 0.25) / 0.06) ** 2)[..., None] * (a[..., 3:] > 0), 0, 1)
    return a

def phone_photo(prod, W=900, H=1200, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W] / max(W, H)
    bg = np.stack([0.55 + 0.1*yy, 0.5 + 0.05*xx, 0.45 + 0*xx], -1).astype(np.float32)
    bg += rng.normal(0, 0.02, bg.shape).astype(np.float32)
    bg = cv2.GaussianBlur(bg, (0, 0), 2)
    ph, pw = prod.shape[:2]; x0, y0 = (W - pw)//2 + 20, (H - ph)//2 + 40
    out = bg.copy(); a = prod[..., 3:]
    out[y0:y0+ph, x0:x0+pw] = out[y0:y0+ph, x0:x0+pw]*(1-a) + prod[..., :3]*a
    out += rng.normal(0, 0.01, out.shape).astype(np.float32)
    return np.clip(out, 0, 1)

def fake_ai_render(prod_rgb, prod_a, W=768, H=768, seed=1, scale=0.55, cx=0.47, base=0.88, angle=3.0):
    """What a generator does: nice scene, product roughly right, label mangled."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W] / max(W, H)
    scene = np.stack([0.25 + 0.4*yy, 0.35 + 0.2*xx, 0.55 - 0.2*yy], -1).astype(np.float32)
    scene[int(H*0.8):] *= 0.7
    scene = cv2.GaussianBlur(scene + rng.normal(0, 0.03, scene.shape).astype(np.float32), (0, 0), 3)
    # garble label: blur + scramble text strip
    g = prod_rgb.copy(); h, w = g.shape[:2]
    lab = g[int(h*0.38):int(h*0.8), int(w*0.12):int(w*0.88)]
    lab = cv2.GaussianBlur(lab, (0, 0), 3)
    strip = lab.reshape(-1, 3); idx = rng.permutation(len(strip))[: len(strip)//6]
    strip[idx] = strip[idx][::-1]
    g[int(h*0.38):int(h*0.8), int(w*0.12):int(w*0.88)] = lab
    g = np.clip(g * np.array([1.05, 0.95, 1.12], np.float32), 0, 1)  # colour cast
    s = scale * H / h
    M = cv2.getRotationMatrix2D((w/2, h), angle, s)
    M[0, 2] += cx*W - w/2; M[1, 2] += base*H - h
    pr = cv2.warpAffine(g, M, (W, H)); pa = cv2.warpAffine(prod_a, M, (W, H))
    out = scene*(1-pa[..., None]) + pr*pa[..., None]
    return np.clip(out, 0, 1), M
