"""TrueShot core: cutout, staging, product localisation, pixel-true restore,
harmonisation, shadows and fidelity QC. numpy + OpenCV only.

Images are float32 RGB in [0, 1], shape (H, W, 3). Masks are float32 (H, W) in [0, 1].
"""
from __future__ import annotations

import math

import cv2
import numpy as np


# ------------------------------------------------------------------ helpers
def to_u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def to_f(img):
    return img.astype(np.float32) / 255.0


def bbox(mask, thr=0.5):
    ys, xs = np.where(mask > thr)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def ssim(a, b, mask=None):
    """Gray SSIM (Gaussian window) optionally averaged inside a mask."""
    a = cv2.cvtColor(to_u8(a), cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(to_u8(b), cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = g(a), g(b)
    saa, sbb, sab = g(a * a) - mu_a ** 2, g(b * b) - mu_b ** 2, g(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    if mask is not None:
        w = (mask > 0.5).astype(np.float64)
        w = cv2.erode(w, np.ones((7, 7), np.uint8))
        if w.sum() < 50:
            w = (mask > 0.5).astype(np.float64)
        return float((m * w).sum() / max(1.0, w.sum()))
    return float(m.mean())


# ------------------------------------------------------------------- cutout
_REMBG = None


def cutout(img, method="auto"):
    """Product matte. Uses rembg (IS-Net / BiRefNet) when available, else a
    GrabCut matte seeded from colour distance to the border."""
    global _REMBG
    if method in ("auto", "rembg"):
        try:
            if _REMBG is None:
                from rembg import new_session
                _REMBG = new_session("isnet-general-use")
            from rembg import remove
            rgba = remove(to_u8(img), session=_REMBG, post_process_mask=True)
            a = rgba[..., 3].astype(np.float32) / 255.0
            if a.max() > 0.5 and 0.01 < (a > 0.5).mean() < 0.95:
                return refine_alpha(img, a)
        except Exception as e:  # pragma: no cover - optional dependency
            if method == "rembg":
                raise
            print(f"[TrueShot] rembg unavailable ({type(e).__name__}); using GrabCut")
    return grabcut_matte(img)


def grabcut_matte(img):
    u8 = to_u8(img)
    h, w = u8.shape[:2]
    # seed: pixels far (in Lab) from the border colour model are probable foreground
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = max(2, min(h, w) // 40)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    k = min(5, len(border))
    _, lab_c, ctr = cv2.kmeans(border, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                               3, cv2.KMEANS_PP_CENTERS)
    d = np.min(np.linalg.norm(lab.reshape(-1, 1, 3) - ctr[None], axis=2), axis=1).reshape(h, w)
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    thr_hi = np.percentile(d, 80)
    gc[d > max(18.0, thr_hi * 0.6)] = cv2.GC_PR_FGD
    gc[:b], gc[-b:], gc[:, :b], gc[:, -b:] = cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD
    fg_core = cv2.erode((d > thr_hi).astype(np.uint8), np.ones((9, 9), np.uint8))
    gc[fg_core > 0] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(u8, gc, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        pass
    m = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    # keep the largest component, fill holes (labels, reflections)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lbl == big).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    return refine_alpha(img, filled.astype(np.float32))


def refine_alpha(img, a):
    """Light edge feather so composites don't look cut with scissors."""
    a = np.clip(a, 0, 1).astype(np.float32)
    hard = (a > 0.5).astype(np.float32)
    soft = cv2.GaussianBlur(hard, (0, 0), 1.2)
    return np.clip(np.maximum(soft * (cv2.dilate(hard, np.ones((3, 3), np.uint8)) > 0), a * hard), 0, 1)


def crop_product(img, alpha, pad=0.04):
    x0, y0, x1, y1 = bbox(alpha)
    p = int(round(max(x1 - x0, y1 - y0) * pad))
    h, w = alpha.shape
    x0, y0, x1, y1 = max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p)
    return img[y0:y1, x0:x1].copy(), alpha[y0:y1, x0:x1].copy()


# ------------------------------------------------------------------ staging
def place(prod, alpha, W, H, height_frac=0.62, cx=0.5, base_y=0.86):
    """Scale the product so its height is `height_frac` of the frame, centred
    at cx with its base on base_y. Returns rgb, alpha and the 3x3 transform
    (product-crop coords -> frame coords)."""
    ph, pw = alpha.shape
    s = min(height_frac * H / ph, 0.8 * W / pw)
    nw, nh = max(1, int(round(pw * s))), max(1, int(round(ph * s)))
    x0 = int(round(cx * W - nw / 2))
    y0 = int(round(base_y * H - nh))
    M = np.array([[s, 0, x0], [0, s, y0], [0, 0, 1]], np.float64)
    rgb = cv2.warpAffine(prod, M[:2], (W, H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT)
    a = cv2.warpAffine(alpha, M[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rgb, a, M


def contact_shadow(alpha, strength=0.55, softness=0.035, lift=0.12):
    """Soft ground shadow from the silhouette: squash the mask onto the floor
    plane under the product and blur it."""
    H, W = alpha.shape
    x0, y0, x1, y1 = bbox(alpha)
    hgt = max(1, y1 - y0)
    # flatten silhouette vertically about its base
    M = np.array([[1, 0, 0], [0, lift, y1 * (1 - lift)]], np.float32)
    sh = cv2.warpAffine(alpha, M, (W, H))
    k = max(3, int(softness * max(W, H)) | 1)
    sh = cv2.GaussianBlur(sh, (k, k), 0)
    # tight occlusion right under the base
    occ = np.zeros_like(alpha)
    cv2.ellipse(occ, ((x0 + x1) // 2, y1), (max(2, (x1 - x0) // 2), max(2, hgt // 40)), 0, 0, 360, 1.0, -1)
    occ = cv2.GaussianBlur(occ, (0, 0), max(1.0, hgt / 120))
    return np.clip(strength * sh + 0.6 * strength * occ, 0, 0.9) * (1 - alpha)


def marketplace(prod, alpha, size=1200, height_frac=0.78, bg=(1.0, 1.0, 1.0)):
    """Deterministic white-background listing photo: no generation, so the
    product is 100% real pixels."""
    rgb, a, _ = place(prod, alpha, size, size, height_frac=height_frac, base_y=0.89)
    canvas = np.ones((size, size, 3), np.float32) * np.array(bg, np.float32)
    sh = contact_shadow(a, strength=0.35)
    canvas = canvas * (1 - sh[..., None])
    out = canvas * (1 - a[..., None]) + rgb * a[..., None]
    return np.clip(out, 0, 1).astype(np.float32)


def stage(prod, alpha, W, H, height_frac, cx=0.5, base_y=0.86, bg=0.5):
    """Reference image for the generator: product placed on a neutral card
    where we want it in the final frame."""
    rgb, a, M = place(prod, alpha, W, H, height_frac, cx, base_y)
    canvas = np.full((H, W, 3), bg, np.float32)
    out = canvas * (1 - a[..., None]) + rgb * a[..., None]
    return out.astype(np.float32), a, M


# ------------------------------------------------------------- localisation
def _gray(img):
    return cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2GRAY)


def locate(prod, alpha, gen, prior_M=None, min_inliers=12):
    """Find where the product landed in the generated image.

    1) SIFT on the product (inside its matte) vs the generated frame,
       ratio test, RANSAC homography constrained to a near-similarity.
    2) Fallback: multi-scale masked template matching around the staged
       position (the generator usually keeps the layout of the reference).
    Returns (H 3x3 product->gen, info dict) or (None, info).
    """
    info = {"method": None, "inliers": 0}
    g1, g2 = _gray(prod), _gray(gen)
    m1 = (alpha > 0.5).astype(np.uint8) * 255
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    k1, d1 = sift.detectAndCompute(g1, m1)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is not None and d2 is not None and len(k1) >= 8 and len(k2) >= 8:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        good = []
        for pair in bf.knnMatch(d1, d2, k=2):
            if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance:
                good.append(pair[0])
        if len(good) >= min_inliers:
            src = np.float32([k1[m.queryIdx].pt for m in good])
            dst = np.float32([k2[m.trainIdx].pt for m in good])
            Hm, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC,
                                         4.0, maxIters=5000, confidence=0.999)
            if Hm is not None:
                n_in = int(inl.sum())
                if n_in >= min_inliers and _sane(Hm, alpha.shape, gen.shape):
                    # refine with a partial-affine fit on inliers (more stable for small products)
                    A, _ = cv2.estimateAffinePartial2D(src[inl.ravel() > 0], dst[inl.ravel() > 0])
                    info.update(method="features", inliers=n_in, matches=len(good))
                    Hp = np.vstack([A, [0, 0, 1]]) if A is not None else Hm
                    # keep full homography only when it explains clearly more
                    return (Hm if _reproj(Hm, src, dst) + 0.5 < _reproj(Hp, src, dst) else Hp), info
    # template fallback
    best = None
    ph, pw = alpha.shape
    Hg, Wg = g2.shape
    if prior_M is not None:
        s0 = prior_M[0, 0]
    else:
        s0 = min(0.6 * Hg / ph, 0.6 * Wg / pw)
    for s in s0 * np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4]):
        tw, th = int(pw * s), int(ph * s)
        if tw < 16 or th < 16 or tw >= Wg or th >= Hg:
            continue
        t = cv2.resize(g1, (tw, th), interpolation=cv2.INTER_AREA)
        tm = cv2.resize(m1, (tw, th), interpolation=cv2.INTER_NEAREST)
        r = cv2.matchTemplate(g2, t, cv2.TM_CCORR_NORMED, mask=tm)
        r = np.nan_to_num(r, nan=0, posinf=0, neginf=0)
        _, v, _, loc = cv2.minMaxLoc(r)
        if best is None or v > best[0]:
            best = (v, s, loc)
    if best and best[0] > 0.6:
        v, s, (x, y) = best
        info.update(method="template", score=round(float(v), 3))
        return np.array([[s, 0, x], [0, s, y], [0, 0, 1]], np.float64), info
    if prior_M is not None:
        info.update(method="prior")
        return prior_M.copy(), info
    return None, info


def _reproj(Hm, src, dst):
    p = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    return float(np.median(np.linalg.norm(p - dst, axis=1)))


def _sane(Hm, pshape, gshape):
    """Reject degenerate/flipped/wildly-perspective homographies."""
    ph, pw = pshape
    c = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(c, Hm).reshape(-1, 2)
    if not cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2)):
        return False
    area = cv2.contourArea(q.astype(np.float32))
    if area < 0.002 * gshape[0] * gshape[1] or area > 1.2 * gshape[0] * gshape[1]:
        return False
    # aspect and angle sanity
    w1, w2 = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
    h1, h2 = np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])
    if min(w1, w2) / max(w1, w2, 1e-6) < 0.6 or min(h1, h2) / max(h1, h2, 1e-6) < 0.6:
        return False
    ar0 = pw / ph
    ar1 = ((w1 + w2) / 2) / max(1e-6, (h1 + h2) / 2)
    return 0.6 < ar1 / ar0 < 1.6


# -------------------------------------------------------------- harmonise
def harmonise(prod_rgb, prod_a, scene, amount=0.55):
    """Shift the real product's colour/brightness toward the scene lighting
    measured in a ring around it (Lab mean transfer, partial). Keeps texture
    and label detail; only global tone moves."""
    a = prod_a > 0.5
    if a.sum() < 20:
        return prod_rgb
    ring = (cv2.dilate(a.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & ~a
    if ring.sum() < 20:
        return prod_rgb
    lab_p = cv2.cvtColor(prod_rgb.astype(np.float32), cv2.COLOR_RGB2LAB)
    lab_s = cv2.cvtColor(scene.astype(np.float32), cv2.COLOR_RGB2LAB)
    # what the generator rendered *as the product* tells us the scene's light on it
    mp, ms = lab_p[a].mean(0), lab_s[a].mean(0)
    sp, ss = lab_p[a].std(0) + 1e-3, lab_s[a].std(0) + 1e-3
    delta_mean = (ms - mp) * np.array([amount, amount * 0.6, amount * 0.6])
    gain_L = 1 + (ss[0] / sp[0] - 1) * amount * 0.5
    out = lab_p.copy()
    out[..., 0] = (lab_p[..., 0] - mp[0]) * np.clip(gain_L, 0.8, 1.25) + mp[0] + delta_mean[0]
    out[..., 1] += delta_mean[1]
    out[..., 2] += delta_mean[2]
    out[..., 0] = np.clip(out[..., 0], 0, 100)
    return np.clip(cv2.cvtColor(out, cv2.COLOR_LAB2RGB), 0, 1)


# ---------------------------------------------------------------- restore
def restore(prod, alpha, gen, prior_M=None, harmonise_amount=0.55, edge_px=2.0):
    """Paste the *real* product pixels back onto the generated scene.

    Returns restored image, warped alpha, and a report with label-fidelity
    scores before/after (SSIM of the product region vs. the real product)."""
    Hg, Wg = gen.shape[:2]
    Hm, info = locate(prod, alpha, gen, prior_M)
    if Hm is None:
        return gen.copy(), np.zeros((Hg, Wg), np.float32), {**info, "ok": False}
    warp = lambda x, interp: cv2.warpPerspective(x, Hm, (Wg, Hg), flags=interp,  # noqa: E731
                                                 borderMode=cv2.BORDER_CONSTANT)
    p_rgb = warp(prod, cv2.INTER_CUBIC if abs(Hm[0, 0]) > 1 else cv2.INTER_AREA)
    p_a = warp(alpha, cv2.INTER_LINEAR)
    region = p_a > 0.5
    before = ssim(gen, p_rgb, p_a)
    p_rgb = harmonise(p_rgb, p_a, gen, harmonise_amount)
    # soft edge so the seam disappears into the generated rim
    a = cv2.GaussianBlur((p_a > 0.5).astype(np.float32), (0, 0), edge_px) * (p_a > 0.02)
    a = np.minimum(a, p_a + 0.0)
    a = np.maximum(a, cv2.erode((p_a > 0.5).astype(np.float32), np.ones((5, 5), np.uint8)))
    out = gen * (1 - a[..., None]) + p_rgb * a[..., None]
    after = ssim(out, warp(prod, cv2.INTER_CUBIC), p_a)
    x0, y0, x1, y1 = bbox(p_a)
    info.update(ok=True, fidelity_before=round(before, 3), fidelity_after=round(after, 3),
                box=[x0, y0, x1, y1], coverage=round(float(region.mean()), 4))
    return np.clip(out, 0, 1).astype(np.float32), p_a.astype(np.float32), info


# ------------------------------------------------------------------ visuals
def compare_card(real_crop, gen, restored, box, label=("AI only", "TrueShot")):
    """Side-by-side zoom on the product: generated vs restored."""
    x0, y0, x1, y1 = box
    pad = int(0.08 * max(x1 - x0, y1 - y0))
    H, W = gen.shape[:2]
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    a = gen[y0:y1, x0:x1]
    b = restored[y0:y1, x0:x1]
    h = 512
    s = h / max(1, a.shape[0])
    a = cv2.resize(a, (max(1, int(a.shape[1] * s)), h))
    b = cv2.resize(b, (max(1, int(b.shape[1] * s)), h))
    bar = np.ones((h, 12, 3), np.float32)
    card = np.concatenate([a, bar, b], 1)
    head = np.full((56, card.shape[1], 3), 0.07, np.float32)
    u8 = to_u8(head)
    cv2.putText(u8, label[0], (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 110, 110), 2, cv2.LINE_AA)
    cv2.putText(u8, label[1], (a.shape[1] + 24, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 255, 160), 2, cv2.LINE_AA)
    return np.concatenate([to_f(u8), card], 0)


def contact_sheet(images, names=None, cols=3, cell=420):
    tiles = []
    for i, im in enumerate(images):
        h, w = im.shape[:2]
        s = cell / max(h, w)
        t = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        c = np.full((cell + 34, cell, 3), 0.08, np.float32)
        y, x = (cell - t.shape[0]) // 2, (cell - t.shape[1]) // 2
        c[y:y + t.shape[0], x:x + t.shape[1]] = t
        if names:
            u8 = to_u8(c)
            cv2.putText(u8, str(names[i])[:38], (8, cell + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1,
                        cv2.LINE_AA)
            c = to_f(u8)
        tiles.append(c)
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        while len(row) < cols:
            row.append(np.full_like(tiles[0], 0.08))
        rows.append(np.concatenate(row, 1))
    return np.concatenate(rows, 0)
