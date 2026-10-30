import sys, os, time
sys.path[:0] = [os.path.dirname(os.path.dirname(__file__)), os.path.dirname(__file__)]
import numpy as np, cv2
from synth import product_rgba, phone_photo, fake_ai_render
from trueshot import core
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ts_test"; os.makedirs(OUT, exist_ok=True)
save = lambda n, im: cv2.imwrite(f"{OUT}/{n}.png", cv2.cvtColor(core.to_u8(im), cv2.COLOR_RGB2BGR))
p = product_rgba(); photo = phone_photo(p)
t = time.time(); a = core.cutout(photo, "grabcut"); print("cutout", round(time.time()-t, 2), "s")
# matte quality vs ground truth
gt = np.zeros(photo.shape[:2], np.float32); ph, pw = p.shape[:2]; x0, y0 = (900-pw)//2+20, (1200-ph)//2+40
gt[y0:y0+ph, x0:x0+pw] = p[..., 3]
iou = ((a > .5) & (gt > .5)).sum() / ((a > .5) | (gt > .5)).sum(); print("matte IoU", round(float(iou), 3)); assert iou > 0.9
prod, pa = core.crop_product(photo, a)
save("marketplace", core.marketplace(prod, pa, 800))
for i, (sc, ang) in enumerate([(0.55, 3.0), (0.4, -6.0), (0.7, 0.0)]):
    gen, M = fake_ai_render(prod, pa, scale=sc, angle=ang, seed=i)
    t = time.time(); out, wa, rep = core.restore(prod, pa, gen); dt = time.time()-t
    print(i, rep["method"], rep.get("inliers"), "fidelity", rep["fidelity_before"], "->", rep["fidelity_after"], f"{dt:.2f}s")
    assert rep["ok"] and rep["fidelity_after"] > rep["fidelity_before"] + 0.2
    save(f"gen{i}", gen); save(f"restored{i}", out); save(f"compare{i}", core.compare_card(prod, gen, out, rep["box"]))
print("OK")
