# TrueShot on a free Kaggle T4 — paste into ONE notebook cell (or use the 10-line bootstrap in the README).
# Notebook settings: Accelerator = GPU T4 x2 (or P100), Internet = On.
# ---------------------------------------------------------------- settings
IMAGE_URL = ""   # direct link to your product photo. Empty = built-in demo product.
PRODUCT   = "a bottle of LUMA vitamin C serum"
BRIEF     = "on a marble bathroom counter with eucalyptus | on wet black stones with water splashes | on a sunlit beach towel next to sunglasses"
FORMATS   = "lifestyle,lifestyle,lifestyle,story,banner"
VARIANTS  = 1
# -------------------------------------------------------------------------
import os, subprocess, sys, time, urllib.request, shutil, glob

W = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
ROOT = "/tmp/bc"
COMFY = f"{ROOT}/ComfyUI"
os.makedirs(ROOT, exist_ok=True)

def stream(cmd, shell=False):
    """Run a command and stream its output into the notebook cell."""
    p = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()

def sh(cmd, check=True):
    print("$", cmd, flush=True)
    rc = stream(cmd, shell=True)
    if check and rc:
        raise SystemExit(f"command failed: {cmd}")

t0 = time.time()
if not os.path.isdir(COMFY):
    sh(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git {COMFY}")
# keep Kaggle's CUDA torch/torchvision/torchaudio; install everything else
# (careful: torchsde etc. also start with "torch")
import re
KEEP = {"torch", "torchvision", "torchaudio"}
req = []
for l in open(f"{COMFY}/requirements.txt").read().splitlines():
    l = l.strip()
    if not l or l.startswith("#"):
        continue
    name = re.split(r"[<>=!~\[; ]", l, 1)[0].lower()
    if name not in KEEP:
        req.append(l)
open(f"{ROOT}/req.txt", "w").write("\n".join(req))
sh(f"{sys.executable} -m pip install -q -r {ROOT}/req.txt")
nodes = f"{COMFY}/custom_nodes/ComfyUI-TrueShot"
if os.path.isdir(nodes):
    sh(f"git -C {nodes} pull -q")
else:
    sh(f"git clone --depth 1 https://github.com/nedsoft057/ComfyUI-TrueShot.git {nodes}")
sh(f"{sys.executable} -m pip install -q -r {nodes}/requirements.txt huggingface_hub")
# optional, better cutouts (falls back to GrabCut if this fails)
sh(f"{sys.executable} -m pip install -q rembg onnxruntime", check=False)

# ---------------------------------------------------------------- models
from huggingface_hub import hf_hub_download
token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

MODELS = [  # (folder, final name, [(repo, path-in-repo), ...fallbacks])
    ("diffusion_models", "flux-2-klein.safetensors",
     [("black-forest-labs/FLUX.2-klein-4b-fp8", "flux-2-klein-4b-fp8.safetensors"),
      ("black-forest-labs/FLUX.2-klein-4B", "flux-2-klein-4b.safetensors"),
      ("black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors")]),
    ("text_encoders", "qwen_3_4b.safetensors",
     [("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors")]),
    ("vae", "flux2-vae.safetensors",
     [("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors")]),
]
for folder, name, cands in MODELS:
    dst = f"{COMFY}/models/{folder}/{name}"
    if os.path.exists(dst):
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for repo, path in cands:
        try:
            print(f"downloading {repo}/{path}", flush=True)
            p = hf_hub_download(repo, path, token=token, cache_dir=f"{ROOT}/hf")
            os.symlink(p, dst)
            open(dst + ".source", "w").write(path)
            break
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {str(e)[:200]}")
    else:
        raise SystemExit(f"Could not download {name}. If it is gated, add an HF_TOKEN Kaggle secret.")
print(f"setup done in {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- inputs
if IMAGE_URL:
    image = f"{ROOT}/product" + (os.path.splitext(IMAGE_URL.split("?")[0])[1] or ".jpg")
    req = urllib.request.Request(IMAGE_URL, headers={"User-Agent": "Mozilla/5.0"})
    open(image, "wb").write(urllib.request.urlopen(req).read())
else:
    image = f"{ROOT}/demo_product.png"
    sh(f"{sys.executable} {nodes}/examples/make_demo_product.py {image}")

# ---------------------------------------------------------------- server
import socket
def up():
    try:
        socket.create_connection(("127.0.0.1", 8188), 1).close(); return True
    except OSError:
        return False
import torch
if not torch.cuda.is_available():
    raise SystemExit("NO GPU in this session. Notebook panel -> Session options -> Accelerator: GPU T4 x2, then run again.")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
if not up():
    log = open(f"{ROOT}/comfy.log", "w")
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch",
                      "--disable-pinned-memory", "--cache-none"], cwd=COMFY, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(180):
        if up(): break
        if proc.poll() is not None:
            print(open(f"{ROOT}/comfy.log").read()[-4000:]); raise SystemExit("ComfyUI crashed on start (log above)")
        time.sleep(2)
    else:
        print(open(f"{ROOT}/comfy.log").read()[-4000:]); raise SystemExit("ComfyUI did not start")
print("ComfyUI is up", flush=True)

# ---------------------------------------------------------------- render
klein_src = open(f"{COMFY}/models/diffusion_models/flux-2-klein.safetensors.source").read()
steps, cfg = ("10", "3.5") if "base" in klein_src else ("4", "1.0")
print("keyframe model:", klein_src, "steps", steps, "cfg", cfg, flush=True)
args = [sys.executable, f"{nodes}/scripts/run_trueshot.py", "--comfy-dir", COMFY, "--image", image,
        "--product", PRODUCT, "--brief", BRIEF, "--formats", FORMATS, "--variants", str(VARIANTS),
        "--klein-steps", steps, "--klein-cfg", cfg]
rc = stream(args)
for d in glob.glob(f"{COMFY}/output/TrueShot/*"):
    shutil.copytree(d, f"{W}/TrueShot/{os.path.basename(d)}", dirs_exist_ok=True)
if rc:
    print("---- last ComfyUI log lines ----"); print(open(f"{ROOT}/comfy.log").read()[-5000:])
print(f"total {time.time() - t0:.0f}s — results in {W}/TrueShot/")
for f in sorted(glob.glob(f"{W}/TrueShot/*/*.png")): print(f)
