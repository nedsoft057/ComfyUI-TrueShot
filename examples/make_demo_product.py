"""Write the built-in demo 'phone photo' (a labelled serum bottle on a table)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
import cv2
from synth import product_rgba, phone_photo
out = sys.argv[1] if len(sys.argv) > 1 else "demo_product.png"
img = phone_photo(product_rgba())
cv2.imwrite(out, cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2BGR))
print("wrote", out)
