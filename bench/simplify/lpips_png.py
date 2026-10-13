# Copyright (c) 2026 jesus luque.
"""LPIPS (AlexNet) between two 8-bit sRGB PNGs, for validate.py: a Python with torch, torchvision and
lpips (LPIPS_PYTHON). Prints the distance last on its line.

    python lpips_png.py reference.png frame.png
"""
import sys

import lpips
import numpy as np
import torch
from PIL import Image

torch.set_num_threads(2)


def tensor(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None] * 2.0 - 1.0


net = lpips.LPIPS(net="alex", verbose=False)
with torch.no_grad():
    print(f"lpips {net(tensor(sys.argv[1]), tensor(sys.argv[2])).item():.4f}")
