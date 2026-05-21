"""
Utility functions for AudioEar standalone inference.
"""
import os

import cv2
import numpy as np
import torch
from torchvision import transforms


def save_obj(path, verts, faces):
    """Save mesh as Wavefront OBJ."""
    verts = verts.detach().cpu().numpy() if torch.is_tensor(verts) else verts
    faces = faces.detach().cpu().numpy() if torch.is_tensor(faces) else faces
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for fa in faces:
            f.write(f"f {fa[0]+1} {fa[1]+1} {fa[2]+1}\n")


def load_state_dict_flexible(path, map_location="cpu"):
    """Load state_dict from plain dict or wrapped dict."""
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model" in ckpt):
        ckpt = ckpt.get("state_dict", ckpt.get("model"))
    if isinstance(ckpt, dict):
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    return ckpt


def preprocess_image(img_path, img_size=256):
    """Load and preprocess an ear image."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img = img[:, :, ::-1]  # BGR -> RGB
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_t = transforms.ToTensor()(img).unsqueeze(0)
    return img_t


def preprocess_mask(mask_path, img_size=256):
    """Load and preprocess a binary mask."""
    if mask_path is None or not os.path.exists(mask_path):
        m = np.ones((img_size, img_size), dtype=np.float32)
    else:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        m = (m > 127).astype(np.float32)
    return torch.from_numpy(m)[None, None]
