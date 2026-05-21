#!/usr/bin/env python3
"""
ONNX Runtime inference for AudioEar.

Requires: onnxruntime (pip install onnxruntime)

Pipeline:
    Image + Mask
        -> FCRN ONNX       -> depth + skip features
        -> Encoder ONNX    -> shape_vec
        -> 3DMM decode     -> mesh vertices (numpy, no PyTorch3D)
        -> (optional) Refiner ONNX -> corrected z
        -> save OBJ
"""
import os
import argparse

import cv2
import numpy as np
import onnxruntime as ort
from torchvision import transforms


def preprocess_image(img_path, img_size=256):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img = img[:, :, ::-1]
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_t = transforms.ToTensor()(img).unsqueeze(0).numpy()
    return img_t.astype(np.float32)


def preprocess_mask(mask_path, img_size=256):
    if mask_path is None or not os.path.exists(mask_path):
        m = np.ones((img_size, img_size), dtype=np.float32)
    else:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        m = (m > 127).astype(np.float32)
    return m[np.newaxis, np.newaxis, ...].astype(np.float32)


def save_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for fa in faces:
            f.write(f"f {fa[0]+1} {fa[1]+1} {fa[2]+1}\n")


def look_at_rotation_np(camera_position, at=(0, 0, 0), up=(0, 1, 0)):
    """Numpy look-at rotation (same logic as PyTorch3D)."""
    cam = np.array(camera_position, dtype=np.float32)
    at = np.array(at, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    z_axis = at - cam
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-5)

    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-5)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-5)

    if np.allclose(x_axis, 0, atol=5e-3):
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-5)

    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    return R.T


def decode_3dmm(shape_vec, mu, U, V, faces,
                cam_pos=(0.0, 0.0, 0.0),
                look_at=(0.0, 0.0, 1.0),
                up=(0.0, 1.0, 0.0),
                scale_factor=120.0):
    """
    Decode shape_vec to 3D mesh vertices using UHM 3DMM.
    Pure numpy — no PyTorch3D.
    """
    K = U.shape[1]
    shape_vec = shape_vec.reshape(1, K)

    # verts = mu + U @ (shape_vec * V).T
    coeffs = (shape_vec * V).T  # (K, 1)
    verts = mu + U @ coeffs     # (N*3, 1)
    verts = verts.reshape(-1, 3)

    # Scale
    verts = np.maximum(0.0, scale_factor) * verts

    # Look-at rotation
    R = look_at_rotation_np(cam_pos, look_at, up)
    T = -(R.T @ np.array(cam_pos)).reshape(1, 3)

    verts = verts @ R + T
    return verts


class AudioEarONNX:
    """End-to-end AudioEar inference using ONNX Runtime."""

    def __init__(self, fcrn_onnx, encoder_onnx, uhm_pkl,
                 refiner_onnx=None, ssm_dir=None,
                 providers=None):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.fcrn_sess = ort.InferenceSession(fcrn_onnx, providers=providers)
        self.enc_sess = ort.InferenceSession(encoder_onnx, providers=providers)

        # Load UHM 3DMM
        import pickle
        with open(uhm_pkl, "rb") as f:
            ear_model = pickle.load(f)
        self.mu = ear_model["Mean"].reshape(-1, 3).astype(np.float32)
        self.mu = (self.mu - self.mu.mean(axis=0)).reshape(-1, 1)
        self.U = ear_model["Eigenvectors"].astype(np.float32)
        self.V = ear_model["EigenValues"].astype(np.float32)
        self.faces = ear_model["Trilist"].astype(np.int32)

        # Optional refiner
        self.refiner_sess = None
        self.mean_flat = None
        self.B = None
        if refiner_onnx is not None and ssm_dir is not None:
            self.refiner_sess = ort.InferenceSession(refiner_onnx, providers=providers)
            eigenvalues = np.load(os.path.join(ssm_dir, "pca_eigenvalues.npy"))
            B_full = np.load(os.path.join(ssm_dir, "pca_basis.npy"))
            valid_mask = eigenvalues > 1e-6
            K = int(valid_mask.sum())
            self.B = B_full[:K, :].astype(np.float32)
            mean_shape = np.load(os.path.join(ssm_dir, "mean_shape.npy"))
            self.mean_flat = mean_shape.reshape(-1).astype(np.float32)

    def predict(self, img_path, mask_path=None, left_ear=False, return_raw=False):
        img = preprocess_image(img_path)
        mask = preprocess_mask(mask_path)

        if left_ear:
            img = np.flip(img, axis=3)
            mask = np.flip(mask, axis=3)

        masked_img = img * mask

        # ---- FCRN ----
        fcrn_out = self.fcrn_sess.run(None, {"image": masked_img})
        depth = fcrn_out[0]
        feat1 = fcrn_out[2]   # fcrn_feat[1] -> encoder feat1
        feat2 = fcrn_out[3]   # fcrn_feat[2] -> encoder feat2
        feat3 = fcrn_out[4]   # fcrn_feat[3] -> encoder feat3
        feat4 = fcrn_out[5]   # fcrn_feat[4] -> encoder feat4

        # ---- Encoder ----
        enc_out = self.enc_sess.run(
            None,
            {"image": masked_img, "feat1": feat1, "feat2": feat2,
             "feat3": feat3, "feat4": feat4}
        )
        shape_vec = enc_out[0]  # (1, 236)

        # ---- 3DMM decode ----
        verts = decode_3dmm(shape_vec[0], self.mu, self.U, self.V, self.faces)

        # ---- Optional refinement ----
        if self.refiner_sess is not None:
            raw_verts = verts.copy()
            N_VERTS = self.mean_flat.shape[0] // 3
            verts_flat = verts.reshape(-1)
            z_pred = self.B @ (verts_flat - self.mean_flat)

            ref_out = self.refiner_sess.run(None, {"z_pred": z_pred.astype(np.float32)[np.newaxis, :]})
            z_refined = ref_out[0][0]

            delta = z_refined @ self.B
            verts_refined_flat = self.mean_flat + delta
            verts = verts_refined_flat.reshape(N_VERTS, 3)

            if return_raw:
                return verts, self.faces, raw_verts

        return verts, self.faces


def main():
    parser = argparse.ArgumentParser(description="AudioEar ONNX inference")
    parser.add_argument("--img", required=True)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--out_dir", default="./audioear_onnx_output")
    parser.add_argument("--fcrn", default="weights/onnx/fcrn.onnx")
    parser.add_argument("--encoder", default="weights/onnx/encoder.onnx")
    parser.add_argument("--uhm", default="weights/ear_model.pkl")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--refiner", default="weights/onnx/refiner.onnx")
    parser.add_argument("--ssm_dir", default="weights/ssm_audioear3d_v4")
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--left_ear", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = AudioEarONNX(
        fcrn_onnx=args.fcrn,
        encoder_onnx=args.encoder,
        uhm_pkl=args.uhm,
        refiner_onnx=args.refiner if args.refine else None,
        ssm_dir=args.ssm_dir if args.refine else None,
    )

    print(f"Processing: {args.img}")
    result = model.predict(args.img, mask_path=args.mask,
                           left_ear=args.left_ear, return_raw=args.save_raw)

    if args.save_raw and args.refine:
        verts, faces, raw_verts = result
        save_obj(os.path.join(args.out_dir, "audioear_raw.obj"), raw_verts, faces)
        print(f"  Saved raw mesh")
    else:
        verts, faces = result

    save_obj(os.path.join(args.out_dir, "final.obj"), verts, faces)
    print(f"Saved refined mesh")

    h = float(verts[:, 1].max() - verts[:, 1].min())
    w = float(verts[:, 0].max() - verts[:, 0].min())
    d = float(verts[:, 2].max() - verts[:, 2].min())
    print(f"Dimensions: H={h:.1f}mm W={w:.1f}mm D={d:.1f}mm")


if __name__ == "__main__":
    main()
