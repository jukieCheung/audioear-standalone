"""
Main inference pipeline for AudioEar standalone.

Usage:
    model = AudioEarInference(
        fcrn_weights="weights/frcn.pt",
        encoder_weights="weights/encoder.pt",
        uhm_pkl="weights/ear_model.pkl",
        device="cuda"
    )
    verts, faces = model.predict("image.jpg", mask="mask.jpg")
    model.save_obj("output.obj", verts, faces)
"""
import os
import pickle

import numpy as np
import torch

from .fcrn import ResNet as FCRNResNet
from .models import FitModel, LatentRefiner, ResNet_FCRN
from .utils import load_state_dict_flexible, preprocess_image, preprocess_mask


class AudioEarInference:
    """
    End-to-end AudioEar inference pipeline (no PyTorch3D).

    Args:
        fcrn_weights: Path to FCRN depth network weights (.pt)
        encoder_weights: Path to ResNet_FCRN encoder weights (.pt)
        uhm_pkl: Path to UHM 3DMM pickle file
        device: "cuda" or "cpu"
        img_size: Input image size (default 256)
    """

    def __init__(
        self,
        fcrn_weights,
        encoder_weights,
        uhm_pkl,
        device="cuda",
        img_size=256,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = img_size

        # Load FCRN depth network
        self.fcrn_model = FCRNResNet(layers=34, output_size=(256, 256))
        self.fcrn_model.load_state_dict(
            load_state_dict_flexible(fcrn_weights, map_location="cpu"),
            strict=True,
        )
        self.fcrn_model.to(self.device).eval()

        # Load encoder
        self.encoder = ResNet_FCRN()
        self.encoder.load_state_dict(
            load_state_dict_flexible(encoder_weights, map_location="cpu"),
            strict=True,
        )
        self.encoder.to(self.device).eval()

        # Load 3DMM
        with open(uhm_pkl, "rb") as f:
            ear_model = pickle.load(f)
        mu = torch.tensor(ear_model["Mean"]).float().reshape(-1, 3)
        mu = (mu - mu.mean(0)).reshape(-1, 1).to(self.device)
        self.mu = mu
        self.U = torch.tensor(ear_model["Eigenvectors"]).float().to(self.device)
        self.V = torch.tensor(ear_model["EigenValues"]).float().to(self.device)
        self.faces = torch.from_numpy(ear_model["Trilist"]).to(self.device)

        # Optional refiner
        self.refiner = None
        self.mean_flat = None
        self.B = None

    def load_refiner(self, ssm_dir, checkpoint):
        """
        Load optional SSM latent refinement model (v4).

        Args:
            ssm_dir: Directory containing mean_shape.npy, pca_basis.npy, pca_eigenvalues.npy
            checkpoint: Path to trained refiner checkpoint (.pt)
        """
        mean_shape = np.load(os.path.join(ssm_dir, "mean_shape.npy"))
        B = np.load(os.path.join(ssm_dir, "pca_basis.npy"))
        eigenvalues = np.load(os.path.join(ssm_dir, "pca_eigenvalues.npy"))

        valid_mask = eigenvalues > 1e-6
        K = int(valid_mask.sum())
        eigenvalues = eigenvalues[:K]
        B = B[:K, :]

        self.B = torch.from_numpy(B).float().to(self.device)
        self.mean_flat = torch.from_numpy(mean_shape.reshape(-1)).float().to(
            self.device
        )

        self.refiner = LatentRefiner(n_modes=K)
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.refiner.load_state_dict(ckpt["model_state_dict"])
        self.refiner.to(self.device).eval()

    @torch.no_grad()
    def predict(self, img_path, mask_path=None, left_ear=False, return_raw=False):
        """
        Run inference on a single image.

        Args:
            img_path: Path to input ear image
            mask_path: Path to binary mask (optional)
            left_ear: Whether this is a left ear (will mirror horizontally)
            return_raw: If True and refiner is loaded, also return raw AudioEar vertices

        Returns:
            verts: (N, 3) mesh vertices as numpy array
            faces: (F, 3) face indices as numpy array
            raw_verts: (N, 3) raw AudioEar vertices (only if return_raw=True and refiner loaded)
        """
        img_t = preprocess_image(img_path, self.img_size).to(self.device)
        mask_t = preprocess_mask(mask_path, self.img_size).to(self.device)

        if left_ear:
            img_t = torch.flip(img_t, dims=[3])
            mask_t = torch.flip(mask_t, dims=[3])

        masked_img = img_t * mask_t

        depth, fcrn_feat = self.fcrn_model(masked_img)
        pos_vec, tex, shape_vec = self.encoder(masked_img, fcrn_feat)

        model = FitModel(
            ear_mu=self.mu,
            ear_eigenvectors=self.U,
            V=self.V,
            shape_vec=shape_vec,
            shape_vec_value="pred",
        ).to(self.device)
        verts, R, T = model()
        verts_np = verts.detach().cpu().numpy()

        if self.refiner is not None:
            raw_verts = verts_np.copy()
            N_VERTS = self.mean_flat.shape[0] // 3
            verts_flat = verts.reshape(-1)
            z_pred = self.B @ (verts_flat - self.mean_flat)
            z_refined = self.refiner(z_pred.unsqueeze(0))[0]
            delta = z_refined @ self.B
            verts_refined_flat = self.mean_flat + delta
            verts = verts_refined_flat.view(N_VERTS, 3)
            verts_np = verts.detach().cpu().numpy()
            if return_raw:
                return verts_np, self.faces.cpu().numpy(), raw_verts

        return verts_np, self.faces.cpu().numpy()

    def save_obj(self, path, verts, faces):
        """Save vertices and faces to a Wavefront OBJ file."""
        from .utils import save_obj as _save_obj

        _save_obj(path, verts, faces)
