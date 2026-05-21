#!/usr/bin/env python3
"""
Export AudioEar models to ONNX.

Outputs:
    fcrn.onnx       - FCRN depth network
    encoder.onnx    - ResNet_FCRN encoder (image + skip features -> shape_vec)
    refiner.onnx    - Latent correction MLP (optional)

Note: 3DMM decode (FitModel) is NOT exported — it is pure matrix math
      and is trivial to reimplement in numpy or any target language.
"""
import os
import sys
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from audioear_standalone.fcrn import ResNet as FCRNResNet
from audioear_standalone.models import ResNet_FCRN, LatentRefiner
from audioear_standalone.utils import load_state_dict_flexible

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256


def export_fcrn(fcrn_weights, out_path):
    """Export FCRN depth network."""
    model = FCRNResNet(layers=34, output_size=(IMG_SIZE, IMG_SIZE))
    model.load_state_dict(load_state_dict_flexible(fcrn_weights), strict=True)
    model.to(DEVICE).eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    # FCRN returns (depth, [f0, f1, f2, f3, f4]) — ONNX doesn't like list outputs well
    # We wrap it to return a flat tuple
    class FCRNWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            depth, feats = self.model(x)
            return (depth, *feats)

    wrapper = FCRNWrapper(model).to(DEVICE).eval()
    with torch.no_grad():
        outputs = wrapper(dummy_input)
    output_names = ["depth", "feat0", "feat1", "feat2", "feat3", "feat4"]

    torch.onnx.export(
        wrapper,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["image"],
        output_names=output_names,
        dynamic_axes={"image": {0: "batch_size"}},
    )
    print(f"Exported FCRN -> {out_path}")


def export_encoder(encoder_weights, out_path):
    """Export ResNet_FCRN encoder.

    ONNX does not support list-of-tensor inputs well, so we flatten to
    6 separate inputs: image + 5 feature maps.
    """
    model = ResNet_FCRN()
    model.load_state_dict(load_state_dict_flexible(encoder_weights), strict=True)
    model.to(DEVICE).eval()

    # Wrapper that takes flat inputs instead of a list
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, image, f0, f1, f2, f3, f4):
            feats = [f0, f1, f2, f3, f4]
            pos_vec, tex, shape_vec = self.model(image, feats)
            # Only export shape_vec (the one used for mesh decode)
            return shape_vec

    wrapper = EncoderWrapper(model).to(DEVICE).eval()

    dummy_image = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    # Order matches FCRN output: [feat0, feat1, feat2, feat3, feat4]
    # ResNet_FCRN uses feat[-1], feat[-2], feat[-3], feat[-4]
    dummy_f0 = torch.randn(1, 256, 8, 8, device=DEVICE)      # feat[0] (unused by encoder)
    dummy_f1 = torch.randn(1, 128, 16, 16, device=DEVICE)    # feat[1] -> fcrn_feat[-4]
    dummy_f2 = torch.randn(1, 64, 32, 32, device=DEVICE)     # feat[2] -> fcrn_feat[-3]
    dummy_f3 = torch.randn(1, 32, 64, 64, device=DEVICE)     # feat[3] -> fcrn_feat[-2]
    dummy_f4 = torch.randn(1, 16, 128, 128, device=DEVICE)   # feat[4] -> fcrn_feat[-1]

    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_f0, dummy_f1, dummy_f2, dummy_f3, dummy_f4),
        out_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["image", "feat0", "feat1", "feat2", "feat3", "feat4"],
        output_names=["shape_vec"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "feat0": {0: "batch_size"},
            "feat1": {0: "batch_size"},
            "feat2": {0: "batch_size"},
            "feat3": {0: "batch_size"},
            "feat4": {0: "batch_size"},
            "shape_vec": {0: "batch_size"},
        },
    )
    print(f"Exported Encoder -> {out_path}")


def export_refiner(ssm_dir, checkpoint, out_path):
    """Export LatentRefiner to ONNX."""
    import numpy as np
    eigenvalues = np.load(os.path.join(ssm_dir, "pca_eigenvalues.npy"))
    valid_mask = eigenvalues > 1e-6
    K = int(valid_mask.sum())

    model = LatentRefiner(n_modes=K)
    ckpt = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()

    dummy_input = torch.randn(1, K, device=DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["z_pred"],
        output_names=["z_refined"],
        dynamic_axes={"z_pred": {0: "batch_size"}, "z_refined": {0: "batch_size"}},
    )
    print(f"Exported Refiner -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fcrn", default="weights/frcn.pt")
    parser.add_argument("--encoder", default="weights/encoder.pt")
    parser.add_argument("--ssm_dir", default="weights/ssm_audioear3d_v4")
    parser.add_argument("--refiner_ckpt", default="weights/best_latent_only_v4.pt")
    parser.add_argument("--out_dir", default="weights/onnx")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    export_fcrn(args.fcrn, os.path.join(args.out_dir, "fcrn.onnx"))
    export_encoder(args.encoder, os.path.join(args.out_dir, "encoder.onnx"))
    export_refiner(args.ssm_dir, args.refiner_ckpt,
                   os.path.join(args.out_dir, "refiner.onnx"))

    print(f"\nAll ONNX models saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
