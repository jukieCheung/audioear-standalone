#!/usr/bin/env python3
"""
CLI entry point for AudioEar standalone inference.

Usage:
    audioear-inference \
        --img ear.jpg \
        --mask mask.jpg \
        --fcrn weights/frcn.pt \
        --encoder weights/encoder.pt \
        --uhm weights/ear_model.pkl \
        --out_dir output/

With SSM refinement:
    audioear-inference \
        --img ear.jpg \
        --mask mask.jpg \
        --fcrn weights/frcn.pt \
        --encoder weights/encoder.pt \
        --uhm weights/ear_model.pkl \
        --refine \
        --ssm_dir weights/ssm_audioear3d_v4 \
        --refiner weights/best_latent_only_v4.pt \
        --out_dir output/
"""
import argparse
import os

import numpy as np

from audioear_standalone import AudioEarInference


def main():
    parser = argparse.ArgumentParser(description="AudioEar 3D ear reconstruction")
    parser.add_argument("--img", required=True, help="Input ear image path")
    parser.add_argument("--mask", default=None, help="Binary mask path (optional)")
    parser.add_argument("--out_dir", default="./audioear_output", help="Output directory")
    parser.add_argument("--fcrn", default="weights/frcn.pt", help="FCRN weights")
    parser.add_argument("--encoder", default="weights/encoder.pt", help="Encoder weights")
    parser.add_argument("--uhm", default="weights/ear_model.pkl", help="UHM 3DMM pickle")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--left_ear", action="store_true", help="Mirror for left ear")

    # Refinement options
    parser.add_argument("--refine", action="store_true", help="Apply SSM refinement")
    parser.add_argument("--ssm_dir", default="weights/ssm_audioear3d_v4", help="SSM directory")
    parser.add_argument("--refiner", default="weights/best_latent_only_v4.pt", help="Refiner checkpoint")
    parser.add_argument("--save_raw", action="store_true", help="Also save raw AudioEar mesh")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Validate weight paths
    for p, name in [(args.fcrn, "FCRN"), (args.encoder, "Encoder"), (args.uhm, "UHM")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} weights not found: {p}")

    # Load model
    print("Loading AudioEar model...")
    model = AudioEarInference(
        fcrn_weights=args.fcrn,
        encoder_weights=args.encoder,
        uhm_pkl=args.uhm,
        device=args.device,
        img_size=args.img_size,
    )

    # Load refiner if requested
    if args.refine:
        if not os.path.exists(args.ssm_dir):
            raise FileNotFoundError(f"SSM directory not found: {args.ssm_dir}")
        if not os.path.exists(args.refiner):
            raise FileNotFoundError(f"Refiner checkpoint not found: {args.refiner}")
        print("Loading SSM refinement model...")
        model.load_refiner(args.ssm_dir, args.refiner)

    # Inference
    print(f"Processing: {args.img}")
    result = model.predict(
        args.img,
        mask_path=args.mask,
        left_ear=args.left_ear,
        return_raw=args.save_raw,
    )

    if args.save_raw and args.refine:
        verts, faces, raw_verts = result
        raw_path = os.path.join(args.out_dir, "audioear_raw.obj")
        model.save_obj(raw_path, raw_verts, faces)
        print(f"  Saved raw mesh: {raw_path}")
    else:
        verts, faces = result

    out_path = os.path.join(args.out_dir, "final.obj")
    model.save_obj(out_path, verts, faces)
    print(f"Saved refined mesh: {out_path}")

    # Print dimensions
    h = float(verts[:, 1].max() - verts[:, 1].min())
    w = float(verts[:, 0].max() - verts[:, 0].min())
    d = float(verts[:, 2].max() - verts[:, 2].min())
    print(f"Dimensions: H={h:.1f}mm W={w:.1f}mm D={d:.1f}mm")


if __name__ == "__main__":
    main()
