#!/usr/bin/env python3
"""
Example: Python API usage for AudioEar standalone inference.
"""
from audioear_standalone import AudioEarInference

# ---------------------------------------------------------------------------
# 1. Basic inference (no refinement)
# ---------------------------------------------------------------------------
model = AudioEarInference(
    fcrn_weights="weights/frcn.pt",
    encoder_weights="weights/encoder.pt",
    uhm_pkl="weights/ear_model.pkl",
    device="cuda",
)

verts, faces = model.predict("ear.jpg", mask="mask.jpg")
model.save_obj("output.obj", verts, faces)

# ---------------------------------------------------------------------------
# 2. With SSM latent refinement (better morphology)
# ---------------------------------------------------------------------------
model.load_refiner(
    ssm_dir="weights/ssm_audioear3d_v4",
    checkpoint="weights/best_latent_only_v4.pt",
)

verts, faces, raw_verts = model.predict(
    "ear.jpg",
    mask="mask.jpg",
    left_ear=False,
    return_raw=True,
)

model.save_obj("output_refined.obj", verts, faces)
model.save_obj("output_raw.obj", raw_verts, faces)

# ---------------------------------------------------------------------------
# 3. Batch inference
# ---------------------------------------------------------------------------
import glob

for img_path in glob.glob("images/*.jpg"):
    name = img_path.split("/")[-1].replace(".jpg", "")
    verts, faces = model.predict(img_path)
    model.save_obj(f"outputs/{name}.obj", verts, faces)
    print(f"Done: {name}")
