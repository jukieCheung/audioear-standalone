# AudioEar Standalone Inference

Standalone Python package for **AudioEar 3D ear reconstruction(https://github.com/seanywang0408/AudioEar)** — with **no PyTorch3D dependency**.

This package provides a clean, portable inference pipeline that can be installed via `pip` and run on any machine with PyTorch. It also includes an optional **SSM latent refinement** module (trained on AudioEar3D) that improves morphological accuracy.

---

## Features

- **No PyTorch3D**: Pure PyTorch + NumPy implementation
- **Two modes**:
  - **Base**: Original AudioEar reconstruction
  - **Refined**: AudioEar + SSM latent correction for better morphology
- **CLI + Python API**: Use from command line or import into your code
- **Real-world scale**: Output meshes at true ear dimensions (~60 mm height)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/audioear-standalone.git
cd audioear-standalone
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

This registers the `audioear-inference` CLI command.

### 3. Download model weights

See [`weights/README.md`](weights/README.md). For weight download: (https://drive.google.com/file/d/1z_LDMSJJdjTAsSTKgDFwqPUbiEvvmNrQ/view?usp=sharing). Place them in the `weights/` directory.

---

## Quick Start

### CLI

```bash
# Basic inference
audioear-inference \
  --img ear.jpg \
  --mask mask.jpg \
  --fcrn weights/frcn.pt \
  --encoder weights/encoder.pt \
  --uhm weights/ear_model.pkl \
  --out_dir output/

# With SSM refinement (better morphology)
audioear-inference \
  --img ear.jpg \
  --mask mask.jpg \
  --fcrn weights/frcn.pt \
  --encoder weights/encoder.pt \
  --uhm weights/ear_model.pkl \
  --refine \
  --ssm_dir weights/ssm_audioear3d_v4 \
  --refiner weights/best_latent_only_v4.pt \
  --save_raw \
  --out_dir output/
```

### Python API

```python
from audioear_standalone import AudioEarInference

# Load model
model = AudioEarInference(
    fcrn_weights="weights/frcn.pt",
    encoder_weights="weights/encoder.pt",
    uhm_pkl="weights/ear_model.pkl",
    device="cuda",
)

# Optional: load SSM refinement
model.load_refiner(
    ssm_dir="weights/ssm_audioear3d_v4",
    checkpoint="weights/best_latent_only_v4.pt",
)

# Inference
verts, faces = model.predict("ear.jpg", mask="mask.jpg")
model.save_obj("output.obj", verts, faces)
```

See [`examples/example.py`](examples/example.py) for more usage patterns.

---

## Output Files

| File | Description |
|------|-------------|
| `final.obj` | Final mesh (refined if `--refine` is used) |
| `audioear_raw.obj` | Raw AudioEar output (only if `--save_raw`) |

All meshes are at **real-world scale** (height ~55-65 mm).

---

## pip Package vs ONNX Deployment

This package is distributed as a **Python pip package**. Here's how it compares to ONNX:

| Aspect | pip Package (this repo) | ONNX |
|---|---|---|
| **Runtime** | Python + PyTorch | ONNX Runtime (C/C++) |
| **Platforms** | Linux / macOS / Windows (with PyTorch) | Any (mobile, web, embedded) |
| **Setup** | `pip install .` | Link ONNX Runtime library |
| **Model parts** | Single Python package | Multiple `.onnx` files + glue code |
| **3DMM decode** | In Python (PyTorch tensors) | Must rewrite in target language |
| **Best for** | Researchers, Python developers | Production deployment, mobile apps |

### When to use pip

- You work in Python / PyTorch
- You want to quickly experiment with the model
- You need the full flexibility of the PyTorch ecosystem

### When to use ONNX

- You need to deploy on **mobile devices** (iOS/Android)
- You want to run in a **web browser** (via ONNX.js)
- Your production stack is **C++ / Java / C#**
- You need **maximum inference speed** with minimal dependencies

### Can this model be converted to ONNX?

**Partially.** The neural network parts (FCRN + Encoder + Refiner MLP) can be exported to ONNX using `torch.onnx.export()`. However, the **3DMM decode** step (`verts = mean + eigenvectors @ shape_vec`) is pure matrix algebra — you would need to reimplement it in your target language or export it as an ONNX `MatMul` node.

For most users, the pip package is the easiest path. ONNX conversion is recommended only if you have a specific cross-platform deployment need.

---

## ONNX Inference (Optional)

If you prefer ONNX Runtime over PyTorch:

### 1. Export ONNX models

```bash
python scripts/export_onnx.py \
  --fcrn weights/frcn.pt \
  --encoder weights/encoder.pt \
  --ssm_dir weights/ssm_audioear3d_v4 \
  --refiner_ckpt weights/best_latent_only_v4.pt \
  --out_dir weights/onnx
```

Outputs:
- `weights/onnx/fcrn.onnx` — Depth network
- `weights/onnx/encoder.onnx` — Shape encoder
- `weights/onnx/refiner.onnx` — Latent correction (optional)

### 2. Run ONNX inference

```bash
python scripts/onnx_inference.py \
  --img ear.jpg \
  --mask mask.jpg \
  --fcrn weights/onnx/fcrn.onnx \
  --encoder weights/onnx/encoder.onnx \
  --uhm weights/ear_model.pkl \
  --refine \
  --refiner weights/onnx/refiner.onnx \
  --ssm_dir weights/ssm_audioear3d_v4 \
  --out_dir output/
```

### Requirements for ONNX

```bash
pip install onnxruntime  # or onnxruntime-gpu for CUDA
```

> Note: The 3DMM decode step (`verts = mu + U @ shape_vec`) is **not exported to ONNX** — it is pure matrix math (~20 lines of numpy) and is included directly in `scripts/onnx_inference.py`. This keeps the ONNX graphs small and avoids dynamic-shape complications.

---

## Project Structure

```
audioear-standalone/
├── audioear_standalone/     # Core package
│   ├── __init__.py
│   ├── inference.py         # AudioEarInference class
│   ├── models.py            # ResNet_FCRN, FitModel, LatentRefiner
│   ├── fcrn.py              # FCRN depth network
│   └── utils.py             # save_obj, preprocess helpers
├── scripts/
│   └── run_inference.py     # CLI entry point
├── examples/
│   └── example.py           # Python API examples
├── tests/
│   └── test_import.py       # Import sanity check
├── weights/                 # Model weights (not in git)
├── setup.py
├── requirements.txt
└── README.md
```

---

## Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- torchvision >= 0.11.0
- OpenCV >= 4.5.0
- NumPy >= 1.21.0

---

## License

MIT License. See original AudioEar paper for model licensing.

---

## Citation

If you use this package, please cite the original AudioEar work:

```bibtex
@article{huang2023audioear,
  title={AudioEar: Single-View Ear Reconstruction for Personalized Spatial Audio},
  author={Huang, Xiaoyang and Wang, Yanjun and Liu, Yang and Ni, Bingbing and Zhang Wenjun and Liu Jinxian and Li, Teng},
  journal={arXiv preprint arXiv:2301.12613},
  year={2023}
}
```
