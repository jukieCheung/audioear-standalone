# Model Weights

This directory should contain the following model weights. They are **not included in the repository** due to file size.

## Required Weights

| File | Size | Source |
|------|------|--------|
| `frcn.pt` | ~90 MB | AudioEar FCRN depth network |
| `encoder.pt` | ~50 MB | AudioEar ResNet_FCRN encoder |
| `ear_model.pkl` | ~20 MB | UHM 3DMM (mean, eigenvectors, eigenvalues, faces) |

## Optional Refinement Weights (for better morphology)

| File | Size | Source |
|------|------|--------|
| `ssm_audioear3d_v4/mean_shape.npy` | ~70 KB | SSM mean shape |
| `ssm_audioear3d_v4/pca_basis.npy` | ~240 KB | SSM PCA basis |
| `ssm_audioear3d_v4/pca_eigenvalues.npy` | ~1 KB | SSM eigenvalues |
| `best_latent_only_v4.pt` | ~150 KB | Trained latent correction model |

## How to Obtain

1. **Original AudioEar weights**: Download from the AudioEar project page or train using their codebase.
2. **SSM + refinement weights**: These were trained on the AudioEar3D dataset. Contact the authors or train your own using `build_ssm_audioear3d_v4.py` and `train_latent_only_v4.py`.

## Place Files Here

Place all `.pt`, `.pkl`, and `.npy` files in this directory (or subdirectories) so the scripts can find them:

```
weights/
├── frcn.pt
├── encoder.pt
├── ear_model.pkl
├── ssm_audioear3d_v4/
│   ├── mean_shape.npy
│   ├── pca_basis.npy
│   └── pca_eigenvalues.npy
└── best_latent_only_v4.pt
```
