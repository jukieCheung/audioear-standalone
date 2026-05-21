"""
Model definitions for AudioEar standalone inference.

Includes:
- ResNet_FCRN: Encoder that predicts 3DMM parameters from image + depth features
- FitModel: 3DMM decoder with look-at camera rotation
- LatentRefiner: Optional SSM-based latent correction (v4)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ResNet_FCRN (copied from AudioEar/models.py — no PyTorch3D dependency)
# ---------------------------------------------------------------------------
class ResNet_FCRN(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        self.layer1_conv = nn.Conv2d(64 + 16, 64, 3, 1, 1)
        self.layer2_conv = nn.Conv2d(64 + 32, 64, 3, 1, 1)
        self.layer3_conv = nn.Conv2d(128 + 64, 128, 3, 1, 1)
        self.layer4_conv = nn.Conv2d(256 + 128, 256, 3, 1, 1)

        self.fc = nn.Linear(512, 34)
        self.fc_tex = nn.Linear(512, 50)
        self.shape_fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 236),
        )

    def forward(self, x, fcrn_feat):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1_conv(torch.cat([x, self.maxpool(fcrn_feat[-1])], 1))
        x = self.layer1(x)
        x = self.layer2_conv(torch.cat([x, fcrn_feat[-2]], 1))
        x = self.layer2(x)
        x = self.layer3_conv(torch.cat([x, fcrn_feat[-3]], 1))
        x = self.layer3(x)
        x = self.layer4_conv(torch.cat([x, fcrn_feat[-4]], 1))
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        pos_vec = self.fc(x)
        tex = self.fc_tex(x)
        shape_vec = self.shape_fc(x)
        return pos_vec, tex, shape_vec


# ---------------------------------------------------------------------------
# Hand-rolled look_at_rotation (replaces pytorch3d.renderer.look_at_rotation)
# ---------------------------------------------------------------------------
def look_at_rotation(camera_position, at=(0, 0, 0), up=(0, 1, 0), device="cpu"):
    """
    Build rotation matrix from look-at vectors.
    Returns (N, 3, 3) rotation matrices.
    """
    cam = torch.as_tensor(camera_position, dtype=torch.float32, device=device)
    at_t = torch.as_tensor(at, dtype=torch.float32, device=device)
    up_t = torch.as_tensor(up, dtype=torch.float32, device=device)

    if cam.ndim == 1:
        cam = cam.unsqueeze(0)
    if at_t.ndim == 1:
        at_t = at_t.unsqueeze(0)
    if up_t.ndim == 1:
        up_t = up_t.unsqueeze(0)

    z_axis = F.normalize(at_t - cam, eps=1e-5)
    x_axis = F.normalize(torch.cross(up_t, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)

    is_close = torch.isclose(
        x_axis, torch.tensor(0.0, device=device), atol=5e-3
    ).all(dim=1, keepdim=True)
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)

    R = torch.cat(
        (x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1
    )
    return R.transpose(1, 2)


# ---------------------------------------------------------------------------
# Hand-rolled FitModel (replaces s2mtest.FitModel)
# ---------------------------------------------------------------------------
class FitModel(nn.Module):
    def __init__(self, ear_mu, ear_eigenvectors, V, shape_vec, shape_vec_value):
        super().__init__()
        self.cam_pos = nn.Parameter(torch.tensor((0.0, 0, 0)))
        self.look_at = nn.Parameter(torch.tensor((0.0, 0, 1.0)))
        self.up = nn.Parameter(torch.tensor((0.0, 1.0, 0.0)))
        self.scale_factor = nn.Parameter(torch.tensor(120.0))
        self.register_buffer("ear_mu", ear_mu)
        self.register_buffer("ear_eigenvectors", ear_eigenvectors)
        self.register_buffer("ear_eigenvalues", V)

        K = ear_eigenvectors.shape[1]
        if shape_vec_value == "optim":
            self.shape_vec = nn.Parameter(torch.zeros(1, K, device=ear_mu.device))
        elif shape_vec_value == "avg":
            self.register_buffer(
                "shape_vec", torch.zeros(1, K, device=ear_mu.device)
            )
        else:
            self.register_buffer("shape_vec", shape_vec)

    def forward(self):
        verts = self.ear_mu + self.ear_eigenvectors.mm(
            (self.shape_vec * self.ear_eigenvalues).permute(1, 0)
        )
        verts = verts.view(-1, 3)
        verts = F.relu(self.scale_factor) * verts
        R = look_at_rotation(
            self.cam_pos[None, :],
            at=self.look_at[None, :],
            up=self.up[None, :],
            device=verts.device,
        )
        T = -torch.bmm(R.transpose(1, 2), self.cam_pos[None, :, None])[:, :, 0]
        R, T = R.squeeze(0), T.squeeze(0)
        verts = verts.mm(R) + T
        return verts, R, T


# ---------------------------------------------------------------------------
# SSM Latent Refiner (optional)
# ---------------------------------------------------------------------------
class LatentRefiner(nn.Module):
    def __init__(self, n_modes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_modes, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_modes),
        )
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_pred):
        return z_pred + self.net(z_pred)
