from __future__ import annotations
import torch
from system.box import Box
from system.topology import Topology

class Harmonic_Angle:
    """Harmonic angle for an MD system.
    """

    def __init__(self, theta_0: float | torch.Tensor, kappa: float | torch.Tensor, label: str, *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype

        self.theta_0 = torch.as_tensor(theta_0, device=self.device, dtype=self.dtype)
        self.kappa = torch.as_tensor(kappa, device=self.device, dtype=self.dtype)
        self.label = label

        if self.theta_0.ndim != 0 or self.kappa.ndim != 0:
            raise ValueError("theta_0 and kappa must be scalar (0D) tensors")

    def energy(self, pos: torch.Tensor, top: Topology, box: Box, node_features: dict[str, torch.Tensor]) -> torch.Tensor:
        vecs = pos[:, top.get_tensor(3, self.label)]  # (B, M, 3, 3)
    
        # Vectors: a1 to a2 and a3 to a2 (central atom is at index 1)
        v1 = box.minimum_image(vecs[:, :, 0] - vecs[:, :, 1])  # (B, M, 3)
        v2 = box.minimum_image(vecs[:, :, 2] - vecs[:, :, 1])  # (B, M, 3)
    
        # Normalize vectors
        v1_norm = torch.nn.functional.normalize(v1, dim=-1)
        v2_norm = torch.nn.functional.normalize(v2, dim=-1)
    
        # Compute cosine of angle and clamp
        cos_theta = (v1_norm * v2_norm).sum(dim=-1).clamp(-1.0, 1.0)  # (B, M)
    
        # Get angle in radians
        theta = torch.acos(cos_theta)  # (B, M)
    
        return self.kappa * (theta - self.theta_0)**2  # (B, M)
