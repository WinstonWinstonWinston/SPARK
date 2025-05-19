from __future__ import annotations
import torch
from system.box import Box
from system.topology import Topology

class Dihedral:
    """Dihedral angle for an MD system.
    """

    def __init__(self, phi_0: float | torch.Tensor, kappa: float | torch.Tensor, n: int | torch.Tensor, label: str, *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
    
        self.phi_0 = torch.as_tensor(phi_0, device=self.device, dtype=self.dtype)
        self.kappa = torch.as_tensor(kappa, device=self.device, dtype=self.dtype)
        self.n     = torch.as_tensor(n, device=self.device, dtype=self.dtype)
        self.label = label
    
        if any(t.ndim != 0 for t in [self.phi_0, self.kappa, self.n]):
            raise ValueError("phi_0, kappa, and n must be scalar (0D) tensors")

    def energy(self, pos: torch.Tensor, top: Topology, box: Box, node_features: dict[str, torch.Tensor]) -> torch.Tensor:
        vecs = pos[:, top.get_tensor(4, self.label)]  # (B, M, 4, 3)
        
        # Unpack positions
        r1 = vecs[:, :, 0]
        r2 = vecs[:, :, 1]
        r3 = vecs[:, :, 2]
        r4 = vecs[:, :, 3]
    
        # Compute bond vectors with PBC
        b1 = box.minimum_image(r2 - r1)  # (B, M, 3)
        b2 = box.minimum_image(r3 - r2)
        b3 = box.minimum_image(r4 - r3)
    
        # Normal vectors to planes
        n1 = torch.cross(b1, b2, dim=-1)  # (B, M, 3)
        n2 = torch.cross(b2, b3, dim=-1)
    
        # Normalize
        n1 =  torch.nn.functional.normalize(n1, dim=-1)
        n2 =  torch.nn.functional.normalize(n2, dim=-1)
        b2n = torch.nn.functional.normalize(b2, dim=-1)
    
        # Compute cosine and sine of angle
        cos_phi = (n1 * n2).sum(dim=-1).clamp(-1.0, 1.0)  # (B, M)
        sin_phi = (torch.cross(n1, n2, dim=-1) * b2n).sum(dim=-1)  # (B, M)
    
        # Signed angle
        phi = torch.atan2(sin_phi, cos_phi)  # (B, M)
    
        # AMBER-style torsion: U = k * (cos(n*phi + phi0))
        return self.kappa * (1 + torch.cos(self.n * phi - self.phi_0))  # (B, M)
