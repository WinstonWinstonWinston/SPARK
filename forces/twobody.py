from __future__ import annotations
import torch
from system.box import Box
from system.topology import Topology

class Harmonic_Bond:
    """Harmonic bond for an MD system.
    """

    def __init__(self, r_0: float | torch.Tensor, kappa: float | torch.Tensor, label: str, *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype

        self.r_0 = torch.as_tensor(r_0, device=self.device, dtype=self.dtype)
        self.kappa = torch.as_tensor(kappa, device=self.device, dtype=self.dtype)
        self.label = label

        if self.r_0.ndim != 0 or self.kappa.ndim != 0:
            raise ValueError("r_0 and kappa must be scalar (0D) tensors")

    def energy(self,pos: torch.Tensor, top: Topology, box: Box, node_features: dict[str, torch.Tensor]) -> torch.Tensor:
        vecs = pos[:,top.get_tensor(2,self.label)] # (B, M, 2, 3)
        r = torch.norm(box.minimum_image(vecs[:, :, 1] - vecs[:, :, 0]), dim=-1)    # (B, M)
        return self.kappa*(r - self.r_0)**2 # (B, M)        

class Mie:
    """Mie potential for an MD system. https://doi.org/10.1002/andp.19033160802.
    """

    def __init__(self, n: float | torch.Tensor, sigma: float | torch.Tensor, epsilon: float | torch.Tensor, label: str, *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype

        self.n = torch.as_tensor(n, device=self.device, dtype=self.dtype)
        self.sigma = torch.as_tensor(sigma, device=self.device, dtype=self.dtype)
        self.epsilon = torch.as_tensor(epsilon, device=self.device, dtype=self.dtype)
        self.label = label

        if self.n.ndim != 0 or self.sigma.ndim != 0 or self.epsilon.ndim != 0:
            raise ValueError("Mie parameters must be scalar (0D) tensors")

        self.m = torch.tensor(6.0, device=self.device, dtype=self.dtype)  # fixed m

        # Precompute constant prefactor
        self.prefactor = (
            self.epsilon * self.n / (self.n - self.m) *
            (self.n / self.m) ** (self.m / (self.n - self.m))
        )

    def energy(self, pos: torch.Tensor, top: Topology, box: Box, node_features: dict[str, torch.Tensor], eps:float = 1e-12) -> torch.Tensor:
        vecs = pos[:, top.get_tensor(2, self.label)]                                # (B, M, 2, 3)
        r = torch.norm(box.minimum_image(vecs[:, :, 0] - vecs[:, :, 1]), dim=-1)    # (B, M)
        sigma_over_r = self.sigma / (r + eps)                                       # avoid div-by-zero
        sr_n = sigma_over_r ** self.n
        sr_m = sigma_over_r ** self.m
        return self.prefactor * (sr_n - sr_m)                                       # (B, M)

class MixtureLJ:
    """Mixture LJ potential for an MD system. Assumes the system object has the node features "sigma" and "epsilon".
    Applies standard mixing rules of sigma_12 = (sigma_1 + sigma_2) and epsilon_12 = sqrt(epsilon_1 * epsilon_2)
    """

    def __init__(self,  fudge, label: str, *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype

        self.label = label
        self.m = torch.tensor(6.0, device=self.device, dtype=self.dtype)  # fixed m
        self.n = torch.tensor(12.0, device=self.device, dtype=self.dtype)  # fixed n
        self.fudge = torch.tensor(fudge, device=self.device, dtype=self.dtype)  # fixed fudge

    def energy(self, pos: torch.Tensor, top: Topology, box: Box, node_features: dict[str, torch.Tensor], eps:float = 1e-12) -> torch.Tensor:
        idx = top.get_tensor(2, self.label)
        vecs = pos[:, idx]                                # (B, M, 2, 3)
        r = torch.norm(box.minimum_image(vecs[:, :, 1] - vecs[:, :, 0]), dim=-1)    # (B, M)
        sigma = 0.5*(node_features["sigma"][:,idx][:,:,0] + node_features["sigma"][:,idx][:,:,1])
        epsilon = torch.sqrt(node_features["epsilon"][:,idx][:,:,0]*node_features["epsilon"][:,idx][:,:,1])
        sigma_over_r = sigma / (r + eps)                                       # avoid div-by-zero
        sr_n = sigma_over_r ** self.n
        sr_m = sigma_over_r ** self.m
        prefactor = 4 * epsilon
        return self.fudge * prefactor * (sr_n - sr_m)                                       # (B, M)

class MixtureCoulomb:
    """MixtureCoulomb potential for an MD system. Assumes the system object has the node features "charge".
    """

    def __init__(self, alpha: float | torch.Tensor, label: str, *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype

        self.alpha = torch.tensor(alpha, device=self.device, dtype=self.dtype)  # coeff alpha
        self.label = label

    def energy(self, pos: torch.Tensor, top: Topology, box: Box, node_features: dict[str, torch.Tensor], eps:float = 1e-12) -> torch.Tensor:
        idx = top.get_tensor(2, self.label)
        vecs = pos[:, idx]                                # (B, M, 2, 3)
        r = torch.norm(box.minimum_image(vecs[:, :, 0] - vecs[:, :, 1]), dim=-1)    # (B, M)
        charge = (node_features["charge"][:,idx][:,:,0]*node_features["charge"][:,idx][:,:,1])
        return self.alpha * charge / (r + eps)                                      # (B, M)