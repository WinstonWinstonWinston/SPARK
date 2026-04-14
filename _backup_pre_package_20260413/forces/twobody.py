from __future__ import annotations
import torch
from system.box import Box
from system.topology import Topology
import math

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
    """MixtureCoulomb potential for an MD system. Assumes the system object has the node features "charge". Only implemented for short
    range interaction. The result will be off for systems where periodicity matters. Use with caution.
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

class MixtureEwaldCoulomb:
    """
    Ewald split with self & background terms ‒ conventions match https://arxiv.org/pdf/2412.03281v2
    """

    def __init__(self, alpha: float | torch.Tensor, rcut: float | torch.Tensor, kcut: float | torch.Tensor, sigma: float | torch.Tensor,
                 box: Box, charges: torch.Tensor,
                 label: str, *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype

        self.alpha = torch.tensor(alpha, device=self.device, dtype=self.dtype)  # coeff alpha
        self.rcut = torch.tensor(rcut, device=self.device, dtype=self.dtype)    # maximum value for real space sum
        self.kcut = torch.tensor(kcut, device=self.device, dtype=self.dtype)    # maximum value for momentum space sum
        self.sigma = torch.tensor(sigma, device=self.device, dtype=self.dtype)  # smearing coeff in ewald technique
        self.charges = torch.tensor(charges, device=self.device, dtype=self.dtype) # Sum of square charges in system
        self.q2_sum = torch.tensor((charges**2).sum(), device=self.device, dtype=self.dtype) # Sum of square charges in system
        self.q_sum = torch.tensor(charges.sum(), device=self.device, dtype=self.dtype) # Sum of charges in system

        # figure out all possible shifts we need for the periodic copies
        # assumes the boxes have fixed lengths in the simulation, i.e. NVT or NVE. 

        # real-space
        L = box.edges #(,3)
        n_repeats  = torch.ceil(rcut / L).to(torch.int64) #(,3)
        nx = torch.arange(-n_repeats[0], n_repeats[0] + 1, device=self.device)
        ny = torch.arange(-n_repeats[1], n_repeats[1] + 1, device=self.device)
        nz = torch.arange(-n_repeats[2], n_repeats[2] + 1, device=self.device)
        shift = torch.cartesian_prod(nx * L[0], ny * L[1], nz * L[2])  # (P,3)
        keep   = (shift.square().sum(-1) <= (rcut + 0.5*L.max()).pow(2)) # Only keep those shifts which produce an image which is smaller than the cutoff
        self.shift  = shift[keep]

        # reciprocal-space
        L_star = 2 * torch.pi / L                         # (3,)
        n_repeats = torch.ceil(self.kcut / L_star).to(torch.int64)   # (3,)
        nx = torch.arange(0, n_repeats[0] + 1, device=self.device)
        ny = torch.arange(0, n_repeats[1] + 1, device=self.device)
        nz = torch.arange(0, n_repeats[2] + 1, device=self.device)
        kshift = torch.cartesian_prod(nx * L_star[0],  ny * L_star[1], nz * L_star[2])    # (P,3)
        keep = (kshift.square().sum(-1) <= self.kcut**2) & (kshift.abs().sum(-1) > 0)
        self.kshift = kshift[keep]  # (P_k, 3)
        knorm_sq = (self.kshift ** 2).sum(dim=1)                       # (P_k,)
        self.V_LR_kernel = (4 * torch.pi) * torch.exp(-0.5 * (self.sigma**2) * knorm_sq) / knorm_sq   # (P_k,)
        
        self.label = label

        # Compute analytical potential from SR and LR self energies as well as charge energy
        # These are total energies of the whole system and they are independent of the 
        # positions so they do not affect forces.
        # SR
        nonzero   = (self.shift.square().sum(-1) > 0)
        r_l       = torch.norm(self.shift[nonzero], dim=-1)        # (P_nonzero,)
        self.E_sr_self = self.alpha * self.q2_sum* torch.sum(torch.special.erfc(r_l/(math.sqrt(2)*self.sigma)) / r_l) # (scalar)
        self.E_charge = - self.alpha * math.pi * self.sigma ** 2 * self.q_sum ** 2 / box.volume
        self.E_lr_self = self.alpha * (-0.5) * math.sqrt(2/math.pi)  * self.q2_sum / self.sigma # (scalar)
    
    def energy(self, pos: torch.Tensor, top: Topology, box: Box, node_features: dict[str, torch.Tensor], eps:float = 1e-12, chunk:int = 65536) -> torch.Tensor:
        idx = top.get_tensor(2, self.label)
        vecs = pos[:, idx]                                           # (B, M, 2, 3)
        rij_0 = box.minimum_image(vecs[:, :, 0] - vecs[:, :, 1])    # (B, M, 3)
        charge2 = (node_features["charge"][:,idx][:,:,0]*node_features["charge"][:,idx][:,:,1]) # (B, M)

        # expand in the second to last dim, add the shifts to obtain many periodic copies
        rij = rij_0.unsqueeze(-2) + self.shift.to(pos.dtype) # (B,M,P,3)
        r2   = (rij ** 2).sum(-1)                    # (B,M,P)
        mask = r2 < self.rcut ** 2
        r    = torch.sqrt(torch.clamp(r2, min=eps))  # (B,M,P)

        # ##### LONG RANGE  (wrong, double counts cos) #####
        # # (P_k, 3), (B, M, 3) -> B, M, P_k 
        # kdotr = torch.einsum("kp,bmp->bmk",
        #              self.kshift,   # (P_k,3)
        #              rij_0)                       # (B,M,3)
        # vlr   = torch.cos(kdotr) * self.V_LR_kernel.unsqueeze(0).unsqueeze(0)  # broadcast to (B,M,P_k)

        # E_batch_LR = self.alpha*(charge2.unsqueeze(dim=-1)*vlr).sum(dim=(-1))/box.volume               # (B,M, )

        ##### LONG RANGE CHUNKED ALONG P_k  (wrong, double counts cos) #####
        E_batch_LR = torch.zeros_like(charge2, dtype=rij_0.dtype)   # (B, M)
        
        # for start in range(0, self.kshift.shape[0], chunk):
        #     k_slice       = self.kshift[start:start+chunk]          # (c, 3)
        #     kernel_slice  = self.V_LR_kernel[start:start+chunk]     # (c,)

        #     print(k_slice.size())
        #     print(kernel_slice.size())
        #     # (c,3) × (B,M,3) → (B,M,c)
        #     kdotr = torch.einsum("kp,bmp->bmk", k_slice, rij_0)
        #     vlr   = torch.cos(kdotr) * kernel_slice.view(1, 1, -1)  # broadcast
        
        #     E_batch_LR += (charge2.unsqueeze(-1) * vlr).sum(dim=-1) # accumulate
        
        # E_batch_LR = self.alpha * E_batch_LR / box.volume           # (B, M)

        ##### LONG RANGE CHUNKED ALONG M #####
        for s in range(0, rij_0.shape[1], chunk):
            e = s + chunk
        
            # current slice of pairs
            rij_slice   = rij_0[:, s:e]              # (B, m, 3)
            q2_slice    = charge2[:, s:e]            # (B, m)
        
            # (P_k,3), (B,m,3)  ->  (B,m,P_k)
            kdotr = torch.einsum("kp,bmp->bmk", self.kshift, rij_slice)
            vlr   = torch.cos(kdotr) * self.V_LR_kernel.view(1, 1, -1)
        
            # write energies for this slice straight into the output tensor
            E_batch_LR[:, s:e] = self.alpha * (q2_slice.unsqueeze(-1) * vlr).sum(dim=-1) / box.volume

        
        ##### SHORT RANGE #####
        vsr  = torch.special.erfc(r/(math.sqrt(2)*self.sigma)) / r
        vsr  = vsr * mask                            # zero out long images

        # sum over images with charge squared to convert from potential to potential energy
        E_batch_SR = self.alpha*(charge2.unsqueeze(dim=-1) * vsr).sum(dim=(-1))               # (B,M, )


        ##### CONSTANT SHIFT #####
        E_const_per_pair = (self.E_sr_self + self.E_lr_self + self.E_charge) / charge2.shape[1]

        return E_batch_LR + E_batch_SR + E_const_per_pair                           # (B,M)