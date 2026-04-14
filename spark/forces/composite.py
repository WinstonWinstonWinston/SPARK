from __future__ import annotations
import torch
from ..system.box import Box
from ..system.topology import Topology


class InterpolatingPotential:
    """Linearly interpolates between two potentials:

        U(r; λ) = (1 - λ) * U_a(r) + λ * U_b(r),   λ ∈ [0, 1]

    λ is read from node_features["lambda"] of shape (B, N). All atoms in a
    batch element share the same value so only column 0 is used -> (B,).

    Both sub-forces must return the same shape (B, M) — i.e. they share the
    same interaction set in the topology (e.g. two Mie potentials with the
    same pair list but different parameters). λ is applied element-wise across
    the M interactions before returning, so System's .sum(dim=1) gives the
    correct interpolated total.

    The topology must contain the labels of both sub-forces so each can call
    top.get_tensor(arity, label) as normal. energy_dict maps a single label
    to this object.

    Usage
    -----
        interp = InterpolatingPotential(mie1, mie2, label="mie1", ...)
        # topology has "mie1" and "mie2" entries
        # energy_dict = {"mie1": interp, "mie2": dummy_zero_force}
        # node_features["lambda"]: (B, N) — same value per row
    """

    def __init__(
        self,
        force_a,
        force_b,
        label: str,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype  = dtype
        self.force_a = force_a
        self.force_b = force_b
        self.label   = label

    def energy(
        self,
        pos: torch.Tensor,
        top: Topology,
        box: Box,
        node_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Returns (B, M_a + M_b). System's .sum(dim=1) gives (1-λ)*ΣE_a + λ*ΣE_b."""
        lam = node_features["lambda"][:, 0].unsqueeze(1)                           # (B, 1)
        E_a = self.force_a.energy(pos, top, box, node_features)                    # (B, M_a)
        E_b = self.force_b.energy(pos, top, box, node_features)                    # (B, M_b)
        return torch.cat([(1 - lam) * E_a, lam * E_b], dim=1)                     # (B, M_a + M_b)
