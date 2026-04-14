from __future__ import annotations
import torch
from ..system.box import Box
from ..system.topology import Topology


class EinsteinCrystal:
    """Harmonic tethering potential anchoring each atom to a reference position.

    U_EC(r^N) = (alpha / 2) * sum_i |r_i - R_i|^2

    Parameters
    ----------
    R : torch.Tensor
        Reference positions of shape (N, 3) or (B, N, 3).
    alpha : float or torch.Tensor
        Spring constant (coupling strength).
    label : str
        Interaction label matching the topology entry.
    """

    def __init__(
        self,
        R: torch.Tensor,
        alpha: float | torch.Tensor,
        label: str,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.label = label
        self.alpha = torch.as_tensor(alpha, device=self.device, dtype=self.dtype)
        self.R = R.to(device=self.device, dtype=self.dtype)

    def energy(
        self,
        pos: torch.Tensor,
        top: Topology,
        box: Box,
        node_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return per-atom Einstein Crystal energy.

        Parameters
        ----------
        pos : torch.Tensor
            Current positions of shape (B, N, 3).

        Returns
        -------
        torch.Tensor
            Per-atom energies of shape (B, N). Summing over dim=1 gives the
            total Einstein Crystal energy per batch element.
        """
        R = self.R
        if R.ndim == 2:
            R = R.unsqueeze(0)          # (1, N, 3) -> broadcasts over B
        return (self.alpha / 2) * (pos - R).pow(2).sum(dim=-1)   # (B, N)
