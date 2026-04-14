from __future__ import annotations
import torch

"""box.py
Minimal rectangular prism simulation box for NVT/NVE runs. Stored numbers are assumed to already be in simulation units.

* angles fixed at 90 deg, 90 deg, 90 deg
* Helpers (`volume`, `wrap`, `minimum_image`)
"""

class Box:
    """Axis‑aligned simulation box.

    Parameters
    ----------
    edges : (3,) array‑like [Lx, Ly, Lz] – edge lengths in simulation units.
    device, dtype : torch kwargs for internal tensor representation.
    """
    edges: torch.Tensor
    # --- construction --------------------------------------------------------
    def __init__(self, edges: tuple[float, float, float] | torch.Tensor, bcs: tuple[str, str, str],*, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32):
        e = torch.as_tensor(edges, device=device, dtype=dtype).flatten()
        if e.numel() != 3:
            raise ValueError("Box expects three edge lengths [Lx, Ly, Lz].")
        if not torch.all(e > 0):
            raise ValueError("All box edge lengths must be positive.")
        
        if len(bcs) != 3:
            raise ValueError("Box expects three boundary conditions (e.g., ['p', 'p', 's']).")
        if not all(bc in {'p', 's'} for bc in bcs):
            raise ValueError("Boundary conditions must be either 'p' (periodic) or 's' (shrinkwrap).")

        self.all_s = all(bc in {'s'} for bc in bcs)
        self.edges = e
        self.bcs = bcs
        self.device = device
        self.dtype = dtype
        self.bcs_bool = torch.tensor([bc == 'p' for bc in self.bcs], device=self.device)

    # --- properties --------------------------------------------------------
    @property
    def Lx(self) -> torch.Tensor:
        return self.edges[0]

    @property
    def Ly(self) -> torch.Tensor:
        return self.edges[1]

    @property
    def Lz(self) -> torch.Tensor:
        return self.edges[2]

    @property
    def volume(self) -> torch.Tensor:
        return self.edges.prod()

    # --- PBC helpers --------------------------------------------------------
    def wrap(self, pos: torch.Tensor) -> torch.Tensor:
        """Return positions wrapped into the primary cell (mod box)."""
        return pos - torch.floor(pos / self.edges) * self.edges

    def minimum_image(self,delta: torch.Tensor) -> torch.Tensor:
        """
        Apply minimum image convention to displacement vectors.
    
        Parameters
        ----------
        delta : torch.Tensor
            Displacement vectors of shape (..., 3)
    
        Returns
        -------
        torch.Tensor
            Minimum image corrected displacement vectors
        """
        # image_shift = -self.edges * torch.round(delta / self.edges)
        # return delta + image_shift * self.bcs_bool
        if self.all_s:
            return delta
        fractional = delta / self.edges
        image_shift = -self.edges * ((fractional + 0.5).floor())
        return delta + image_shift * self.bcs_bool


    # --- misc --------------------------------------------------------
    def __repr__(self):
        Lx, Ly, Lz = self.edges.tolist()
        bcx, bcy, bcz = self.bcs
        return f"Box({Lx:g}{bcx}, {Ly:g}{bcy}, {Lz:g}{bcz})"
