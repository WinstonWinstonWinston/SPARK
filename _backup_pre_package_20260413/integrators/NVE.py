import torch

class NVE:
    """
    Minimal velocity-Verlet (V.V.) integrator for an NVE ensemble.

    Parameters
    ----------
    dt : float
        Time step (simulation units).

    Expected `system` interface
    ---------------------------
    system.pos      : returns (B, N, 3) tensor – current positions
    system.mom      : returns (B, N, 3) tensor – current momenta  (p = m v)
    system.mass     : returns (B, N)    tensor – atomic masses
    system.force    : returns (B, N, 3) tensor – forces (negative potential gradient wrt to pos)
    """
    def __init__(self, dt: float, *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32,):
        self.dt = torch.as_tensor(dt, device=device, dtype=dtype)
        self.dtype = dtype
        self.device = device

    def step(self, system) -> None:
        system.mom = system.mom + 0.5 * self.dt * system.force()
        system.pos = system.pos + self.dt *  system.mom / system.mass.unsqueeze(-1)
        system.reset_cache()
        system.mom = system.mom + 0.5 * self.dt * system.force()

    def __repr__(self) -> str:
        return f"NVE(dt={self.dt.item():.3g})"
