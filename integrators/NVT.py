import torch

class NVT:
    """
    Minimal Bussi Parinello Langevin integrator for an NVT ensemble. 10.1103/PhysRevE.75.056707

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
    def __init__(self, dt: float, gamma: float, T: float, *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32,):
        self.dt = torch.as_tensor(dt, device=device, dtype=dtype)
        self.T = torch.as_tensor(T, device=device, dtype=dtype)
        self.gamma = torch.as_tensor(gamma, device=device, dtype=dtype)
        self.dtype = dtype
        self.device = device

    def step(self, system) -> None:
        c_1 = torch.exp(-self.gamma*self.dt/2)
        c_2 = (system.mass.unsqueeze(-1)*system.units.kB*self.T*(1-c_1**2))**0.5
        
        # system.mom += 0.5 * self.dt * system.force()
        momentum_t_plus = c_1*system.mom + c_2*torch.randn(system.mom.size(),device=self.device)
        old_forces = system.force()
        system.pos = system.pos + self.dt *  system.mom / system.mass.unsqueeze(-1) + 0.5 * old_forces*(self.dt**2) / system.mass.unsqueeze(-1)
        system.reset_cache()
        momentum_t_minus_plus_dt = momentum_t_plus + 0.5*(old_forces + system.force())*self.dt
        system.mom = c_1*momentum_t_minus_plus_dt + c_2*torch.randn(system.mom.size(),device=self.device)

    def __repr__(self) -> str:
        return f"NVT(dt={self.dt.item():.3g}, gamma={self.gamma.item():.3g}, T={self.T.item():.3g})"
