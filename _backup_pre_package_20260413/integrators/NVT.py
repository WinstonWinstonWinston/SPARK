import torch

class NVT:
    """
    Minimal Bussi Parinello Langevin integrator for an NVT ensemble. 10.1103/PhysRevE.75.056707

    Parameters
    ----------
    dt : float
        Time step (simulation units).
    gamma : float
        Friction coefficient.
    T : float
        Target temperature.
    remove_com_mom : bool
        If True, remove centre-of-mass momentum every step.
    remove_com_pos : bool
        If True, remove centre-of-mass position drift every step.

    Expected `system` interface
    ---------------------------
    system.pos      : returns (B, N, 3) tensor – current positions
    system.mom      : returns (B, N, 3) tensor – current momenta  (p = m v)
    system.mass     : returns (B, N)    tensor – atomic masses
    system.force    : returns (B, N, 3) tensor – forces (negative potential gradient wrt to pos)
    """
    def __init__(self, dt: float, gamma: float, T: float, *, remove_com_mom: bool = False, remove_com_pos: bool = False, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32,):
        self.dt = torch.as_tensor(dt, device=device, dtype=dtype)
        self.T = torch.as_tensor(T, device=device, dtype=dtype)
        self.gamma = torch.as_tensor(gamma, device=device, dtype=dtype)
        self.remove_com_mom = remove_com_mom
        self.remove_com_pos = remove_com_pos
        self.dtype = dtype
        self.device = device

    def zero_com_momentum(self, system) -> None:
        """Remove centre-of-mass momentum from all batch elements.

        Computes v_com = sum(p_i) / M_total and subtracts m_i * v_com from
        each atom's momentum so the total momentum is exactly zero.
        """
        M_total = system.mass.sum(dim=1, keepdim=True)              # (B, 1)
        p_com   = system.mom.sum(dim=1, keepdim=True)               # (B, 1, 3)
        v_com   = p_com / M_total.unsqueeze(-1)                     # (B, 1, 3)
        system.mom = system.mom - system.mass.unsqueeze(-1) * v_com

    def zero_com_position(self, system) -> None:
        """Remove centre-of-mass drift while preserving the initial COM.

        On first call, stores the current COM as the target COM.
        On later calls, shifts all atoms so the COM stays at that target.
        """
        M_total = system.mass.sum(dim=1, keepdim=True)  # (B, 1)

        r_com = (
            system.mass.unsqueeze(-1) * system.pos
        ).sum(dim=1, keepdim=True) / M_total.unsqueeze(-1)  # (B, 1, 3)

        if not hasattr(system, "_com_target"):
            system._com_target = r_com.clone()

        system.pos = system.pos - (r_com - system._com_target)
        
    def step(self, system) -> None:
        c_1 = torch.exp(-self.gamma*self.dt/2)
        c_2 = (system.mass.unsqueeze(-1)*system.units.kB*self.T*(1-c_1**2))**0.5

        momentum_t_plus = c_1*system.mom + c_2*torch.randn(system.mom.size(),device=self.device)
        old_forces = system.force()
        system.pos = system.pos + self.dt *  system.mom / system.mass.unsqueeze(-1) + 0.5 * old_forces*(self.dt**2) / system.mass.unsqueeze(-1)
        system.reset_cache()
        momentum_t_minus_plus_dt = momentum_t_plus + 0.5*(old_forces + system.force())*self.dt
        system.mom = c_1*momentum_t_minus_plus_dt + c_2*torch.randn(system.mom.size(),device=self.device)

        if self.remove_com_mom:
            self.zero_com_momentum(system)
        if self.remove_com_pos:
            self.zero_com_position(system)

    def __repr__(self) -> str:
        return f"NVT(dt={self.dt.item():.3g}, gamma={self.gamma.item():.3g}, T={self.T.item():.3g}, remove_com_mom={self.remove_com_mom}, remove_com_pos={self.remove_com_pos})"
