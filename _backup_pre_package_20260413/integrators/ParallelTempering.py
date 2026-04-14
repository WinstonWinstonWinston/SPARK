import torch

class ParallelTempering:
    """
    Minimal Bussi-Parrinello Langevin integrator for a set of parallel-tempered ensembles.

    This class implements two main methods: `step_NVT` and `swap`.

    - step_NVT integrates Langevin dynamics at fixed temperature using the Bussi-Parrinello thermostat,
      evolving each replica toward local equilibrium.
    - swap attempts Monte Carlo exchanges between neighboring temperature replicas. Swaps are accepted
      according to the Metropolis criterion based on potential energy differences and inverse temperatures,
      and momenta are resampled from the Maxwell-Boltzmann distribution at the new temperature. Applys even
      odd critetia.

    Typical usage alternates between step_NVT and swap to ensure both local and global exploration
    of phase space.
    
    Parameters
    ----------
    dt : float
        Time step in simulation units.
    gamma : float
        Friction coefficient for Langevin dynamics.
    T : torch.Tensor
        Temperature tensor of shape (B,) for B parallel replicas.
    device : str or torch.device, optional
        Device to place internal tensors on (default: "cuda").
    dtype : torch.dtype, optional
        Data type for all tensors (default: torch.float32).

    Expected `system` interface
    ---------------------------
    system.pos   : (B, N, 3) tensor – current positions
    system.mom   : (B, N, 3) tensor – current momenta (p = m * v)
    system.mass  : (B, N)    tensor – atomic masses
    system.force : (B, N, 3) tensor – forces (negative gradient of potential wrt positions)
    """
    def __init__(self, dt: float, gamma: float, T: list | torch.Tensor, D: int, *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32):
        self.dt = torch.as_tensor(dt, device=device, dtype=dtype)
        self.gamma = torch.as_tensor(gamma, device=device, dtype=dtype)
        self.T = torch.as_tensor(T, device=device, dtype=dtype)
        if self.T.ndim != 1:
            raise ValueError(f"T must be a 1D tensor of shape (B,), but got shape {self.T.shape}")
        self.dtype = dtype
        self.device = device
        self.offset = 0
        self.last_accept_percent = 0

        self.B = self.T.numel()
        if self.B % D != 0: 
            raise ValueError(f"D={D} must divide number of replicas B={B}")
        self.D = D 
        self.n_blocks = self.B // D # n_pairs = n_blocks - 1 

    def step_NVT(self, system) -> None:
        
        c_1 = torch.exp(-self.gamma*self.dt/2)
        c_2 = (system.mass.unsqueeze(-1) * system.units.kB * self.T.view(-1, 1, 1) * (1 - c_1**2))**0.5
        
        # system.mom += 0.5 * self.dt * system.force()
        momentum_t_plus = c_1*system.mom + c_2*torch.randn(system.mom.size(),device=self.device)
        old_forces = system.force()
        system.pos = system.pos + self.dt *  system.mom / system.mass.unsqueeze(-1) + 0.5 * old_forces*(self.dt**2) / system.mass.unsqueeze(-1)
        system.reset_cache()
        momentum_t_minus_plus_dt = momentum_t_plus + 0.5*(old_forces + system.force())*self.dt
        system.mom = c_1*momentum_t_minus_plus_dt + c_2*torch.randn(system.mom.size(),device=self.device)

    
    def swap(self,system) -> None:
        # get swap indexes
        blocks = torch.arange(self.offset,
                              self.n_blocks - 1,
                              2,
                              device=self.device) 
        
        # for each block b and each chain index in [0..D-1], form replica indices:
        i_idx   = torch.arange(self.D, device=self.device)  # (D,)
        j_rep   = blocks[:, None]*self.D + i_idx[None, :]   # (n_pairs, D)
        i_rep   = (blocks+1)[:,None]*self.D + i_idx[None,:] # (n_pairs, D)

        j = j_rep.reshape(-1)  # (n_pairs*D,)
        i = i_rep.reshape(-1)  # (n_pairs*D,)

        # swap_idx = torch.arange(self.offset, len(self.T) -1, 2, device=self.device)
        # i, j     = swap_idx, swap_idx+1

        # compute change in energy
        dBeta = 1/(system.units.kB*self.T[i]) - 1/(system.units.kB*self.T[j])
        E = system.potential_energy()
        dE = E[i] - E[j]
 
        # apply metropolis criteria
        accept  = torch.rand_like(dE).log() < dBeta * dE
        swap_mask = accept.nonzero(as_tuple=True)[0]

        # perform swaps for accepted pairs
        i_acc = i[swap_mask]   
        j_acc = j[swap_mask] 

        self.last_accept_percent = len(i_acc)/len(i)
    
        # swap positions
        system.pos[i_acc], system.pos[j_acc] = system.pos[j_acc].clone(), system.pos[i_acc].clone()

        # resample momentum using correct temperature for each replica (T[i_acc] and T[j_acc])
        system.mom[i_acc] = torch.randn_like(system.mom[i_acc]) * (system.mass[i_acc].unsqueeze(-1) * system.units.kB * self.T[i_acc].view(-1,1,1)).sqrt() 
        system.mom[j_acc] = torch.randn_like(system.mom[j_acc]) * (system.mass[j_acc].unsqueeze(-1) * system.units.kB * self.T[j_acc].view(-1,1,1)).sqrt()

        # move offset for next call to swap()
        self.offset = (1 + self.offset) % 2

        def __repr__(self) -> str: 
            return (f"ParallelTempering(dt={self.dt.item():.3g}, "
                    f"gamma={self.gamma.item():.3g}, "
                    f"D={self.D}, replicas={self.T.numel()})")


        

        