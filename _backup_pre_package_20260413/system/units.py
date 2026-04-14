from __future__ import annotations

"""units.py
Generic unit system for MD.

* Pick three base‑unit scale factors (L, E, M) as conversion factors to SI.
* Helpers `UnitSystem.from_SI`, `.akma()`, `.gmx()` instantiate common sets.
"""

from dataclasses import dataclass
from functools import cached_property
import math

# --- SI constants --------------------------------------------------------
_AVOGADRO = 6.022_140_76e23         # mol^{-1} (exact)
_BOLTZMANN = 1.380_649e-23          # J K^{-1}  (per particle)
_EPS0      = 8.854_187_8128e-12     # F m^{-1}  (C^{2} J^{-1} m^{-1})
_ELEM_Q    = 1.602_176_634e-19      # C (exact)
_AMU       = 1.660_539_066_60e-27   # kg (exact)


@dataclass(frozen=True)
class UnitSystem:
    """Define one simulation unit for length, energy, mass, kb, charge.

    Parameters (all are conversion factors)
    ----------
    L : float
        Metres per simulation length‑unit.
    E : float
        Joules per simulation energy‑unit.
    M : float
        Kilograms per simulation mass‑unit.
    kB : float
        Simulation‑energy units per unit temperature.
    q_unit : float
        Simulation charge‑units per elementary charge e.
    """

    L: float = 1.0
    E: float = 1.0
    M: float = 1.0
    kB: float = 1.0
    Q: float = 1.0

    # --- derived scales --------------------------------------------------------
    @cached_property
    def time(self) -> float:              # seconds per sim time‑unit
        return math.sqrt(self.M * self.L ** 2 / self.E)

    @cached_property
    def velocity(self) -> float:          # m s^{-1} per sim velocity‑unit
        return self.L / self.time

    @cached_property
    def force(self) -> float:             # newtons per sim force‑unit
        return self.E / self.L

    @cached_property
    def pressure(self) -> float:          # pascals per sim pressure‑unit
        return self.E / self.L ** 3

    # --- convenient defaults --------------------------------------------------------
    @classmethod
    def from_SI(cls, *, L: float, E: float, M: float) -> "UnitSystem":
        """Create UnitSystem and derive kB & q_unit from SI constants."""
        kB_sim = _BOLTZMANN / E
        q_unit_e = math.sqrt(4 * math.pi * _EPS0 * E * L) / _ELEM_Q
        return cls(L=L, E=E, M=M, kB=kB_sim, Q=q_unit_e)

    @classmethod
    def akma(cls) -> "UnitSystem":
        """AA, kcal mol^{-1}, amu with Kelvin & elementary charges."""
        L = 1.0e-10                                   # 1 AA
        E = 4184.0 / _AVOGADRO                        # 1 kcal mol^{-1} per particle
        M = _AMU                                      # 1 amu
        return cls.from_SI(L=L, E=E, M=M)

    @classmethod
    def gmx(cls) -> "UnitSystem":
        """nm, kJ mol^{-1}, amu (GROMACS style)."""
        L = 1.0e-9                                    # 1 nm
        E = 1000.0 / _AVOGADRO                        # 1 kJ mol^{-1} per particle
        M = _AMU
        return cls.from_SI(L=L, E=E, M=M)

    # --- misc --------------------------------------------------------
    def __repr__(self):
        return (
            f"UnitSystem(L={self.L:.3g} m/uL, E={self.E:.3g} J/uE, "
            f"M={self.M:.3g} kg/um, kB={self.kB:.3g} uE/K, Q={self.Q:.3g} e/uQ)"
        )
