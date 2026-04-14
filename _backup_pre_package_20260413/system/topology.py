from __future__ import annotations
from typing import Sequence
import torch

class Topology:
    """Topology for an MD system, it defines a set of arity groupings for usage on a potential energy surface.
    """
    # --- construction --------------------------------------------------------
    def __init__( self, arity_table: tuple[Sequence[str], Sequence[int]], *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32):
        """
        Parameters
        ----------
        arity_table : tuple of (keys, arities)
            labels: list of term types (e.g., "bond", "angle", "dihedral")
            arities: list of corresponding arities (2 for bond, 3 for angle, etc.)
        """
        keys, arities = arity_table
        if len(keys) != len(arities):
            raise ValueError("arity_table must have same-length key and arity lists")

        # Construct the nested hyperedge storage
        self.edges = {}
        for key, arity in zip(keys, arities):
            self.edges.setdefault(arity, {})[key] = set()

        # Cache of tensor objects
        self._tensor_cache = {}
    
        self.device = torch.device(device)
        self.dtype = dtype
    
    # --- helpers --------------------------------------------------------
    def add(self, arity: int, label: str, indices: tuple[int]):
        """
        Add a hyperedge of a given type and arity to the topology.

        Parameters
        ----------
        arity : int
            The number of atoms in the interaction (e.g. 2 for bonds, 3 for angles).
        label : str
            The label or type of the interaction (e.g. "bond", "angle").
        indices : tuple[int]
            A tuple of atom indices participating in the interaction.
        """
        self.edges[arity][label].add(indices)

    def get(self, arity: int, label: str) -> set[tuple[int]]:
        """
        Retrieve all hyperedges of a given type and arity.

        Parameters
        ----------
        arity : int
            The number of atoms in the interaction (e.g. 2 for bonds, 3 for angles).
        label : str
            The label or type of the interaction (e.g. "bond", "angle").

        Returns
        -------
        set[tuple[int]]
            A set of atom index tuples defining the stored interactions.
        """
        return self.edges[arity][label]

    def get_arity(self, arity: int) -> dict[set[tuple[int]]]:
        """
        Retrieve all hyperedges of a given type and arity.

        Parameters
        ----------
        arity : int
            The number of atoms in the interaction (e.g. 2 for bonds, 3 for angles).
        Returns
        -------
        dict[set[tuple[int]]]
            A dict of a sets of atom index tuples defining the stored interactions.
        """
        return self.edges[arity]

    def get_tensor(self, arity: int, label: str, *, overwrite: bool = False) -> torch.Tensor:
        """
        Return interaction indices as a (M, arity) LongTensor and cache it.
    
        Parameters
        ----------
        arity : int
            Arity of the interaction (e.g., 2 for bonds).
        label : str
            Type label of the interaction (e.g., "bond", "angle").
        overwrite : bool, optional
            If True, always recompute from the internal set and replace any cached tensor.
            If False, return cached tensor if it exists.
    
        Returns
        -------
        torch.Tensor
            Tensor of shape (M, arity) with atom index tuples.
        """
        key = (arity, label)
    
        if not overwrite and key in self._tensor_cache:
            return self._tensor_cache[key]
    
        # Recompute from set and overwrite cache
        idx_set = self.get(arity, label)
        tensor = torch.tensor(
            list(idx_set),
            dtype=torch.long,
            device=self.device
        )
        self._tensor_cache[key] = tensor
        return tensor

    # --- misc --------------------------------------------------------
    def __repr__(self):
        entries = [
            f"{label} (arity {arity})"
            for arity, group in self.edges.items()
            for label in group
        ]
        return "Topology(" + ", ".join(entries) + ")"