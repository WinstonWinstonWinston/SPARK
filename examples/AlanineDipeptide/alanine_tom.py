import os
import parmed as pmd
import torch
import spark.system.units as units
import spark.system.box as box
import spark.system.system as sys
from spark.utils import *
from tqdm import tqdm
dtype=torch.float32
device="cuda"

top, node_features, mass, energy_dict = build_top_and_features("alanine-dipeptide.prmtop")
B = 512*16
print(B)
pos = torch.tensor(pmd.load_file("alanine-dipeptide.pdb").coordinates,dtype=dtype,device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
atomic_numbers = [a.atomic_number for a in pmd.load_file("alanine-dipeptide.pdb").atoms]
b = box.Box([1000,1000,1000],["s","s","s"])
u = units.UnitSystem.akma()
mom = 0.5*torch.randn_like(pos)

S = sys.System(pos, mom, mass, top, b, energy_dict, u, node_features)
S.potential_energy()
S.compile_force_fn()

S.pos = pos + 0.01*torch.randn_like(pos)
print("Force", S.force())
print("Energy:",S.potential_energy())
S.reset_cache()


for i in tqdm(range(1000)):
    S.pos = pos + 0.01*torch.randn_like(pos)
    S.force()
    # print("Force", S.force())
    # print("Energy:",S.potential_energy())
    S.reset_cache()
