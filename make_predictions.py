
from mace.calculators import mace_mp
from ase import build
from ase.io import read

macemp = mace_mp(model="mace-mp-large_2024.model",device='cuda',default_dtype='float32')

# atoms = read(<some file with structure info>)
atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)

atoms.calc = macemp

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
