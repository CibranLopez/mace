import numpy  as np
import pandas as pd
import warnings
import os

from pymatgen.io.vasp.outputs    import Vasprun
from pymatgen.core               import Structure
from mace.calculators            import mace_mp
from ase                         import units
from ase.md                      import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.vasp                 import read_vasp, write_vasp, write_vasp_xdatcar
from ase.io.trajectory           import Trajectory
from ase.optimize                import BFGS
from ase.constraints             import ExpCellFilter

# To suppress warnings for clearer output
warnings.simplefilter('ignore')

def is_relaxation_folder_valid(path_to_relaxation):
    """Determines whether path_to_relaxation contains a vasprun.xml file or not.
    It returns True (valid) if the file does exist.
    
    Args:
        path_to_relaxation (str): path to the folder containing VASP file from an ionic relaxation.
    
    Returns:
        (bool): true if the relaxation is valid, else false.
    """
    
    if os.path.exists(f'{path_to_relaxation}/vasprun.xml'):
        return True
    return False


def clean_vasprun(path_to_relaxation):
    """Rewrite those lines from the vasprun like '<field>  </field>', as somentimes invalid characters appear.
    
    Args:
        path_to_relaxation (str): path to the folder which contains the vasprun.xml file.
    
    Returns:
        None
    """
    
    # Load lines
    with open(f'{path_to_relaxation}/vasprun.xml', 'r') as file:
        lines = file.readlines()

    # Rewrite them, avoiding invalid characters
    with open(f'{path_to_relaxation}/vasprun.xml', 'w') as file:
        for line in lines:
            split_line = line.split()
            if split_line[0][:7] == '<field>':
                    if (len(split_line) == 2) and (split_line[0][1][:7]):
                        file.write('     <field>  </field>\n')
                    else:
                        file.write(line)
            else:
                file.write(line)


def extract_vaspruns_GdCeO2(path_to_dataset, ionic_steps_to_skip=0):
    """Generates a Pandas DataFrame with the data from each simulation in the path (identifier, strucutre, energy, forces, stresses, charge). It gathers different relaxation steps under the same charge state, and different deformations of the charge state under the same defect state (just as different ionic steps). It assumes the following disposition:

    Stoichiometricity
        Temperature
            Configuration
                run
                    vasprun.xml (with several ionic steps)

    And the dataframe will be disposed as:

    Stoichiometricity
        Temperature
            Configuration

    Args:
        path_to_dataset     (str): Path to tree database containing different level of theory calculations.
        ionic_steps_to_skip (int): Number of ionic steps to skip between each load.

    Returns:
        m3gnet_dataset (Pandas DataFrame): DataFrame with information of simulations in multicolumn format.
    """

    # Initialize the data dictionary
    data = {}

    # Initialize dataset with MP format
    columns = ['structure', 'energy', 'force', 'stress']

    # Iterate over materials and relaxations in the dataset
    for stoichiometricity in os.listdir(path_to_dataset):
        # Define path to material
        path_to_stoichiometricity = f'{path_to_dataset}/{stoichiometricity}'

        # Check if it is a folder
        if not os.path.isdir(path_to_stoichiometricity):
            continue

        print()
        print(stoichiometricity)

        # Run over all stoichiometricities
        for temperature in os.listdir(path_to_stoichiometricity):
            # Define path to temperature
            path_to_temperature = f'{path_to_stoichiometricity}/{temperature}'

            # Check if it is a folder
            if not os.path.isdir(path_to_temperature):
                continue

            print()
            print(temperature)

            # Run over all configurations
            for configuration in os.listdir(path_to_temperature):
                # Define path to configuration
                path_to_configuration = f'{path_to_temperature}/{configuration}'

                # Check if it is a folder
                if not os.path.isdir(path_to_configuration):
                    continue

                print()
                print(configuration)

                structure_list = []
                energy_list = []
                forces_list = []
                stress_list = []
                # Run over all runs
                for run in os.listdir(path_to_configuration):
                    # Define path to run
                    path_to_run = f'{path_to_configuration}/{run}'

                    # Check if it is a folder
                    if not os.path.isdir(path_to_configuration):
                        continue

                    print()
                    print(configuration)

                    # Load data from relaxation
                    try:
                        # Try to load those unfinished relaxations as well
                        vasprun = Vasprun(f'{path_to_run}/vasprun.xml', exception_on_bad_xml=False)
                    except:
                        print('Error: vasprun not correctly loaded.')
                        continue

                    # Run over ionic steps
                    skip_ionic_step = 0
                    for ionic_step_idx, ionic_step in enumerate(vasprun.ionic_steps):
                        if skip_ionic_step-1 == ionic_steps_to_skip:
                            skip_ionic_step = 0

                        if not skip_ionic_step:
                            # Extract data from each ionic step
                            temp_structure = ionic_step['structure']
                            temp_energy    = ionic_step['e_fr_energy']
                            temp_forces    = ionic_step['forces']
                            temp_stress    = ionic_step['stress']

                            # Stresses obtained from VASP calculations (default unit is kBar) should be multiplied by -0.1
                            # to work directly with the model
                            temp_stress = np.array(temp_stress)
                            temp_stress *= -0.1
                            temp_stress = temp_stress.tolist()

                            # Append data
                            structure_list.append(temp_structure)
                            energy_list.append(temp_energy)
                            forces_list.append(temp_forces)
                            stress_list.append(temp_stress)

                            # Counter to zero
                            skip_ionic_step = 0

                        # New step added
                        skip_ionic_step += 1

                # Update main data object
                data.update({(stoichiometricity, temperature, configuration):
                                [structure_list, energy_list, forces_list, stress_list]})

    # Convert to Pandas DataFrame
    m3gnet_dataset = pd.DataFrame(data, index=columns)
    return m3gnet_dataset


def extract_vaspruns_dataset(path_to_dataset, load_stresses=False, energy_threshold=None, ionic_steps_to_skip=1):
    """Generates a xyz file database from each simulation in the path.
    
   Args:
        path_to_dataset  (str):         Path to tree database containing different level of theory calculations.
        energy_threshold (float, None): If set, maximum energy of some ionic step to be included.
    
    Returns:
        m3gnet_dataset (Pandas DataFrame): DataFrame with information of simulations in multicolumn format (material, defect state, ionic step).
    """
    
    # Find all paths to any folder containing a vasprun.xml file within path_to_dataset
    paths_to_vaspruns = []
    for root, _, files in os.walk(path_to_dataset):
        if 'vasprun.xml' in files:
            paths_to_vaspruns.append(root)

    # Gather relaxations from different deformations as different ionic steps
    for paths_to_vasprun in paths_to_vaspruns:
        # Remove invalid characters from the vasprun.xml file
        #clean_vasprun(paths_to_vasprun)  # Uncomment is it happens to you as well!!

        # Load data from relaxation
        try:
            # Try to load those unfinished relaxations as well
            vasprun = Vasprun(f'{paths_to_vasprun}/vasprun.xml', exception_on_bad_xml=False)
        except:
            print('Error: vasprun not correctly loaded.')
            continue

        with open(f'{path_to_dataset}/file.xyz', 'w') as file:
            # Run over ionic steps
            for ionic_step_idx, ionic_step in enumerate(vasprun.ionic_steps):
                if ionic_step_idx%ionic_steps_to_skip == 0:
                    # Extract data from each ionic step
                    structure = ionic_step['structure']
                    energy    = ionic_step['e_fr_energy']
                    forces    = ionic_step['forces']
                    #stress    = ionic_step['stress']
                    
                    if load_stresses:
                        # Stresses obtained from VASP calculations, default unit is kBar, to eV/A^3
                        # https://github.com/ACEsuit/mace/discussions/542
                        stress = np.array(stress)
                        stress = stress / 1602

                    # Write the number of atoms
                    file.write(f"{len(structure)}\n")

                    lattice    = ' '.join(map(str, structure.lattice.matrix.flatten()))
                    if load_stresses:
                        stress_str = ' '.join(map(str, stress.flatten()))

                    # Write the metadata
                    if load_stresses:
                        file.write(f"Lattice=\"{lattice}\" Properties=species:S:1:pos:R:3:forces:R:3 energy={energy} stress=\"{stress_str}\" pbc=\"T T T\"\n")
                    else:
                        file.write(f"Lattice=\"{lattice}\" Properties=species:S:1:pos:R:3:forces:R:3 energy={energy} pbc=\"T T T\"\n")

                    # Write atom data
                    for idx, _ in enumerate(structure):
                        atom = str(structure[idx].specie)
                        position = " ".join(map(str, structure[idx].coords))
                        force = " ".join(map(str, forces[idx]))
                        file.write(f"{atom} {position} {force}\n")


def extract_vaspruns_GdCeO2(path_to_dataset, energy_threshold=None, ionic_steps_to_skip=0):
    """Generates a Pandas DataFrame with the data from each simulation in the path (identifier, strucutre, energy, forces, stresses, charge). It gathers different relaxation steps under the same charge state, and different deformations of the charge state under the same defect state (just as different ionic steps). It assumes the following disposition:
    
    Concentration
        Temperature
            Configuration
                vasprun.xml (with several ionic steps)
    
    And the dataframe will be disposed likewise.
    
    Args:
        path_to_dataset     (str):         Path to tree database containing different level of theory calculations.
        energy_threshold    (float, None): If set, maximum energy of some ionic step to be included.
        ionic_steps_to_skip (int, 0):      If set, we skip those ionic steps each time we save one.
    
    Returns:
        dataset (Pandas DataFrame): DataFrame with information of simulations in multicolumn format (material, defect state, ionic step).
    """
    
    # Initialize the data dictionary
    data = {}
    
    # Initialize dataset with MP format
    columns = ['structure', 'energy', 'force', 'stress']
    
    # Iterate over concentrations of Gd-Ce
    for concentration in os.listdir(path_to_dataset):
        # Define path to material
        path_to_concentration = f'{path_to_dataset}/{concentration}'

        # Check if it is a folder
        if not os.path.isdir(path_to_concentration):
            continue

        print()
        print(concentration)
        
        # Run over temperatures
        for temperature in os.listdir(path_to_concentration):
            # Define path to material
            path_to_temperature = f'{path_to_concentration}/{temperature}'
    
            # Check if it is a folder
            if not os.path.isdir(path_to_temperature):
                continue
            
            print(f'\t{temperature}')
            
            # Run over configurations
            for configuration in os.listdir(path_to_temperature):
                # Define path to relaxation loading every relaxation step of a same defect state in the same data column
                path_to_configuration = f'{path_to_temperature}/{configuration}'
                
                # Avoiding non-directories (such as .DS_Store)
                if not os.path.isdir(path_to_configuration):
                    continue
            
                print(f'\t{configuration}')
                
                # Remove invalid characters from the vasprun.xml file
                clean_vasprun(path_to_configuration)
                
                # Load data from relaxation
                try:
                    # Try to load those unfinished relaxations as well
                    vasprun = Vasprun(f'{path_to_configuration}/vasprun.xml', exception_on_bad_xml=False)
                except:
                    print('Error: vasprun not correctly loaded.')
                    continue

                # Run over ionic steps
                skip_ionic_step = 0
                for ionic_step_idx, ionic_step in enumerate(vasprun.ionic_steps):
                    if skip_ionic_step-1 == ionic_steps_to_skip:
                        skip_ionic_step = 0

                    if not skip_ionic_step:
                        # Extract data from each ionic step
                        temp_structure = ionic_step['structure']
                        temp_energy    = ionic_step['e_fr_energy']
                        temp_forces    = ionic_step['forces']
                        temp_stress    = ionic_step['stress']

                        # Stresses obtained from VASP calculations (default unit is kBar) should be multiplied by -0.1
                        # to work directly with the model
                        temp_stress = np.array(temp_stress)
                        temp_stress *= -0.1
                        temp_stress = temp_stress.tolist()

                        # Generate a dictionary object with the new data
                        new_data = {(concentration, temperature, configuration, ionic_step_idx):
                                    [temp_structure, temp_energy, temp_forces, temp_stress]}

                        # Update in the main data object
                        data.update(new_data)

                        # Counter to zero
                        skip_ionic_step = 0

                    # New step added
                    skip_ionic_step += 1

    # Convert to Pandas DataFrame and return
    return pd.DataFrame(data, index=columns)


def extract_OUTCAR_dataset(path_to_dataset):
    """Generates a Pandas DataFrame with the data from each simulation in the path (identifier, strucutre, energy, forces, stresses). It gathers different relaxation steps under the same charge state, and different deformations of the charge state under the same defect state (just as different ionic steps).
    
    Args:
        path_to_dataset (str): Path to tree database containing different folder with a calculation.
    
    Returns:
        m3gnet_dataset (Pandas DataFrame): DataFrame with information of simulations in multicolumn format (material, defect state, ionic step).
    """

    # Initialize the data dictionary
    data = {}

    # Initialize dataset with MP format
    #columns = ['structure', 'energy', 'force', 'stress']
    columns = ['structure', 'energy', 'force']

    # Iterate over materials and relaxations in the dataset
    for dir_name in os.listdir(path_to_dataset):
        # Define path to material
        path_to_dir = f'{path_to_dataset}/{dir_name}'

        # Check if it is a folder
        if not os.path.isdir(path_to_dir):
            continue

        # Check if POSCAR is in folder
        POSCAR_filename = f'{path_to_dir}/POSCAR'
        if not os.path.exists(POSCAR_filename):
            POSCAR_filename = f'{path_to_dir}/CONTCAR'
            if not os.path.exists(POSCAR_filename):
                continue

        # Check if OUTCAR is in folder
        OUTCAR_filename = f'{path_to_dir}/OUTCAR'
        if not os.path.exists(OUTCAR_filename):
            continue

        print()
        print(dir_name)

        try:
            with open(POSCAR_filename, 'r') as file:
                for _ in range(5):
                    _ = file.readline()

                composition   = file.readline().split()
                concentration = np.array(file.readline().split(), dtype=int)
                n_atoms       = np.sum(concentration)
                species       = [val for val, count in zip(composition, concentration) for _ in range(count)]

            with open(OUTCAR_filename, 'r') as file:
                ionic_step_idx = 0
                line = file.readline()
                while True:
                    ### Read cell


                    while line != ' VOLUME and BASIS-vectors are now :\n':
                        line = file.readline()
                        if not line: break

                    # Skip intermediate lines
                    for _ in range(4):
                        file.readline()

                    # Append cell
                    temp_cell = []
                    for _ in range(3):
                        temp_cell.append(file.readline().split()[:3])

                    # Convert to arrays
                    temp_cell = np.array(temp_cell, dtype=float)


                    ### Read forces


                    while line != ' POSITION                                       TOTAL-FORCE (eV/Angst)\n':
                        line = file.readline()
                        if not line: break

                    # Skip intermediate lines
                    file.readline()

                    # Read data if available
                    temp_positions = []
                    temp_forces    = []

                    for _ in range(n_atoms):
                        # Read line and split
                        line       = file.readline()
                        split_line = line.split()

                        # Append positions and forces
                        temp_positions.append(split_line[:3])
                        temp_forces.append(np.array(split_line[3:], dtype=float).tolist())

                    # Convert to arrays
                    temp_positions = np.array(temp_positions, dtype=float).tolist()
                    temp_forces    = np.array(temp_forces,    dtype=float)

                    ### Read energy


                    while line != '  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n':
                        line = file.readline()
                        if not line: break

                    # Skip intermediate lines
                    for _ in range(3):
                        file.readline()

                    # Read energy
                    temp_energy = float(file.readline().split()[-1])


                    ### Generate data entry


                    # Create the Structure object
                    temp_structure = Structure(temp_cell, species, temp_positions)

                    # Generate a dictionary object with the new data
                    #new_data = {
                    #    (dir_name, str(ionic_step_idx)): [temp_structure, temp_energy, temp_forces, temp_stress]}
                    new_data = {
                        (dir_name, str(ionic_step_idx)): [temp_structure, temp_energy, temp_forces]}

                    # Update in the main data object
                    data.update(new_data)

                    # Update ionic step
                    ionic_step_idx += 1
        except:
            continue
    # Convert to Pandas DataFrame
    m3gnet_dataset = pd.DataFrame(data, index=columns)
    return m3gnet_dataset


def compute_offset(computed_energies, predicted_energies):
    """Computes how accurate the predictions are globally (the offset between predicted and computed energies), defined as:
    
    d_1 = || E^{DFT} - E^{ML-IAP} ||
    
    Args:
        computed_energies  (1D array): DFT computed energies at different ionic steps (typically in eV/supercell).
        predicted_energies (1D array): ML-IAP computed energies at different ionic steps (typically in eV/supercell).
    
    Returns:
        offset (float): euclidean distance between both curves (typically in eV/supercell).
    """
    
    # Euclidean definition
    offset = np.mean(computed_energies - predicted_energies)
    return offset


def compute_accuracy(computed_energies, predicted_energies, offset):
    """Computes how accurate the predictions are in terms of curve reproduction (the difference between predicted and computed energies minus the offset), defined as:
    
    d_2 = || E^{DFT} - E^{ML-IAP} - d_1 ||
    
    Args:
        computed_energies  (1D array): DFT computed energies at different ionic steps (typically in eV/supercell).
        predicted_energies (1D array): ML-IAP computed energies at different ionic steps (typically in eV/supercell).
        offset (float): euclidean distance between both curves (typically in eV/supercell).
    
    Returns:
        accuracy (float): euclidean distance between both curves extracting the offset (typically in eV/supercell).
    """
    
    # Euclidean definition
    accuracy = np.mean(computed_energies - predicted_energies - offset)
    return accuracy


def structural_relaxation(
        path_to_structure,
        model_load_path='large',
        device='cuda',
        dispersion=False,
        relax_cell=True,
        fmax=0.05,
        output_folder='./'
):
    """
        Perform structural relaxation of a molecular or crystalline structure.

        This function facilitates the relaxation of a structure file using a pre-trained
        machine learning potential. The relaxation can be constrained to keep the simulation
        cell fixed or allow the relaxation of both positions and the cell itself. The relaxed
        structure is saved to a specified output directory.

        Parameters:
            path_to_structure (str):   Path to the file containing the structure in VASP format.
            model_load_path   (str):   Path to the pre-trained model.
            relax_cell        (bool):  A boolean value indicating whether to relax the simulation cell
                along with atomic positions. Defaults to True.
            fmax              (float): Maximum force tolerance in eV/Å for stopping the relaxation process.
                Defaults to 0.05.

        Returns:
            atoms (Atoms): ASE Atoms object representing the relaxed structure.
    """

    # Load the relaxed structure
    atoms = read_vasp(file=path_to_structure)

    # Load the pre-trained model
    atoms.calc = mace_mp(model=model_load_path, device=device, dispersion=dispersion, default_dtype='float64')

    # Check whether to relax the cell
    if relax_cell:
        atoms = ExpCellFilter(atoms)

    # Relax the structure
    dyn = BFGS(atoms, trajectory=f'{output_folder}/run.traj')
    dyn.run(fmax=fmax)

    if relax_cell:
        atoms = atoms.atoms

    write_vasp(f'{output_folder}/CONTCAR', atoms=atoms, direct=True, sort=True)
    return atoms


def single_shot_energy_calculation(
        path_to_structure,
        model_load_path='large',
        device='cuda',
        dispersion=False
):
    """
    Determine the energy, forces, and stress of a molecular structure using a pre-trained MACE model.

    This function loads a pre-trained molecular model, reads a relaxed molecular structure from a file,
    and computes the potential energy, atomic forces, and stress tensor for the given molecular configuration.

    Parameters:
        path_to_structure (str):  Path to the file containing the molecular structure
            in VASP format.
        model_load_path   (str):  Path to the pre-trained MACE model file. Default is the 'large' model.
        device            (str):  Device to run the computations on, e.g., 'cuda' for GPU or
            'cpu' for CPU. Default is 'cuda'.
        dispersion        (bool): Whether to include the D3 dispersion correction in the model.

    Returns:
        float:         The computed potential energy of the molecular structure.
        numpy.ndarray: The computed forces on every atom in the molecular structure.
        numpy.ndarray: The computed stress tensor of the molecular structure.

    Raises:
        ValueError:   If the molecular structure file is invalid or cannot be read.
        RuntimeError: If the MACE model fails to run the computations due to an
            incompatible model or device.
    """

    # Load the relaxed structure
    atoms = read_vasp(file=path_to_structure)

    # Load the pre-trained model
    atoms.calc = mace_mp(model=model_load_path, device=device, dispersion=dispersion, default_dtype='float64')

    # Determine energy
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    return energy, forces, stress


def molecular_dynamics(
        path_to_structure,
        model_load_path='large',
        device='cuda',
        dispersion=False,
        temperature=300,
        timestep=1,
        friction=0.001,
        n_steps=200,
        output_folder='./'
):
    """
    Conducts a molecular dynamics simulation on a given atomic structure using a pre-trained model
    and Langevin dynamics for an NVT ensemble. Initializes atomic velocities based on a Maxwell-Boltzmann
    distribution, applies the provided force field, and evolves the system for a specified number of steps.

    Parameters:
        path_to_structure (str):            Path to the input atomic structure file in VASP format.
        model_load_path   (str, optional):  Path or identifier to load the pre-trained model. Defaults to 'large'.
        device            (str, optional):  Device used for computation, e.g., 'cuda' or 'cpu'. Defaults to 'cuda'.
        dispersion        (bool, optional): Specifies whether to include dispersion corrections in the model.
            Defaults to False.
        temperature       (float, optional): Initial temperature in Kelvin. Defaults to 300.
        timestep          (float, optional): Timestep for the dynamics in femtoseconds. Defaults to 1.
        friction          (float, optional): Friction coefficient for Langevin dynamics. Defaults to 0.001.
        n_steps           (int, optional):   Number of simulation steps. Defaults to 200.

    Raises:
        Various exceptions may occur during file reading, model initialization, or dynamics execution.
    """

    # Load the relaxed structure
    atoms = read_vasp(file=path_to_structure)

    # Load the pre-trained model
    atoms.calc = mace_mp(model=model_load_path, device=device, dispersion=dispersion, default_dtype='float64')

    # Set units
    timestep    *= units.fs
    temperature *= units.kB
    friction    *= 1/units.fs

    # Initialize velocities.
    MaxwellBoltzmannDistribution(atoms, temperature)

    # Set up the Langevin dynamics engine for NVT ensemble
    dyn = Langevin(atoms, timestep=timestep, temperature=temperature, friction=friction, trajectory=f'{output_folder}/run.traj')
    dyn.run(n_steps)
    
    traj_to_XDATCAR(output_folder)
    write_vasp(f'{output_folder}/CONTCAR', atoms=atoms, direct=True, sort=True)
    return atoms


def traj_to_XDATCAR(path_to_simulation='./', traj_name='run.traj'):
    traj = Trajectory(f'{path_to_simulation}/{traj_name}')
    
    write_vasp_xdatcar(f'{path_to_simulation}/XDATCAR', traj)

