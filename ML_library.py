import numpy  as np
import pandas as pd
import matgl
import warnings
import os

from sklearn.model_selection     import train_test_split
from pymatgen.io.vasp.outputs    import Vasprun
from pymatgen.io.vasp.inputs     import Poscar
from pymatgen.core               import Structure
from matgl.ext.ase               import M3GNetCalculator, Relaxer
from pymatgen.io.ase             import AseAtomsAdaptor

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


def split_data(data, test_size=0.2, validation_size=0.2, random_state=None):
    """
    Split a Pandas DataFrame into training, validation, and test sets.

    Args:
        data (DataFrame): The input dataset to be split.
        test_size (float): The proportion of data to include in the test set (default: 0.2).
        validation_size (float): The proportion of data to include in the validation set (default: 0.2).
        random_state (int or None): Seed for the random number generator (default: None).

    Returns:
        train_set (DataFrame): The training dataset.
        validation_set (DataFrame): The validation dataset.
        test_set (DataFrame): The test dataset.
    """
    
    # First, split the data into training and temporary data (temporary_data = validation + test)
    train_data, temp_data = train_test_split(data, test_size=(validation_size + test_size), random_state=random_state)

    # Next, split the temporary data into validation and test data
    validation_data, test_data = train_test_split(temp_data, test_size=(test_size / (validation_size + test_size)), random_state=random_state)

    return train_data, validation_data, test_data


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


def extract_vaspruns_dataset(path_to_dataset, energy_threshold=None, ionic_steps_to_skip=1):
    """Generates a Pandas DataFrame with the data from each simulation in the path.
    
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
                    stress    = ionic_step['stress']

                    # Write the number of atoms
                    file.write(f"{len(structure)}\n")

                    lattice    = ' '.join(map(str, structure.lattice.matrix.flatten()))
                    stress_str = ' '.join(map(str, np.array(stress).flatten()))

                    # Write the metadata
                    file.write(f"Lattice=\"{lattice}\" Properties=species:S:1:pos:R:3:forces:R:3 energy={energy} stress=\"{stress_str}\" pbc=\"T T T\"\n")

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


def structural_relaxation(path_to_POSCAR, model_load_path, verbose=True, relax_cell=True):
    """
    Perform structural relaxation on a given structure.

    Args:
        path_to_POSCAR  (str):  Path to the input structure (POSCAR).
        model_load_path (str):  Path to the pre-trained model for relaxation.
        verbose         (bool): Verbosity of the relaxation process.
        relax_cell      (bool): Whether to relax the lattice cell.

    Returns:
        poscar_relaxed (pymatgen structure): Relaxed structure saved as a POSCAR object.
    """


    # Load the structure to be relaxed
    atoms_ini = Structure.from_file(f'{path_to_POSCAR}/POSCAR')

    # Load the default pre-trained model
    try:
        pot = matgl.load_model(model_load_path)
    except ValueError:
        pot = matgl.load_model('M3GNet-MP-2021.2.8-PES')
        pot.model.load(model_load_path)
    
    relaxer = Relaxer(potential=pot, relax_cell=relax_cell)

    # Relax the structure
    relax_atoms_ini = relaxer.relax(atoms_ini, verbose=verbose)
    atoms = relax_atoms_ini['final_structure']

    # Save the relaxed structure as a POSCAR file
    poscar_relaxed = Poscar(atoms)
    poscar_relaxed.write_file(f'{path_to_POSCAR}/CONTCAR')
    return poscar_relaxed


def single_shot_energy_calculations(path_to_structure, model_load_path):
    """
    Calculate the potential energy of a relaxed structure using a pre-trained model.

    Args:
        path_to_structure (str): Path to the relaxed structure (CONTCAR).
        model_load_path   (str): Path to the pre-trained model for energy calculation.

    Returns:
        ssc_energy (float): Potential energy of the structure.
    """
    
    # Load the relaxed structure
    atoms = Structure.from_file(f'{path_to_structure}')
    
    # Load the default pre-trained model
    pot = matgl.load_model(model_load_path)
    relaxer = Relaxer(potential=pot)

    # Define the M3GNet calculator
    calc = M3GNetCalculator(pot)

    # Load atoms adapter and adapt structure
    ase_adaptor = AseAtomsAdaptor()
    adapted_atoms = ase_adaptor.get_atoms(atoms)

    # Calculate potential energy
    adapted_atoms.set_calculator(calc)
    
    # Extract the energy
    ssc_energy = float(adapted_atoms.get_potential_energy())
    return ssc_energy
