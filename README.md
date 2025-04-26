To extract a dataset just run:

mll.extract_vaspruns_dataset('/home/claudio/cibran/Work/UCL/mace/data/GdCeO2/test', ionic_steps_to_skip=5)

Logo para correr:

mll.molecular_dynamics(path_to_structure='/home/claudio/Desktop/comparison-finetuning/foundational-model/POSCAR', output_folder='/home/claudio/Desktop/comparison-finetuning/foundational-model', temperature=1500, model_load_path='mace-mpa-0-medium.model', n_steps=80000)


for dirpath, dirnames, filenames in os.walk('/home/claudio/Desktop/GdCeO2-MLMD-poscars'):
    if "POSCAR" in filenames:
        mll.molecular_dynamics(path_to_structure=f'{folder}/POSCAR', output_folder=folder, temperature=1500, model_load_path='mace-mpa-0-medium.model', n_steps=80000)
