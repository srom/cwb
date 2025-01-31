from pathlib import Path
import random
import re
import subprocess
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem
import gemmi


def generate_ccd_from_mol(
    mol : Chem.Mol, 
    ligand_name : str, 
    output_path : Path = None,
    random_seed : int = 42,
    obabel_fallback : bool = False,
) -> str:
    """
    Generate a user-provided CCD file for AlphaFold 3 using RDKit for parsing
    and Gemmi for writing the mmCIF format.
    """
    try:
        conformer = mol.GetConformer()
        generate_conformation_bool = not conformer.Is3D()
    except ValueError:
        generate_conformation_bool = True

    if generate_conformation_bool:
        mol = generate_conformation(mol, random_seed, obabel_fallback)
        conformer = mol.GetConformer()

    # Extract atom and bond information
    atoms = []
    bonds = []
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    molecular_weight = Chem.rdMolDescriptors.CalcExactMolWt(mol)

    for atom in mol.GetAtoms():
        coords = conformer.GetAtomPosition(atom.GetIdx())
        atoms.append({
            "id": atom.GetIdx(),
            "element": atom.GetSymbol(),
            "charge": atom.GetFormalCharge(),
            "coords": coords,
            "leaving": "N"  # Default: not a leaving atom
        })

    bond_type_mapping = {
        Chem.BondType.SINGLE: ("SING", "N"),
        Chem.BondType.DOUBLE: ("DOUB", "N"),
        Chem.BondType.TRIPLE: ("TRIP", "N"),
        Chem.BondType.AROMATIC: ("AROM", "Y"),
    }

    for bond in mol.GetBonds():
        bond_type, aromatic_flag = bond_type_mapping.get(bond.GetBondType(), ("SING", "N"))
        bonds.append({
            "atom1": bond.GetBeginAtomIdx(),
            "atom2": bond.GetEndAtomIdx(),
            "order": bond_type,
            "aromatic": aromatic_flag
        })

    # Create the mmCIF document
    doc = gemmi.cif.Document()
    block = doc.add_new_block(f"{ligand_name}")

    # Add _chem_comp metadata
    block.set_pair('_chem_comp.id', ligand_name)
    block.set_pair('_chem_comp.name', ligand_name)
    block.set_pair('_chem_comp.type', 'non-polymer')
    block.set_pair('_chem_comp.formula', formula)
    block.set_pair('_chem_comp.formula_weight', f"{molecular_weight:.3f}")
    block.set_pair('_chem_comp.mon_nstd_parent_comp_id', '?')
    block.set_pair('_chem_comp.pdbx_synonyms', '?')

    # Add atoms to _chem_comp_atom
    atom_loop = block.init_loop('_chem_comp_atom.', [
        'comp_id', 'atom_id', 'type_symbol', 'charge',
        'pdbx_model_Cartn_x_ideal', 'pdbx_model_Cartn_y_ideal', 
        'pdbx_model_Cartn_z_ideal', 'pdbx_leaving_atom_flag',
    ])
    for atom in atoms:
        atom_loop.add_row([
            ligand_name, f"{atom['element']}{atom['id'] + 1}", atom['element'], str(atom['charge']),
            f"{atom['coords'].x:.3f}", f"{atom['coords'].y:.3f}", f"{atom['coords'].z:.3f}", atom['leaving']
        ])

    # Add bonds to _chem_comp_bond
    bond_loop = block.init_loop('_chem_comp_bond.', [
        'comp_id', 'atom_id_1', 'atom_id_2', 'value_order', 'pdbx_aromatic_flag',
    ])
    for bond in bonds:
        atom1 = f"{atoms[bond['atom1']]['element']}{bond['atom1'] + 1}"
        atom2 = f"{atoms[bond['atom2']]['element']}{bond['atom2'] + 1}"
        bond_loop.add_row([ligand_name, atom1, atom2, bond['order'], bond['aromatic']])

    # Write the mmCIF file
    if output_path is not None:
        doc.write_file(output_path.resolve().as_posix())

    return doc.as_string()


def generate_ccd_from_pdb(input_pdb : Path, ligand_name : str) -> str:
    mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)
    if mol is None:
        raise ValueError("Failed to parse the input PDB file. Ensure it contains valid ligand information.")
    
    return generate_ccd_from_mol(mol, ligand_name)


def generate_ccd_from_smiles(smiles : str, ligand_name : str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Failed to parse the input SMILES. Ensure it contains valid ligand information.")
    
    return generate_ccd_from_mol(mol, ligand_name)


def run_obabel_gen3d(smiles : str, output_path : Path) -> Chem.Mol:
    output_path_str = output_path.resolve().as_posix()
    subprocess.run(
        [
            'obabel', '-:' + smiles, 
            '-O', output_path_str, 
            '--gen3d', 
            '--best',
        ],
        check=True
    )
    return Chem.MolFromPDBFile(output_path_str)


def generate_conformation(mol : Chem.Mol, random_seed : int = 42, obabel_fallback : bool = False):
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    conformer_id = AllChem.EmbedMolecule(mol, params)

    if conformer_id == -1 and obabel_fallback:
        print('RDKit ETKDGv3 failed - falling back on obabel')
        smiles = Chem.MolToSmiles(mol)
        with tempfile.NamedTemporaryFile(suffix='.pdb') as f:
            mol = run_obabel_gen3d(smiles, Path(f.name))
    
    elif conformer_id == -1:
        raise ValueError('RDKit ETKDGv3 failed to generate a 3D conformation')

    conformer = mol.GetConformer()
    assert conformer.Is3D()

    AllChem.UFFOptimizeMolecule(mol)

    return mol


def sanitize_ligand_name(input_string: str, replacement_char: str = '-') -> str:
    """
    Sanitizes a string to make it compatible for use as a file name on POSIX, Windows, and Mac systems.

    Args:
        input_string (str): The input string to sanitize.
        replacement_char (str): The character to replace invalid characters with (default is '_').

    Returns:
        str: A sanitized string safe for use as a file name.
    """
    # Define the invalid characters for file names on POSIX, Windows, and Mac
    invalid_chars = r"[\\/;:*?\"<>|\0\[\]\(\)]"

    # Define reserved names for Windows that cannot be used as file names
    reserved_names = {
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    }

    # Special case for (NAG)(NAM)
    sanitized = input_string.replace(')(', replacement_char)

    # Remove None when no glycans
    sanitized = sanitized.replace('None-', '').replace('-None', '').replace('None', '')

    # Replace invalid characters with the replacement character
    sanitized = re.sub(invalid_chars, replacement_char, sanitized)

    # Strip leading and trailing whitespace or replacement characters
    sanitized = sanitized.strip().strip(replacement_char)

    # Ensure the filename is not a reserved name on Windows
    if sanitized.upper() in reserved_names:
        sanitized = f"{sanitized}{replacement_char}"
    
    # Swap multiple relacement char for a single one
    sanitized = re.sub(f'[{replacement_char}]+', replacement_char, sanitized)

    # Trim replacement characters and spaces
    sanitized = sanitized.strip(replacement_char).strip()

    # Raise if empty
    if not sanitized:
        raise ValueError('Empty string after sanitization')

    return sanitized


def gen_model_seeds(n):
    return [int(random.uniform(1, 100)) for _ in range(n)]
