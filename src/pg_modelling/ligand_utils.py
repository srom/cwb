import xml.etree.ElementTree as ET
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile

import pandas as pd
from Bio.PDB import PDBIO, PDBParser
import gemmi
from posebusters import PoseBusters
from rdkit import Chem
from rdkit.Chem import AllChem


POSEBUSTERS_CHECKS = [
    'mol_pred_loaded', 'mol_cond_loaded', 'sanitization', 'inchi_convertible', 'all_atoms_connected',
    'bond_lengths', 'bond_angles', 'internal_steric_clash', 'aromatic_ring_flatness', 'double_bond_flatness',
    'internal_energy', 'protein-ligand_maximum_distance', 'minimum_distance_to_protein', 
    'minimum_distance_to_organic_cofactors', 'minimum_distance_to_inorganic_cofactors',
    'minimum_distance_to_waters', 'volume_overlap_with_protein', 'volume_overlap_with_organic_cofactors', 
    'volume_overlap_with_inorganic_cofactors', 'volume_overlap_with_waters',
]


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


def generate_conformation(
    mol : Chem.Mol, 
    random_seed : int = 42, 
    obabel_fallback : bool = False,
    n_cpus : int = 1,
):
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = n_cpus
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
        replacement_char (str): The character to replace invalid characters with (default is '-').

    Returns:
        str: A sanitized string safe for use as a file name.
    """
    # Define invalid characters
    invalid_chars = r"[\\/;:*?\"<>|\0\[\]\(\)]"

    # Remove None from name
    sanitized = input_string.replace('None-', '').replace('-None', '').replace('None', '')

    # Replace invalid characters with the replacement character
    sanitized = re.sub(invalid_chars, replacement_char, sanitized)

    # Strip leading and trailing whitespace or replacement characters
    sanitized = sanitized.strip().strip(replacement_char)
    
    # Swap multiple relacement char for a single one
    sanitized = re.sub(f'[{replacement_char}]+', replacement_char, sanitized)

    # Trim replacement characters and spaces
    sanitized = sanitized.strip(replacement_char).strip()

    # Raise if empty
    if not sanitized:
        raise ValueError('Empty string after sanitization')

    return sanitized


def sanitize_protein_id(input_string: str, replacement_char: str = '-') -> str:
    return sanitize_ligand_name(input_string, replacement_char)


def run_pose_busters(input_mmcif : Path, ligand_id : str, full_report : bool = False):
    """
    Plausibility checks for generated molecule poses with [PoseBusters](https://github.com/maabuu/posebusters).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        protein_pdb_path = Path(tmpdir) / 'protein.pdb'
        ligand_sdf_path = Path(tmpdir) / 'ligand.sdf'
        extract_protein_and_ligand_from_mmcif(input_mmcif, protein_pdb_path, ligand_sdf_path)
        return run_pose_busters_from_ligand_and_protein(
            ligand_sdf_path,
            protein_pdb_path,
            ligand_id,
            full_report,
        )


def run_pose_busters_from_ligand_and_protein(
    ligand_sdf_path : Path, 
    protein_pdb_path : Path, 
    ligand_id : str, 
    full_report : bool = False,
):
    buster = PoseBusters(config='dock')
    res_df = buster.bust(
        mol_pred=ligand_sdf_path, 
        mol_cond=protein_pdb_path,
        mol_true=None,
        full_report=full_report,
    )

    boolean_columns = POSEBUSTERS_CHECKS
    res_df['ligand_id'] = ligand_id
    res_df = res_df.set_index('ligand_id')
    res_df['score'] = res_df.apply(lambda row: sum([row[c] for c in boolean_columns]), axis=1)
    res_df['max_score'] = len(boolean_columns)
    res_df['perfect_score'] = res_df['score'] == res_df['max_score']
    return res_df


def extract_protein_and_ligand_from_mmcif(
    input_mmcif : Path, 
    output_protein_pdb : Path,
    output_ligand_sdf : Path,
    output_pdb : Path = None,
):
    """
    Load mmCIF file and extract protein and ligand into PDB and SDF files respectively.
    This is first and foremost a helper function to prep inputs for PoseBusters.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pdb_ligand = Path(tmpdir) / 'ligand.pdb'

        if output_pdb is None:
            pdb_complex = Path(tmpdir) / 'complex.pdb'
        else:
            pdb_complex = output_pdb

        convert_mmcif_to_pdb(input_mmcif, pdb_complex)

        # Load the PDB file
        parser = PDBParser()
        structure = parser.get_structure('complex', pdb_complex.as_posix())

        # Extract protein (chain A)
        io = PDBIO()
        io.set_structure(structure[0]['A'])
        io.save(output_protein_pdb.as_posix())

        if not output_protein_pdb.is_file():
            raise ValueError('Error converting protein to PDB')

        # Extract ligand (chain B)
        io = PDBIO()
        io.set_structure(structure[0]['B'])
        io.save(temp_pdb_ligand.as_posix())

        if not temp_pdb_ligand.is_file():
            raise ValueError('Error converting ligand to PDB')

        res = subprocess.run([
            'obabel', 
            '-ipdb', temp_pdb_ligand.as_posix(),
            '-osdf', '-O', output_ligand_sdf.as_posix(),
        ])
        if res.returncode != 0 or not output_ligand_sdf.is_file():
            raise ValueError('Error converting ligand to SDF')


def run_plip(input_pdb : Path, output_dir : Path) -> int:
    return subprocess.run(
        [
            'plip', 
            '-f', input_pdb.resolve().as_posix(),
            '-o', output_dir.resolve().as_posix(),
            '-pxty',
        ],
        stdout=sys.stdout, 
        stderr=sys.stderr,
    ).returncode


def parse_plip_output(xml_file : Path) -> pd.DataFrame:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the interactions element (under the bindingsite element)
    interactions = root.find(".//bindingsite/interactions")

    data = {
        'interaction_type': [],
        'residue_name': [],
        'residue_number': [],
    }
    for group in interactions:
        for interaction in group:
            interaction_type = interaction.tag
            resnr_elem = interaction.find("resnr")
            restype_elem = interaction.find("restype")
            
            if resnr_elem is not None and restype_elem is not None:
                resnr = int(resnr_elem.text)
                restype = restype_elem.text
                data['interaction_type'].append(interaction_type)
                data['residue_name'].append(restype)
                data['residue_number'].append(resnr)
    
    out_df = pd.DataFrame(data)
    out_df['n_interactions'] = 1
    out_df = out_df.groupby(['interaction_type', 'residue_name', 'residue_number']).sum()
    return out_df.reset_index().set_index('interaction_type')


def convert_mmcif_to_pdb(input_mmcif : Path, output_pdb : Path) -> None:
    res = subprocess.run([
        'obabel', 
        '-icif', input_mmcif.as_posix(),
        '-opdb', '-O', output_pdb.as_posix(),
    ])
    if res.returncode != 0 or not output_pdb.is_file():
        raise ValueError('Error converting complex to PDB')


def gen_model_seeds(n):
    return [int(random.uniform(1, 1000)) for _ in range(n)]
