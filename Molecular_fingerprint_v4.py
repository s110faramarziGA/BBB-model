# refactored_molecular_fp.py

import os
import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO)

# standard atomic masses
MM_of_Elements = {
    'H': 1.0079, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998,
    'S': 32.06, 'P': 30.974, 'Cl': 35.45, 'Br': 79.904, 'I': 126.90
}

def molecular_mass(formula: dict):
    return sum(MM_of_Elements.get(el, 0) * count for el, count in formula.items())

def get_distance_matrix(coords):
    return cdist(coords, coords)

def get_element_counts(df):
    return df['element_symbol'].value_counts().to_dict()

def extract_rdkit_features(mol):
    return {
        'mlogp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'n_rotatable': Lipinski.NumRotatableBonds(mol),
        'n_h_acceptors': Lipinski.NumHAcceptors(mol),
        'n_h_donors': Lipinski.NumHDonors(mol)
    }

SMARTS_PATTERNS = {
    'amide': '[CX3](=[OX1])[NX3H2]',
    'carboxylic_acid': 'C(=O)[OH]',
    'ester': 'C(=O)O[C]',
    'ether': 'C-O-C',
    'ketone': 'C(=O)C',
    'primary_amine': '[NX3H2]',
    'secondary_amine': '[NX3H1]',
    'tertiary_amine': '[NX3]([C])([C])[C]',
    'benzene': 'c1ccccc1'
    # you can add more as needed
}

def detect_functional_groups(mol):
    results = {}
    for name, smarts in SMARTS_PATTERNS.items():
        patt = Chem.MolFromSmarts(smarts)
        matches = mol.GetSubstructMatches(patt)
        results[name] = len(matches)
    return results

def mol_fp(pdb_file):
    logging.info(f"Processing: {pdb_file}")
    ppdb = PandasPdb().read_pdb(pdb_file)
    atoms = ppdb.df['HETATM'] if not ppdb.df['HETATM'].empty else ppdb.df['ATOM']

    if atoms.empty:
        raise ValueError(f"No atoms found in {pdb_file}")

    coords = atoms[['x_coord', 'y_coord', 'z_coord']].values
    d_matrix = get_distance_matrix(coords)
    element_counts = get_element_counts(atoms)
    mw = molecular_mass(element_counts)

    # convert to RDKit molecule
    mol = Chem.rdmolfiles.MolFromPDBFile(pdb_file, removeHs=False)
    if not mol:
        raise ValueError(f"RDKit failed to parse {pdb_file}")

    features = extract_rdkit_features(mol)
    functional_groups = detect_functional_groups(mol)

    # merge features
    fp = {
        'molecular_mass': mw,
        **features,
        **functional_groups
    }

    return pd.DataFrame([fp])

def run_all(fp_dir, output_file='fpArray_2.csv'):
    fp_array = pd.DataFrame()
    success_count = 0
    failure_count = 0
    failures = []

    for fname in os.listdir(fp_dir):
        full_path = os.path.join(fp_dir, fname)
        if os.path.isfile(full_path):
            try:
                fp = mol_fp(full_path)
                fp_array = pd.concat([fp_array, fp])
                success_count += 1
                logging.info(f"Finished {success_count}: {fname}")
            except Exception as e:
                failure_count += 1
                failures.append(fname)
                logging.warning(f"Skipping {fname}: {e}")

    fp_array.to_csv(output_file, index=False)
    logging.info(f"Saved results to {output_file}")
    logging.info(f"Summary: {success_count} fingerprints generated successfully.")
    logging.info(f"Summary: {failure_count} failures.")
    if failures:
        logging.warning(f"The following files failed: {failures}")


if __name__ == "__main__":
    working_dir = os.getcwd()
    pdb_dir = os.path.join(working_dir, 'Structures', 'pdb')
    run_all(pdb_dir)
