import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

from molecules.conversion import (
    mols_from_smiles, mol_to_smiles, mols_to_smiles, canonicalize)
from molecules.fragmentation import fragment_iterative, reconstruct
from molecules.properties import add_property
from molecules.structure import (
    add_atom_counts, add_bond_counts, add_ring_counts)
from utils.config import DATA_DIR, get_dataset_info


def fetch_dataset(name):
    info = get_dataset_info(name)
    filename = Path(info['filename'])
    url = info['url']
    unzip = info['unzip']

    folder = Path("/home/aita130/drug_discovery/ldm/fragment-based-dgm-master-copy/temp").absolute()
    if not folder.exists():
        os.makedirs(folder)

    os.system(f'wget -P {folder} {url}')
    #正在连接 raw.githubusercontent.com (raw.githubusercontent.com)|::|:443... 失败：拒绝连接。

    raw_path = DATA_DIR / name / 'RAW'
    if not raw_path.exists():
        os.makedirs(raw_path)

    processed_path = DATA_DIR / name / 'PROCESSED'
    if not processed_path.exists():
        os.makedirs(processed_path)

    path = folder / filename

    if unzip is True:
        if ".tar.gz" in info['url']:
            os.system(f'tar xvf {path}.tar.gz -C {folder}')
        elif '.zip' in info['url']:
            os.system(f'unzip {path.with_suffix(".zip")} -d {folder}')
        elif '.gz' in info['url']:
            os.system(f'gunzip {path}.gz')

    source = folder / filename
    dest = raw_path / filename

    shutil.move(source, dest)
    shutil.rmtree(folder)


def break_into_fragments(mol, smi):
    frags = fragment_iterative(mol)

    if len(frags) == 0:
        return smi, np.nan, 0

    if len(frags) == 1:
        return smi, smi, 1

    fragments = mols_to_smiles(frags)
    return smi, " ".join(fragments), len(frags)


def read_and_clean_dataset(info):
    raw_path = DATA_DIR / info['name'] / 'RAW'

    dataset = pd.read_csv(
        raw_path / info['filename'],
        index_col=info['index_col'])
    data_HTR1A = pd.read_csv("./DATA/7e2x/RAW/HTR1A.csv", index_col=False)
    data_EGFR = pd.read_csv("./DATA/egfr/RAW/EGFR.csv", index_col=False)

    dataset = pd.concat([dataset,data_HTR1A,data_EGFR])

    if info['drop'] != []:
        dataset = dataset.drop(info['drop'], axis=1)

    if info['name'] == 'ZINC':
        dataset = dataset.replace(r'\n', '', regex=True)

    smiles = dataset.smiles.tolist()
    dataset.smiles = [canonicalize(smi, clear_stereo=True) for smi in smiles]
    dataset = dataset[dataset.smiles.notnull()].reset_index(drop=True)

    return dataset


def read_and_clean_dataset_transfer(info):
    raw_path = DATA_DIR / info['name'] / 'RAW'
    dataset = pd.read_csv(
        raw_path / info['filename'],
        index_col=info['index_col'])

    if info['drop'] != []:
        dataset = dataset.drop(info['drop'], axis=1)
    
    if info['name'] == 'HTR1A':
        dataset = dataset.replace(r'\n', '', regex=True)
        
    if info['name'] == 'EGFR':
        dataset = dataset.replace(r'\n', '', regex=True)

    smiles = dataset.smiles.tolist()
    dataset.smiles = [canonicalize(smi, clear_stereo=True) for smi in smiles]
    dataset = dataset[dataset.smiles.notnull()].reset_index(drop=True)

    return dataset


def add_fragments(dataset, info, n_jobs):
    smiles = dataset.smiles.tolist()
    mols = mols_from_smiles(smiles)
    pjob = Parallel(n_jobs=n_jobs, verbose=1)
    fun = delayed(break_into_fragments)
    results = pjob(fun(m, s) for m, s in zip(mols, smiles))
    smiles, fragments, lengths = zip(*results)
    dataset["smiles"] = smiles
    dataset["fragments"] = fragments
    dataset["n_fragments"] = lengths

    return dataset


def save_dataset(dataset, info):
    dataset = dataset[info['column_order']]
    testset = dataset[dataset.fragments.notnull()]
    trainset = testset[testset.n_fragments >= info['min_length']]
    trainset = trainset[trainset.n_fragments <= info['max_length']]
    processed_path = DATA_DIR / info['name'] / 'PROCESSED'
    trainset.to_csv(processed_path / 'train.smi')
    dataset.to_csv(processed_path / 'test.smi')


def preprocess_dataset(name, n_jobs):
    info = get_dataset_info(name)
    dataset = read_and_clean_dataset(info)
    dataset = add_atom_counts(dataset, info, n_jobs)
    dataset = add_bond_counts(dataset, info, n_jobs)
    dataset = add_ring_counts(dataset, info, n_jobs)

    for prop in info['properties']:
        if prop not in dataset.columns:
            dataset = add_property(dataset, prop, n_jobs)

    dataset = add_fragments(dataset, info, n_jobs)

    save_dataset(dataset, info)


def preprocess_dataset_transfer(name, n_jobs):
    info = get_dataset_info(name)
    dataset = read_and_clean_dataset_transfer(info)
    dataset = add_atom_counts(dataset, info, n_jobs)
    dataset = add_bond_counts(dataset, info, n_jobs)
    dataset = add_ring_counts(dataset, info, n_jobs)

    for prop in info['properties']:
        if prop not in dataset.columns:
            dataset = add_property(dataset, prop, n_jobs)

    dataset = add_fragments(dataset, info, n_jobs)

    save_dataset(dataset, info)