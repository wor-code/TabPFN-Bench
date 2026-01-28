from functools import partial
from collections import defaultdict
import os.path as osp

from dgl import backend as F
from dgllife.utils import BaseAtomFeaturizer
from dgllife.utils import CanonicalAtomFeaturizer, ConcatFeaturizer, atom_degree_one_hot, \
    atom_formal_charge, atom_num_radical_electrons, \
    atom_total_num_H_one_hot
from dgllife.utils.featurizers import BaseBondFeaturizer, atom_type_one_hot, atom_hybridization_one_hot
from dgllife.utils import one_hot_encoding
from dgllife.utils import atom_type_one_hot, atom_hybridization_one_hot
from dgllife.utils.featurizers import atom_is_aromatic
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import numpy as np


def alchemy_nodes(mol):

    atom_feats_dict = defaultdict(list)
    is_donor = defaultdict(int)
    is_acceptor = defaultdict(int)

    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    mol_feats = mol_featurizer.GetFeaturesForMol(mol)

    for i in range(len(mol_feats)):
        if mol_feats[i].GetFamily() == 'Donor':
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_donor[u] = 1
        elif mol_feats[i].GetFamily() == 'Acceptor':
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_acceptor[u] = 1

    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        atom = mol.GetAtomWithIdx(u)
        atom_type = atom.GetAtomicNum()
        num_h = atom.GetTotalNumHs()
        atom_feats_dict['node_type'].append(atom_type)

        h_u = []
        h_u += atom_type_one_hot(atom, ['H', 'C', 'N', 'O', 'F', 'S', 'Cl'])
        h_u.append(atom_type)
        h_u.append(is_acceptor[u])
        h_u.append(is_donor[u])
        h_u += atom_is_aromatic(atom)
        h_u += atom_hybridization_one_hot(atom, [Chem.rdchem.HybridizationType.SP,
                                                 Chem.rdchem.HybridizationType.SP2,
                                                 Chem.rdchem.HybridizationType.SP3])
        h_u.append(num_h)
        atom_feats_dict['n_feat'].append(F.tensor(np.array(h_u).astype(np.float32)))

    atom_feats_dict['n_feat'] = F.stack(atom_feats_dict['n_feat'], dim=0)
    atom_feats_dict['node_type'] = F.tensor(np.array(
        atom_feats_dict['node_type']).astype(np.int64))

    return atom_feats_dict


def alchemy_edges(mol, self_loop=False):
    """Featurization for all bonds in a molecule.
    The bond indices will be preserved.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    self_loop : bool
        Whether to add self loops. Default to be False.
    Returns
    -------
    bond_feats_dict : dict
        Dictionary for bond features
    """
    bond_feats_dict = defaultdict(list)

    AllChem.EmbedMolecule(mol)
    mol_conformers = mol.GetConformers()
    geom = mol_conformers[0].GetPositions()

    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        for v in range(num_atoms):
            if u == v and not self_loop:
                continue

            e_uv = mol.GetBondBetweenAtoms(u, v)
            if e_uv is None:
                bond_type = None
            else:
                bond_type = e_uv.GetBondType()
            bond_feats_dict['e_feat'].append([
                float(bond_type == x)
                for x in (Chem.rdchem.BondType.SINGLE,
                          Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE,
                          Chem.rdchem.BondType.AROMATIC, None)
            ])
            bond_feats_dict['distance'].append(
                np.linalg.norm(geom[u] - geom[v]))

    bond_feats_dict['e_feat'] = F.tensor(
        np.array(bond_feats_dict['e_feat']).astype(np.float32))
    bond_feats_dict['distance'] = F.tensor(
        np.array(bond_feats_dict['distance']).astype(np.float32)).reshape(-1, 1)

    return bond_feats_dict


def chirality(atom):
    """Get Chirality information for an atom.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    Returns
    -------
    list of 3 boolean values
        The 3 boolean values separately indicate whether the atom
        has a chiral tag R, whether the atom has a chiral tag S and
        whether the atom is a possible chiral center.
    """
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
            [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


node_featurizer = BaseAtomFeaturizer(
    featurizer_funcs={
        'hv': ConcatFeaturizer([
            partial(
                atom_type_one_hot,
                allowable_set=[
                    'B', 'C', 'N', 'O', 'F',
                    'Si', 'P', 'S', 'Cl', 'As',
                    'Se', 'Br', 'Te', 'I', 'At'],
                encode_unknown=True
            ),
            partial(atom_degree_one_hot, allowable_set=list(range(6))),
            atom_formal_charge,
            atom_num_radical_electrons,
            partial(atom_hybridization_one_hot, encode_unknown=True),
            lambda atom: [0],  # A placeholder for aromatic information,
            atom_total_num_H_one_hot,
            chirality
        ], )
    }
)

edge_featurizer = BaseBondFeaturizer({
    'he': lambda bond: [0 for _ in range(10)]
})
