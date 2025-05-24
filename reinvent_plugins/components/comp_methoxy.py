from __future__ import annotations

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import ops
from rdkit import Chem, RDLogger
from typing import List
from openbabel import pybel
import subprocess
from pydantic import Field
from pydantic.dataclasses import dataclass

from reinvent_plugins.components.add_tag import add_tag
from rdkit import Chem
from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.components.SAScore.sascorer import calculateScore

RDLogger.DisableLog('rdApp.info')  

__all__ = ["methoxy"]

def fraction_methoxy_groups(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan

    methoxy_smarts = Chem.MolFromSmarts("[OX2][CH3]")
    matches = mol.GetSubstructMatches(methoxy_smarts)
    num_bonds = mol.GetNumBonds()
    return len(matches)/num_bonds

@add_tag("__component")
class Methoxy:
    def __init__(self, *args, **kwargs):
        pass 

    def __call__(self, smiles_list: List[str]) -> np.array:
        rewards = 1 - np.array(list(map(fraction_methoxy_groups, smiles_list)))
        return ComponentResults([rewards])