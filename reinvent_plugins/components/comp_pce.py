from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import ops
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from typing import List
from openbabel import pybel
import subprocess
from pydantic import Field
from pydantic.dataclasses import dataclass

from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.components.SAScore.sascorer import calculateScore

RDLogger.DisableLog('rdApp.info')  

__all__ = ["pce"]

def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = ops.mean(x)
    my = ops.mean(y)
    xm, ym = x-mx, y-my
    r_num = ops.sum(tf.multiply(xm, ym))
    r_den = ops.sqrt(tf.multiply(ops.sum(ops.square(xm)), ops.sum(ops.square(ym))))
    r = r_num / r_den

    r = ops.maximum(ops.minimum(r, 1.0), -1.0)
    return ops.square(r)

@add_tag("__parameters")
@dataclass
class Parameters:
    work_dir: List[str]
    models_dir: List[str]
    acceptor: List[str]
    sa_score_lambda: List[float]

@add_tag("__component")
class PCE:
    def __init__(self, params: Parameters):
        self.models_list = []
        for fileName in os.listdir(params.models_dir[0]):
            self.models_list.append(keras.models.load_model(params.models_dir[0] + fileName, custom_objects={"correlation_coefficient": correlation_coefficient}))
        
        self.acceptor = params.acceptor[0]
        self.work_dir = params.work_dir[0]
        self.lambda_ = params.sa_score_lambda[0]

    def create_rep(self, smiles):
        allowed_acceptors = ["PC61BM", "PC71BM", "TiO2"]
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
        mol = Chem.MolFromSmiles(smiles)
        fp = mfpgen.GetFingerprintAsNumPy(mol)
        oh_acc = np.zeros(len(allowed_acceptors), dtype=int)
        oh_acc[allowed_acceptors.index(self.acceptor)] = 1
        rep = np.concatenate((oh_acc, fp))
        return rep

    def __call__(self, smiles_list: List[str]) -> np.array:
        results = pd.DataFrame({
            "SMILES":  len(smiles_list) * [None],
            "VALID": len(smiles_list) * [False],
            "ACCEPTOR": len(smiles_list) * [None],
            "AVG_PCE": len(smiles_list) * [np.nan],
            "STD_PCE": len(smiles_list) * [np.nan],
            "SA_SCORE": len(smiles_list) * [np.nan],
            "REWARD": len(smiles_list) * [np.nan],
        })

        reps_list = []
        for i, smiles in enumerate(smiles_list):
            try:
                rdkit_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                rep = self.create_rep(rdkit_smiles)
                reps_list.append(rep)
                results["SMILES"][i] = rdkit_smiles
                results["VALID"][i] = True
                results["SA_SCORE"][i] = calculateScore(Chem.MolFromSmiles(rdkit_smiles))
            except:
                results["SMILES"][i] = smiles

        x_pred = np.vstack(reps_list)
        preds_array = np.vstack([model.predict(x_pred).flatten() for model in self.models_list])

        avg_pce = np.mean(preds_array, axis=0)
        std_pce = np.std(preds_array, axis=0)

        valid_smiles_list = results["SMILES"][results["VALID"]]
        for i, smiles in enumerate(valid_smiles_list):
            results["ACCEPTOR"][results["SMILES"] == smiles] = self.acceptor
            results["AVG_PCE"][results["SMILES"] == smiles] = avg_pce[i]
            results["STD_PCE"][results["SMILES"] == smiles] = std_pce[i]

        results["REWARD"] = [(results["AVG_PCE"][i] - 2 * results["STD_PCE"][i] + self.lambda_ * (10 - results["SA_SCORE"][i]))/20.0 for i in range(len(smiles_list))]

        epoch = 1
        while f"results_epoch_{epoch}.csv" in os.listdir(self.work_dir):
            epoch += 1
        
        [os.remove(file) for file in os.listdir(self.work_dir) if ("results_epoch" not in file and file != "sscan")]
        results.to_csv(self.work_dir + f"results_epoch_{epoch}.csv")
        return ComponentResults([np.array(results["REWARD"])])
  




