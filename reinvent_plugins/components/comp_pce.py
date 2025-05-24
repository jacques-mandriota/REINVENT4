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

def save_dataframe_to_csv(df, csv_filename, write_mode):
    """
    Saves a DataFrame to a CSV file

    Arguments:
        df (pd.DataFrame): DataFrame to save
        csv_filename (str): Name of the CSV file to save the DataFrame to
        write_mode (str): Write mode for saving the DataFrame ('w+' for overwrite, 'a' for append)
    """
    if write_mode == 'w+':
        df.to_csv(csv_filename, header=True, index=False, mode=write_mode)
    elif write_mode == 'a':
        df.to_csv(csv_filename, header=False, index=False, mode=write_mode)

def load_dataframe_from_csv(csv_filename):
    """
    Loads a DataFrame from a CSV file

    Arguements:
        csv_filename (str): Name of the CSV file to load the DataFrame from
    """
    df = pd.read_csv(csv_filename)
    return df

def csv2numpy(filename):
    """
    Read CSV file and convert to numpy array
    
    Arguments:
        filename (str): Path to the CSV file

    Returns:
        np.array: Numpy array containing the data from the CSV file with shape (n_samples, n_features)
    """
    df = pd.read_csv(filename)
    df = df.iloc[1:].reset_index(drop=True)
    data = df.to_dict()
    
    sortednames = sorted(data.keys(), key=lambda x: x.lower())
    
    data_x = []
    
    for key in sortednames:
        if key == 'Donors' or key == 'Acceptors':
            continue
        else:
            data_x.append(np.array(list(data[key].values())))
    
    return np.array(data_x).T

def get_smiles(database):
    """
    Extract the smiles from the database
    
    Arguments:
        database (str): Path to the CSV file containing the database
    
    Returns:
        smiles_list (list): List of SMILES strings extracted from the database
    """
    database = pd.read_csv(database)
    smiles_list = []
    for index, row in database.iterrows():
        smiles = row['SMILES']
        smiles_list.append(smiles)
    return smiles_list

def smi2mol(smiles, molfile_name='temp.mol'):
    """
    Converts a SMILES string to a .mol file. 

    Arguments:
        smiles (str): SMILES string to convert
        molfile_name (str): Name of the output .mol file (default: 'temp.mol')

    Returns:
        bool: True if conversion is successful with .mol file, False otherwise
    """
    try:
        mol = pybel.readstring("smi", smiles)
        mol.make3D()
        mol.write("mol", molfile_name, overwrite=True)
        print("Succesfully converted SMILES to MOL file")
        return True
    except Exception as e:
        print(f"Error converting SMILES to MOL: {e}")
        return False

def read_molsig_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    signatures = {}
    for line in lines:
        parts = line.strip().split()
        if parts and not parts[0] == "0.0":
            signature = parts[1]
            count = float(parts[0])
            signatures[signature] = count

    df = pd.DataFrame([signatures])
    return df  

def generate_molecular_signatures(molfile_name, height_range=(1, 4)):
    """
    Generate molecular signatures for heights 1-4 and return combined dataframe 

    Arguments:
        molfile_name (str): Name of the input .mol file
        height_range (tuple): Range of heights for molecular signatures (default: (1, 4))

    Returns:
        pd.DataFrame: DataFrame containing combined molecular signatures
    """
    combined_signatures_df = pd.DataFrame()
    
    for height in range(height_range[0], height_range[1] + 1):
        sig_file = f"sig{height}"
        command = f"./sscan {molfile_name} {sig_file}"
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"Generated {sig_file} signatures")
            
            sig_df = read_molsig_file(f"temp.{sig_file}")
            
            if combined_signatures_df.empty:
                combined_signatures_df = sig_df
            else:
                combined_signatures_df = pd.concat([combined_signatures_df, sig_df], axis=1)
                
        except subprocess.CalledProcessError as e:
            print(f"Error generating {sig_file}: {e}")
    
    return combined_signatures_df

def find_missing_signatures(database_df, new_signatures_df):
    """
    Find molecular descriptors in new_signatures_df that are not present in database_df

    Arguments:
        database_df (pd.DataFrame): DataFrame containing the database of molecular signatures 
        new_signatures_df (pd.DataFrame): DataFrame containing the new molecular signatures
    
    Returns:
        set: Set of molecular descriptors in new_signatures_df that are not in database_df
    """

    database_columns = set(database_df.columns)
    new_columns = set(new_signatures_df.columns)

    missing_signatures = new_columns - database_columns

    return missing_signatures

def database_update(smiles, database_df, new_signatures_df):
    """
    Updates the .csv database with the generated molecules and the signatures that are contained in the training set. 

    Arguments:
        smiles (str): SMILES string of the molecule to add
        database_df (pd.DataFrame): DataFrame containing the database of molecular signatures 
        new_signatures_df (pd.DataFrame): DataFrame containing the new molecular signatures

    Returns:
        pd.DataFrame: Updated DataFrame containing the database of molecular signatures
    """

    all_acceptors = ['PC61BM', 'PC71BM', 'TiO2', 'C60', 'PDI', 'ICB']
    zero_acceptors = ['C60', 'PDI', 'ICB']

    acceptors = [acc for acc in all_acceptors if acc not in zero_acceptors]

    results = []
    
    missing_signatures = find_missing_signatures(database_df, new_signatures_df)

    new_signatures_df = new_signatures_df.drop(columns=missing_signatures)
    print("Dropped new signatures")

    for acc in acceptors:
        row = {
            'Donors': smiles,
            'Acceptors': acc
        }

        for acceptor in all_acceptors:
            row[acceptor] = 1 if acceptor == acc else 0

        for col in database_df.columns:
            if col in ['Donors', 'Acceptors'] + all_acceptors:
                continue
            elif col in new_signatures_df.columns:
                row[col] = new_signatures_df[col].iloc[0]  
            else:
                row[col] = 0

        results.append(row)

    results_df = pd.DataFrame(results)

    updated_database_df = pd.concat([database_df, results_df], ignore_index=False, sort=False)
    updated_database_df = updated_database_df.iloc[1:]

    return updated_database_df

def duplicate_smiles(smiles):
    """
    Duplicates smiles for each acceptor in the list of acceptors

    Arguments:
        smiles (list): List of SMILES strings

    Returns:
        results_donors (list): List of SMILES strings duplicated for each acceptor
        results_acceptors (list): List of acceptors corresponding to the duplicated SMILES strings
    """
    acceptors = ['PC61BM', 'PC71BM', 'TiO2']
    results_donors = []
    results_acceptors = []

    for smi in smiles:
        for acc in acceptors:
            results_donors.append(smi)
            results_acceptors.append(acc)

    return results_donors, results_acceptors

def extract_highest_pce(csv_file):

    df = pd.read_csv(csv_file)

    donor_col = "smiles"
    pce_col = "models avg"
    std_col = "models std"

    idx = df.groupby(donor_col)[pce_col].idxmax()

    result = df.loc[idx, [donor_col, pce_col, std_col]].copy()

    result.columns = ["Donors", "Highest PCE", "Std PCE"]

    return result


@add_tag("__parameters")
@dataclass
class Parameters:
    work_dir: List[str]
    models_dir: List[str]
    template_file: List[str]
    acceptor: List[str]
    sa_score_lambda: List[float]

@add_tag("__component")
class PCE:
    """
    def __init__(self, params: Parameters):
        self.models_list = list(map( 
            lambda s: keras.models.load_model(params.models_dir[0] + s, custom_objects={"correlation_coefficient": correlation_coefficient}), 
            os.listdir(params.models_dir[0])
        ))
        
        self.database_template = load_dataframe_from_csv(params.template_file[0]).iloc[0]
        self.work_dir = params.work_dir[0]
        self.lambda_ = params.sa_score_lambda[0]
    
    def __call__(self, smiles_list: List[str]) -> np.array:
        results = pd.DataFrame({
            "SMILES": [],
            "VALID": len(smiles_list) * [False],
            "ACCEPTOR": len(smiles_list) * [None],
            "AVG_PCE": len(smiles_list) * [np.nan],
            "STD_PCE": len(smiles_list) * [np.nan],
            "SA_SCORE": len(smiles_list) * [np.nan],
            "REWARD": len(smiles_list) * [np.nan],
        })

        database = self.database_template
        save_dataframe_to_csv(database, 
            self.work_dir + 'database.csv', 'w+'
        )

        for i, smiles in enumerate(smiles_list):
            try:
                rdkit_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                smi2mol(rdkit_smiles, 'temp.mol')
                new_signatures = generate_molecular_signatures('temp.mol', (1, 4))
                updated_database_df = database_update(rdkit_smiles, database, new_signatures)
                save_dataframe_to_csv(
                    updated_database_df, 
                    self.work_dir + 'database.csv', 'a'
                )
                results["SMILES"].append(rdkit_smiles)
                results["VALID"][i] = True
                results["SA_SCORE"][i] = calculateScore(Chem.MolFromSmiles(rdkit_smiles))
            except:
                results["SMILES"].append(smiles)

        x_pred = csv2numpy(self.work_dir + 'database.csv')
        preds_array = np.vstack([model.predict(x_pred).flatten() for model in self.models_list])
        valid_smiles_list, acceptors_list = duplicate_smiles(results["SMILES"][results["VALID"]])

        avg_pce = np.mean(preds_array, axis=0)
        std_pce = np.std(preds_array, axis=0)
        pce_lb = avg_pce - 2 * std_pce

        num_acceptors = len(valid_smiles_list)//len(results["SMILES"][results["VALID"]])

        for i, smiles in enumerate(valid_smiles_list[::num_acceptors]):
            j = i * num_acceptors
            k = j + np.argmax(pce_lb[j:j+num_acceptors])
            results["ACCEPTOR"][results["SMILES"] == smiles] = acceptors_list[k]
            results["AVG_PCE"][results["SMILES"] == smiles] = avg_pce[k]
            results["STD_PCE"][results["SMILES"] == smiles] = std_pce[k]

        results["REWARD"] = [(results["AVG_PCE"][i] - 2 * results["STD_PCE"][i] + self.lambda_ * (10 - results["SA_SCORE"][i]))/20.0 for i in range(len(smiles_list))]

        epoch = 1
        while f"results_epoch_{epoch}.csv" in os.listdir(self.work_dir):
            epoch += 1
        
        [os.remove(file) for file in os.listdir(self.work_dir) if ("results_epoch" not in file and file != "sscan")]
        results.to_csv(self.work_dir + f"results_epoch_{epoch}.csv")
        return ComponentResults([np.array(results["REWARD"])])
    """ 
    def __init__(self, params: Parameters):
        self.models_list = []
        for fileName in os.listdir(params.models_dir[0]):
            # with open(params.models_dir[0] + fileName, "rb") as f:
                # self.models_list.append(pickle.load(f))
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
  




