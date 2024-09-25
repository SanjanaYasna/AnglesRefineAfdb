import json
import sys
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Model import Transformer
import time
import os
from PDB2Angles import extract_backbone_model
from joblib import Parallel, delayed

os.environ["LIBCIFPP_DATA_DIR"] = "/kuhpc/work/slusky/s300y051/dssp/libdssp/mmcif_pdbx"
#input
pre_relax = "/home/s300y051/scratch/alphafoldSwiss"
relax = "/home/s300y051/scratch/relax_afdb"
angles = ["psi", "phi", "omega", "CCN", "CNC", "NCC"]
minlength,maxlength = 5,37

#to do: change input list
files = os.listdir(relax)

def make_angles_csv(file_path):
    file_id = file_path.split(".")[0].split("_relaxed")[0]
    angles_dir = "/home/s300y051/scratch/angles_refine_afdb_final/"+ file_id
    #check if file_id +".pdb" exists in pre_relax
    if file_id + ".pdb" not in os.listdir(pre_relax):
        print(file_id + ".pdb not in pre_relax or angles already present")
        return
   # os.mkdir(angles_dir)
    try:
        model_structure_geo_pre_relax = extract_backbone_model(pre_relax + "/" + file_id + ".pdb", angles_dir + '/source.csv' ,pre_relax = True)
        model_structure_geo_relax = extract_backbone_model(relax + "/" + file_id + "_relaxed_0001.pdb", angles_dir + '/target.csv')
    except:
        print("Error in extracting angles for protein id ", file_id)
        return

if __name__ == "__main__":
    
    #hopefully this doesn't break. Be wary of n_jobs
    Parallel(n_jobs=-1)(delayed(make_angles_csv)(f) for f in tqdm(files))