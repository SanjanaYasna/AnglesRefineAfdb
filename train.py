import json
import sys
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch.nn as nn
import torch.optim as optim
from Model import Transformer
import time
import os
from PDB2Angles import extract_backbone_model
#from tensorboardX import SummaryWriter

from typing import Any

import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def load_input(path,angle, length):
    source_file= path + angle + '/source.json'
    target_file= path +angle + '/target.json'
    with open(source_file, 'r') as sf:
        with open(target_file, 'r') as tf:
            sl = json.load(sf)[length]
            tl= json.load(tf)[length]
        tf.close()
    sf.close()
    return sl,tl

def convert_to_Longtensor(data):
    data = torch.LongTensor(data)
    return data

def make_decoderinput(data):
    for i in range(len(data)):
        data[i].insert(0, 1001)
        # data[i].append(36002)
    return data

def recover_target(data):
    for i in range(len(data)):
        del data[i][0]
    return data

def make_decoderoutput(data):
    for i in range(len(data)):
        data[i].append(1002)
    return data



class GetAngleForSSData(Data.Dataset):
    def __init__(self, data_path, ss, angle, batch_size = 4):
        self.data_path = data_path
        self.ss = ss
        self.angle = angle
        self.proteins = os.listdir(data_path)
    def __len__(self):
        return len(self.proteins)
    def __getitem__(self, index) -> Any:
        protein = self.proteins[index]
        # train_enc_inputs = convert_to_Longtensor(source_angle)
        target = pd.read_csv(os.path.join(self.data_path, protein,"target.csv"), usecols=["SS", self.angle])
        target_angle = target[target["SS"] == self.ss][self.angle].values
        #find the corresponding indices in target.csv that have ss equal to desired self.ss
        target_indices = target[target["SS"] == self.ss].index
        #get the corresponding source angle values
        source = pd.read_csv(os.path.join(self.data_path, protein,"source.csv"), usecols=[ self.angle])
        source_angle = source.iloc[target_indices][self.angle].values
        # train_dec_inputs = convert_to_Longtensor(make_decoderinput(target_angle))
        # train_dec_outputs = convert_to_Longtensor(make_decoderoutput(recover_target(target_angle)))
        
        return source_angle, target_angle#train_dec_inputs, train_dec_outputs
    
 # #enc angle and inflate values for training convenience 
def collate_fn(batch):
    train_enc_inputs =  [list(x) for x in list(zip(*batch))[1]]
    #remove any empty arrays
    train_enc_inputs = [x for x in train_enc_inputs if x != []]
    train_enc_inputs = [convert_to_Longtensor(x) for x in train_enc_inputs]
    #remove any -1 values
    target_angles = [list(x) for x in list(zip(*batch))[1]]
    #remove any empty angles
    target_angles = [x for x in target_angles if x != []]
    target_enc_inputs = make_decoderinput(target_angles)
    target_enc_inputs = [convert_to_Longtensor(x) for x in target_enc_inputs]
    #recover then decode for the target_dec
    target_dec_inputs = make_decoderinput(recover_target(target_angles))
    target_dec_inputs = [convert_to_Longtensor(x) for x in target_dec_inputs]
    return train_enc_inputs, target_enc_inputs, target_dec_inputs


if __name__ == "__main__":

    train_path = "/home/s300y051/scratch/angles_refine_afdb_final/"
    test_path = "/home/s300y051/scratch/angles_refine_afdb_test"
    os.environ["LIBCIFPP_DATA_DIR"] = "/kuhpc/work/slusky/s300y051/dssp/libdssp/mmcif_pdbx"
    angles = ["psi_im1", "phi", "omega", "N_CA_C_angle", "C_N_CA_angle", "CA_C_N_angle"]
    ss = ["H", "E", "C"]
    for secondary_structure in ss:
        for angle in angles:
            train_batchsize = 32
            
            #initialize dataset for the given angle and structure 
            dataset = GetAngleForSSData(train_path, secondary_structure, angle)
            testset = GetAngleForSSData(test_path, secondary_structure, angle)
            train_loader = DataLoader(dataset, batch_size = train_batchsize, shuffle = True, collate_fn = collate_fn)
            test_loader = DataLoader(testset, batch_size = train_batchsize, shuffle = True, collate_fn = collate_fn)
            #load in the data
            