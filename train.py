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
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pad_sequence

from typing import Any

import pandas as pd
#arg parse
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer().to(device)
#used to be CrossEntropyLoss without any reduction argument
criterion = nn.L1Loss()
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
def data_convertion(data, maxAngle, minAngle):
    normlist = []
    for val in data:
        val = float(val)
        normVal = (val - minAngle) / (maxAngle - minAngle)
        normlist.append(round(normVal, 3))

    normlist = [i * 1000 for i in normlist]
    return normlist

class GetAngleForSSData(Data.Dataset):
    def __init__(self, data_path, ss, angle, batch_size = 4):
        self.data_path = data_path
        self.ss = ss
        self.angle = angle
        self.proteins = os.listdir(data_path)
    def __len__(self):
        return len(self.proteins)
    
    #return source and target angles list (these values are in the range of [0, 1000] after being normalized from 0 to 1 (in terms of min and max angle range) * 1000)
    def __getitem__(self, index) -> Any:
        protein = self.proteins[index]
        # #enc angle and inflate values for training convenience 
        # train_enc_inputs = convert_to_Longtensor(source_angle)
        target = pd.read_csv(os.path.join(self.data_path, protein,"target.csv"), usecols=["SS", self.angle])
        target_indices = target[target["SS"] == self.ss].index
        #put indices in gropus of consecutive indices
        target_angle = target[target["SS"] == self.ss][self.angle].values 
        #find the corresponding indices in target.csv that have ss equal to desired self.ss
        #get the corresponding source angle values
        source = pd.read_csv(os.path.join(self.data_path, protein,"source.csv"), usecols=[ self.angle])
        source_angle = source.iloc[target_indices][self.angle].values 
        # train_dec_inputs = convert_to_Longtensor(make_decoderinput(target_angle))
        # train_dec_outputs = convert_to_Longtensor(make_decoderoutput(recover_target(target_angle)))
        if self.angle == "omega" or self.angle == "psi_im1" or self.angle == "phi":
            target_angle = data_convertion(target_angle, 180.0, -180.0)
            source_angle = data_convertion(source_angle, 180.0, -180.0)
        else:
            target_angle = data_convertion(target_angle, 180.0, 0.0)
            source_angle = data_convertion(source_angle, 180.0, 0.0)
        return source_angle, target_angle#train_dec_inputs, train_dec_outputs
    
#Archived
#  # #enc angle and inflate values for training convenience 
# def collate_fn(batch):
#     train_enc_inputs =  [list(x) for x in list(zip(*batch))[1]]
#     #remove any empty arrays
#     train_enc_inputs = [x for x in train_enc_inputs if x != []]
#     train_enc_inputs = [convert_to_Longtensor(x) for x in train_enc_inputs]
#     #remove any -1 values
#     target_angles = [list(x) for x in list(zip(*batch))[1]]
#     #remove any empty angles
#     target_angles = [x for x in target_angles if x != []]
#     target_enc_inputs = make_decoderinput(target_angles)
#     target_enc_inputs = [convert_to_Longtensor(x) for x in target_enc_inputs]
#     #recover then decode for the target_dec
#     target_dec_inputs = make_decoderinput(recover_target(target_angles))
#     target_dec_inputs = [convert_to_Longtensor(x) for x in target_dec_inputs]
#     return train_enc_inputs, target_enc_inputs, target_dec_inputs

#collate with padding, and limits
#set sequence range to [5, 200]
def collate_fn_padded(batch):
    source = [torch.LongTensor(item[0]) for item in batch if len(item[0]) < 201 and len(item[0]) > 4]
    if len(source) == 0:
        return None, None, None
    source = pad_sequence(source, batch_first=True, padding_value=0)
    target = [list(item[1]) for item in batch if len(item[1]) < 201 and len(item[1]) > 4]
    #decode target inputs and pad
    target_dec_inputs = make_decoderinput(target)
    target_dec_inputs = pad_sequence([torch.LongTensor(item) for item in target_dec_inputs], batch_first=True, padding_value=0)
    #recover and decode target outputs and pad
    target_dec_outputs = make_decoderoutput(recover_target(target))
    target_dec_outputs = pad_sequence([torch.LongTensor(item) for item in target_dec_outputs], batch_first=True, padding_value=0)
    return source, target_dec_inputs, target_dec_outputs

def train():
    model.train()  # Turn on the train mode
    total_loss = 0.
    t_loss = 0.
    start_time = time.time()
    for step, (enc_inputs, dec_inputs, dec_outputs) in enumerate(train_loader):
        if enc_inputs is None:
            continue
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        if outputs.view(-1).shape != dec_outputs.view(-1).shape:
            continue
        loss = criterion(outputs.view(-1), dec_outputs.view(-1))
        # print(outputs,len(outputs),len(outputs[0]))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        t_loss += loss.item()
        log_interval = 100
        if step % log_interval == 0:
            if( step != 0 ):
                cur_loss = total_loss / log_interval
                elapsed = (time.time() - start_time) / log_interval
            else:
                cur_loss = total_loss
                elapsed = time.time() - start_time
            print('| epoch {:3d} | batches {:5d} | '
                  'lr {:0.5f} | s/batch {:5.2f} | '
                  'loss {:7.4f} '.format(epoch, step, lr_scheduler.get_last_lr()[0], elapsed , cur_loss))
            total_loss = 0
            start_time = time.time()
    return t_loss / len(train_loader)


def evaluate(eval_model, psi_valid_loader):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for enc_inputs, dec_inputs, dec_outputs in psi_valid_loader:
            if enc_inputs is None:
                continue
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            if outputs.view(-1).shape != dec_outputs.view(-1).shape:
                continue
            total_loss = criterion(outputs.view(-1), dec_outputs.view(-1))
            # print(criterion(outputs, dec_outputs.view(-1)).item())
        print("total loss: ", total_loss)
    return total_loss / len(psi_valid_loader)

if __name__ == "__main__":

    train_path = "/home/s300y051/scratch/angles_refine_afdb_final/"
    test_path = "/home/s300y051/scratch/angles_refine_afdb_test/"
    os.environ["LIBCIFPP_DATA_DIR"] = "/kuhpc/work/slusky/s300y051/dssp/libdssp/mmcif_pdbx"
    angles = ["psi_im1", "phi", "omega", "N_CA_C_angle", "C_N_CA_angle", "CA_C_N_angle"]
    #create parser for SS
    parser = argparse.ArgumentParser()
    parser.add_argument("--ss", type=str, help="Secondary structure type")
    args = parser.parse_args()
    secondary_structure= args.ss
    ss = ["H", "E", "C"]
    ss_name = ""
    if secondary_structure == "H":
        ss_name = "Helix"
    elif secondary_structure == "E":
        ss_name = "Sheet"
    else: 
        ss_name = "Coil"
    for angle in angles:
        train_batchsize = 64
        print('\n-----Training %s On %s-----' % (ss_name, angle))
        RESUME = False
        EPOCH = 50
        start_epoch = 0
        modeldir = "models/%s/%s_model" % (ss_name, angle)
        best_val_loss = float("inf")
        plotPath = "models/%s/plot/%s_plot/loss/" % (ss_name, angle)
        
        #resume code. ADJUST accordingly
        if RESUME:
            path_checkpoint = "%s/ckpt_best.pth" % modeldir
            checkpoint = torch.load(path_checkpoint)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_schedule'])
            loss = checkpoint['loss']
            best_val_loss = loss
            
        loss_show = []
        trainPlotPath = plotPath + "train"
        validPlotPath = plotPath + "valid"
        train_writer = SummaryWriter(trainPlotPath)
        val_writer = SummaryWriter(validPlotPath)
        
        #initialize dataset for the given angle and structure 
        dataset = GetAngleForSSData(train_path, secondary_structure, angle)
        testset = GetAngleForSSData(test_path, secondary_structure, angle)
        train_loader = DataLoader(dataset, batch_size = train_batchsize, shuffle = True, collate_fn = collate_fn_padded)
        test_loader = DataLoader(testset, batch_size = train_batchsize, shuffle = True, collate_fn = collate_fn_padded)
        
        #can remove if tensorboard works sufficiently as expected...
        txt_loss_record = "models/%s/txt_loss_record/loss.txt" % (ss_name)
        with open(txt_loss_record, "a") as f:
            f.write('\n-----Training %s On %s-----\n' % (ss_name, angle))
            f.close()
        #main train loop
        
        for epoch in range(start_epoch + 1, EPOCH + 1):
            epoch_start_time = time.time()
            train_loss = train()
            val_loss = evaluate(model, test_loader)


            train_loss = float('%.4f' % train_loss)
            val_loss = float('%.4f' % val_loss)
            
            #write to txt_loss_record
            with open(txt_loss_record, "a") as f:
                f.write("Epoch: %d, Train Loss: %f, Validation Loss: %f\n" % (epoch, train_loss, val_loss))
                f.close()

            # ----------------------------------------tensorboard----------------------------------------
            train_writer.add_scalar("loss_train", train_loss, epoch)
            val_writer.add_scalar("loss_validation", val_loss, epoch)
            # ----------------------------------------tensorboard----------------------------------------

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:7.4f} | valid loss {:7.4f} |  '
                  .format(epoch, (time.time() - epoch_start_time), train_loss, val_loss))
            print('-' * 89)
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "lr_schedule": lr_scheduler.state_dict(),
                "loss": val_loss
            }

            if not os.path.exists(modeldir):
                os.makedirs(modeldir)
            lr_scheduler.step()
            val_loss = float('%.4f' % val_loss)
            best_val_loss = float('%.4f' % best_val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                torch.save(checkpoint, '%s/ckpt_best.pth' % modeldir)

        best_path_checkpoint = '%s/ckpt_best.pth' % modeldir
        bestCheckpoint = torch.load(best_path_checkpoint)
        print('\n-----%s-%s-----Done' % (secondary_structure, angle))
        print('\nBEST EPOCH:    LOSS    ', bestCheckpoint['epoch'], ':    ', bestCheckpoint['loss'])
