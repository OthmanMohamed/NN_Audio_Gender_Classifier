from torch.utils.data import Dataset, DataLoader
import os
import random
import pickle
import numpy as np
import torch
import json

# LOADING CONFIGURATIONS
f = open("config.json")
config = json.load(f)
f.close()
max_mfcc_len = config['max_mfcc_len']
mfccs_path = config['mfccs_path']
train_percent = config['train_percent']
test_percent = config['test_percent']
val_percent = config['val_percent']
hp_shuffle_ds = config['data_shuffle']

class mfcc_ds(Dataset):

  def __init__(self, train_list, lbls_path):
    self.train_list = train_list
    with open(lbls_path, 'rb') as f:
        self.lbls_dict = pickle.load(f)
    if hp_shuffle_ds: 
        random.shuffle(self.train_list)
        
  def __len__(self):
    return len(self.train_list)

  def __getitem__(self, idx):
    mfcc = np.load(self.train_list[idx]) #loading mfcc numpy file
    label = self.lbls_dict[self.train_list[idx]]
    if mfcc.shape[1] < max_mfcc_len:
        mfcc = np.pad(mfcc, ((0,0),(0, max_mfcc_len - mfcc.shape[1]))) #pad if less than max len
    else:
        mfcc = mfcc[:, :max_mfcc_len] #truncate at length equal max len
    return torch.Tensor(mfcc), torch.Tensor([label])


if __name__ == "__main__":
    lbls_file_path = os.path.join(mfccs_path, "lbls.pkl")
    with open(lbls_file_path, 'rb') as f:
        lbls_dict = pickle.load(f)
    mfccs_list = list(lbls_dict.keys())
    train_list = mfccs_list[ : int(len(mfccs_list)*train_percent)]
    test_list = mfccs_list[int(len(mfccs_list)*train_percent) : int(len(mfccs_list)*(train_percent+test_percent))]
    val_list = mfccs_list[int(len(mfccs_list)*(train_percent+test_percent)) : ]


    ds_train = mfcc_ds(train_list, lbls_file_path)
    train_loader = DataLoader(ds_train, batch_size=64, shuffle=True)
    ds_test = mfcc_ds(test_list, lbls_file_path)
    test_loader = DataLoader(ds_test, batch_size=64, shuffle=True) 
    ds_val = mfcc_ds(val_list, lbls_file_path)
    val_loader = DataLoader(ds_val, batch_size=64, shuffle=True) 

    for d, l in train_loader:
        print(d.shape)
        break
    for d, l in val_loader:
        print(d.shape)
        break