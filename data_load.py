from torch.utils.data import Dataset, DataLoader
import os
import random
import pickle
import numpy as np
import torch

max_mfcc_len = 250
mfccs_path = "/data/datasets/mfccs"
train_percent = 0.8
test_percent = 0.1
val_percent = 0.1

class mfcc_ds(Dataset):

  def __init__(self, train_list, lbls_path):
    self.train_list = train_list
    with open(lbls_path, 'rb') as f:
        self.lbls_dict = pickle.load(f)
    hp_shuffle_ds = True
    if hp_shuffle_ds: 
        random.shuffle(self.train_list)
        
  def __len__(self):
    return len(self.train_list)

  def __getitem__(self, idx):
    mfcc = np.load(self.train_list[idx])
    label = self.lbls_dict[self.train_list[idx]]
    if mfcc.shape[1] < max_mfcc_len:
        mfcc = np.pad(mfcc, ((0,0),(0, max_mfcc_len - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_mfcc_len]
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