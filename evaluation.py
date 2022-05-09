from data_load import mfcc_ds
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import torch
import torch.nn as nn
import json
from sklearn.metrics import precision_recall_fscore_support

f = open("config.json")
config = json.load(f)
f.close()

n_mfccs = config['n_mfccs']
network_type = config['network_type']
test_model_path = config['test_model_path']
mfccs_path = config['mfccs_path']
train_percent = config['train_percent']
test_percent = config['test_percent']
val_percent = config['val_percent']
batch_size = config['batch_size']

lbls_file_path = os.path.join(mfccs_path, "lbls.pkl")
with open(lbls_file_path, 'rb') as f:
    lbls_dict = pickle.load(f)
mfccs_list = list(lbls_dict.keys())
test_list = mfccs_list[int(len(mfccs_list)*train_percent) : int(len(
    mfccs_list)*(train_percent+test_percent))]

ds_test = mfcc_ds(test_list, lbls_file_path)
test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False) 

model = torch.load(test_model_path)
model.eval()
print(model)
pred = torch.Tensor()
ground_truth = torch.Tensor()

for batch_id, (mfccs, lbls) in enumerate(test_loader):
    if network_type == "RNN":
        mfccs = torch.reshape(mfccs, (mfccs.shape[0], -1, n_mfccs))
    elif network_type == "CNN":
        mfccs = torch.reshape(mfccs, (mfccs.shape[0], 1, -1, n_mfccs))
    outputs = model(mfccs)
    predicted = torch.gt(outputs, 0.5)
    pred = torch.cat((pred, predicted))
    ground_truth = torch.cat((ground_truth, lbls))

    # running_correct += (predicted == lbls).sum().item()
    # running_samples += predicted.size(0)
else: 
    samples = pred.size(0)
    correct = (pred == ground_truth).sum().item()
    precision, recall, f1_score, _ = precision_recall_fscore_support(ground_truth, pred)
    print (f'RUN TESTING ON {network_type} MODEL {test_model_path}')
    print (f'Accuracy : {correct/samples:.4f}')
    print (f'MALE CLASS ---> Percision : {precision[0]:.4f}, Recall : {recall[0]:.4f}, F1 Score : {f1_score[0]:.4f}')
    print (f'FEMALE CLASS ---> Percision : {precision[1]:.4f}, Recall : {recall[1]:.4f}, F1 Score : {f1_score[1]:.4f}')