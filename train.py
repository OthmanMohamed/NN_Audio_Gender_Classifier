from models import GenderClassifier_RNN, GenderClassifier_CNN
from data_load import mfcc_ds
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle
import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
import json

today = datetime.now()
today = today.strftime('%m-%d-%H-%M')
f = open("config.json")
config = json.load(f)
f.close()

out_path = "out/"
if not os.path.isdir(out_path): os.mkdir(out_path)
out_path = os.path.join(out_path, today)
if not os.path.isdir(out_path): os.mkdir(out_path)
n_mfccs = config['n_mfccs']
mfccs_path = config['mfccs_path']
batch_size = config['batch_size']
data_shuffle = config['data_shuffle']
train_percent = config['train_percent']
test_percent = config['test_percent']
val_percent = config['val_percent']
network_type = config['network_type']
learning_rate = config['learning_rate']
max_epochs = config['max_epochs']
log_interval = config['log_interval']
save_checkpoint_path  = config["save_checkpoint_path"]
save_checkpoint_interval = config["save_checkpoint_interval"]


lbls_file_path = os.path.join(mfccs_path, "lbls.pkl")
with open(lbls_file_path, 'rb') as f:
    lbls_dict = pickle.load(f)
mfccs_list = list(lbls_dict.keys())
train_list = mfccs_list[ : int(len(mfccs_list)*train_percent)]
test_list = mfccs_list[int(len(mfccs_list)*train_percent) : int(len(mfccs_list)*(train_percent+test_percent))]
val_list = mfccs_list[int(len(mfccs_list)*(train_percent+test_percent)) : ]

ds_train = mfcc_ds(train_list, lbls_file_path)
train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=data_shuffle)
ds_test = mfcc_ds(test_list, lbls_file_path)
test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=data_shuffle) 
ds_val = mfcc_ds(val_list, lbls_file_path)
val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=data_shuffle) 

if network_type == "RNN":
    model = GenderClassifier_RNN()
elif network_type == "CNN":
    model = GenderClassifier_CNN()

writer = SummaryWriter(out_path)
example = iter(test_loader)
d,l = example.next()
if network_type == "RNN":
    writer.add_graph(model, torch.reshape(d, (d.shape[0], -1, n_mfccs)))
elif network_type == "CNN":
    writer.add_graph(model, torch.reshape(d, (d.shape[0], 1, -1, n_mfccs)))
writer.close()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

train_loss = []
val_loss = []
n_total_steps = len(train_loader)
num_epochs = max_epochs
running_correct = 0
running_samples = 0
e=0
while(e < num_epochs):
    model.train()
    for batch_id, (mfccs, lbls) in enumerate(train_loader): 
        if network_type == "RNN":
            mfccs = torch.reshape(mfccs, (mfccs.shape[0], -1, n_mfccs))
        elif network_type == "CNN":
            mfccs = torch.reshape(mfccs, (mfccs.shape[0], 1, -1, n_mfccs))
        outputs = model(mfccs)
        predicted = torch.gt(outputs, 0.5)
        running_correct += (predicted == lbls).sum().item()
        running_samples += predicted.size(0)
        loss = criterion(outputs, lbls)
        train_loss.append(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (e*n_total_steps + batch_id) % save_checkpoint_interval == 0:
            model_name = network_type + "_model_" + str(e*n_total_steps + batch_id) + "Steps.pt"
            p = os.path.join(save_checkpoint_path, model_name)
            print("SAVING MODEL CHECPOINT TO " + str(p))
            torch.save(model, p)
            
        if batch_id%log_interval == 0:
            writer.add_scalar('training loss', loss, e * n_total_steps + batch_id)
            writer.add_scalar('avg training loss', sum(train_loss)/ len(train_loss), e * n_total_steps + batch_id)
            running_accuracy = running_correct / running_samples
            writer.add_scalar('train accuracy', running_accuracy, e * n_total_steps + batch_id)
            print (f'TRAINING Epoch [{e+1}/{max_epochs}], Step [{batch_id+1}/{n_total_steps}], Global Step {e*n_total_steps + batch_id % save_checkpoint_interval}, Loss: {loss.item():.4f}, AVGLoss: {sum(train_loss)/ len(train_loss)}, Running Accuracy : {running_accuracy}')
            running_correct = 0
            running_samples = 0
            
    model.eval()
    for batch_id, (mfccs, lbls) in enumerate(val_loader):
        if network_type == "RNN":
            mfccs = torch.reshape(mfccs, (mfccs.shape[0], -1, n_mfccs))
        elif network_type == "CNN":
            mfccs = torch.reshape(mfccs, (mfccs.shape[0], 1, -1, n_mfccs))
        outputs = model(mfccs)
        predicted = torch.gt(outputs, 0.5)
        running_correct += (predicted == lbls).sum().item()
        running_samples += predicted.size(0)
        loss = criterion(outputs, lbls)
        val_loss.append(loss.detach())
    else:
        writer.add_scalar('avg val loss', sum(val_loss)/ len(val_loss), e)
        writer.add_scalars('TRAIN VS VAL LOSS', {'avg val loss' : sum(val_loss)/ len(val_loss), 'avg train loss' : sum(train_loss)/ len(train_loss)}, e)
        running_accuracy = running_correct / running_samples
        writer.add_scalar('val accuracy', running_accuracy, e)
        print (f'VALIDATION Epoch [{e+1}/{max_epochs}], AVGLoss: {sum(val_loss)/ len(val_loss)}, Running Accuracy : {running_accuracy}')
        running_correct = 0
        running_samples = 0
    e += 1