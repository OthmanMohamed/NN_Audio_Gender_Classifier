import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json 

f = open("config.json")
config = json.load(f)
f.close()
n_mfccs = config['n_mfccs']
max_mfcc_len = config['max_mfcc_len']
batch_size = config['batch_size']

class GenderClassifier_RNN(nn.Module):
    def __init__(self):
        super(GenderClassifier_RNN, self).__init__()    
        self.LSTM1 = nn.LSTM(n_mfccs, 2 * n_mfccs, num_layers=3, batch_first=True)
        self.Linear = nn.Linear(2* int(n_mfccs), 1)
        
    def forward(self, x):
        x, _ = self.LSTM1(x.float())
        x =  F.relu(self.Linear(x[:,x.size(1)-1]))
        x = torch.sigmoid(x)
        return x



class GenderClassifier_CNN(nn.Module):
    def __init__(self):
        super(GenderClassifier_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(int(16 * 59 * (n_mfccs)/10), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.float()
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 16 * 59 * int(n_mfccs/10))   
        x = self.dropout(x)        
        x = F.relu(self.fc1(x))               
        x = F.relu(self.fc2(x))               
        x = self.fc3(x)                       
        return torch.sigmoid(x)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    rnn = GenderClassifier_RNN().to(device)
    cnn = GenderClassifier_CNN().to(device)

    dummy_batch = np.random.rand(batch_size, n_mfccs, max_mfcc_len)
    dummy_batch = torch.from_numpy(dummy_batch)
    print(rnn(torch.reshape(dummy_batch, (dummy_batch.shape[0], -1, 20))))

    dummy_batch = np.random.rand(batch_size, n_mfccs, max_mfcc_len)
    dummy_batch = torch.from_numpy(dummy_batch)
    print(cnn(torch.reshape(dummy_batch, (dummy_batch.shape[0], 1, -1, 20))))