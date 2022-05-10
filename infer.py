import os
import librosa
import numpy as np
import torch
import json
import sys
import wave
import soundfile as sf

# LOADING CONFIGURATIONS
f = open("config.json")
config = json.load(f)
f.close()
n_mfccs = config['n_mfccs']
network_type = config['network_type']
SR = config['SR']
max_mfcc_len = config['max_mfcc_len']
infer_model_path = config['infer_model_path']

def inference(infer_file):
    sound_file, _ = librosa.core.load(infer_file)
    try:
        with wave.open(infer_file, "rb") as wave_file:
            sr = wave_file.getframerate()
    except:
        x, sr = librosa.load(infer_file)
    print(f"INPUT FILE SAMPLING RATE : {sr}")
    if sr!=SR:
        sound_file = librosa.resample(sound_file, orig_sr=sr, target_sr=SR)
    mfcc = librosa.feature.mfcc(y=sound_file, sr=SR, n_mfcc=n_mfccs)
    mfcc = torch.Tensor(mfcc)
    if mfcc.shape[1] < max_mfcc_len:
        mfcc = np.pad(mfcc, ((0,0),(0, max_mfcc_len - mfcc.shape[1]))) #pad if less than max len
    else:
        mfcc = mfcc[:, :max_mfcc_len] #truncate at length equal max len
    mfcc = torch.reshape(mfcc, (1, mfcc.shape[0], mfcc.shape[1]))
    print(mfcc.shape)

    #LOADING THE MODEL
    model = torch.load(infer_model_path)
    model.eval()
    pred = torch.Tensor()
    ground_truth = torch.Tensor()

    if network_type == "RNN":
        mfcc = torch.reshape(mfcc, (mfcc.shape[0], -1, n_mfccs))
    elif network_type == "CNN":
        mfcc = torch.reshape(mfcc, (mfcc.shape[0], 1, -1, n_mfccs))
    outputs = model(mfcc)
    predicted = torch.gt(outputs, 0.5)
    if predicted == 0:
        print("MALE")
    else:
        print("FEMALE")

if __name__ == "__main__":
    infer_file = sys.argv[1]
    inference(infer_file)
