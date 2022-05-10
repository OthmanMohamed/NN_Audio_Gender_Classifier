import os
import glob
import librosa
import random
import numpy as np
from tqdm import tqdm
import pickle
import json

# LOADING CONFIGURATIONS
f = open("config.json")
config = json.load(f)
f.close()
TIMIT_path = config['TIMIT_path']
TIMIT_path = os.path.join(TIMIT_path, "data")
LIBRI_path = config['LIBRI_path']
mfccs_path = config['mfccs_path']
SR = config['SR']
n_mfccs = config['n_mfccs']
max_mfcc_len = config['max_mfcc_len']

# INITIALIZE NEEDED LISTS
wav_files_male_train_timit = []
wav_files_female_train_timit = []
wav_files_male_test_timit = []
wav_files_female_test_timit = []
wav_files_male_libri = []
wav_files_female_libri = []

# PROCESS TIMIT TRAIN SUBSET
path = os.path.join(TIMIT_path, "TRAIN")
for DR in os.listdir(path):
    DR_dir = os.path.join(path, DR)
    for speaker_dir in os.listdir(DR_dir):
        # add the wav path to the required list
        if speaker_dir[0] == 'F':
            speaker_dir = os.path.join(DR_dir, speaker_dir)
            wav_files_female_train_timit.extend([x for x in glob.glob(os.path.join(speaker_dir, "*.wav")) if 'wav' in x])
        elif speaker_dir[0] == 'M' :
            speaker_dir = os.path.join(DR_dir, speaker_dir)
            wav_files_male_train_timit.extend([x for x in glob.glob(os.path.join(speaker_dir, "*.wav")) if 'wav' in x])

# PROCESS TIMIT TEST SUBSET
path = os.path.join(TIMIT_path, "TEST")
for DR in os.listdir(path):
    DR_dir = os.path.join(path, DR)
    for speaker_dir in os.listdir(DR_dir):
        # add the wav path to the required list
        if speaker_dir[0] == 'F':
            speaker_dir = os.path.join(DR_dir, speaker_dir)
            wav_files_female_test_timit.extend([x for x in glob.glob(os.path.join(speaker_dir, "*.wav")) if 'wav' in x])
        elif speaker_dir[0] == 'M' :
            speaker_dir = os.path.join(DR_dir, speaker_dir)
            wav_files_male_test_timit.extend([x for x in glob.glob(os.path.join(speaker_dir, "*.wav")) if 'wav' in x])

wav_files_male_full_timit = wav_files_male_test_timit + wav_files_male_train_timit
wav_files_female_full_timit = wav_files_female_test_timit + wav_files_female_train_timit

# READ LIBRISPEECH SPEAKERS INFO FILE TO GET GENDER LABELS
f = open(os.path.join(LIBRI_path, "SPEAKERS.TXT"), "r")
lines = f.readlines()
Libri_speakers_dict = dict()
for l in lines:
    if not ';' in l:
        split_line = l.replace(' ', '').split('|')
        Libri_speakers_dict[split_line[0]] = split_line[1]

# PROCESS LIBRISPEECH DATA
path = os.path.join(LIBRI_path, "train-clean-100")
for speaker_dir in os.listdir(path):
    speaker_files = []
    speaker_dir = os.path.join(path, speaker_dir)
    #loop over books directories
    for book_dir in os.listdir(speaker_dir):
        book_dir = os.path.join(speaker_dir, book_dir)
        # add book files to speaker files list
        speaker_files += glob.glob(os.path.join(book_dir, "*.flac"))
    # get 10 random files for each speaker
    speaker_subset = random.sample(speaker_files, 10)
    speaker_id = os.path.basename(speaker_subset[0]).split('-')[0]
    if Libri_speakers_dict[speaker_id] == 'F':
        wav_files_female_libri += speaker_subset
    elif Libri_speakers_dict[speaker_id] == 'M':
        wav_files_male_libri += speaker_subset      

wav_files_male_full = wav_files_male_full_timit + wav_files_male_libri
wav_files_female_full = wav_files_female_full_timit + wav_files_female_libri
wav_files_full = []
# create full files list, with gender labels (0 for male and 1 for female)
for w in wav_files_female_full:
    wav_files_full.append((w, 1))
for w in wav_files_male_full:
    wav_files_full.append((w, 0))
random.shuffle(wav_files_full)  

# SAVE OUTPUT TO MFCC PATH
if not os.path.isdir(mfccs_path): os.mkdir(mfccs_path)
labels_dict = dict()
max_len = 0
max_duration = 0
for i, w in enumerate(tqdm(wav_files_full)):
    sound_file, _ = librosa.core.load(w[0], sr=SR)
    mfccs = librosa.feature.mfcc(y=sound_file, sr=SR, n_mfcc=n_mfccs)
    if mfccs.shape[1] > max_len: max_len = mfccs.shape[1]
    if len(sound_file)/SR > max_duration: max_duration = len(sound_file)/SR 
            
    spk_id = os.path.basename(os.path.abspath(os.path.join(w[0], os.pardir)))
    np_name = spk_id + "_" + os.path.basename(w[0][:w[0].find('.')]) + "_mfcc.npy" 
    np_name = os.path.join(mfccs_path, np_name)
    labels_dict[np_name] = w[1]
    np.save(np_name, mfccs)
print("MAX FILE DURATION : ", max_duration, " SEC")
print("MAX MFCC LEN : ", max_len)

# SAVE LABELS FILE, WITH FILE NAME AND GENDER LABEL
lbls_file_path = os.path.join(mfccs_path, "lbls.pkl")
lbls_file = open(lbls_file_path, "wb")
pickle.dump(labels_dict, lbls_file)
lbls_file.close()