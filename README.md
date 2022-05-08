# NN Audio Gender Classifier
This project implement neural based classifiers for gender classification from audio files
RNN models and CNN models are implemented and compared

## Setting Up Environment
Requirements needed can be found in requirements.txt
It is recommended to set up a new Anaconda environment or python virtual env and install the required requirements.

## Preparing Data
#### Downloading Datasets
For training and validation, We have used TIMIT dataset and the 100 hr subset of LibriSpeech Dataset.
To Download The data:
+ TIMIT: https://goo.gl/l0sPwz
+ 100 hr Subset LibriSpeech: https://www.openslr.org/resources/12/train-clean-100.tar.gz

#### Feature Extraction
Mel-frequency cepstrum (MFCCs) are calculated for each frame of the data audio files      
To extract mfcc files, modify the Datasets paths and mfccs_path in `config.json` Output Mfccs will be saved in `"mfccs_path"`
Then run the command	`python train.py`
It is expected for the paths to be passed to be structured as follow


    .
    ├── ...
    └─archive			##TIMIT_path                  
		├── PHONCODE.DOC
		├── PROMPTS.TXT
		├── README.DOC
		├── SPKRINFO.TXT
		├── SPKRSENT.TXT
		├── test_data.csv
		├── TESTSET.DOC
		├── TIMITDIC.DOC
		├── TIMITDIC.TXT
		├── train_data.csv
		└── data
			   ├── TEST
			   └── TRAIN
     ...
    .
    └─── LibriSpeech		#LibriSpeech Path          
		├──BOOKS.TXT
		├──CHAPTERS.TXT
		├──LICENSE.TXT
		├──README.TXT
		├──SPEAKERS.TXT
		└──train-clean-100 ...
		
## Configurations File
Modify the needed hyper parameters in `config.json` file, you can modify files Sampling Rate, Learning Rate, Number of epochs, and other parameters
You can also choose the model type RNN or CNN by setting the parameter `network_type`
This is an example of `config.json`
```
    {
    "out_path": "out/",
    "mfccs_path": "/data/datasets/mfccs",
    "TIMIT_path" : "/data/datasets/archive",
    "LIBRI_path" : "/data/datasets/train-clean-100/LibriSpeech",
    "SR" : 16000,
    "n_mfccs" : 20,
    "max_mfcc_len" : 250,
    "batch_size": 64,
    "data_shuffle": true,
    "train_percent": 0.8,
    "test_percent": 0.1,
    "val_percent": 0.1,
    "network_type": "RNN",
    "learning_rate": 0.001,
    "max_epochs": 100,
    "log_interval": 10,
    "save_checkpoint_path" : "out/",
    "save_checkpoint_interval" : 200
    }
```

## Training The Network
You can run training by running the script `python train.py`
Output to monitor losses and accuracies will be written in the terminal, and a tensorboard log out will be saved in out_path


