# NN Audio Gender Classifier
This project implement neural based classifiers for gender classification from audio files
RNN models and CNN models are implemented and compared

### Setting Up Environment
Requirements needed can be found in requirements.txt
It is recommended to set up a new Anaconda environment or python virtual env and install the required requirements.

### Preparing Data
#### Downloading Datasets
For training and validation, We have used TIMIT dataset and the 100 hr subset of LibriSpeech Dataset.
To Download The data:
+ TIMIT: https://goo.gl/l0sPwz
+ 100 hr Subset LibriSpeech: https://www.openslr.org/resources/12/train-clean-100.tar.gz

#### Feature Extraction
Mel-frequency cepstrum (MFCCs) are calculated for each frame of the data audio files      
To extract mfcc files, modify the Datasets paths in `config.json` file
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
