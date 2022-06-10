import json
import os
import math
import librosa
from scipy.io import wavfile
import warnings

DATASET_PATH = "Clean_Audio_Data_3"
JSON_PATH = "Clean_Audio_Data_3/Audio_Data_MFCC.json"
SAMPLE_RATE = 22050

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512):
    
    """Extracts MFCCs from music dataset and saves them into a json file along with genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """
        
        # dictionary to store mapping, labels, and MFCCs
    data = {
            "mapping": [],
            "labels": [],
            "mfcc": []
            }
    
    # loop through all guitar folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        """
         i: count
         dirpath: path to folder we are currently in.
         dirnames: name of all folders within the folder currently in (Stratocaster and Acoustic)
         filenames: all the files in dirpath
        """   
        # ensure we're processing a guitar sub-folder level
        if dirpath is not dataset_path:
        
            # save the semantic
            dirpath_components = dirpath.split("/") # if we have CleanGuitar/Stratocaster => ["CleanGuitar", "Stratocaster"]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))
            
            # proccess files for specific guitar
            for f in filenames:
                
                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
                mfcc = librosa.feature.mfcc(signal, sr = SAMPLE_RATE, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
                mfcc = mfcc.T
                
                data["mfcc"].append(mfcc.tolist())
                data["labels"].append(i-1)
                print("{}".format(file_path))
                
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
            

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    save_mfcc(DATASET_PATH, JSON_PATH)

        
