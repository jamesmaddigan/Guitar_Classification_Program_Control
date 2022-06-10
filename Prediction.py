# Prediction.py>

import pyaudio
import wave
import os
import numpy as np
from scipy.io.wavfile import read
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
from python_speech_features import mfcc, logfbank
from librosa.util import fix_length
import pandas as pd
import pickle
import warnings
import Midi as midi


DATA_PATH = "data_10.json"
Pkl_model = "Clean_Audio_Data_3/MFCC_Model_1.pkl"

required_device = "Bela"        #0 = virtual, 1 = bela, 2 = audiobox

""" virtual port name = 'IAC Driver Virtual Port'
    bela port name = 'Bela'
    
"""

p = pyaudio.PyAudio()
interface_index = 0
filename = "Recording.wav"
processed_file = "newAudio.wav"
# set the chunk size of 1024 samples
chunk = 1024
# sample format
FORMAT = pyaudio.paInt16
# mono, change to 2 if you want stereo
channels = 1
# 44100 samples per second
sample_rate = 22050
audio_seconds = 3
midi_channel = 1             # 1-16



def record(filename, record_seconds, chunk, interface_index, rate):
    # open stream object as input & output
    stream = p.open(format=FORMAT,
                channels=channels,
                rate=rate,
                input_device_index = interface_index,
                input=True,
                output=True,
                frames_per_buffer=chunk)
    frames = []
    print("\nRecording...")
    for i in range(int(sample_rate / chunk * audio_seconds)):
        data = stream.read(chunk)
        frames.append(data)
    print("Finished recording.")
    
    # stop and close stream
    stream.stop_stream()
    stream.close()
    # terminate pyaudio object
    p.terminate()
    # save audio file
    # open the file in 'write bytes' mode
    wf = wave.open(filename, "wb")
    # put newly opened wav file at the end of the path to specific directory

    # set the channels
    wf.setnchannels(channels)
    # set the sample format
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # set the sample rate
    wf.setframerate(sample_rate)
    # write the frames as bytes
    data = wf.writeframes(b"" .join(frames))
    # os.path.join(path, filename)
    # close the file
    wf.close()
    
    return filename
    



def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


    original_signal, rate = librosa.load(filename,  sr=rate)
    mask = envelope(original_signal, rate, 0.0035)
    clean_signal = original_signal[mask]

    # audio, sf = librosa.load(file_path, sr=rate)   # mono=True converts stereo audio to mono
    clean_padded_signal = fix_length(clean_signal, size=audio_seconds*rate)     # librosa function zero pads signal to required length


    print('Array length before padding', np.shape(clean_signal))
    print('Audio length before padding in seconds', (np.shape(clean_signal)[0]/rate))
    print('Array length after padding', np.shape(clean_padded_signal))
    print('Audio length after padding in seconds', (np.shape(clean_padded_signal)[0]/rate))
    
    
    wavfile.write(filename = 'newAudio.wav', rate=rate, data=clean_padded_signal)

    
def process_data(filename, rate, required_audio_size, new_file):
    

   original_signal, rate = librosa.load(filename,  sr=rate)
   mask = envelope(original_signal, rate, 0.0035)
   clean_signal = original_signal[mask]

   # audio, sf = librosa.load(file_path, sr=rate)   # mono=True converts stereo audio to mono
   clean_padded_signal = fix_length(clean_signal, size=required_audio_size*rate)     # librosa function zero pads signal to required length

   wavfile.write(filename=new_file, rate=rate, data=clean_padded_signal)
   
   
def plot_signal_features(y, rate):
    
    
    n = len(y)
    freq = np.fft.rfftfreq(n,d=1/rate)
    Y= abs(np.fft.rfft(y)/n)**4
    fbank = logfbank(y[:rate], rate, nfilt=26, nfft=2048).T
    mel = mfcc(y[:rate], rate, numcep=26, nfilt=26, nfft=2048).T
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
    axs[0, 0].plot(y)
    axs[0, 0].set_title('Signal', fontsize = 25.0)
    axs[0, 0].set_xlabel('Time', fontsize = 15)
    axs[0, 0].set_ylabel('Magnitude', fontsize = 15)
    axs[0, 1].plot(freq, Y)
    axs[0, 1].set_title('fft', fontsize = 25.0)
    axs[0, 1].set_xlabel('Frequency [Hz]', fontsize = 15)
    axs[0, 1].set_ylabel('Power', fontsize = 15)
    axs[1, 0].imshow(fbank, cmap='hot', interpolation='nearest')
    axs[1, 0].set_title('Mel frequency Spectrogram', fontsize = 25.0)
    axs[1, 0].set_xlabel('Time (ms)', fontsize = 15)
    axs[1, 0].set_ylabel('Mel-Frequency', fontsize = 15)
    axs[1, 1].imshow(mel, cmap='hot', interpolation='nearest')
    axs[1, 1].set_title('mfcc' , fontsize = 25.0)
    axs[1, 1].set_xlabel('Time (ms)', fontsize = 15)
    axs[1, 1].set_ylabel('Index', fontsize = 15)

    
def prepare_data_for_model(filename, rate):
    
    newAudio, rate = librosa.load("newAudio.wav", sr=rate)
    mfcc = librosa.feature.mfcc(newAudio, sr = sample_rate, n_fft=2048, n_mfcc=13, hop_length=512)
    X = mfcc.T
    X = X[..., np.newaxis] # array shape (1, 130, 13, 1)
    X = X[np.newaxis, ...]
    
    return X, newAudio
   
def delete_wav(filename):
    
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print("The file does not exist")
        
    

def get_input_devices():
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
  
def prediction(filename, mfcc):
    
    with open(Pkl_model, 'rb') as file:
        model = pickle.load(file)
    
    prediction = model.predict(mfcc)
    
    predicted_index = np.argmax(prediction, axis=1)
    
    if (predicted_index == 0):
        print ("\nPlaying Acoustic Guitar")
        program_index = 0
        
    elif (predicted_index == 1):
        print("\nplaying Stratocaster")
        program_index = 1
        
    return program_index
  

    
if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    
    get_input_devices()
    record(filename=filename, record_seconds=audio_seconds, chunk=chunk, interface_index=interface_index, rate=sample_rate)
    process_data(filename, rate=sample_rate, required_audio_size=audio_seconds, new_file=processed_file)
    X, newAudio = prepare_data_for_model(filename=filename, rate=sample_rate)
    program_index = prediction(filename=filename, mfcc=X)
    plot_signal_features(newAudio, sample_rate)
    delete_wav(filename=filename)
    delete_wav(filename="newAudio.wav")
    
    required_device_index = midi.get_midi_ports(required_device= required_device)
    #midi.print_midi_bytes()
    channel_hex = midi.midi_message_info(midi_channel=midi_channel, program_num=program_index)
    midi.send_midi_message(midi_port_index=required_device_index, channel_num=channel_hex, program_num=program_index)
    

    
    


