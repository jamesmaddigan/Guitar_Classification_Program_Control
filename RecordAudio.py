import pyaudio
import wave
import os
import numpy as np
from numpy.fft import fft
from scipy.io.wavfile import read
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
from librosa.util import fix_length
import pandas as pd


signals = {}
fft = {}
p = pyaudio.PyAudio()
interface_index = 0
filename = "Recording.wav"
# set the chunk size of 1024 samples
chunk = 1024
# sample format
FORMAT = pyaudio.paInt16
# mono, change to 2 if you want stereo
channels = 1
# 44100 samples per second
sample_rate = 44100
audio_seconds = 3

number =2100
num = f"{number:04}"



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
    print("Recording...")
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


def zero_padding(file_path, rate, required_audio_size):

    audio, sf = librosa.load(file_path, sr=rate)   # mono=True converts stereo audio to mono
    padded_audio = fix_length(audio, size=required_audio_size*rate)     # array size is required_audio_size*sampling frequency


    print('Array length before padding', np.shape(audio))
    print('Audio length before padding in seconds', (np.shape(audio)[0]/rate))
    print('Array length after padding', np.shape(padded_audio))
    print('Audio length after padding in seconds', (np.shape(padded_audio)[0]/rate))
    
    return padded_audio

def plot_signal_and_fft(original_signal, clean_signal, rate):
    
    n = len(clean_signal)
    freq = np.fft.rfftfreq(n,d=1/rate)
    Y= abs(np.fft.rfft(clean_signal)/n)**4
    
    fft=np.fft.fft(clean_signal)
    magnitude = np.abs(fft)**4
    frequency = np.linspace(0, rate, len(magnitude))
    
    magnitude = magnitude[:int(len(frequency)/2)]
    frequency = frequency[:int(len(frequency)/2)]

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
    axs[0].plot(original_signal)
    axs[0].set_title('Original Signal', fontsize = 25.0)
    axs[0].set_xlabel('Sample', fontsize = 15)
    axs[0].set_ylabel('Magnitude', fontsize = 15)
    axs[1].plot(clean_signal)
    axs[1].set_title('Clean Signal', fontsize = 25.0)
    axs[1].set_xlabel('Sample', fontsize = 15)
    axs[1].set_ylabel('Magnitude', fontsize = 15)
    axs[2].plot(freq, Y)
    axs[2].set_title('Fast Fourier Transform', fontsize = 25.0)
    axs[2].set_xlabel('Frequency [Hz]', fontsize = 15)
    axs[2].set_ylabel('Power', fontsize = 15)
    
    fig.tight_layout()
    
def write_file(filename, rate, num, required_audio_size):
    
   input_data = read(filename)
   audio = input_data[1]
   original_signal, rate = librosa.load(filename,  sr=rate)
   mask = envelope(original_signal, rate, 0.0035)
   clean_signal = original_signal[mask]

   # audio, sf = librosa.load(file_path, sr=rate)   # mono=True converts stereo audio to mono
   clean_padded_signal = fix_length(clean_signal, size=required_audio_size*rate)     # librosa function zero pads signal to required length


   print('Array length before padding', np.shape(clean_signal))
   print('Audio length before padding in seconds', (np.shape(clean_signal)[0]/rate))
   print('Array length after padding', np.shape(clean_padded_signal))
   print('Audio length after padding in seconds', (np.shape(clean_padded_signal)[0]/rate))
    
   wavfile.write(filename = 'Clean_Audio_Data_3/Stratocaster/Strat_O.'+ num +'.wav', rate=rate, data=original_signal)
   wavfile.write(filename = 'Clean_Audio_Data_3/Stratocaster/Strat_C.'+ num +'.wav', rate=rate, data=clean_padded_signal)
    
    
   print("Writing data for sample " + num + ".") 
   
   if os.path.exists(filename):
       os.remove(filename)
   else:
       print("The file does not exist")
       
   
   plot_signal_and_fft(original_signal = original_signal, clean_signal =  clean_padded_signal, rate=sample_rate)
   plt.show()
   
   
    

def get_input_devices():
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    
    
if __name__ == "__main__":
    
    get_input_devices()
    
    record(filename=filename, record_seconds=audio_seconds, chunk=chunk, interface_index=interface_index, rate=sample_rate)
    write_file(filename=filename, rate=sample_rate, num = num, required_audio_size=audio_seconds)
    
    