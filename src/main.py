from email.mime import audio
from queue import Queue
import io
import os
import wave
import time
import queue
import threading
import numpy as np
import pyaudiowpatch as pyaudio
#import sounddevice as sd
#from faster_whisper import WhisperModel
from tkinter import *
from tkinter import ttk

filename = "loopback_record_class.wav"
data_format = pyaudio.paInt24

RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt32
SECONDS = 5
OUTPUT = "output.wav"

def find_divices(p):
    for i in range(0, p.get_host_api_info_by_index(0).get('deviceCount') - 1):
        #if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        #    print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels')) > 0:
            print("Output Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

def record_audio():
    with pyaudio.PyAudio() as p:
        find_divices(p)
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        output=True,
                        frames_per_buffer=1024)

        frames = []
        for i in range(0, int(RATE / 1024 * SECONDS)):
            data = stream.read(1024)
            frames.append(data)
            stream.write(data)

        stream.stop_stream()
        stream.close()

    with wave.open(OUTPUT, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def record_text():
    text = "Start"
    if text == "Record":
        text = "Stop"
        record_audio()

if __name__ == "__main__":
    root = Tk()
    root.title("Voice to Text")
    root.geometry("400x300")

    label = Label(root, text="Press the button to start recording")
    label.pack(pady=20)

    record_button = Button(root, text="Start", command=record_text())
    record_button.pack(pady=10)

    root.mainloop()
