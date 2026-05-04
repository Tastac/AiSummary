from email.mime import audio
from queue import Queue
import io
import os
import site
import wave
import time
import queue
import threading
from xml.parsers.expat import model
import numpy as np
import pyaudiowpatch as pyaudio
import keyboard
from faster_whisper import WhisperModel
from tkinter import *
from tkinter import ttk

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set to "" to disable GPU, or "0", "1", etc. to specify a GPU device
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
#SECONDS = 5
MODEL_SIZE = "large-v3"  # use "tiny" if too slow, "small" if your PC handles it

CHUNK_SECONDS = 2.0
OVERLAP_SECONDS = 0.5
TARGET_RATE = 16000
FRAMES_PER_BUFFER = 1024

audio = pyaudio.PyAudio()

class AudioRecorder:
    def __init__(self):
        ...

    def stereo_to_mono_float32(raw_bytes, channels):
        audio_i16 = np.frombuffer(raw_bytes, dtype=np.int16)

        if channels > 1:
            audio_i16 = audio_i16.reshape(-1, channels)
            audio_i16 = audio_i16.mean(axis=1)

        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        return audio_f32

    def resample_to_16k(audio_f32, source_rate):
        if source_rate == TARGET_RATE:
            return audio_f32

        old_indexes = np.arange(len(audio_f32))
        new_length = int(len(audio_f32) * TARGET_RATE / source_rate)
        new_indexes = np.linspace(0, len(audio_f32) - 1, new_length)

        return np.interp(new_indexes, old_indexes, audio_f32).astype(np.float32)

    def find_loopback_device(audio):
        for i in range(audio.get_device_count()):
            dev = audio.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice", False):
                return dev
        raise RuntimeError("No Speaker device found.")

    def capture_audio(audio_queue, stop_event):
        device = AudioRecorder.find_loopback_device(audio)

        channels = int(device["maxInputChannels"])
        source_rate = int(device["defaultSampleRate"])

        print("Capturing speakers:")
        print(device["name"])
        print(f"Rate: {source_rate}, channels: {channels}")

        stream = audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=source_rate,
            input=True,
            frames_per_buffer=1024,
            input_device_index=int(device["index"]),
        )

        try:
            while not stop_event.is_set():
                raw = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                mono = AudioRecorder.stereo_to_mono_float32(raw, channels)
                mono_16k = AudioRecorder.resample_to_16k(mono, source_rate)
                audio_queue.put(mono_16k)
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    def transcribe_loop(audio_queue, stop_event):
        model = WhisperModel(
            MODEL_SIZE,
            device=("cuda"),
            compute_type="float16",
        )

        buffer = np.array([], dtype=np.float32)

        chunk_samples = int(TARGET_RATE * CHUNK_SECONDS)
        overlap_samples = int(TARGET_RATE * OVERLAP_SECONDS)

        last_text = ""

        while not stop_event.is_set():
            try:
                audio_piece = audio_queue.get(timeout=1)
                buffer = np.concatenate((buffer, audio_piece))
            except queue.Empty:
                continue

            if len(buffer) < chunk_samples:
                continue

            audio_chunk = buffer[-chunk_samples:]

            segments, info = model.transcribe(
                audio_chunk,
                language="en",
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )

            text = " ".join(segment.text.strip() for segment in segments).strip()

            if text and text != last_text:
                print(text)
                last_text = text

            buffer = buffer[-overlap_samples:]
        #segments, _ = model.transcribe("jschlatt.mp3", word_timestamps=True)



        #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        #for segment in segments:
        #    for word in segment.words:
        #        print(word.word)
                #print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))

if __name__ == "__main__":
    audio_queue = queue.Queue()
    stop_event = threading.Event()

    #capture_thread = threading.Thread(target=AudioRecorder.capture_audio, args=(audio_queue, stop_event), daemon=True)
    #transcribe_thread = threading.Thread(target=AudioRecorder.transcribe_loop, args=(audio_queue, stop_event), daemon=True)

    #AudioRecorder().capture_audio(audio_queue)
    #print("Starting transcription...")

    capture_thread = threading.Thread(
        target=AudioRecorder.capture_audio,
        args=(audio_queue, stop_event),
        daemon=True
    )

    transcribe_thread = threading.Thread(
        target=AudioRecorder.transcribe_loop,
        args=(audio_queue, stop_event),
        daemon=True
    )

    capture_thread.start()
    transcribe_thread.start()

    try:
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        stop_event.set()
        print("\nStopped.")

