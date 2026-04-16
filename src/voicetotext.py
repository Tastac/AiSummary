import io
import os
import wave
import time
import queue
import threading
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu")

segments, info = model.transcribe("loopback_record_class.wav", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")