from email.mime import audio, text
from html import entities
from queue import Queue
import io
import os
import site
import wave
import time
import queue
import threading
import numpy as np
import pyaudiowpatch as pyaudio
import keyboard
import spacy
import numpy as np
import json
import os
from faster_whisper import WhisperModel
from tkinter import *
from tkinter import ttk

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CHANNELS = 2
FORMAT = pyaudio.paInt16
TARGET_RATE = 16000
FRAMES_PER_BUFFER = 1024

MODEL_SIZE = "medium"
CHUNK_SECONDS = 1.5
OVERLAP_SECONDS = 0.1

audio = pyaudio.PyAudio()

headphone_text = ""

PATH_FILE = "campaign.json"

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

    def is_similar(a, b):
        return a in b or b in a

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
        #text = ""

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
                temperature=0.0,
                vad_filter=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.65,
                log_prob_threshold=0.8,
                compression_ratio_threshold=2.4 ,
                word_timestamps=True,
            )

            for segment in segments:
                for word in segment.words:
                    print(word.word)
                entities = Sorter.extract_entities(segment.text)
                print(entities)
                    #headphone_text = word.word

            buffer = buffer[-overlap_samples:]

class Sorter:
    nlp = spacy.load("en_core_web_sm")

    #ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})

    RACES = [
        "human",
        "elf",
        "dwarf",
        "orc",
        "tiefling",
        "goblin",
        "gnome",
        "halfling",
        "dragonborn",
        "warforged",
        "plasmoid",
        "goliath",
    ]
    GENDER_WORDS = {
        "male": ["man", "male", "boy", "guy", "gentleman"],
        "female": ["woman", "female", "girl", "lady"]
    }

    CITY_WORDS = [
        "waterdeep",
        "daggerford",
        "baldur's gate",
        "neverwinter"
    ]

    BUILDINGS = [
        "tavern",
        "inn",
        "castle",
        "temple",
        "blacksmith",
        "church",
        "shop"
    ]

    MONSTERS = [
        "goblin",
        "orc",
        "dragon",
        "mind flayer",
        "beholder"
    ]

    def current_city(window):
        text = " ".join(window).lower()
        for city in Sorter.CITY_WORDS:
            if city in text:
                return city
        return None

    def detect_gender(window):
        text = " ".join(window).lower()
        for gender, words in Sorter.GENDER_WORDS.items():
            for word in words:
                if word in text:
                    return gender
        return None

    def extract_entities(text):
        doc = Sorter.nlp(text)

        npcs = []
        cities = []
        buildings = []
        monsters = []

        npc = ""
        city = ""
        building = ""

        words = [token.text for token in doc]

        for ent in doc.ents:

            # PERSON / NPC
            if ent.label_ == "PERSON":

                start = max(0, ent.start - 5)
                end = min(len(doc), ent.end + 5)

                window = [token.text.lower() for token in doc[start:end]]

                race = None

                for word in window:
                    if word in Sorter.RACES:
                        race = word
                        break

                gender = Sorter.detect_gender(window)

                npcs.append({
                    "name": ent.text,
                    "race": race,
                    "gender": gender,
                })

            # CITY DETECTION
            if ent.label_ in ("GPE", "LOC"):

                start = max(0, ent.start - 3)
                end = min(len(doc), ent.end + 3)

                window = [token.text.lower() for token in doc[start:end]]

                is_city = False

                for word in window:
                    if word in Sorter.CITY_WORDS:
                        is_city = True
                        city = word
                        break

                if is_city or ent.text[0].isupper():
                    cities.append(ent.text)
                    Memory.add_city(PATH_FILE, ent.text)

        # simple word scan
        lowered = text.lower()

        for building in Sorter.BUILDINGS:
            if building in lowered:
                buildings.append(building)
                Memory.add_building(PATH_FILE, city, building)

        for monster in Sorter.MONSTERS:
            if monster in lowered:
                monsters.append(monster)

        # remove duplicates
        cities = list(dict.fromkeys(cities))
        buildings = list(dict.fromkeys(buildings))
        monsters = list(dict.fromkeys(monsters))

        unique_npcs = []

        seen = set()

        for npc in npcs:
            if npc["name"] not in seen:
                unique_npcs.append(npc)
                seen.add(npc["name"])

        return {
            "npcs": unique_npcs,
            "cities": cities,
            "buildings": buildings,
            "monsters": monsters,
        }

class Memory:
    def load_memory(self, path):
        if not os.path.exists(path):
            return {"cities": {}}
        with open(PATH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_memory(path):
        with open(PATH_FILE, "w", encoding="utf-8") as f:
            json.dump(path, f, indent=2)

    def add_city(self, memory, city):
        if city not in memory["cities"]:
            memory["cities"][city] = {"buildings": {}}
            self.save_memory()
    
    def add_building(self, memory, city, building):
        cities = memory["cities"]

        if city not in cities:
            cities[city] = {"buildings": {}}

        buildings = cities[city]["buildings"]

        if building not in buildings:
            buildings[building] = {"npcs": []}
            self.save_memory()

    def add_npc(self, memory, city, building, npc_name, race=None, gender=None):
        cities = memory["cities"]

        if city not in cities:
            cities[city] = {"buildings": {}}

        buildings = cities[city]["buildings"]

        if building not in buildings:
            buildings[building] = {"npcs": []}

        npcs = buildings[building]["npcs"]

        if npc_name not in npcs:
            npcs[npc_name] = {
                "race": race,
                "gender": gender
            }
        else:
            if race:
                npcs[npc_name]["race"] = race
            if gender:
                npcs[npc_name]["gender"] = gender

        self.save_memory()

class GUI:
    def __init__(self):
        ...

    def setup(self):
        self.root = Tk()
        self.root.title("Live Transcription")

        return self.root

    def update(self, new_text):
        frm = ttk.Frame(gui)
        frm.grid()
        
        self.text.delete(1.0, END)
        self.text.insert(END, new_text)


if __name__ == "__main__":
    gui = Tk()
    gui.title("Live Transcription")
    #ttk.Label(frm, text=voice, font=("Helvetica", 16)).pack(pady=10)

    '''
    gui.text = Text(gui, wrap=WORD)
    gui.text.bind("Hallo")
    gui.text.delete(1.0, END)
    gui.text.insert(END, headphone_text)
    gui.text.pack(expand=True, fill=BOTH)
    '''
    audio_queue = queue.Queue()
    stop_event = threading.Event()

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

    gui_thread = threading.Thread(
        target=gui.update,
        args=(),
        daemon=True
    )


    capture_thread.start()
    transcribe_thread.start()

    #gui.mainloop()

    try:
        while True:
            time.sleep(0.2)
            #print("1")
    
    except KeyboardInterrupt:
        #stop_event.set()
        print("\nStopped.")