from dotenv import load_dotenv
import os
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("TRANSLATION_CREDENTIALS_PATH")

import speech_recognition as sr

def run_quickstart(file):
    r = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio = r.record(source)  # read th5e entire audio file

    # recognize speech using Sphinx
    try:
        print("Sphinx thinks you said " + r.recognize_sphinx(audio))
    except sr.UnknownValueError:
        print("Sphinx could not understand audio")
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))


file='recordings/speech.mp3'
run_quickstart(file)