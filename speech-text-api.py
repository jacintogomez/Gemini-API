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


def speech_to_text():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use default microphone as the audio source
    with sr.Microphone() as source:
        print("Listening...")

        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)

        # Capture the audio input
        audio_data = recognizer.listen(source)

        print("Processing...")

        try:
            # Recognize speech using Google Speech Recognition
            text = recognizer.recognize_google(audio_data)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand the audio.")
            return None
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return None



# Example usage
text = speech_to_text()



# file='recordings/machine.wav'
# run_quickstart(file)