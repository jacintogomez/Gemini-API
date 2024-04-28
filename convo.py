import google.generativeai as genai
from google.cloud import translate_v2 as translate
import pygame
import torch
import sounddevice as sd
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from scipy.io.wavfile import write
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google.cloud import texttospeech

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
openai_api_key=os.getenv("OPENAI_API_KEY")
client=OpenAI()

llm=ChatOpenAI()
prompt=ChatPromptTemplate.from_messages([
    ("system", "Respond conversationally to the given input, in whatever language it is given in"),
    ("user", "{input}")
])
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

device="cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
model_id="openai/whisper-base"
openaimodel=AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,torch_dtype=torch_dtype,
    use_safetensors=True
)
openaimodel.to(device)
processor=AutoProcessor.from_pretrained(model_id)
pipe=pipeline(
    "automatic-speech-recognition",
    model=openaimodel,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

genai.configure(api_key=GOOGLE_API_KEY)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("TRANSLATION_CREDENTIALS_PATH")
english='en'

# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)

model = genai.GenerativeModel('gemini-pro')

def record_audio(filename,duration=5,fs=44100):
    print('recording...')
    recording=sd.rec(int(duration*fs),samplerate=fs,channels=1)
    sd.wait()
    write(filename,fs,recording)
    result=pipe(filename,generate_kwargs={'language':lang})
    return result['text']

def play_audio(file):
    sound=pygame.mixer.Sound(file)
    recordlength=int(sound.get_length()*1000)
    sound.play()
    pygame.time.wait(recordlength)

def make_speech_file(speech_file_path,text):
    googclient = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice=texttospeech.VoiceSelectionParams(
        language_code=lang, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = googclient.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    with open(speech_file_path, "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

def translate_to(lang,text):
    transclient=translate.Client()
    if isinstance(text,bytes):
        text=text.decode('utf-8')
    result=transclient.translate(text,target_language=lang)
    return result['translatedText']

def generate_response(input_text):
    prompt='Please respond logically to the following sentence in a conversation: '+input_text
    generation=model.generate_content(prompt).text
    #print(generation)
    return generation

def machine_turn(text):
    machine_file='recordings/machine.wav'
    if text=='':
        eng='Hello, how are you today?'
        talk=translate_to(lang,eng)
    else:
        talk=generate_response(text)
    make_speech_file(machine_file,talk)
    eng=translate_to(english,machine_file)
    play_audio(machine_file)
    print('Computer: '+talk+' ('+eng+')')

def human_turn():
    file='recordings/human.wav'
    talk=record_audio(file)
    eng=translate_to(english,file)
    print('Me: '+talk+' ('+eng+')')
    return talk

def conversation():
    human_response=''
    x=0
    while x!=5:
        machine_turn(human_response)
        human_response=human_turn()
        x+=1

lang='en'
pygame.init()
conversation()
pygame.quit()