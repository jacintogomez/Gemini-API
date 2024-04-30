import vertexai

from vertexai.generative_models import GenerativeModel, ChatSession

from dotenv import load_dotenv
import os
load_dotenv()

project_id=os.getenv("CLOUD_PROJECT_ID")
location=os.getenv("CLOUD_PROJECT_LOCATION")
vertexai.init(project=project_id, location=location)
model=GenerativeModel(model_name="gemini-1.0-pro-002")
chat=model.start_chat()

def get_chat_response(chat: ChatSession, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)

prompt = "Hello."
print(get_chat_response(chat, prompt))

prompt = "What are all the colors in a rainbow?"
print(get_chat_response(chat, prompt))

prompt = "Why does it appear when it rains?"
print(get_chat_response(chat, prompt))