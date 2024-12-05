from bs4 import BeautifulSoup
import requests
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json
import pvporcupine
import pvrecorder
import pyttsx3
import openai
import sounddevice as sd
import numpy as np

import scipy.io.wavfile as wavfile
import os
import dotenv

# Load open_ai_API key from .env file
dotenv.load_dotenv()
open_ai_api_key = os.getenv("OPENAI_API_KEY")
porcupine_access_key = os.getenv("PORCUPINE_ACCESS_KEY")

# Function to fetch all text from a URL
def fetch_all_text_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = ' '.join(element.get_text(separator=" ", strip=True) for element in soup.find_all(['p', 'li', 'h1', 'h2', 'h3', 'h4']))
        return text
    else:
        return None

# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", 
                 openai_api_key=open_ai_api_key, 
                 temperature=0.7)

# Define a prompt template for extracting recipe information
prompt_template = PromptTemplate(
    input_variables=["raw_text"],
    template=(
        "You are a recipe extraction assistant. I have some unstructured text from a recipe that has a lot of extraneous information in it:\n\n"
        "{raw_text}\n\n"
        "Please analyze this text and extract the following details:\n"
        "- Name of the dish (e.g., 'Chocolate Cake')\n"
        "- Ingredients (in list form)\n"
        "- Steps (in numbered list form)\n"
        "- Any additional relevant information (e.g., cooking time, serving size, special notes)\n\n"
        "Return the result in the following JSON format:\n"
        "{{ 'title': 'Dish Name', 'ingredients': ['ingredient1', 'ingredient2'], 'steps': ['Step 1', 'Step 2'], 'additional_info': 'Other details if any' }}"
    )
)

# Chain the prompt template and LLM
recipe_extraction_chain = prompt_template | llm

# Function to extract recipe details from raw text
def extract_recipe_details(raw_text):
    response = recipe_extraction_chain.invoke({"raw_text": raw_text})
    structured_recipe = response.content
    structured_recipe_json = json.loads(structured_recipe)
    return structured_recipe_json

# Initialize Porcupine for wake word detection
porcupine = pvporcupine.create(
    access_key="EfQ5zAXVwcoyqMxsRwYlIZoC0gPnerl6qJwNiYCZrWZvrMfv9Gszlw==",
    keyword_paths=["C:\\Users\\johan\\Dropbox\\GitHub\\soo_chef\\wake_word\\Hey-Chef_en_windows_v3_0_0.ppn"]
)
recorder = pvrecorder.PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
recorder.start()

# Initialize TTS engine
tts_engine = pyttsx3.init()

def speak(text):
    print(f"Assistant: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen_for_wake_word():
    print("Listening for wake word...")
    while True:
        pcm = recorder.read()
        if porcupine.process(pcm) >= 0:
            print("Wake word detected!")
            return True

def listen_for_question(client):
    print("Listening for question...")
    duration = 5  # Adjust duration as needed
    sample_rate = 16000
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    audio_data = np.squeeze(audio_data)

    # Save audio as a .wav file for Whisper API
    wavfile.write("question.wav", sample_rate, audio_data)

    # Use the latest OpenAI Whisper API call
    with open("question.wav", "rb") as f:
        response = client.audio.transcriptions.create(model="whisper-1", file=f)
        
    question = response.text
    print(f"User asked: {question}")
    return question

def query_llm_with_recipe(client, recipe, question):
    prompt = (
        f"You are a helpful cooking assistant. Here is a recipe:\n\n"
        f"Title: {recipe['title']}\n\n"
        f"Ingredients:\n" + "\n".join(recipe["ingredients"]) + "\n\n"
        f"Steps:\n" + "\n".join(recipe["steps"]) + "\n\n"
        f"Additional Information: {recipe['additional_info']}\n\n"
        f"User Question: {question}\n\n"
        "Answer the question as accurately and concisely as possible based on the recipe above."
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7
    )
    answer = response.choices[0].message.content
    print(f"LLM Answer: {answer}")
    return answer

# Main assistant loop
def main(recipe):
    client = openai.OpenAI(api_key=open_ai_api_key)
    
    try:
        while True:
            if listen_for_wake_word():
                speak("I'm listening. What would you like to know?")
                question = listen_for_question(client)
                answer = query_llm_with_recipe(client, recipe, question)
                speak(answer)
    
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        recorder.stop()
        recorder.delete()

# Example usage
if __name__ == "__main__":
    url = input("Please enter the URL of the recipe: ")
    raw_text = fetch_all_text_from_url(url)
    if raw_text:
        recipe_details = extract_recipe_details(raw_text)
        main(recipe_details)
    else:
        print("Failed to fetch text from the provided URL.")