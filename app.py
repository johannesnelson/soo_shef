from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests
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
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from gtts import gTTS
import platform


# Load environment variables
dotenv.load_dotenv()
open_ai_api_key = os.getenv("OPENAI_API_KEY")
porcupine_access_key = os.getenv("PORCUPINE_ACCESS_KEY")

# Initialize Flask app
app = Flask(__name__)

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=open_ai_api_key, temperature=0.7)

# Initialize Porcupine for wake word detection
porcupine = pvporcupine.create(
    access_key=porcupine_access_key,
    keyword_paths=["C:\\Users\\johan\\Dropbox\\GitHub\\soo_chef\\wake_word\\Hey-Chef_en_windows_v3_0_0.ppn"]
)
recorder = pvrecorder.PvRecorder(device_index=-1, frame_length=porcupine.frame_length)

# Define the recipe prompt template
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
recipe_extraction_chain = prompt_template | llm


# Utility functions
def fetch_all_text_from_url(url):
    """Fetch text content from a recipe URL."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            text = ' '.join(element.get_text(separator=" ", strip=True) for element in soup.find_all(['p', 'li', 'h1', 'h2', 'h3', 'h4']))
            return text
        return None
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return None


def extract_recipe_details(raw_text):
    """Extract recipe details using LLM."""
    try:
        response = recipe_extraction_chain.invoke({"raw_text": raw_text})
        structured_recipe = response.content
        structured_recipe_json = json.loads(structured_recipe)
        return structured_recipe_json
    except Exception as e:
        print(f"Error extracting recipe details: {e}")
        return None


def speak(text):
    """Speak text using gTTS."""
    print(f"Speaking: {text}")
    tts = gTTS(text=text, lang="en")
    tts.save("output.mp3")  # Save the audio to a file
    
    # Play the audio file (cross-platform)
    if platform.system() == "Windows":
        os.system("start output.mp3")
    elif platform.system() == "Darwin":  # macOS
        os.system("afplay output.mp3")
    else:  # Linux
        os.system("mpg123 output.mp3")
    
    print("Speech playback complete.")

def listen_for_wake_word():
    """Listen for the wake word using Porcupine."""
    recorder.start()
    print("Listening for wake word...")
    while True:
        pcm = recorder.read()
        if porcupine.process(pcm) >= 0:
            print("Wake word detected!")
            recorder.stop()
            return True


def listen_for_question(client):
    """Listen for a voice question."""
    print("Listening for question...")
    duration = 5  # Adjust duration as needed
    sample_rate = 16000
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    audio_data = np.squeeze(audio_data)

    # Save audio as a .wav file for Whisper API
    wavfile.write("question.wav", sample_rate, audio_data)

    # Use OpenAI Whisper for transcription
    with open("question.wav", "rb") as f:
        response = client.audio.transcriptions.create(model="whisper-1", file=f)    
        return response.text


def query_llm_with_recipe(client, recipe, question):
    """Query the LLM with a recipe and user question."""
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
    return response.choices[0].message.content


# Flask routes
@app.route('/process_recipe', methods=['POST'])
def process_recipe():
    """Process a recipe URL and enable voice interactions."""
    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Fetch and process the recipe
    raw_text = fetch_all_text_from_url(url)
    if not raw_text:
        return jsonify({"error": "Failed to fetch text from the provided URL"}), 400

    recipe_details = extract_recipe_details(raw_text)
    if not recipe_details:
        return jsonify({"error": "Failed to extract recipe details"}), 500

    # Enable voice interaction
    client = openai.OpenAI(api_key=open_ai_api_key)

    if listen_for_wake_word():
        speak("I'm listening. What would you like to know?")
        print("Calling listen_for_question now...")
        question = listen_for_question(client)
        answer = query_llm_with_recipe(client, recipe_details, question)
        speak(answer)
        return jsonify({"answer": answer})

    return jsonify(recipe_details)


@app.route('/', methods=['GET'])
def home():
    """Welcome route."""
    return "Welcome to the Voice-Enabled Recipe Assistant!"


# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
