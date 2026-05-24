import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# నీ API Key ని కాన్ఫిగర్ చేస్తున్నాం
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# అందుబాటులో ఉన్న మోడల్స్ ని లిస్ట్ చేస్తున్నాం
print("Available models supporting embedding:")
for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(f"Model Name: {m.name}")