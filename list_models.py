import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Configure the library with your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List available models
print("Available models:")
for m in genai.list_models():
    print(m.name)