import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")

# Test direct call
response = model.generate_content("What is 2+2? Answer in one word.")
print("Direct API test:", response.text)