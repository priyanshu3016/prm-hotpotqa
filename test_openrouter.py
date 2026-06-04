import os
from dotenv import load_dotenv
import requests

load_dotenv()

key = os.getenv("OPENROUTER_API_KEY")

resp = requests.get(
    "https://openrouter.ai/api/v1/key",
    headers={
        "Authorization": f"Bearer {key}"
    }
)

print(resp.status_code)
print(resp.text)