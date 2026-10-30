import os
from google import genai
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

models_to_test = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash-exp",
    "gemini-2.5-flash"
]

for m in models_to_test:
    try:
        print(f"Testing {m}...")
        chat = client.chats.create(model=m)
        resp = chat.send_message("Say hello!")
        print(f"Success {m}: {resp.text}")
    except Exception as e:
        print(f"Failed {m}: {e}")
