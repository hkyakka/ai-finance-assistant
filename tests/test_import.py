from __future__ import annotations
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    print("Attempting to import google.genai...")
    try:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)
        print("Successfully imported and initialized google.genai")
        print("\n--- Available Models ---")
        for m in client.models.list():
            if "generateContent" in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
