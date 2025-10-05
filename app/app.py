# pip install google-genai python-dotenv

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()  # Loads GEMINI_API_KEY from .env if present

MODEL_NAME = "gemini-2.5-flash"


def build_prompt(temp_c, feels_c, humidity, wind_ms, rain_prob, uv_index):
    return f"""
Respond in English only.

You are a concise outdoor-planning assistant. Use the numbers below to
describe the weather in simple, friendly English and suggest activities.

Constraints:
- Be short (120–180 words), structured, and practical.
- First: a 1–2 sentence overview (comfort, risks).
- Then: bullet points with tips (clothing, hydration, sunscreen).
- Finally: 3 recommended activities and 3 to avoid (with 5–8 word reasons).
- Use °C, m/s, %, UV as given. No warnings about consulting professionals.

Data:
- Temperature: {temp_c:.1f} °C
- Feels like:  {feels_c:.1f} °C
- Humidity:    {humidity:.0f} %
- Wind:        {wind_ms:.1f} m/s
- Rain prob.:  {rain_prob:.0f} %
- UV index:    {uv_index:.1f}

Return only the answer, no preface.
""".strip()


def generate_weather_description(temp_c, feels_c, humidity, wind_ms, rain_prob, uv_index):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY env var (or put it in a .env file).")

    client = genai.Client(api_key=api_key)
    prompt = build_prompt(temp_c, feels_c, humidity, wind_ms, rain_prob, uv_index)

    # If your installed google-genai version doesn't support ThinkingConfig,
    # you can remove the 'config=' argument entirely (see fallback below).
    try:
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1)
        )
        chunks = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=config,
        )
    except Exception:
        # Fallback without thinking_config if your SDK version doesn't support it
        chunks = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        )

    out = []
    for ch in chunks:
        if getattr(ch, "text", None):
            print(ch.text, end="")  # streaming to console (optional)
            out.append(ch.text)
    return "".join(out)


if __name__ == "__main__":
    desc = generate_weather_description(
        temp_c=24.2,
        feels_c=25.0,
        humidity=58,
        wind_ms=3.6,
        rain_prob=22,
        uv_index=6.5,
    )
    print("\n\n---\nRESULT:\n", desc)
