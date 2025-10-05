# app.py
# pip install fastapi uvicorn google-genai python-dotenv pydantic

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, confloat
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()  # Loads GEMINI_API_KEY from .env if present

MODEL_NAME = "gemini-2.5-flash"

# ---------- Prompt ----------

def build_prompt(temp_c: float, feels_c: float, humidity: float,
                 wind_ms: float, rain_prob: float, uv_index: float,
                 activity: Optional[str] = None) -> str:
    activity_hint = f"\nAlso, evaluate suitability specifically for: **{activity}**." if activity else ""
    return f"""
Respond in English only.

You are a concise outdoor-planning assistant. Use the numbers below to
describe the weather in simple, friendly English and suggest activities.

Constraints:
- Be short (120–180 words), structured, and practical.
- First: a 1–2 sentence overview (comfort, risks).
- Then: bullet points with tips (clothing, hydration, sunscreen).
- Finally: 3 recommended activities and 3 to avoid (with 5–8 word reasons).
- Use °C, m/s, %, UV as given. No warnings about consulting professionals.{activity_hint}

Data:
- Temperature: {temp_c:.1f} °C
- Feels like:  {feels_c:.1f} °C
- Humidity:    {humidity:.0f} %
- Wind:        {wind_ms:.1f} m/s
- Rain prob.:  {rain_prob:.0f} %
- UV index:    {uv_index:.1f}

Return only the answer, no preface.
""".strip()

def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY env var (or put it in a .env file).")

    client = genai.Client(api_key=api_key)

    # Intentamos con ThinkingConfig; si tu versión no lo soporta, usamos el fallback.
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
        chunks = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        )

    out = []
    for ch in chunks:
        if getattr(ch, "text", None):
            out.append(ch.text)
    return "".join(out).strip()

# ---------- FastAPI ----------

app = FastAPI()

class WeatherRequest(BaseModel):
    temp_c: float = Field(..., description="Air temperature in °C")
    feels_c: float = Field(..., description="Feels-like temperature in °C")
    humidity: confloat(ge=0, le=100) = Field(..., description="Humidity in % (0-100)")
    wind_ms: float = Field(..., description="Wind speed in m/s")
    rain_prob: confloat(ge=0, le=100) = Field(..., description="Rain probability in % (0-100)")
    uv_index: float = Field(..., description="UV index")
    activity: Optional[str] = Field(None, description="Optional activity (e.g., hiking, picnic, cycling)")

class WeatherResponse(BaseModel):
    description: str

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "Weather Explainer API"}

@app.post("/describe", response_model=WeatherResponse, summary="Describe weather and suggest activities")
def describe_weather(payload: WeatherRequest):
    try:
        prompt = build_prompt(
            temp_c=payload.temp_c,
            feels_c=payload.feels_c,
            humidity=payload.humidity,
            wind_ms=payload.wind_ms,
            rain_prob=payload.rain_prob,
            uv_index=payload.uv_index,
            activity=payload.activity,
        )
        text = call_gemini(prompt)
        return WeatherResponse(description=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 