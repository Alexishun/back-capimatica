# app.py
# pip install fastapi uvicorn google-genai python-dotenv pydantic

import os
from typing import Optional

from fastapi import FastAPI, HTTPException,Query
from pydantic import BaseModel, Field, confloat
from google import genai
from google.genai import types
from dotenv import load_dotenv
import math
import datetime as dt

from pathlib import Path
import sys, json, subprocess

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


CITIES = [
    {"name": "lima",     "lat": -12.0464, "lon": -77.0428, "table": "lima_feature"},
    {"name": "cusco",    "lat": -13.5320, "lon": -71.9675, "table": "cuzco_feature"}, 
    {"name": "piura",    "lat":  -5.1945, "lon": -80.6328, "table": "piura_feature"},
    {"name": "puno",     "lat": -15.8402, "lon": -70.0219, "table": "puno_feature"},
    {"name": "tarapoto", "lat":  -6.4921, "lon": -76.3655, "table": "tarapoto_feature"},
]
def run_prediccion(table: str, fecha_str: str, rango_horas: float, timeout_s: int = 30) -> dict:
    """
    Ejecuta: py prediccion.py --tabla=<table> --fecha "<YYYY-MM-DD HH:MM>" --rango <horas>
    y devuelve el dict con el JSON de stdout.
    """
    script = Path(__file__).parent / "prediccion.py"  # ajusta si está en otra ruta
    if not script.exists():
        raise RuntimeError(f"No se encuentra prediccion.py en {script}")

    # Usa el mismo intérprete que ejecuta FastAPI
    cmd = [
        sys.executable,
        str(script),
        f"--tabla={table}",
        "--fecha", fecha_str,
        "--rango", str(rango_horas),
    ]

    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,   # lanza CalledProcessError si exit != 0
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"prediccion.py falló ({e.returncode}): {e.stderr.strip()}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"prediccion.py timeout tras {timeout_s}s") from e

    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Salida no es JSON válido: {res.stdout[:400]}") from e



def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def closest_city(lat: float, lon: float):
    best = None
    for c in CITIES:
        d = haversine_km(lat, lon, c["lat"], c["lon"])
        if best is None or d < best["dist_km"]:
            best = {**c, "dist_km": round(d, 2)}
    return best  # {'name','lat','lon','table','dist_km'}


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


@app.get("/weather", summary="Get nearest city's weather by lat/lon/date/hour")
def get_weather(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    date: str = Query(..., description="Date YYYY-MM-DD (assumed UTC)"),
    hour: int = Query(..., ge=0, le=23, description="Hour 0-23 (assumed UTC)"),
    window_minutes: int = Query(60, ge=0, le=360, description="Time window +/- minutes to search"),
):
    try:
        city = closest_city(lat, lon)
        table = city["table"]

        dt_utc = dt.datetime.strptime(f"{date} {hour:02d}:00", "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)
        fecha_str = dt_utc.strftime("%Y-%m-%d %H:%M")

        rango_horas_int = max(1, int(math.ceil(window_minutes / 60)))

        result = run_prediccion(table=table, fecha_str=fecha_str, rango_horas=rango_horas_int)
        print(result)
        return {
            "city": city["name"],
            "city_distance_km": city["dist_km"],
            "table": table,
            "target_dt_utc": dt_utc.isoformat(),
            "query_params": {"lat": lat, "lon": lon, "date": date, "hour": hour, "window_minutes": window_minutes},
            "rango_horas": rango_horas_int,
            "prediction": result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))