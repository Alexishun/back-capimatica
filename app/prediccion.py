# prediccion_historica_unahora_api.py
import argparse
import pandas as pd
import numpy as np
from conectar_mysql import obtener_datos

# ---------- UMBRALES / CONFIG ----------
UMBRAL_LLUVIA_MM_H = 0.1
UMBRAL_HUMEDAD = 0.5
UMBRAL_CALOR_C = 20.0
UMBRAL_FRIO_C = 15.0
UMBRAL_VIENTO_MS = 6.0
UMBRAL_MUY_HUMEDAD = 0.70

# Neblina score params
NEBLINA_WIND_MAX = 2.5
NEBLINA_RADIATION_MAX = 150
NEBLINA_SCORE_THRESHOLD = 0.25

MIN_SAMPLES_WARN = 30
RANGO_HORAS_DEFAULT = 1
FALLBACK_EXPAND_HOURS = 3

# ---------- PREPARAR Y ETIQUETAR ----------
def preparar_df(df):
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'], errors='coerce')
    df = df.dropna(subset=['fecha_hora']).copy()
    df['mes'] = df['fecha_hora'].dt.month
    df['hora'] = df['fecha_hora'].dt.hour

    # Temperatura: detectar Kelvin -> Celsius si aplica
    if 'y_temperatura' not in df.columns:
        raise KeyError("No existe columna 'y_temperatura' en el DataFrame")
    if df['y_temperatura'].mean() > 200:
        df['temp_C'] = df['y_temperatura'] - 273.15
    else:
        df['temp_C'] = df['y_temperatura'].astype(float)

    # Precipitación: kg/m2/s -> mm/h
    if 'y_precipitacion' not in df.columns:
        raise KeyError("No existe columna 'y_precipitacion' en el DataFrame")
    df['precip_mm_h'] = df['y_precipitacion'].astype(float) * 3600.0

    # Humedad: normalizar 0..1 (si está en % lo convierte)
    if 'y_humedad' not in df.columns:
        raise KeyError("No existe columna 'y_humedad' en el DataFrame")
    hum_mean = df['y_humedad'].mean()
    df['humedad'] = (df['y_humedad'] / 100.0) if hum_mean > 2 else df['y_humedad'].astype(float)

    # Viento
    if 'y_speedwind' not in df.columns:
        raise KeyError("No existe columna 'y_speedwind' en el DataFrame")
    df['viento_ms'] = df['y_speedwind'].astype(float)

    # Radiación (y_uv o y_radiacion)
    if 'y_uv' in df.columns:
        df['radiacion'] = df['y_uv'].astype(float)
    elif 'y_radiacion' in df.columns:
        df['radiacion'] = df['y_radiacion'].astype(float)
    else:
        df['radiacion'] = np.nan

    return df

def etiquetar_eventos_y_neblina_score(df):
    # Lluvia (precip + humedad)
    df['et_lluvia'] = (((df['precip_mm_h'] > UMBRAL_LLUVIA_MM_H) |
                        ((df['precip_mm_h'] > 0.0) & (df['humedad'] >= UMBRAL_HUMEDAD)))
                       ).astype(int)

    df['et_calor'] = (df['temp_C'] > UMBRAL_CALOR_C).astype(int)
    df['et_frio'] = (df['temp_C'] < UMBRAL_FRIO_C).astype(int)
    df['et_viento'] = (df['viento_ms'] > UMBRAL_VIENTO_MS).astype(int)
    df['et_muy_humedo'] = (df['humedad'] >= UMBRAL_MUY_HUMEDAD).astype(int)

    # Neblina: dewpoint + humedad + viento + radiación -> score
    a, b = 17.27, 237.7
    T = df['temp_C']
    RH = df['humedad'].clip(0.0001, 1.0)
    alpha = (a * T) / (b + T) + np.log(RH)
    df['dewpoint_C'] = (b * alpha) / (a - alpha)

    humidity_factor = df['humedad'].clip(0,1)
    wind_factor = (np.maximum(0.0, NEBLINA_WIND_MAX - df['viento_ms']) / NEBLINA_WIND_MAX).clip(0,1)
    if df['radiacion'].notna().any():
        rad = df['radiacion'].fillna(NEBLINA_RADIATION_MAX*2)
        radiacion_factor = (np.maximum(0.0, NEBLINA_RADIATION_MAX - rad) / NEBLINA_RADIATION_MAX).clip(0,1)
    else:
        radiacion_factor = np.zeros(len(df))
    diff = (df['temp_C'] - df['dewpoint_C']).clip(lower=-10, upper=20)
    dewpoint_factor = (np.maximum(0.0, 2.0 - diff) / 2.0).clip(0,1)

    # pesos
    w_h, w_w, w_r, w_d = 0.4, 0.25, 0.15, 0.20
    df['neblina_score'] = (w_h*humidity_factor + w_w*wind_factor + w_r*radiacion_factor + w_d*dewpoint_factor).clip(0,1)
    df['et_neblina'] = (df['neblina_score'] >= NEBLINA_SCORE_THRESHOLD).astype(int)

    return df

# ---------- FUNCION CORE ----------
def predict_for_datetime_and_table(fecha_hora_str, tabla="lima_feature", rango_horas=RANGO_HORAS_DEFAULT):
    """
    Retorna probabilidades y stats para la tabla (lugar) y la fecha/hora dada.
    """
    # Obtener datos de la tabla
    df = obtener_datos(tabla=tabla)

    # Preparar y etiquetar
    df = preparar_df(df)
    df = etiquetar_eventos_y_neblina_score(df)

    # Filtrar por mes y hora (± rango)
    fecha = pd.to_datetime(fecha_hora_str, errors='coerce')
    if pd.isna(fecha):
        raise ValueError("Fecha inválida. Usa 'YYYY-MM-DD HH:MM'")

    mes = int(fecha.month)
    hora = int(fecha.hour)
    df_sim = df[(df['mes']==mes) & (df['hora'].between(hora-rango_horas, hora+rango_horas))]

    note = None
    if len(df_sim) == 0:
        df_sim = df[df['mes']==mes]
        if len(df_sim) == 0:
            return {'error': 'No hay datos para ese mes en la tabla solicitada.'}
        note = 'Usando todo el mes (no hubo registros exactos por hora).'

    if len(df_sim) < MIN_SAMPLES_WARN:
        df_sim2 = df[(df['mes']==mes) & (df['hora'].between(hora-FALLBACK_EXPAND_HOURS, hora+FALLBACK_EXPAND_HOURS))]
        if len(df_sim2) >= MIN_SAMPLES_WARN:
            df_sim = df_sim2
            note = f'Ampliado rango horario a ±{FALLBACK_EXPAND_HOURS}h por pocas muestras.'
        else:
            note = (note or '') + f' Pocas muestras ({len(df_sim)}). Baja confianza.'

    out = {
        'fecha_consulta': fecha_hora_str,
        'tabla': tabla,
        'n_muestras': int(len(df_sim)),
        'prob_lluvia': float(df_sim['et_lluvia'].mean()),
        'prob_calor': float(df_sim['et_calor'].mean()),
        'prob_frio': float(df_sim['et_frio'].mean()),
        'prob_viento': float(df_sim['et_viento'].mean()),
        'prob_muy_humedo': float(df_sim['et_muy_humedo'].mean()),
        'prob_neblina': float(df_sim['et_neblina'].mean()),
        'note': note,
        'stats': {
            'temp_C_mean': float(df_sim['temp_C'].mean()),
            'precip_mm_h_mean': float(df_sim['precip_mm_h'].mean()),
            'viento_ms_mean': float(df_sim['viento_ms'].mean()),
            'humedad_mean': float(df_sim['humedad'].mean())
        }
    }
    return out

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Probabilidades históricas por tabla (lugar) + fecha/hora")
    parser.add_argument("--tabla", "-t", type=str, default="lima_feature", help="Tabla/lugar a usar (ej: lima_feature)")
    parser.add_argument("--fecha", "-f", type=str, required=True, help="Fecha/hora 'YYYY-MM-DD HH:MM'")
    parser.add_argument("--rango", "-r", type=int, default=RANGO_HORAS_DEFAULT, help="Rango horario ±horas")
    args = parser.parse_args()

    res = predict_for_datetime_and_table(args.fecha, tabla=args.tabla, rango_horas=args.rango)
    import json
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()