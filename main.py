# main.py
# ===========================================
# MERRA-2: descarga de archivos .nc4 (endpoint DATA, no OPeNDAP),
# extracción por punto más cercano y exportación a CSV (horario/diario/semanal).
# ===========================================

import os
import re
import sys
import math
import glob
import json
import time
import shutil
import getpass
import logging
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from calendar import monthrange
import requests
from requests.adapters import HTTPAdapter, Retry

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
YEARS = [2007, 2008, 2009, 2010, 2011]            # Años
FIELD_ID = "T2M"                                   # Variable (ej. T2M = 2-m air temperature)
FIELD_NAME = "temperature"                         # Nombre de salida
DATABASE_NAME = "M2I1NXASM"                        # Colección (inst1_2d_asm_Nx)
DATABASE_ID = "inst1_2d_asm_Nx"
VERSION = "5.12.4"                                 # Versión MERRA-2

# Ubicaciones: (nombre, lat, lon)
LOCS = [
    ("maputo", -25.9629, 32.5732),
    ("cdelgado", -12.3335, 39.3206),
    ("manica", -18.9438, 32.8649),
    ("gaza", -23.0222, 32.7181),
    ("sofala", -19.2039, 34.8624),
    ("tete", -16.1328, 33.6364),
    ("zambezia", -16.5639, 36.6094),
    ("nampula", -15.1266, 39.2687),
    ("niassa", -12.7826, 36.6094),
    ("inhambane", -23.8662, 35.3827),
]

# Conversión de Kelvin a °C
def to_celsius(k):
    return k - 273.15

AGGREGATOR = "mean"  # "sum" | "mean" | "min" | "max"

# Carpetas
RAW_DIR = "raw_nc4"                    # Aquí se guardan .nc4 descargados (compartidos para todas las ubicaciones)
OUT_DIR = FIELD_NAME                   # Carpeta raíz para CSVs por ubicación

# Credenciales por variables de entorno (recomendado)
EARTHDATA_USER = os.getenv("EARTHDATA_USER")
EARTHDATA_PASS = os.getenv("EARTHDATA_PASS")

# -----------------------------
# UTILIDADES
# -----------------------------
def file_number_for_year(year: int) -> str:
    """MERRA-2 file_number por año."""
    if 1980 <= year < 1992:
        return "100"
    if 1992 <= year < 2001:
        return "200"
    if 2001 <= year < 2011:
        return "300"
    if year >= 2011:
        return "400"
    raise ValueError("Año fuera de rango para MERRA-2")

def build_data_urls(years, database_name, version, database_id):
    """
    Construye URLs del endpoint DATA (no OPeNDAP):
    https://goldsmr4.gesdisc.eosdis.nasa.gov/data/{database_name}.{version}/{YYYY}/{MM}/MERRA2_{num}.{database_id}.{YYYYMMDD}.nc4
    """
    urls = []
    for y in years:
        num = file_number_for_year(y)
        for m in range(1, 12 + 1):
            _, ndays = monthrange(y, m)
            for d in range(1, ndays + 1):
                yyyy = f"{y:04d}"
                mm = f"{m:02d}"
                dd = f"{d:02d}"
                fname = f"MERRA2_{num}.{database_id}.{yyyy}{mm}{dd}.nc4"
                url = (
                    f"https://goldsmr4.gesdisc.eosdis.nasa.gov/data/"
                    f"{database_name}.{version}/{yyyy}/{mm}/{fname}"
                )
                urls.append((yyyy, mm, dd, fname, url))
    return urls

def make_session(user: str, pwd: str) -> requests.Session:
    """Crea sesión con auth básica + reintentos. Earthdata gestiona redirecciones."""
    s = requests.Session()
    s.auth = (user, pwd)
    retries = Retry(
        total=5,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "merra2-downloader/1.0"})
    return s

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def download_all_nc4(url_tuples, raw_dir, session) -> int:
    """
    Descarga todos los .nc4 al directorio raw_dir.
    url_tuples: lista de (yyyy, mm, dd, fname, url)
    """
    ensure_dir(raw_dir)
    downloaded = 0
    for (yyyy, mm, dd, fname, url) in url_tuples:
        outpath = os.path.join(raw_dir, fname)
        if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
            continue
        r = session.get(url, allow_redirects=True, stream=True, timeout=180)
        # Si Earthdata requiere aprobación inicial, 302/401 pueden ocurrir
        if r.status_code == 401:
            raise RuntimeError(
                "401 Unauthorized: inicia sesión en el navegador con tus credenciales y "
                "aprueba la app para gesdisc.eosdis.nasa.gov, o revisa EARTHDATA_USER/EARTHDATA_PASS."
            )
        r.raise_for_status()
        with open(outpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
        downloaded += 1
    return downloaded

def find_lat_lon_names(ds: xr.Dataset):
    """Detecta nombres de coordenadas lat/lon en el Dataset."""
    lat_name = None
    lon_name = None
    for cand in ["lat", "latitude", "Lat", "LAT"]:
        if cand in ds.coords:
            lat_name = cand
            break
    for cand in ["lon", "longitude", "Lon", "LON"]:
        if cand in ds.coords:
            lon_name = cand
            break
    if not lat_name or not lon_name:
        raise KeyError("No se encontraron coordenadas lat/lon en el dataset.")
    return lat_name, lon_name

def open_and_extract_point(nc_path: str, var: str, lat: float, lon: float) -> pd.DataFrame:
    """
    Abre un NetCDF, selecciona el punto más cercano a (lat, lon) para la variable `var`
    y devuelve DataFrame con columnas [var, date, time].
    """
    with xr.open_dataset(nc_path) as ds:
        if var not in ds:
            raise KeyError(f"La variable '{var}' no está en {nc_path}. Vars disponibles: {list(ds.data_vars)}")
        lat_name, lon_name = find_lat_lon_names(ds)

        # Selección nearest del punto
        da_point = ds[var].sel({lat_name: lat, lon_name: lon}, method="nearest")

        # Asegurar que exista 'time' en coords
        if "time" not in da_point.coords:
            # Algunos ficheros podrían tener 'Time' u otro nombre
            tname = None
            for cand in ["Time", "time", "TIME"]:
                if cand in da_point.coords:
                    tname = cand
                    break
            if not tname:
                raise KeyError(f"No se encontró coordenada temporal en {nc_path}")
            da_point = da_point.rename({tname: "time"})

        df = da_point.to_dataframe().reset_index()
        # Normalizar columnas: quedarse con time y el valor
        if "time" not in df.columns:
            raise KeyError("No se pudo extraer la columna 'time' desde el NetCDF.")
        df = df[["time", var]]
        # date = solo la fecha
        df["date"] = pd.to_datetime(df["time"]).dt.date
        # formateamos 'time' como HH:MM:SS
        df["time"] = pd.to_datetime(df["time"]).dt.time
        return df

# -----------------------------
# MAIN
# -----------------------------
def main():
    # Credenciales
    user = EARTHDATA_USER
    pwd = EARTHDATA_PASS
    if not user:
        user = input("EARTHDATA_USER no configurado. Ingresa tu usuario Earthdata: ").strip()
    if not pwd:
        pwd = getpass.getpass("EARTHDATA_PASS no configurado. Ingresa tu password Earthdata: ").strip()

    print("DOWNLOADING DATA FROM MERRA")
    print(f"Predicted time: {len(YEARS)*len(LOCS)*6} minutes")
    print("=====================")

    # 1) Construir lista de URLs de DATA
    url_list = build_data_urls(YEARS, DATABASE_NAME, VERSION, DATABASE_ID)

    # 2) Sesión y descarga a RAW_DIR
    sess = make_session(user, pwd)
    ensure_dir(RAW_DIR)
    new_files = download_all_nc4(url_list, RAW_DIR, sess)
    print(f"Descargas nuevas: {new_files} archivos (el resto ya existían).")

    # 3) Procesamiento por ubicación
    print("\nCLEANING AND MERGING DATA")
    print(f"Predicted time: {len(YEARS)*len(LOCS)*0.1} minutes")
    print("=====================")

    ensure_dir(OUT_DIR)

    # Listado de todos los .nc4 disponibles en RAW_DIR
    all_nc4 = sorted(glob.glob(os.path.join(RAW_DIR, "*.nc4")))
    if not all_nc4:
        raise RuntimeError("No hay .nc4 descargados en 'raw_nc4/'. Revisa credenciales o conectividad.")

    for loc, lat, lon in LOCS:
        print(f"Cleaning and merging {FIELD_NAME} data for {loc}")
        out_loc_dir = os.path.join(OUT_DIR, loc)
        ensure_dir(out_loc_dir)

        hourly_rows = []
        for ncfile in all_nc4:
            try:
                df = open_and_extract_point(ncfile, FIELD_ID, lat, lon)
                hourly_rows.append(df)
            except Exception as e:
                # Si un archivo falla, lo reportamos y continuamos
                print(f"Issue with file {os.path.basename(ncfile)}: {e}")
                continue

        if not hourly_rows:
            print(f"Saltando {loc}: no se pudo extraer ningún dato.")
            continue

        df_hourly = pd.concat(hourly_rows, ignore_index=True)

        # Renombrar columna de variable a FIELD_NAME
        df_hourly.rename(columns={FIELD_ID: FIELD_NAME}, inplace=True)

        # Conversión de unidades
        df_hourly[FIELD_NAME] = df_hourly[FIELD_NAME].apply(to_celsius)

        # Ordenar por fecha/hora
        df_hourly["date"] = pd.to_datetime(df_hourly["date"])
        # reconstruir 'time' como string para el CSV
        df_hourly["time"] = df_hourly["time"].astype(str)
        df_hourly.sort_values(["date", "time"], inplace=True)

        # Guardar HORARIO
        hourly_csv = os.path.join(OUT_DIR, f"{loc}_hourly.csv")
        df_hourly.to_csv(hourly_csv, index=False, header=[FIELD_NAME, "date", "time"])

        # DIARIO
        # Convertimos a datetime completo para agrupar por día
        dfh = df_hourly.copy()
        dfh["datetime"] = pd.to_datetime(dfh["date"].astype(str) + " " + dfh["time"])
        dfh.set_index("datetime", inplace=True)
        # Agregación por día
        if AGGREGATOR == "sum":
            df_daily = dfh.resample("D")[FIELD_NAME].sum().to_frame()
        elif AGGREGATOR == "min":
            df_daily = dfh.resample("D")[FIELD_NAME].min().to_frame()
        elif AGGREGATOR == "max":
            df_daily = dfh.resample("D")[FIELD_NAME].max().to_frame()
        else:  # mean
            df_daily = dfh.resample("D")[FIELD_NAME].mean().to_frame()
        df_daily["date"] = df_daily.index.date
        daily_csv = os.path.join(OUT_DIR, f"{loc}_daily.csv")
        df_daily.reset_index(drop=True).to_csv(daily_csv, index=False, header=[FIELD_NAME, "date"])

        # SEMANAL (ISO year-week)
        df_weekly = df_daily.copy()
        idx = pd.to_datetime(df_weekly["date"])
        df_weekly["Year"] = idx.dt.isocalendar().year
        df_weekly["Week"] = idx.dt.isocalendar().week

        if AGGREGATOR == "sum":
            dfw = df_weekly.groupby(["Year", "Week"])[FIELD_NAME].sum().to_frame()
        elif AGGREGATOR == "min":
            dfw = df_weekly.groupby(["Year", "Week"])[FIELD_NAME].min().to_frame()
        elif AGGREGATOR == "max":
            dfw = df_weekly.groupby(["Year", "Week"])[FIELD_NAME].max().to_frame()
        else:
            dfw = df_weekly.groupby(["Year", "Week"])[FIELD_NAME].mean().to_frame()

        dfw["Year"] = dfw.index.get_level_values(0)
        dfw["Week"] = dfw.index.get_level_values(1)
        weekly_csv = os.path.join(OUT_DIR, f"{loc}_weekly.csv")
        dfw.reset_index(drop=True).to_csv(weekly_csv, index=False)

    print("\nFINISHED")

# -----------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
