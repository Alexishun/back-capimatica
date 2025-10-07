import os
import pymysql
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

def obtener_datos(tabla="lima_feature", limit=None):
    """
    Conecta a la DB y devuelve un DataFrame con los datos de la tabla solicitada.
    tabla: string, por ejemplo 'lima_feature', 'cuzco_feature', etc.
    limit: opcional, limita filas (útil para debug)
    """
    # Establecer la conexión
    timeout = 10
    connection = pymysql.connect(
    charset="utf8mb4",
    connect_timeout=timeout,
    cursorclass=pymysql.cursors.DictCursor,
    db=os.getenv("DB"),
    host=os.getenv("HOST"),
    password=os.getenv("PASSWORD"),
    read_timeout=timeout,
    port=12284,
    user=os.getenv("USER"),
    write_timeout=timeout,
    )
    try:
        with connection.cursor() as cursor:
            # DEBUG: mostrar tablas (puedes comentar esto luego)
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()
            # print("Tablas disponibles:", tables)

            consulta = f"SELECT * FROM {tabla}"
            if limit:
                consulta += f" LIMIT {int(limit)};"
            else:
                consulta += ";"

            cursor.execute(consulta)
            rows = cursor.fetchall()
            df = pd.DataFrame(rows)

            # print(f"\nPrimeras filas de la tabla {tabla}:")
            # print(df.head())

            return df
    finally:
        connection.close()