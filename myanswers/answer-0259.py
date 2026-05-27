import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


def extraer_dinamica_temporal(df, col_sensor, ventana=12):
    """
    Extrae features dinámicas sobre una ventana deslizante y las escala
    con RobustScaler para resistir outliers térmicos.

    Parámetros
    ----------
    df         : pd.DataFrame — datos originales (no se modifica)
    col_sensor : str          — nombre de la columna a analizar
    ventana    : int          — tamaño de la ventana deslizante (default 12)

    Retorna
    -------
    pd.DataFrame con columnas [asimetria_movil, rango_movil, tasa_cambio],
    escaladas con RobustScaler, manteniendo el índice original.
    """
    serie = df[col_sensor].copy()

    # 1. Features sobre ventana deslizante
    rolling = serie.rolling(window=ventana)

    resultado = pd.DataFrame(index=df.index)
    resultado["asimetria_movil"] = rolling.skew()
    resultado["rango_movil"]     = rolling.max() - rolling.min()
    resultado["tasa_cambio"]     = serie.diff()

    # 2. Eliminar NaNs
    resultado = resultado.dropna()

    # 3. Escalado robusto
    scaler = RobustScaler()
    columnas = ["asimetria_movil", "rango_movil", "tasa_cambio"]
    resultado[columnas] = scaler.fit_transform(resultado[columnas])

    return resultado