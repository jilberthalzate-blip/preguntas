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
    pd.DataFrame con columnas [skew_movil, rango_movil, tasa_cambio],
    escaladas con RobustScaler, manteniendo el índice original.
    """
    serie = df[col_sensor].copy()

    # 1. Features sobre ventana deslizante
    rolling = serie.rolling(window=ventana)

    skew_movil   = rolling.skew()
    rango_movil  = rolling.max() - rolling.min()
    tasa_cambio  = serie.diff()

    # 2. Ensamblar y limpiar NaNs
    features = pd.DataFrame({
        "skew_movil"  : skew_movil,
        "rango_movil" : rango_movil,
        "tasa_cambio" : tasa_cambio,
    }, index=df.index).dropna()

    # 3. Escalado robusto (preserva índice)
    scaler = RobustScaler()
    valores_escalados = scaler.fit_transform(features)

    resultado = pd.DataFrame(
        valores_escalados,
        columns=["skew_movil", "rango_movil", "tasa_cambio"],
        index=features.index
    )

    return resultado