import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


def entrenar_evaluar_arbol(df, feature_cols, target_col, max_depth):
    """
    Entrena un árbol de decisión regresor y evalúa su desempeño
    sobre los mismos datos de entrenamiento (baseline rápido).

    Parámetros
    ----------
    df          : pd.DataFrame — dataset completo
    feature_cols: list[str]   — columnas usadas como features
    target_col  : str         — columna objetivo
    max_depth   : int         — profundidad máxima del árbol

    Retorna
    -------
    tuple: (mse, r2)
        mse : float — Error Cuadrático Medio
        r2  : float — Coeficiente de determinación
    """
    # 1. Extraer X e y
    X = df[feature_cols].values
    y = df[target_col].values

    # 2. Entrenar modelo
    modelo = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    modelo.fit(X, y)

    # 3. Predicciones sobre los mismos datos
    y_pred = modelo.predict(X)

    # 4. Métricas
    mse = mean_squared_error(y, y_pred)
    r2  = r2_score(y, y_pred)

    # 5. Retornar tupla
    return (mse, r2)