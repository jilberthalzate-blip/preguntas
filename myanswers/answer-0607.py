import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


def comparar_umbrales_precision_recall(X, y, umbrales):
    """
    Entrena una regresión logística y evalúa precision, recall y F1
    para cada umbral de decisión dado.

    Parámetros
    ----------
    X : array-like de forma (n_samples, n_features)
    y : array-like de forma (n_samples,)  — etiquetas binarias
    umbrales : lista de floats en [0, 1]

    Retorna
    -------
    pd.DataFrame con columnas [umbral, precision, recall, f1],
    valores redondeados a 4 decimales e índice reiniciado.
    """
    # 1. Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Escalado (fit solo sobre train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Entrenar modelo
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # 4. Probabilidades de la clase positiva
    probs = model.predict_proba(X_test)[:, 1]

    # 5. Calcular métricas para cada umbral
    rows = []
    for umbral in umbrales:
        y_pred = (probs >= umbral).astype(int)
        rows.append({
            "umbral":    round(float(umbral), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0),        4),
        })

    # 6. Construir y retornar DataFrame
    return pd.DataFrame(rows, columns=["umbral", "precision", "recall", "f1"]).reset_index(drop=True)