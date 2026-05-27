import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def reducir_y_evaluar_pca(df, n_componentes):
    """
    Estandariza un DataFrame, aplica PCA y calcula la varianza explicada acumulada.

    Parámetros
    ----------
    df           : pd.DataFrame  — datos originales (no se modifica)
    n_componentes: int           — número de componentes principales a retener

    Retorna
    -------
    tuple:
        - np.ndarray : datos transformados, shape (n_samples, n_componentes)
        - float      : varianza explicada acumulada por los n_componentes
    """
    X = df.copy().values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_componentes)
    X_reducido = pca.fit_transform(X_scaled)

    varianza_acumulada = float(np.sum(pca.explained_variance_ratio_))

    return X_reducido, varianza_acumulada