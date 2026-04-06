###Firma esperada:
import pandas as pd

def diagnosticar_sesgo_varianza(
    X: np.ndarray,
    y: np.ndarray,
    modelo,
    cv: int,
    n_puntos: int
) -> dict: ...
###Generador de caso de uso:
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

def generar_caso_de_uso_diagnosticar_sesgo_varianza():
    np.random.seed(np.random.randint(0, 9999))
    modo = np.random.choice(["regresion","clasificacion"])
    n    = np.random.randint(400, 900)
    nf   = np.random.randint(5, 20)
    X    = np.random.randn(n, nf)
    if modo == "regresion":
        coef = np.random.randn(nf)
        y = X @ coef + np.random.randn(n) * np.random.uniform(1,5)
        d = np.random.choice([2, None])
        modelo = DecisionTreeRegressor(max_depth=d, random_state=42)
    else:
        y = (X[:,0] + X[:,1] > 0).astype(int)
        ne = np.random.choice([5, 200])
        modelo = RandomForestClassifier(n_estimators=ne, random_state=42)
    cv = np.random.choice([3, 5])
    n_puntos = np.random.randint(5, 10)
    return X, y, {"modelo": modelo, "cv": cv, "n_puntos": n_puntos}
