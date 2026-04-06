###Firma esperada:
import pandas as pd

def pipeline_mixto_clasificacion(
    df: pd.DataFrame,
    target_col: str,
    test_size: float
) -> dict: ...
###Generador de caso de uso:
def generar_caso_de_uso_pipeline_mixto_clasificacion():
    np.random.seed(np.random.randint(0, 9999))
    n = np.random.randint(300, 700)
    industrias = np.random.choice(["Salud","Finanzas","Retail","Mfg"], n)
    regiones   = np.random.choice(["Norte","Sur","Centro","Este"], n)
    df = pd.DataFrame({
        "edad":   np.where(np.random.rand(n)<0.1, np.nan,
                  np.random.randint(18,70,n).astype(float)),
        "ingresos": np.random.normal(5000,1500,n).round(2),
        "deuda":  np.where(np.random.rand(n)<0.15, np.nan,
                  np.random.exponential(2000,n).round(2)),
        "industria": np.where(np.random.rand(n)<0.1,None,industrias),
        "region":  regiones,
        "num_prod": np.random.randint(1,10,n).astype(float),
        "target":  np.random.choice([0,1],n,p=[0.7,0.3])
    })
    ts = round(np.random.choice([0.2,0.25,0.3]),2)
    return {"target_col": "target", "test_size": ts}, df
