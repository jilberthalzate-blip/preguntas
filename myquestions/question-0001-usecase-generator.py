###Firma esperada:
import pandas as pd

def resumir_por_segmento(
    df: pd.DataFrame,
    col_segmento: str,
    col_cliente: str,
    col_monto: str
) -> pd.DataFrame: ...
###Generador de caso de uso:
import pandas as pd
import numpy as np

def generar_caso_de_uso_resumir_por_segmento():
    np.random.seed(np.random.randint(0, 9999))
    n = np.random.randint(200, 600)
    segmentos = np.random.choice(
        ["Premium","Estandar","Basico","VIP","Corporativo"], n)
    clientes = np.random.randint(1000, 1100, n)
    montos = np.random.exponential(
        scale=np.random.uniform(100, 500), size=n).round(2)
    df = pd.DataFrame({
        "segmento": segmentos,
        "cliente_id": clientes,
        "monto": montos
    })
    return {"col_segmento": "segmento", "col_cliente": "cliente_id", "col_monto": "monto"}, df


