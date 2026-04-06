###Firma esperada:
import pandas as pd

def deduplicar_pacientes(
    df: pd.DataFrame,
    col_id: str,
    col_fecha: str,
    cols_clinicas: list
) -> pd.DataFrame: ...
###Generador de caso de uso:
def generar_caso_de_uso_deduplicar_pacientes():
    np.random.seed(np.random.randint(0, 9999))
    n_pac = np.random.randint(40, 80)
    ids = np.repeat(
        np.arange(1, n_pac+1),
        np.random.randint(1, 4, n_pac))
    n = len(ids)
    fechas = pd.to_datetime(np.random.choice(
        pd.date_range("2023-01-01","2024-12-31"), n))
    cols = ["glucosa","presion","temperatura","frec_cardiaca"]
    data = {c: np.where(np.random.rand(n)<0.3, np.nan,
            np.random.uniform(60,180,n).round(1)) for c in cols}
    df = pd.DataFrame({
        "paciente_id": ids, "fecha_ingreso": fechas, **data})
    return df, "paciente_id", "fecha_ingreso", cols



