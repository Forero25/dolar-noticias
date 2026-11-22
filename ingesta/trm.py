"""TRM diaria (Superintendencia Financiera) desde datos.gov.co, conjunto 32sa-8pi3.

La TRM vigente un día se calcula con las operaciones del día hábil anterior. Aquí cada fila se fecha
por el día de las operaciones (el día hábil anterior a la vigencia), que es cuando se conoce el precio:
así el modelo nunca usa información del futuro.
"""
import pathlib, time

import pandas as pd
import requests

RAIZ = pathlib.Path(__file__).resolve().parents[1]
URL = "https://www.datos.gov.co/resource/32sa-8pi3.json"


def descargar() -> pd.DataFrame:
    filas, salto = [], 0
    while True:
        r = requests.get(URL, params={"$limit": 50000, "$offset": salto, "$order": "vigenciadesde"}, timeout=120)
        r.raise_for_status()
        lote = r.json()
        if not lote:
            break
        filas += lote
        salto += len(lote)
        time.sleep(1)
    return pd.DataFrame(filas)


def preparar(crudo: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"vigencia": pd.to_datetime(crudo["vigenciadesde"]).dt.normalize(), "trm": pd.to_numeric(crudo["valor"])})
    d = d.drop_duplicates("vigencia").sort_values("vigencia")
    # Cada fila es una publicación: la que rige sábado, domingo y lunes empieza el sábado y se calculó con
    # las operaciones del viernes. Se fecha en el día hábil anterior al inicio de su vigencia.
    d["fecha"] = d["vigencia"] - pd.offsets.BDay(1)
    d = d.drop_duplicates("fecha", keep="last")
    return d[["fecha", "vigencia", "trm"]].reset_index(drop=True)


def main():
    d = preparar(descargar())
    (RAIZ / "datos_publicos").mkdir(exist_ok=True)
    d.to_csv(RAIZ / "datos_publicos" / "trm.csv", index=False)
    print("TRM:", len(d), "días,", d["fecha"].min().date(), "a", d["fecha"].max().date())


if __name__ == "__main__":
    main()
