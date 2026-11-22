"""Eventos codificados de las noticias del mundo: archivos diarios de GDELT 1.0 (data.gdeltproject.org/events).

GDELT lee noticias en más de 100 idiomas y codifica cada hecho que reportan (quién, contra quién, dónde, qué tipo
de hecho en la clasificación CAMEO) con el tono de la cobertura y su número de menciones. La API de búsqueda de
GDELT dejó de responder (devuelve 429 incluso a consultas mínimas); los archivos diarios siguen publicándose.

Cada archivo AAAAMMDD trae los eventos que GDELT agregó ese día. Se fecha por ese día (el de publicación de las
noticias), así el modelo solo usa lo que ya se había reportado. De cada archivo se guardan agregados diarios por
tema; los archivos crudos no se guardan.

La descarga es reanudable: guarda el avance cada 30 días procesados y se detiene antes del límite de tiempo del flujo.
"""
import io, json, pathlib, sys, time, zipfile

import numpy as np
import pandas as pd
import requests

RAIZ = pathlib.Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "datos_publicos" / "noticias.csv"
ESTADO = RAIZ / "datos_publicos" / "noticias_estado.json"
URL = "http://data.gdeltproject.org/events/{}.export.CSV.zip"
INICIO = pd.Timestamp("2017-01-01")
LIMITE_SEGUNDOS = 4 * 3600 + 20 * 60

# Columnas del formato de eventos de GDELT 1.0 (58 desde abril de 2013)
N_COLUMNAS = 58
COL = {"actor1_pais": 7, "actor2_pais": 17, "actor1_tipo": 12, "raiz": 28, "quad": 29, "goldstein": 30,
       "menciones": 31, "tono": 34, "lugar_pais": 51}


def temas(d: pd.DataFrame) -> dict[str, pd.Series]:
    col = d["actor1_pais"].eq("COL") | d["actor2_pais"].eq("COL") | d["lugar_pais"].eq("CO")
    ven = d["actor1_pais"].eq("VEN") | d["actor2_pais"].eq("VEN") | d["lugar_pais"].eq("VE")
    usa = d["actor1_pais"].eq("USA") | d["lugar_pais"].eq("US")
    return {"colombia": col, "colombia_gobierno": col & d["actor1_tipo"].eq("GOV"), "venezuela": ven,
            "eeuu": usa, "mundo": pd.Series(True, index=d.index)}


def agregar(d: pd.DataFrame, fecha: pd.Timestamp) -> pd.DataFrame:
    """Un renglón por tema: tono y Goldstein ponderados por menciones, menciones del tema sobre el total del día,
    y participación de los hechos de conflicto, protesta y violencia."""
    total = float(d["menciones"].sum())
    filas = []
    for tema, m in temas(d).items():
        g = d[m]
        w = g["menciones"].astype(float)
        if w.sum() == 0:
            continue
        filas.append({"fecha": fecha, "tema": tema, "eventos": int(len(g)), "articulos": float(w.sum()), "total": total,
                      "tono": float(np.average(g["tono"], weights=w)),
                      "goldstein": float(np.average(g["goldstein"], weights=w)),
                      "conflicto": float(w[g["quad"].isin([3, 4])].sum() / w.sum()),
                      "protestas": float(w[g["raiz"].eq(14)].sum() / w.sum()),
                      "violencia": float(w[g["raiz"].isin([18, 19, 20])].sum() / w.sum())})
    return pd.DataFrame(filas)


def leer(contenido: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        crudo = z.read(z.namelist()[0])
    primera = crudo.split(b"\n", 1)[0].split(b"\t")
    if len(primera) != N_COLUMNAS:
        sys.exit(f"Formato inesperado: {len(primera)} columnas en lugar de {N_COLUMNAS}.")
    d = pd.read_csv(io.BytesIO(crudo), sep="\t", header=None, usecols=list(COL.values()), dtype=str,
                    quoting=3, on_bad_lines="skip")
    d = d.rename(columns={v: k for k, v in COL.items()})
    for c in ("raiz", "quad"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("goldstein", "menciones", "tono"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["menciones", "tono", "goldstein"])


def bajar(fecha: pd.Timestamp, s: requests.Session) -> bytes | None:
    for intento in range(4):
        try:
            r = s.get(URL.format(fecha.strftime("%Y%m%d")), timeout=120)
            if r.status_code == 404:
                return None  # GDELT no publicó ese día
            if r.ok:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(5 * 2 ** intento)
    return None


def guardar(d: pd.DataFrame, completo: bool, faltantes: list[str]):
    SALIDA.parent.mkdir(exist_ok=True)
    d.sort_values(["tema", "fecha"]).to_csv(SALIDA, index=False)
    ESTADO.write_text(json.dumps({"fuente": "GDELT 1.0, archivos diarios de eventos", "completo": completo,
                                  "hasta": str(d["fecha"].max().date()) if len(d) else None, "dias_sin_archivo": faltantes[-30:],
                                  "filas": int(len(d))}, indent=2, ensure_ascii=False))


def main():
    t0 = time.time()
    ayer = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    d = pd.read_csv(SALIDA, parse_dates=["fecha"]) if SALIDA.exists() else pd.DataFrame()
    if len(d) and "eventos" not in d.columns:
        d = pd.DataFrame()  # archivo viejo de la API de búsqueda: se reemplaza
    hechos = set(d["fecha"]) if len(d) else set()
    pendientes = [f for f in pd.date_range(INICIO, ayer) if f not in hechos]
    print("días por procesar:", len(pendientes), flush=True)
    s, nuevos, faltantes = requests.Session(), [], []
    for i, f in enumerate(pendientes):
        if time.time() - t0 > LIMITE_SEGUNDOS:
            break
        contenido = bajar(f, s)
        if contenido is None:
            faltantes.append(str(f.date()))
            continue
        nuevos.append(agregar(leer(contenido), f))
        if len(nuevos) >= 30:
            d = pd.concat([d, *nuevos], ignore_index=True)
            nuevos = []
            guardar(d, False, faltantes)
            print("guardado hasta", f.date(), f"({i + 1}/{len(pendientes)})", flush=True)
    if nuevos:
        d = pd.concat([d, *nuevos], ignore_index=True)
    hechos = set(d["fecha"]) if len(d) else set()
    restan = [f for f in pd.date_range(INICIO, ayer) if f not in hechos and str(f.date()) not in faltantes]
    guardar(d, completo=not restan, faltantes=faltantes)
    print("listo:", len(d), "filas; días que faltan:", len(restan), flush=True)


if __name__ == "__main__":
    main()
