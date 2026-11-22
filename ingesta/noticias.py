"""Tono y volumen diario de noticias por tema, desde la API DOC 2.0 de GDELT.

GDELT traduce al inglés la prensa de 65 idiomas, así que las búsquedas van en inglés y cubren también la
prensa colombiana en español. Los modos de serie de tiempo llegan hasta el 1 de enero de 2017.
- timelinetone: tono promedio de los artículos del día (negativo = cobertura más negativa).
- timelinevolraw: número de artículos del tema y total de artículos monitoreados ese día.

Las series se guardan en datos_publicos/noticias.csv (datos derivados de GDELT, de uso libre con cita)
y cada corrida solo pide lo nuevo. GDELT pide no hacer más de una petición cada 5 segundos.
"""
import pathlib, sys, time

import pandas as pd
import requests

RAIZ = pathlib.Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "datos_publicos" / "noticias.csv"
API = "https://api.gdeltproject.org/api/v2/doc/doc"
INICIO = pd.Timestamp("2017-01-01")
PAUSA = 6

TEMAS = {
    "peso": '"colombian peso"',
    "banrep": '("Banco de la Republica" OR "Colombian central bank" OR "Colombia central bank")',
    "fiscal": 'Colombia (deficit OR "fiscal rule" OR "tax reform" OR budget)',
    "calificacion": 'Colombia ("credit rating" OR downgrade OR "investment grade")',
    "petroleo_col": "Colombia (oil OR Ecopetrol OR crude)",
    "gobierno": '("Gustavo Petro" OR "Colombian government" OR "Ivan Duque")',
    "orden_publico": "Colombia (protest OR strike OR violence)",
    "prensa_col_economia": "economy sourcecountry:colombia",
    "prensa_col_dolar": "dollar sourcecountry:colombia",
    "fed": '"Federal Reserve"',
    "petroleo_mundo": '"oil prices"',
    "emergentes": '"emerging markets"',
}
MODOS = {"timelinetone": "tono", "timelinevolraw": "volumen"}


def pedir(query: str, modo: str, ini: pd.Timestamp, fin: pd.Timestamp) -> dict:
    params = {"query": query, "mode": modo, "format": "json",
              "startdatetime": ini.strftime("%Y%m%d000000"), "enddatetime": fin.strftime("%Y%m%d235959")}
    ultimo = ""
    for intento in range(6):
        try:
            r = requests.get(API, params=params, timeout=120)
            ultimo = f"HTTP {r.status_code}: {r.text[:300]!r}"
            if r.ok and r.text.lstrip().startswith("{"):
                time.sleep(PAUSA)
                return r.json()
        except requests.RequestException as e:
            ultimo = repr(e)
        print(f"  reintento {intento + 1} ({modo}, {ini.date()}): {ultimo}", flush=True)
        time.sleep(PAUSA * 2 ** intento)  # límite de velocidad o error temporal
    sys.exit(f"GDELT no respondió: {modo} {query} {ini.date()} — última respuesta: {ultimo}")


def a_tabla(js: dict, tema: str, modo: str) -> pd.DataFrame:
    filas = []
    for serie in js.get("timeline", []):
        for p in serie["data"]:
            fila = {"fecha": pd.to_datetime(p["date"][:8]), "tema": tema}
            if modo == "timelinetone":
                fila["tono"] = p["value"]
            else:
                fila["articulos"], fila["total"] = p["value"], p.get("norm")
            filas.append(fila)
    d = pd.DataFrame(filas)
    # Días sin artículos pueden faltar (temas poco cubiertos); una resolución semanal o mensual se nota en que
    # el salto más común entre fechas es mayor que un día.
    if len(d) > 2 and d["fecha"].diff().dropna().dt.days.median() > 1:
        sys.exit(f"GDELT devolvió una resolución menor que diaria para {tema} ({modo}); reducir el tramo.")
    return d


def tramos(ini: pd.Timestamp, fin: pd.Timestamp, dias=180):
    a = ini
    while a <= fin:
        b = min(a + pd.Timedelta(days=dias - 1), fin)
        yield a, b
        a = b + pd.Timedelta(days=1)


def main():
    hoy = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    previo = pd.read_csv(SALIDA, parse_dates=["fecha"]) if SALIDA.exists() else pd.DataFrame()
    nuevas = []
    for tema, q in TEMAS.items():
        # se repiten los últimos 7 días porque GDELT completa días recientes con retraso
        desde = INICIO if previo.empty or tema not in set(previo["tema"]) else previo.loc[previo["tema"] == tema, "fecha"].max() - pd.Timedelta(days=7)
        for a, b in tramos(desde, hoy):
            partes = [a_tabla(pedir(q, modo, a, b), tema, modo) for modo in MODOS]
            partes = [p for p in partes if not p.empty]
            if len(partes) == 2:
                nuevas.append(partes[0].merge(partes[1], on=["fecha", "tema"], how="outer"))
        print(tema, "listo")
    d = pd.concat([previo, *nuevas], ignore_index=True).drop_duplicates(["fecha", "tema"], keep="last")
    d = d.sort_values(["tema", "fecha"])
    SALIDA.parent.mkdir(exist_ok=True)
    d.to_csv(SALIDA, index=False)
    print("noticias:", d.groupby("tema").size().to_dict())


if __name__ == "__main__":
    main()
