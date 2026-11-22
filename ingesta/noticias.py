"""Tono y volumen diario de noticias por tema, desde la API DOC 2.0 de GDELT.

GDELT traduce al inglés la prensa de 65 idiomas, así que las búsquedas van en inglés y cubren también la
prensa colombiana en español. Los modos de serie de tiempo llegan hasta el 1 de enero de 2017.
- timelinetone: tono promedio de los artículos del día (negativo = cobertura más negativa).
- timelinevolraw: número de artículos del tema y total de artículos monitoreados ese día.

Las series se guardan en datos_publicos/noticias.csv (datos derivados de GDELT, de uso libre con cita)
y cada corrida solo pide lo nuevo. GDELT pide no hacer más de una petición cada 5 segundos.
"""
import json, pathlib, sys, time

import pandas as pd
import requests

RAIZ = pathlib.Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "datos_publicos" / "noticias.csv"
API = "https://api.gdeltproject.org/api/v2/doc/doc"
INICIO = pd.Timestamp("2017-01-01")
PAUSA = 10
ESTADO = RAIZ / "datos_publicos" / "noticias_estado.json"


class Frenado(Exception):
    """GDELT limitó las consultas: se guarda lo avanzado y se sigue en la próxima corrida."""

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
    for intento in range(4):
        try:
            r = requests.get(API, params=params, timeout=120)
            ultimo = f"HTTP {r.status_code}: {r.text[:300]!r}"
            if r.ok and r.text.lstrip().startswith("{"):
                time.sleep(PAUSA)
                return r.json()
        except requests.RequestException as e:
            ultimo = repr(e)
        print(f"  reintento {intento + 1} ({modo}, {ini.date()}): {ultimo}", flush=True)
        time.sleep(PAUSA * 3 ** intento)  # límite de velocidad o error temporal: 10, 30, 90, 270 s
    raise Frenado(f"{modo} {query} {ini.date()}: {ultimo[:120]}")


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
        raise ValueError("resolución menor que diaria")
    return d


def bajar_tramo(tema: str, q: str, a: pd.Timestamp, b: pd.Timestamp) -> pd.DataFrame:
    """Pide el tramo completo; si GDELT lo devuelve con resolución semanal, lo parte en dos."""
    try:
        partes = [a_tabla(pedir(q, modo, a, b), tema, modo) for modo in MODOS]
    except ValueError:
        if (b - a).days < 60:
            sys.exit(f"GDELT no entrega resolución diaria ni en tramos cortos para {tema}.")
        m = a + (b - a) / 2
        m = m.normalize()
        return pd.concat([bajar_tramo(tema, q, a, m), bajar_tramo(tema, q, m + pd.Timedelta(days=1), b)])
    partes = [p for p in partes if not p.empty]
    if len(partes) < 2:
        return pd.DataFrame()
    return partes[0].merge(partes[1], on=["fecha", "tema"], how="outer")


def tramos(ini: pd.Timestamp, fin: pd.Timestamp, dias=180):
    a = ini
    while a <= fin:
        b = min(a + pd.Timedelta(days=dias - 1), fin)
        yield a, b
        a = b + pd.Timedelta(days=1)


def guardar(d: pd.DataFrame):
    SALIDA.parent.mkdir(exist_ok=True)
    d.sort_values(["tema", "fecha"]).to_csv(SALIDA, index=False)


def main():
    hoy = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    d = pd.read_csv(SALIDA, parse_dates=["fecha"]) if SALIDA.exists() else pd.DataFrame(columns=["fecha", "tema"])
    completo, motivo = True, ""
    try:
        for tema, q in TEMAS.items():
            hechos = d.loc[d["tema"] == tema, "fecha"]
            # se repiten los últimos 7 días porque GDELT completa los días recientes con retraso
            desde = INICIO if hechos.empty else hechos.max() - pd.Timedelta(days=7)
            for a, b in tramos(desde, hoy, dias=365):
                nuevo = bajar_tramo(tema, q, a, b)
                if not nuevo.empty:
                    d = pd.concat([d, nuevo], ignore_index=True).drop_duplicates(["fecha", "tema"], keep="last")
                    guardar(d)  # se guarda tramo a tramo: si GDELT frena, nada se pierde
            print(tema, "al día", flush=True)
    except Frenado as e:
        completo, motivo = False, str(e)
        print("GDELT frenó las consultas; se sigue en la próxima corrida:", motivo, flush=True)
    ultimo = d.groupby("tema")["fecha"].max() if len(d) else pd.Series(dtype="datetime64[ns]")
    faltan = [t for t in TEMAS if t not in ultimo or ultimo[t] < hoy - pd.Timedelta(days=10)]
    estado = {"completo": completo and not faltan, "temas_pendientes": faltan, "motivo": motivo,
              "filas": int(len(d)), "revisado": pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes")}
    ESTADO.write_text(json.dumps(estado, indent=2, ensure_ascii=False))
    print(estado)


if __name__ == "__main__":
    main()
