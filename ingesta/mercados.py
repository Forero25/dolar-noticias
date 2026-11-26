"""Variables de mercado que la literatura asocia al peso colombiano.

FRED (sin clave, archivo fredgraph.csv): petróleo Brent, dólar amplio, tasas del Tesoro, VIX, tasa de la Fed,
expectativas de inflación y monedas de pares (peso mexicano, real brasileño).
Yahoo Finance: peso chileno, sol peruano, el ETF de bonos emergentes (EMB) y el cierre diario del peso
colombiano (COP=X), que se usa solo para comparar con la TRM, no como variable del modelo.
Los precios no se publican en el repositorio: viven en la caché del flujo.
"""
import io, pathlib, sys, time

import pandas as pd
import requests
import yfinance as yf

RAIZ = pathlib.Path(__file__).resolve().parents[1]
FRED = {"DCOILBRENTEU": "brent", "DTWEXBGS": "dolar_amplio", "DGS10": "tesoro_10a", "DGS2": "tesoro_2a",
        "VIXCLS": "vix", "DFF": "tasa_fed", "T5YIE": "inflacion_esperada_5a", "DEXMXUS": "usd_mxn", "DEXBZUS": "usd_brl"}
YAHOO = {"CLP=X": "usd_clp", "PEN=X": "usd_pen", "EMB": "bonos_emergentes", "COP=X": "usd_cop_cierre"}


def fred(serie: str) -> pd.Series:
    for intento in range(4):
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": serie}, timeout=60)
        if r.ok:
            d = pd.read_csv(io.StringIO(r.text), na_values=".")
            return d.set_index(pd.to_datetime(d.iloc[:, 0])).iloc[:, 1].astype(float)
        time.sleep(2 ** intento)
    sys.exit(f"FRED no respondió para {serie}")


def yahoo(t: str) -> pd.Series:
    d = yf.download(t, start="2016-01-01", interval="1d", auto_adjust=True, progress=False, multi_level_index=False)
    if d.empty:
        sys.exit(f"Yahoo no devolvió {t}")
    return d["Close"]


def main():
    cols = {nombre: fred(s) for s, nombre in FRED.items()}
    cols |= {nombre: yahoo(t) for t, nombre in YAHOO.items()}
    d = pd.DataFrame(cols).sort_index()
    d.index.name = "fecha"
    d = d.loc["2016-01-01":]
    (RAIZ / "datos").mkdir(exist_ok=True)
    d.to_parquet(RAIZ / "datos" / "mercados.parquet")
    print("mercados:", d.shape, d.index.min().date(), "a", d.index.max().date())


if __name__ == "__main__":
    main()
