"""Panel diario para pronosticar la TRM.

Origen del pronóstico: la noche del día hábil t, cuando ya se conocen la TRM negociada ese día, el cierre de
Nueva York y las noticias del día. Objetivo: el cambio logarítmico de la TRM entre t y t+h días hábiles.
Toda variable usa solo información disponible en t (los datos de mercado se llevan hacia adelante, nunca hacia atrás).
"""
import numpy as np
import pandas as pd

HORIZONTES = (1, 5, 20)
PRECIOS = ["brent", "dolar_amplio", "usd_mxn", "usd_brl", "usd_clp", "usd_pen", "bonos_emergentes"]
TASAS = ["tesoro_10a", "tesoro_2a", "tasa_fed", "inflacion_esperada_5a"]


def objetivo(trm: pd.Series, h: int) -> pd.Series:
    l = np.log(trm)
    return l.shift(-h) - l


def de_trm(trm: pd.Series) -> pd.DataFrame:
    l = np.log(trm)
    r = l.diff()
    return pd.DataFrame({"trm_r1": r, "trm_r5": l.diff(5), "trm_r20": l.diff(20),
                         "trm_vol20": r.rolling(20).std(), "trm_desvio_media60": l - l.rolling(60).mean()})


def de_mercado(m: pd.DataFrame, fechas: pd.DatetimeIndex) -> pd.DataFrame:
    m = m.sort_index().ffill().reindex(m.index.union(fechas)).ffill().reindex(fechas)
    out = {}
    for c in PRECIOS:
        l = np.log(m[c])
        out |= {f"{c}_r1": l.diff(), f"{c}_r5": l.diff(5), f"{c}_r20": l.diff(20)}
    for c in TASAS:
        out |= {f"{c}_d5": m[c].diff(5), f"{c}_d20": m[c].diff(20)}
    out |= {"vix": np.log(m["vix"]), "vix_d5": np.log(m["vix"]).diff(5), "pendiente_tesoro": m["tesoro_10a"] - m["tesoro_2a"]}
    return pd.DataFrame(out, index=fechas)


def de_noticias(n: pd.DataFrame, fechas: pd.DatetimeIndex) -> pd.DataFrame:
    """Por tema: tono suavizado y su desviación frente a los 90 días previos; participación del tema en el total de noticias."""
    out = {}
    for tema, g in n.groupby("tema"):
        g = g.set_index("fecha").sort_index().asfreq("D")
        tono = g["tono"].rolling(7, min_periods=3).mean()
        z = (tono - tono.rolling(90, min_periods=30).mean()) / tono.rolling(90, min_periods=30).std()
        cuota = np.log((g["articulos"].rolling(7, min_periods=3).sum() + 1) / g["total"].rolling(7, min_periods=3).sum())
        d = pd.DataFrame({f"n_{tema}_tono7": tono, f"n_{tema}_tono_z": z, f"n_{tema}_cuota": cuota,
                          f"n_{tema}_cuota_d7": cuota.diff(7)})
        out[tema] = d.reindex(d.index.union(fechas)).ffill().reindex(fechas)
    return pd.concat(out.values(), axis=1)


def panel(trm: pd.DataFrame, mercados: pd.DataFrame, noticias: pd.DataFrame) -> pd.DataFrame:
    s = trm.set_index("fecha")["trm"].sort_index()
    s = s[s.index.dayofweek < 5]
    fechas = s.index
    d = pd.concat([de_trm(s), de_mercado(mercados, fechas), de_noticias(noticias, fechas)], axis=1)
    for h in HORIZONTES:
        d[f"y{h}"] = objetivo(s, h)
    d["trm"] = s
    return d


def columnas(d: pd.DataFrame, conjunto: str) -> list[str]:
    base = [c for c in d.columns if c.startswith("trm_")]
    mercado = [c for c in d.columns if not c.startswith(("trm", "n_", "y"))]
    noticias = [c for c in d.columns if c.startswith("n_")]
    return {"propia": base, "mercado": base + mercado, "noticias": base + mercado + noticias, "solo_noticias": base + noticias}[conjunto]
