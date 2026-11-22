"""Proyecciones locales (Jordà, 2005): cuánto cambia la TRM k días después de un tono de noticias una
desviación estándar más negativo que lo normal en un tema, controlando por la propia TRM y el mercado.
Errores estándar de Newey-West con k rezagos, porque los cambios acumulados se traslapan."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

CONTROLES = ["trm_r1", "trm_r5", "dolar_amplio_r1", "brent_r1", "vix_d5", "usd_mxn_r1"]


def respuesta(d: pd.DataFrame, tema: str, k_max=20) -> list[dict]:
    choque = -d[f"n_{tema}_tono_z"]  # signo invertido: choque = noticias más negativas
    l = np.log(d["trm"])
    out = []
    for k in range(1, k_max + 1):
        y = (l.shift(-k) - l) * 100  # en porcentaje
        X = pd.concat([choque.rename("choque"), d[[c for c in CONTROLES if c in d]]], axis=1)
        datos = pd.concat([y.rename("y"), X], axis=1).dropna()
        if len(datos) < 200:
            continue
        r = sm.OLS(datos["y"], sm.add_constant(datos.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": k})
        b, se = r.params["choque"], r.bse["choque"]
        out.append({"k": k, "efecto_pct": float(b), "ic95": [float(b - 1.96 * se), float(b + 1.96 * se)], "n": int(r.nobs)})
    return out
