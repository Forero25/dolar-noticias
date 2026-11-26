"""¿Por qué un modelo con la propia TRM le gana por poco al paseo aleatorio a un día? Hipótesis: porque la TRM
es un promedio de las operaciones del día, no un precio de cierre.

Working (1960): si el precio sigue un paseo aleatorio y se observa su promedio en cada intervalo, los cambios
de ese promedio tienen autocorrelación de primer orden de 0,25 (con muchas operaciones por día) y varianza de
2/3 de la del precio puntual, aunque el precio mismo no sea predecible. Un precio de cierre no tiene ese efecto.

La prueba compara, en las mismas fechas, la TRM con el cierre diario del peso en Yahoo Finance (COP=X):
1. autocorrelación de primer orden de los cambios diarios, con intervalo por bootstrap de bloques;
2. razón de varianzas;
3. pronóstico con un AR(1) reestimado con origen móvil frente al paseo aleatorio, en cada serie.
Si la hipótesis es cierta: TRM con autocorrelación cercana a 0,25 y AR(1) que le gana al paseo; cierre con
autocorrelación cercana a 0 y AR(1) que no le gana.
"""
import numpy as np
import pandas as pd

import evaluacion as ev

WORKING = 0.25


def autocorr(r: np.ndarray) -> float:
    r = r - r.mean()
    return float((r[1:] * r[:-1]).sum() / (r * r).sum())


def bootstrap_bloques(r: np.ndarray, b=20, n=2000, semilla=7) -> list[float]:
    rng = np.random.default_rng(semilla)
    k = int(np.ceil(len(r) / b))
    sims = []
    for _ in range(n):
        ini = rng.integers(0, len(r) - b, k)
        s = np.concatenate([r[i:i + b] for i in ini])[:len(r)]
        sims.append(autocorr(s))
    return [float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))]


def ar1_vs_paseo(r: pd.Series, inicio="2020-01-01", cada=20) -> dict:
    """Pronóstico del cambio de mañana con un AR(1) estimado con datos hasta hoy, frente a cero."""
    r = r.dropna()
    pred, real = [], []
    fechas = r.index[r.index >= pd.Timestamp(inicio)]
    for k in range(0, len(fechas) - 1, cada):
        bloque = fechas[k:k + cada]
        hist = r[r.index < bloque[0]]
        x, y = hist.to_numpy()[:-1], hist.to_numpy()[1:]
        b = np.polyfit(x, y, 1)
        for f in bloque:
            i = r.index.get_loc(f)
            if i + 1 < len(r):
                pred.append(b[0] * r.iloc[i] + b[1])
                real.append(r.iloc[i + 1])
    e_m, e_rw = np.array(real) - np.array(pred), np.array(real)
    return {"n": len(real), "rmse_relativo": ev.rmse(e_m) / ev.rmse(e_rw), "dm": ev.diebold_mariano(e_m, e_rw, 1),
            "direccion": ev.pesaran_timmermann(real, pred)}


def comparar(trm: pd.Series, cierre: pd.Series, desde="2017-01-01") -> dict:
    """trm y cierre indexados por fecha de negociación."""
    lt, lc = np.log(trm), np.log(cierre)
    d = pd.DataFrame({"trm": lt, "cierre": lc}).dropna()
    d = d[d.index >= pd.Timestamp(desde)]
    r = d.diff().dropna()
    r = r[(r.abs() < 0.08).all(axis=1)]  # fuera saltos imposibles en un día (errores de la serie de Yahoo)
    out = {"n": len(r), "desde": str(r.index[0].date()), "hasta": str(r.index[-1].date()), "working_teorico": WORKING,
           "razon_varianzas_trm_cierre": float(r["trm"].var() / r["cierre"].var()), "razon_varianzas_teorica": 2 / 3,
           "correlacion_trm_cierre_mismo_dia": float(r["trm"].corr(r["cierre"])),
           "correlacion_trm_hoy_cierre_ayer": float(r["trm"].corr(r["cierre"].shift(1))),
           "correlacion_trm_hoy_cierre_manana": float(r["trm"].corr(r["cierre"].shift(-1)))}
    for c in ("trm", "cierre"):
        x = r[c].to_numpy()
        out[c] = {"autocorrelacion_1": autocorr(x), "ic95": bootstrap_bloques(x), "ar1": ar1_vs_paseo(r[c])}
    return out
