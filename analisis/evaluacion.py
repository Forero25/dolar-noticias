"""Pruebas para comparar pronósticos.

- Diebold y Mariano (1995) con la corrección de Harvey, Leybourne y Newbold (1997) para muestras pequeñas y
  horizontes con traslape: compara la pérdida cuadrática de dos pronósticos.
- Pesaran y Timmermann (1992): ¿se acierta la dirección más que por azar?
- Cobertura de intervalos conformales: con qué frecuencia el valor real cae dentro del intervalo.
"""
import numpy as np
from scipy import stats


def rmse(e) -> float:
    e = np.asarray(e)
    return float(np.sqrt(np.mean(e ** 2)))


def diebold_mariano(e_modelo, e_rival, h: int) -> dict:
    """Estadístico negativo = el modelo tiene menor error que el rival. Valor p unilateral (H1: el modelo es mejor)."""
    d = np.asarray(e_modelo) ** 2 - np.asarray(e_rival) ** 2
    n, dbar = len(d), d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return {"dm": None, "p": None}
    dm = dbar / np.sqrt(var)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"dm": float(hln), "p": float(stats.t.cdf(hln, df=n - 1))}


def pesaran_timmermann(y, p) -> dict:
    y, p = np.sign(np.asarray(y)), np.sign(np.asarray(p))
    m = (y != 0) & (p != 0)
    y, p = y[m] > 0, p[m] > 0
    n = len(y)
    if n == 0 or p.all() or (~p).all():
        return {"acierto": float(np.mean(y == p)) if n else None, "p": None, "n": n}
    acierto = np.mean(y == p)
    py, pp = y.mean(), p.mean()
    p_star = py * pp + (1 - py) * (1 - pp)
    v_hat = p_star * (1 - p_star) / n
    v_star = ((2 * py - 1) ** 2 * pp * (1 - pp) / n + (2 * pp - 1) ** 2 * py * (1 - py) / n
              + 4 * py * pp * (1 - py) * (1 - pp) / n ** 2)
    est = (acierto - p_star) / np.sqrt(v_hat - v_star) if v_hat > v_star else np.nan
    return {"acierto": float(acierto), "p": float(1 - stats.norm.cdf(est)) if np.isfinite(est) else None, "n": n}


def intervalos_conformales(y, pred, h: int, nivel=0.9, ventana=250) -> dict:
    """Intervalo = pronóstico ± cuantil de los errores absolutos de las últimas `ventana` predicciones cuyo
    resultado ya se conocía (con h días de rezago, para no mirar el futuro)."""
    y, pred = np.asarray(y), np.asarray(pred)
    err = np.abs(y - pred)
    dentro, ancho = [], []
    for i in range(ventana + h, len(y)):
        q = np.quantile(err[i - h - ventana:i - h], nivel)
        dentro.append(abs(y[i] - pred[i]) <= q)
        ancho.append(2 * q)
    return {"nivel": nivel, "cobertura": float(np.mean(dentro)) if dentro else None,
            "ancho_medio": float(np.mean(ancho)) if ancho else None, "n": len(dentro)}
