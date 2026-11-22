"""El modelo debe encontrar una señal de noticias cuando existe y no inventarla cuando no existe."""
import pathlib, sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "analisis"))
import correr, evaluacion as ev, variables  # noqa: E402


def sintetico(senal: float, semilla=0):
    rng = np.random.default_rng(semilla)
    dias = pd.date_range("2016-06-01", "2023-12-31", freq="D")
    habiles = dias[dias.dayofweek < 5]
    # tono diario del tema "fiscal": proceso AR(1) persistente
    tono = np.zeros(len(dias))
    for i in range(1, len(dias)):
        tono[i] = 0.9 * tono[i - 1] + rng.normal()
    tono_s = pd.Series(tono, index=dias)
    z = tono_s.rolling(7, min_periods=3).mean()
    z = ((z - z.rolling(90, min_periods=30).mean()) / z.rolling(90, min_periods=30).std()).reindex(habiles).fillna(0)
    # la TRM sube al día siguiente cuando el tono fiscal es negativo (si senal > 0)
    r = -senal * 0.004 * z.shift(1).fillna(0).to_numpy() + rng.normal(0, 0.006, len(habiles))
    trm = pd.DataFrame({"fecha": habiles, "trm": 3500 * np.exp(np.cumsum(r))})
    merc = pd.DataFrame(index=habiles)
    for c in variables.PRECIOS:
        merc[c] = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(habiles))))
    for c in variables.TASAS:
        merc[c] = 3 + np.cumsum(rng.normal(0, 0.02, len(habiles)))
    merc["vix"] = 18 * np.exp(rng.normal(0, 0.1, len(habiles)))
    noti = pd.concat([pd.DataFrame({"fecha": dias, "tema": t, "tono": v, "articulos": rng.poisson(20, len(dias)), "total": 100000})
                      for t, v in [("fiscal", tono), ("ruido", rng.normal(size=len(dias)))]])
    return variables.panel(trm, merc, noti)


@pytest.fixture(scope="module")
def con_senal():
    return correr.evaluar(sintetico(1.0), inicio="2020-01-01", cada=60)["h1"]


@pytest.fixture(scope="module")
def sin_senal():
    return correr.evaluar(sintetico(0.0, semilla=1), inicio="2020-01-01", cada=60)["h1"]


def test_encuentra_la_senal(con_senal):
    m = con_senal["modelos"]
    assert m["ridge|noticias"]["rmse_relativo"] < 0.97
    assert m["ridge|noticias"]["dm_vs_paseo"]["p"] < 0.01
    assert con_senal["noticias_vs_mercado"]["ridge"]["p"] < 0.01
    assert m["ridge|noticias"]["direccion"]["acierto"] > 0.55


def test_no_inventa_senal(sin_senal):
    for k, v in sin_senal["modelos"].items():
        assert v["rmse_relativo"] > 0.985, k  # nadie le gana con holgura al paseo aleatorio en ruido puro


def test_intervalos_calibrados(con_senal):
    assert abs(con_senal["modelos"]["ridge|noticias"]["intervalo90"]["cobertura"] - 0.9) < 0.04


def test_diebold_mariano_y_direccion():
    rng = np.random.default_rng(3)
    y = rng.normal(size=2000)
    bueno, malo = y - rng.normal(0, 0.5, 2000), y - rng.normal(0, 1.5, 2000)
    assert ev.diebold_mariano(y - bueno, y - malo, 1)["p"] < 0.001
    assert ev.pesaran_timmermann(y, bueno)["p"] < 0.001
    assert ev.pesaran_timmermann(y, rng.normal(size=2000))["p"] > 0.01


def test_sin_mirar_el_futuro():
    d = sintetico(0.0, semilla=2)
    assert d["y1"].iloc[10] == pytest.approx(np.log(d["trm"].iloc[11] / d["trm"].iloc[10]))
    assert d["trm_r1"].iloc[10] == pytest.approx(np.log(d["trm"].iloc[10] / d["trm"].iloc[9]))
