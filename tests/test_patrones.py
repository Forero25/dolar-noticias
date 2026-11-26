"""Los análisis de patrones deben encontrar relaciones plantadas y no inventarlas."""
import pathlib, sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "analisis"))
import patrones  # noqa: E402


def panel(efecto_vol=0.0, efecto_adelanto=0.0, semilla=0, n=2200):
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range("2017-01-02", periods=n)
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = 0.7 * z[i - 1] + rng.normal()
    z = (z - z.mean()) / z.std()
    sigma = 0.006 * np.exp(efecto_vol * np.r_[0, z[:-1]])          # la noticia de ayer cambia el tamaño del movimiento de hoy
    r = efecto_adelanto * 0.003 * np.r_[0, z[:-1]] + sigma * rng.standard_normal(n)
    d = pd.DataFrame({"trm": 3500 * np.exp(np.cumsum(r)), "n_x_z": z, "vix": 18 + rng.normal(0, 1, n)}, index=fechas)
    return d


def test_volatilidad_detecta_noticia_que_mueve_el_tamano():
    con = patrones.volatilidad(panel(efecto_vol=0.6), ["vix"], ["n_x_z"], h=1, cada=60)
    sin = patrones.volatilidad(panel(efecto_vol=0.0, semilla=1), ["vix"], ["n_x_z"], h=1, cada=60)
    assert con["dm_noticias_vs_mercado"]["p"] < 0.01
    assert con["r2_fuera_de_muestra"]["har_mercado_noticias"] > con["r2_fuera_de_muestra"]["har_mercado"] + 0.02
    assert sin["dm_noticias_vs_mercado"]["p"] > 0.05


def test_adelanto_y_granger():
    d = panel(efecto_adelanto=1.0)
    cc = patrones.correlaciones_cruzadas(d, ["n_x_z"], max_rezago=3)["n_x_z"]
    k1 = next(x for x in cc["serie"] if x["k"] == 1)
    assert k1["corr"] > 3 * cc["banda"] and max(abs(x["corr"]) for x in cc["serie"] if x["k"] <= 0) < cc["banda"] * 1.5
    g = patrones.granger(d, ["n_x_z"])["n_x_z"]
    assert g["p_noticia_antecede_trm"] < 0.001 and g["p_trm_antecede_noticia"] > 0.01


def test_movimientos_grandes_y_extremos():
    d = panel(efecto_vol=0.6, semilla=2)
    m = patrones.movimientos_grandes(d, ["vix"], ["n_x_z"], cada=120, n_boot=100)
    assert m["diferencia"] > 0.03 and m["ic95"][0] > 0
    e = patrones.dias_extremos(d, ["n_x_z"], n_boot=50)
    assert e["variables"]["n_x_z"]["dia_anterior"]["diferencia"] > 0.2


def test_por_gobierno_y_series():
    d = panel(efecto_adelanto=1.0, semilla=3)
    g = patrones.por_gobierno(d, "n_x_z")
    assert [x["gobierno"] for x in g][:3] == ["Santos", "Duque", "Petro"] and all(x["ic95"][0] > 0 for x in g)
    s = patrones.series_graficas(d, None)
    assert len(s["trm_mensual"]) > 90 and s["movimiento_medio_mensual_pct"][0]["y"] > 0
