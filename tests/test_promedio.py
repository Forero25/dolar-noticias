"""El efecto Working: promediar un paseo aleatorio crea autocorrelación de 0,25 en sus cambios; el cierre no."""
import pathlib, sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "analisis"))
import promedio  # noqa: E402


def dias_sinteticos(n_dias=3000, por_dia=100, semilla=0):
    rng = np.random.default_rng(semilla)
    p = np.cumsum(rng.normal(0, 0.006 / np.sqrt(por_dia), n_dias * por_dia)).reshape(n_dias, por_dia)
    fechas = pd.bdate_range("2010-01-01", periods=n_dias)
    return pd.Series(np.exp(8 + p.mean(axis=1)), fechas), pd.Series(np.exp(8 + p[:, -1]), fechas)


def test_working_promedio_vs_cierre():
    trm, cierre = dias_sinteticos()
    r = promedio.comparar(trm, cierre, desde="2010-01-01")
    assert r["trm"]["autocorrelacion_1"] == pytest.approx(0.25, abs=0.05)
    assert abs(r["cierre"]["autocorrelacion_1"]) < 0.05
    assert r["razon_varianzas_trm_cierre"] == pytest.approx(2 / 3, abs=0.06)
    assert r["trm"]["ar1"]["rmse_relativo"] < 0.99 and r["trm"]["ar1"]["dm"]["p"] < 0.01
    assert r["cierre"]["ar1"]["rmse_relativo"] > 0.995
