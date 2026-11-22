import pathlib, sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ingesta"))
import noticias, trm  # noqa: E402


def test_trm_se_fecha_en_el_dia_de_las_operaciones():
    crudo = pd.DataFrame({"vigenciadesde": ["2026-11-14T00:00:00.000", "2026-11-17T00:00:00.000", "2026-11-18T00:00:00.000"],
                          "valor": ["3900.5", "3910", "3895.25"]})
    d = trm.preparar(crudo)
    # la TRM que empieza a regir el sábado 14 se calculó el viernes 13
    assert d["fecha"].dt.strftime("%Y-%m-%d").tolist() == ["2026-11-13", "2026-11-16", "2026-11-17"]
    assert d["trm"].tolist() == [3900.5, 3910.0, 3895.25]


def test_gdelt_tono_y_volumen():
    tono = {"timeline": [{"series": "Average Tone", "data": [{"date": "20170101T000000Z", "value": -1.5}, {"date": "20170102T000000Z", "value": 0.5}]}]}
    vol = {"timeline": [{"series": "Article Count", "data": [{"date": "20170101T000000Z", "value": 12, "norm": 90000}]}]}
    t, v = noticias.a_tabla(tono, "peso", "timelinetone"), noticias.a_tabla(vol, "peso", "timelinevolraw")
    assert t["tono"].tolist() == [-1.5, 0.5] and v.loc[0, "articulos"] == 12 and v.loc[0, "total"] == 90000


def test_gdelt_rechaza_resolucion_semanal_pero_acepta_dias_faltantes():
    semanal = {"timeline": [{"series": "Average Tone", "data": [{"date": f"201701{d:02d}T000000Z", "value": 1} for d in (1, 8, 15)]}]}
    with pytest.raises(SystemExit):
        noticias.a_tabla(semanal, "x", "timelinetone")
    con_hueco = {"timeline": [{"series": "Average Tone", "data": [{"date": f"201701{d:02d}T000000Z", "value": 1} for d in (1, 2, 3, 9)]}]}
    assert len(noticias.a_tabla(con_hueco, "x", "timelinetone")) == 4


def test_tramos_cubren_todo_sin_huecos():
    t = list(noticias.tramos(pd.Timestamp("2017-01-01"), pd.Timestamp("2017-12-31"), dias=100))
    assert t[0][0] == pd.Timestamp("2017-01-01") and t[-1][1] == pd.Timestamp("2017-12-31")
    assert all(sig[0] == ant[1] + pd.Timedelta(days=1) for ant, sig in zip(t, t[1:]))
