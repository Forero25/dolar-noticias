import pathlib, sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ingesta"))
import titulares as t  # noqa: E402


def test_texto_idioma_temas_y_tono():
    p = t.texto_de_url("https://www.portafolio.co/economia/dolar-hoy-peso-colombiano-cae-por-reforma-tributaria-5512345.html")
    assert p == ["dolar", "hoy", "peso", "colombiano", "cae", "por", "reforma", "tributaria"]
    assert t.idioma(p) == "es" and t.tono(p) == (0, 1)
    assert t.tiene(p, t.TEMAS["dolar"]) and t.tiene(p, t.TEMAS["fiscal"]) and not t.tiene(p, t.TEMAS["violencia"])
    q = t.texto_de_url("https://www.reuters.com/markets/colombian-peso-falls-after-central-bank-holds-interest-rates-2022-05-01/")
    assert t.idioma(q) == "en" and t.tiene(q, t.TEMAS["politica_monetaria"])
    assert t.texto_de_url("https://example.com/news/12345.html") == []


def test_agregado_diario():
    urls = ["https://a.co/x/dolar-sube-por-temor-en-el-congreso", "https://b.com/y/colombia-oil-output-rises-to-record",
            "https://c.com/12345"]
    f, textos = t.agregar_dia(urls, pd.Timestamp("2026-11-20"))
    assert f["enlaces"] == 3 and f["con_texto"] == 2 and f["es"] == 1 and f["en"] == 1
    assert f["t_dolar"] == 1 and f["t_petroleo"] == 1 and f["t_gobierno"] == 1
    assert f["tono"] == 0.5  # uno con tono 0 (sube y temor) y otro con +1 (rises, record): promedio de los que tienen tono
    assert len(textos) == 2
