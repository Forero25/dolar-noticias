import io, pathlib, sys, zipfile

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ingesta"))
import eventos, trm  # noqa: E402


def test_trm_se_fecha_en_el_dia_de_las_operaciones():
    crudo = pd.DataFrame({"vigenciadesde": ["2026-11-14T00:00:00.000", "2026-11-17T00:00:00.000", "2026-11-18T00:00:00.000"],
                          "valor": ["3900.5", "3910", "3895.25"]})
    d = trm.preparar(crudo)
    # la TRM que empieza a regir el sábado 14 se calculó el viernes 13
    assert d["fecha"].dt.strftime("%Y-%m-%d").tolist() == ["2026-11-13", "2026-11-16", "2026-11-17"]
    assert d["trm"].tolist() == [3900.5, 3910.0, 3895.25]


def fila(a1="", a2="", lugar="", tipo="", raiz=1, quad=1, gold=0.0, menc=1, tono=0.0):
    f = [""] * eventos.N_COLUMNAS
    f[0], f[1] = "1", "20261120"
    v = {"actor1_pais": a1, "actor2_pais": a2, "actor1_tipo": tipo, "raiz": str(raiz), "quad": str(quad),
         "goldstein": str(gold), "menciones": str(menc), "tono": str(tono), "lugar_pais": lugar}
    for k, i in eventos.COL.items():
        f[i] = v[k]
    return "\t".join(f)


def zip_de(filas):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("20261120.export.CSV", "\n".join(filas) + "\n")
    return b.getvalue()


def test_eventos_agregados_por_tema():
    filas = [fila(a1="COL", tipo="GOV", menc=3, tono=-6, gold=-5, quad=4, raiz=19),
             fila(lugar="CO", menc=1, tono=2, gold=3, quad=1, raiz=4),
             fila(a1="USA", menc=6, tono=1, gold=1, quad=1, raiz=4)]
    d = eventos.agregar(eventos.leer(zip_de(filas)), pd.Timestamp("2026-11-20")).set_index("tema")
    c = d.loc["colombia"]
    assert c["eventos"] == 2 and c["articulos"] == 4 and c["total"] == 10
    assert c["tono"] == pytest.approx((3 * -6 + 1 * 2) / 4)
    assert c["violencia"] == pytest.approx(0.75) and c["conflicto"] == pytest.approx(0.75)
    assert d.loc["colombia_gobierno", "eventos"] == 1 and d.loc["mundo", "articulos"] == 10
    assert "venezuela" not in d.index


def test_eventos_rechaza_formato_distinto():
    malo = io.BytesIO()
    with zipfile.ZipFile(malo, "w") as z:
        z.writestr("x.CSV", "\t".join(["1"] * 40) + "\n")
    with pytest.raises(SystemExit):
        eventos.leer(malo.getvalue())
