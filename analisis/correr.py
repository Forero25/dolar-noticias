"""Corre la evaluación completa y escribe resultados/evaluacion.json."""
import datetime as dt, json, pathlib, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import choques, evaluacion as ev, modelos, variables  # noqa: E402

RAIZ = pathlib.Path(__file__).resolve().parents[1]
CONFIG = [("paseo_aleatorio", "propia"), ("media", "propia"), ("ridge", "propia"), ("ridge", "mercado"), ("ridge", "noticias"),
          ("gbm", "mercado"), ("gbm", "noticias")]
TEMAS_CHOQUE = ["fiscal", "calificacion", "gobierno", "banrep", "orden_publico", "prensa_col_dolar"]


def cargar():
    trm = pd.read_csv(RAIZ / "datos_publicos" / "trm.csv", parse_dates=["fecha"])
    merc = pd.read_parquet(RAIZ / "datos" / "mercados.parquet")
    noti = pd.read_csv(RAIZ / "datos_publicos" / "noticias.csv", parse_dates=["fecha"])
    return variables.panel(trm, merc, noti)


def evaluar(d: pd.DataFrame, inicio="2020-01-01", cada=20) -> dict:
    res = {}
    for h in variables.HORIZONTES:
        preds = {f"{m}|{c}": modelos.origen_movil(d, h, c, m, inicio, cada) for m, c in CONFIG}
        comun = sorted(set.intersection(*(set(p.index) for p in preds.values())))
        rw = preds["paseo_aleatorio|propia"].loc[comun]
        e_rw = (rw["y"] - rw["pred"]).to_numpy()
        filas = {}
        for k, p in preds.items():
            p = p.loc[comun]
            e = (p["y"] - p["pred"]).to_numpy()
            por_anio = {str(a): ev.rmse(g["y"] - g["pred"]) / ev.rmse(rw.loc[g.index, "y"]) for a, g in p.groupby(p.index.year)}
            filas[k] = {"rmse_pct": ev.rmse(e) * 100, "rmse_relativo": ev.rmse(e) / ev.rmse(e_rw),
                        "dm_vs_paseo": ev.diebold_mariano(e, e_rw, h) if k != "paseo_aleatorio|propia" else None,
                        "direccion": ev.pesaran_timmermann(p["y"], p["pred"]),
                        "intervalo90": ev.intervalos_conformales(p["y"], p["pred"], h),
                        "rmse_relativo_por_anio": por_anio}
        # ¿las noticias le agregan al mercado? misma familia de modelo, con y sin noticias
        aporte = {}
        for m in ("ridge", "gbm"):
            a, b = preds[f"{m}|noticias"].loc[comun], preds[f"{m}|mercado"].loc[comun]
            aporte[m] = ev.diebold_mariano((a["y"] - a["pred"]).to_numpy(), (b["y"] - b["pred"]).to_numpy(), h)
        res[f"h{h}"] = {"n": len(comun), "desde": str(comun[0].date()), "hasta": str(comun[-1].date()),
                        "modelos": filas, "noticias_vs_mercado": aporte}
    return res


def pronostico_hoy(d: pd.DataFrame, evals: dict) -> dict:
    out = {"fecha_origen": str(d.index[-1].date()), "trm_origen": float(d["trm"].iloc[-1])}
    for h in variables.HORIZONTES:
        m, p, X, y = modelos.ajuste_final(d, h, "noticias", "ridge")
        coefs = m[-1].coef_
        top = sorted(zip(X.columns, coefs), key=lambda x: -abs(x[1]))[:10]
        out[f"h{h}"] = {"modelo": "ridge|noticias", "cambio_pct": p * 100, "trm_central": float(d["trm"].iloc[-1] * np.exp(p)),
                        "variables_con_mas_peso": [{"variable": c, "coef_estandarizado": float(b)} for c, b in top]}
    return out


def main():
    d = cargar()
    out = {"actualizado": dt.date.today().isoformat(), "panel": {"desde": str(d.index[0].date()), "hasta": str(d.index[-1].date()), "dias": len(d)}}
    out["evaluacion"] = evaluar(d)
    out["choques"] = {t: choques.respuesta(d, t) for t in TEMAS_CHOQUE if f"n_{t}_tono_z" in d}
    out["hoy"] = pronostico_hoy(d, out["evaluacion"])
    (RAIZ / "resultados").mkdir(exist_ok=True)
    (RAIZ / "resultados" / "evaluacion.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for h, r in out["evaluacion"].items():
        print(h, {k: round(v["rmse_relativo"], 4) for k, v in r["modelos"].items()}, "noticias vs mercado:", r["noticias_vs_mercado"])


if __name__ == "__main__":
    main()
