"""Más allá del pronóstico de la dirección: ¿las noticias mueven cuánto se mueve el dólar, quién antecede a quién,
qué dominaba las noticias en los días extremos y cambió la relación entre gobiernos?

Todo con origen móvil (reentrenando solo con el pasado) o con inferencia que respeta la dependencia en el tiempo.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests

import evaluacion as ev

GOBIERNOS = [("Santos", "2010-08-07", "2018-08-06"), ("Duque", "2018-08-07", "2022-08-06"),
             ("Petro", "2022-08-07", "2026-08-06"), ("Desde agosto de 2026", "2026-08-07", "2100-01-01")]


def retornos(d: pd.DataFrame) -> pd.Series:
    return np.log(d["trm"]).diff()


# ---------- 1. Volatilidad: ¿las noticias anticipan cuánto se moverá el dólar? ----------
def har(d: pd.DataFrame) -> pd.DataFrame:
    """Modelo HAR (Corsi, 2009) sobre el tamaño del movimiento diario: |r| de ayer, promedio de 5 y de 22 días."""
    a = retornos(d).abs()
    return pd.DataFrame({"har_1": a, "har_5": a.rolling(5).mean(), "har_22": a.rolling(22).mean()}, index=d.index)


def volatilidad(d: pd.DataFrame, cols_mercado: list[str], cols_noticias: list[str], h=1, inicio="2020-01-01", cada=20) -> dict:
    a = retornos(d).abs()
    y = a.rolling(h).mean().shift(-h) * 100  # tamaño medio del movimiento en los próximos h días, en %
    X = pd.concat([har(d) * 100, d[cols_mercado], d[cols_noticias]], axis=1)
    conjuntos = {"har": list(har(d).columns), "har_mercado": list(har(d).columns) + cols_mercado,
                 "har_mercado_noticias": list(har(d).columns) + cols_mercado + cols_noticias}
    datos = X.join(y.rename("y")).dropna(subset=["y", "har_22"])
    prueba = datos.index[datos.index >= pd.Timestamp(inicio)]
    preds = {k: [] for k in conjuntos}
    for k in range(0, len(prueba), cada):
        bloque = prueba[k:k + cada]
        fin = datos.index.get_loc(bloque[0]) - h
        for nombre, cols in conjuntos.items():
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              RidgeCV(alphas=np.logspace(-2, 4, 20), cv=TimeSeriesSplit(5, gap=h)))
            m.fit(datos[cols].iloc[:fin], datos["y"].iloc[:fin])
            preds[nombre].append(pd.Series(m.predict(datos.loc[bloque, cols]), index=bloque))
    yv = datos.loc[prueba, "y"]
    e = {k: (yv - pd.concat(v)).to_numpy() for k, v in preds.items()}
    base = ev.rmse(e["har"])
    r2 = lambda err: 1 - (err ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    return {"h": h, "n": len(yv), "r2_fuera_de_muestra": {k: float(r2(v)) for k, v in e.items()},
            "rmse_relativo_al_har": {k: ev.rmse(v) / base for k, v in e.items()},
            "dm_noticias_vs_mercado": ev.diebold_mariano(e["har_mercado_noticias"], e["har_mercado"], h),
            "dm_mercado_vs_har": ev.diebold_mariano(e["har_mercado"], e["har"], h)}


# ---------- 2. ¿Quién antecede a quién? ----------
def sorpresa(x: pd.Series, p=5) -> pd.Series:
    """Parte no anticipable de una serie persistente: residuo de un AR(p) (preblanqueo de Box y Jenkins).
    Sin esto, una noticia persistente parece correlacionada con el dólar en todos los rezagos a la vez."""
    x = x.dropna()
    X = pd.concat({f"l{i}": x.shift(i) for i in range(1, p + 1)}, axis=1)
    datos = pd.concat([x.rename("y"), X], axis=1).dropna()
    m = sm.OLS(datos["y"], sm.add_constant(datos.drop(columns="y"))).fit()
    return m.resid


def correlaciones_cruzadas(d: pd.DataFrame, cols: list[str], max_rezago=10) -> dict:
    """corr(sorpresa de la noticia en t, cambio de la TRM en t+k). k > 0: la noticia antecede al dólar; k < 0: el dólar
    antecede a la noticia. Banda de ±1,96/√n (Bartlett) bajo la hipótesis de que no hay relación."""
    r = retornos(d)
    out = {}
    for c in cols:
        x = sorpresa(d[c])
        filas = []
        for k in range(-max_rezago, max_rezago + 1):
            par = pd.concat([x, r.shift(-k)], axis=1).dropna()
            filas.append({"k": k, "corr": float(par.iloc[:, 0].corr(par.iloc[:, 1])), "n": len(par)})
        n = min(f["n"] for f in filas)
        out[c] = {"banda": 1.96 / np.sqrt(n), "serie": filas}
    return out


def granger(d: pd.DataFrame, cols: list[str], rezagos=5) -> dict:
    r = retornos(d) * 100
    out = {}
    for c in cols:
        x = d[c].diff() if not c.endswith("_z") else d[c]
        par = pd.concat([r.rename("trm"), x.rename("noticia")], axis=1).dropna()
        if len(par) < 300:
            continue
        p1 = grangercausalitytests(par[["trm", "noticia"]], maxlag=[rezagos])[rezagos][0]["ssr_ftest"][1]
        p2 = grangercausalitytests(par[["noticia", "trm"]], maxlag=[rezagos])[rezagos][0]["ssr_ftest"][1]
        out[c] = {"p_noticia_antecede_trm": float(p1), "p_trm_antecede_noticia": float(p2), "n": len(par)}
    return out


# ---------- 3. Días extremos ----------
def dias_extremos(d: pd.DataFrame, cols: list[str], cuantil=0.95, n_boot=400, semilla=7) -> dict:
    """En los días de mayor subida o bajada de la TRM (5 % extremo), ¿cuánto distinto del resto estaba cada variable
    de noticias el mismo día y el día anterior? Diferencia de medias estandarizada, con bootstrap por bloques mensuales.
    Cada variable se estandariza dentro de su año: la cobertura de GDELT cambia con los años y los días extremos se
    concentran en algunos años (2020, 2022, 2023), así que comparar contra todo el periodo mezclaría las dos tendencias."""
    r = retornos(d)
    a = r.abs()
    umbral = a.quantile(cuantil)
    rng = np.random.default_rng(semilla)
    out = {"umbral_pct": float(umbral * 100), "n_extremos": int((a > umbral).sum()),
           "extremos_por_anio": {str(k): int(v) for k, v in (a > umbral).groupby(a.index.year).sum().items()}, "variables": {}}
    for c in cols:
        g = d[c].groupby(d.index.year)
        z = (d[c] - g.transform("mean")) / g.transform("std")
        res = {}
        for nombre, serie in (("mismo_dia", z), ("dia_anterior", z.shift(1))):
            df = pd.DataFrame({"z": serie, "ext": a > umbral, "mes": d.index.to_period("M")}).dropna()
            dif = df.loc[df.ext, "z"].mean() - df.loc[~df.ext, "z"].mean()
            meses = df["mes"].unique()
            sims = []
            for _ in range(n_boot):
                m = rng.choice(meses, len(meses))
                b = pd.concat([df[df.mes == x] for x in m[:60]])  # 60 meses por réplica: rápido y suficiente
                if b.ext.any() and (~b.ext).any():
                    sims.append(b.loc[b.ext, "z"].mean() - b.loc[~b.ext, "z"].mean())
            res[nombre] = {"diferencia": float(dif), "ic95": [float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))]}
        out["variables"][c] = res
    return out


# ---------- 4. ¿Las noticias anticipan los días de movimientos grandes? ----------
def movimientos_grandes(d: pd.DataFrame, cols_mercado: list[str], cols_noticias: list[str], cuantil=0.9,
                        inicio="2020-01-01", cada=60, n_boot=500, semilla=7) -> dict:
    a = retornos(d).abs()
    X = pd.concat([har(d), d[cols_mercado], d[cols_noticias]], axis=1)
    sin = list(har(d).columns) + cols_mercado
    con = sin + cols_noticias
    datos = X.join(a.shift(-1).rename("a_sig")).dropna(subset=["a_sig", "har_22"])
    prueba = datos.index[datos.index >= pd.Timestamp(inicio)]
    ps, pc, yv = [], [], []
    for k in range(0, len(prueba), cada):
        bloque = prueba[k:k + cada]
        fin = datos.index.get_loc(bloque[0]) - 1
        hist = datos.iloc[:fin]
        umbral = hist["a_sig"].quantile(cuantil)  # el umbral también se fija solo con el pasado
        y_ent = (hist["a_sig"] > umbral).astype(int)
        for cols, dest in ((sin, ps), (con, pc)):
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            m.fit(hist[cols], y_ent)
            dest.append(m.predict_proba(datos.loc[bloque, cols])[:, 1])
        yv.append((datos.loc[bloque, "a_sig"] > umbral).astype(int).to_numpy())
    ps, pc, yv = map(np.concatenate, (ps, pc, yv))
    auc_s, auc_c = roc_auc_score(yv, ps), roc_auc_score(yv, pc)
    rng = np.random.default_rng(semilla)
    difs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(yv), len(yv))
        if yv[i].min() != yv[i].max():
            difs.append(roc_auc_score(yv[i], pc[i]) - roc_auc_score(yv[i], ps[i]))
    return {"n": int(len(yv)), "positivos": int(yv.sum()), "auc_mercado": float(auc_s), "auc_mercado_noticias": float(auc_c),
            "diferencia": float(auc_c - auc_s), "ic95": [float(np.quantile(difs, 0.025)), float(np.quantile(difs, 0.975))]}


# ---------- 5. ¿Cambió la relación entre gobiernos? ----------
def por_gobierno(d: pd.DataFrame, col: str) -> list[dict]:
    """Regresión del cambio de la TRM de mañana sobre la noticia de hoy, por gobierno, con errores de Newey-West."""
    r = retornos(d).shift(-1) * 100
    x = d[col] if col.endswith("_z") else d[col].diff()
    out = []
    for nombre, a, b in GOBIERNOS:
        par = pd.concat([r.rename("y"), x.rename("x")], axis=1).loc[a:b].dropna()
        if len(par) < 120:
            continue
        m = sm.OLS(par["y"], sm.add_constant((par["x"] - par["x"].mean()) / par["x"].std())).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        out.append({"gobierno": nombre, "desde": a, "n": len(par), "efecto_pct": float(m.params["x"]),
                    "ic95": [float(m.conf_int().loc["x", 0]), float(m.conf_int().loc["x", 1])],
                    "volatilidad_diaria_pct": float(retornos(d).loc[a:b].std() * 100)})
    return out


# ---------- Series para las gráficas ----------
def series_graficas(d: pd.DataFrame, titulares: pd.DataFrame | None) -> dict:
    trm_m = d["trm"].resample("ME").mean()
    vol = (retornos(d).abs() * 100).resample("ME").mean()
    out = {"trm_mensual": [{"x": f"{i:%Y-%m}", "y": round(float(v), 2)} for i, v in trm_m.dropna().items()],
           "movimiento_medio_mensual_pct": [{"x": f"{i:%Y-%m}", "y": round(float(v), 4)} for i, v in vol.dropna().items()]}
    if titulares is not None and len(titulares):
        t = titulares.set_index("fecha").sort_index().resample("QE").sum(numeric_only=True)
        out["temas_trimestrales"] = {tema: [{"x": f"{i.year}T{i.quarter}", "y": round(float(v), 4)}
                                            for i, v in (t[f"t_{tema}"] / t["con_texto"]).dropna().items()]
                                     for tema in ["dolar", "fiscal", "petroleo", "elecciones", "violencia", "venezuela", "eeuu", "politica_monetaria", "calificacion", "protestas"]}
        tt = titulares.set_index("fecha").sort_index()["tono"].resample("ME").mean()
        out["tono_titulares_mensual"] = [{"x": f"{i:%Y-%m}", "y": round(float(v), 4)} for i, v in tt.dropna().items()]
        out["titulares_totales"] = int(titulares["con_texto"].sum())
        out["titulares_es"] = int(titulares["es"].sum())
        out["titulares_en"] = int(titulares["en"].sum())
        out["titulares_diarios_por_anio"] = {str(k): float(v) for k, v in titulares.groupby(titulares["fecha"].dt.year)["con_texto"].mean().items()}
        out["menciones_por_tema"] = {k[2:]: int(titulares[k].sum()) for k in titulares.columns if k.startswith("t_")}
    return out
