"""Evaluación con origen móvil: el modelo se reentrena cada `cada` días hábiles usando solo pares
(variables en s, resultado en s+h) con s+h <= t, es decir, resultados ya conocidos en la fecha del pronóstico."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from variables import columnas


def fabrica(nombre: str, h: int):
    if nombre == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-1, 5, 25), cv=TimeSeriesSplit(5, gap=h)))
    if nombre == "gbm":
        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.03, max_depth=3, min_samples_leaf=50,
                                             l2_regularization=1.0, random_state=7)
    raise ValueError(nombre)


def preparar(d: pd.DataFrame, h: int, conjunto: str):
    cols = columnas(d, conjunto)
    datos = d[cols + [f"y{h}"]].dropna(subset=[f"y{h}"])
    datos = datos[datos[cols].notna().mean(axis=1) > 0.8]  # fuera el arranque, cuando faltan casi todas las variables
    return datos[cols], datos[f"y{h}"]


def origen_movil(d: pd.DataFrame, h: int, conjunto: str, modelo: str, inicio="2020-01-01", cada=20) -> pd.DataFrame:
    X, y = preparar(d, h, conjunto)
    prueba = X.index[X.index >= pd.Timestamp(inicio)]
    salida = []
    for k in range(0, len(prueba), cada):
        bloque = prueba[k:k + cada]
        pos = X.index.get_loc(bloque[0])
        fin = pos - h  # el resultado de la fila fin-1 se conoce en bloque[0]
        if modelo == "paseo_aleatorio":
            p = np.zeros(len(bloque))
        elif modelo == "media":
            p = np.full(len(bloque), y.iloc[:fin].mean())
        else:
            m = fabrica(modelo, h).fit(X.iloc[:fin], y.iloc[:fin])
            p = m.predict(X.loc[bloque])
        salida.append(pd.DataFrame({"pred": p, "y": y.loc[bloque]}, index=bloque))
    return pd.concat(salida)


def ajuste_final(d: pd.DataFrame, h: int, conjunto: str, modelo: str):
    """Modelo con toda la historia, para el pronóstico de hoy y para ver qué variables pesan."""
    X, y = preparar(d, h, conjunto)
    m = fabrica(modelo, h).fit(X, y)
    hoy = d[X.columns].iloc[[-1]]
    return m, float(m.predict(hoy)[0]), X, y
