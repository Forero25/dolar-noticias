"""Titulares desde los enlaces de las noticias de GDELT.

Los archivos diarios de eventos de GDELT traen, para cada hecho, el enlace de la noticia que lo reporta
(columna SOURCEURL). En la mayoría de los medios ese enlace contiene el titular:
    https://www.portafolio.co/economia/dolar-hoy-peso-colombiano-cae-por-reforma-tributaria-5512345
De cada noticia sobre Colombia (Colombia como actor o como lugar) se toma ese texto, se detecta el idioma
(español o inglés) y se buscan temas y palabras de tono con diccionarios propios, en los dos idiomas.

Se publican solo agregados diarios en datos_publicos/titulares.csv. Los textos de los enlaces quedan en la
caché del flujo (datos/titulares_texto/), no en el repositorio. Descarga reanudable, como ingesta/eventos.py.
"""
import gzip, io, json, pathlib, re, sys, time, unicodedata, zipfile
from urllib.parse import unquote, urlparse

import pandas as pd
import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import eventos  # noqa: E402

RAIZ = pathlib.Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "datos_publicos" / "titulares.csv"
ESTADO = RAIZ / "datos_publicos" / "titulares_estado.json"
TEXTOS = RAIZ / "datos" / "titulares_texto"
COL_URL = 57

# Temas: prefijos de palabra, palabras exactas y pares de palabras seguidas, en español e inglés
TEMAS = {
    "dolar": {"pref": ["dolar", "dollar", "divisa", "devalu", "revalu"], "pal": ["peso", "pesos", "trm", "usd", "cop"],
              "par": [("tasa", "cambio"), ("exchange", "rate")]},
    "politica_monetaria": {"pref": ["banrep", "inflac", "inflat"], "pal": [],
                           "par": [("banco", "republica"), ("central", "bank"), ("interest", "rate"), ("interest", "rates"),
                                   ("tasas", "interes"), ("tasa", "interes"), ("junta", "directiva")]},
    "petroleo": {"pref": ["petrol", "ecopetrol", "hidrocarbur", "fracking"], "pal": ["oil", "crude", "brent", "wti", "crudo"], "par": []},
    "fiscal": {"pref": ["tribut", "impuest", "fiscal", "deficit", "presupuest", "budget", "deuda"], "pal": ["tax", "taxes", "debt"],
               "par": [("tax", "reform")]},
    "calificacion": {"pref": ["calificad", "calificaci", "downgrad", "moody", "fitch"], "pal": [],
                     "par": [("grado", "inversion"), ("investment", "grade"), ("credit", "rating"), ("standard", "poor"), ("s", "p")]},
    "elecciones": {"pref": ["eleccion", "election", "elector", "candidat", "votacion", "encuesta"], "pal": ["vote", "votes", "voters", "poll", "polls", "votos"], "par": []},
    "gobierno": {"pref": ["presiden", "ministr", "gobierno", "government", "congres"], "pal": ["petro", "duque", "santos", "uribe"], "par": []},
    "violencia": {"pref": ["violen", "asesin", "masacr", "guerrill", "atentad", "terror", "secuestr", "kidnap", "homicid"],
                  "pal": ["eln", "farc", "attack", "attacks", "killed", "kills", "bomb", "muertos", "dead"], "par": []},
    "protestas": {"pref": ["protest", "manifestac", "huelga"], "pal": ["paro", "strike", "marcha", "bloqueos"], "par": [("paro", "nacional")]},
    "venezuela": {"pref": ["venezuel", "maduro", "migrant", "migrac", "migrat"], "pal": [], "par": []},
    "eeuu": {"pref": ["trump", "biden", "arancel", "tariff", "sanction"], "pal": ["fed", "eeuu"],
             "par": [("estados", "unidos"), ("united", "states"), ("federal", "reserve"), ("white", "house"), ("casa", "blanca")]},
}
# Tono: prefijos de palabras positivas y negativas (economía y noticias), en español e inglés
POSITIVAS = ["sube", "subi", "alza", "ganan", "gana", "crece", "crecim", "growth", "grow", "rise", "rises", "rising", "rally", "record",
             "recuper", "recover", "acuerdo", "agreement", "deal", "positiv", "mejor", "improv", "boom", "aprueb", "approv", "gains"]
NEGATIVAS = ["cae", "caen", "cayo", "caida", "desplom", "crisis", "fall", "falls", "drop", "plung", "slump", "loss", "perdid", "riesgo",
             "risk", "amenaz", "threat", "temor", "fear", "negativ", "peor", "worse", "recesi", "recession", "incertid", "uncertain",
             "default", "impago", "alarma", "alarm", "colaps", "collapse", "golpe", "hit", "hits"]
ES = {"de", "la", "el", "en", "los", "las", "del", "por", "para", "con", "que", "una", "un", "se", "al", "su", "y"}
EN = {"the", "of", "in", "to", "for", "and", "on", "with", "as", "is", "by", "from", "after", "over", "its", "at"}
EXT = re.compile(r"\.(html?|php|aspx?|shtml|cms|jsp)$")


def sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def texto_de_url(url: str) -> list[str]:
    """Las palabras del segmento del enlace que parece titular (el más largo con al menos tres palabras)."""
    try:
        ruta = unquote(urlparse(url).path)
    except ValueError:
        return []
    mejor = []
    for seg in ruta.split("/"):
        seg = EXT.sub("", sin_tildes(seg.lower()))
        palabras = [p for p in re.split(r"[-_+.,;:!?\s]+", seg) if p and not any(ch.isdigit() for ch in p)]
        if len(palabras) >= 3 and len(palabras) > len(mejor):
            mejor = palabras
    return mejor


def idioma(p: list[str]) -> str:
    es, en = sum(x in ES for x in p), sum(x in EN for x in p)
    return "es" if es > en else "en" if en > es else "otro"


def tiene(p: list[str], d: dict) -> bool:
    if any(x in d["pal"] for x in p) or any(x.startswith(tuple(d["pref"])) for x in p if d["pref"]):
        return True
    pares = set(zip(p, p[1:])) | set(zip(p, p[2:]))  # admite una palabra en medio: "tasa de cambio"
    return any(par in pares for par in d["par"])


def tono(p: list[str]) -> tuple[int, int]:
    return sum(x.startswith(tuple(POSITIVAS)) for x in p), sum(x.startswith(tuple(NEGATIVAS)) for x in p)


def agregar_dia(urls: list[str], fecha: pd.Timestamp) -> tuple[dict, list[str]]:
    fila = {"fecha": fecha, "enlaces": len(urls), "con_texto": 0, "es": 0, "en": 0, "pos": 0, "neg": 0, "con_tono": 0, "tono": 0.0}
    fila |= {f"t_{t}": 0 for t in TEMAS} | {f"tono_{t}": 0.0 for t in TEMAS}
    textos = []
    for u in urls:
        p = texto_de_url(u)
        if not p:
            continue
        fila["con_texto"] += 1
        fila[idioma(p)] = fila.get(idioma(p), 0) + 1
        po, ne = tono(p)
        fila["pos"] += po > 0
        fila["neg"] += ne > 0
        s = (po - ne) / (po + ne) if po + ne else 0.0
        if po + ne:
            fila["con_tono"] += 1
            fila["tono"] += s
        for t, d in TEMAS.items():
            if tiene(p, d):
                fila[f"t_{t}"] += 1
                fila[f"tono_{t}"] += s
        textos.append(" ".join(p))
    fila.pop("otro", None)
    fila["tono"] = fila["tono"] / fila["con_tono"] if fila["con_tono"] else None
    for t in TEMAS:
        fila[f"tono_{t}"] = fila[f"tono_{t}"] / fila[f"t_{t}"] if fila[f"t_{t}"] else None
    return fila, textos


def urls_colombia(contenido: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        crudo = z.read(z.namelist()[0])
    if len(crudo.split(b"\n", 1)[0].split(b"\t")) != eventos.N_COLUMNAS:
        sys.exit("Formato inesperado en el archivo de eventos.")
    c = eventos.COL
    d = pd.read_csv(io.BytesIO(crudo), sep="\t", header=None, usecols=[c["actor1_pais"], c["actor2_pais"], c["lugar_pais"], COL_URL],
                    dtype=str, quoting=3, on_bad_lines="skip")
    m = d[c["actor1_pais"]].eq("COL") | d[c["actor2_pais"]].eq("COL") | d[c["lugar_pais"]].eq("CO")
    return sorted(set(d.loc[m, COL_URL].dropna()))


def main():
    t0 = time.time()
    ayer = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    d = pd.read_csv(SALIDA, parse_dates=["fecha"]) if SALIDA.exists() else pd.DataFrame()
    hechos = set(d["fecha"]) if len(d) else set()
    pendientes = [f for f in pd.date_range(eventos.INICIO, ayer) if f not in hechos]
    print("días por procesar:", len(pendientes), flush=True)
    TEXTOS.mkdir(parents=True, exist_ok=True)
    s, nuevos, faltantes = requests.Session(), [], []
    for i, f in enumerate(pendientes):
        if time.time() - t0 > eventos.LIMITE_SEGUNDOS:
            break
        contenido = eventos.bajar(f, s)
        if contenido is None:
            faltantes.append(str(f.date()))
            continue
        fila, textos = agregar_dia(urls_colombia(contenido), f)
        nuevos.append(fila)
        with gzip.open(TEXTOS / f"{f:%Y%m%d}.txt.gz", "wt") as g:
            g.write("\n".join(textos))
        if len(nuevos) >= 30:
            d = pd.concat([d, pd.DataFrame(nuevos)], ignore_index=True)
            nuevos = []
            d.sort_values("fecha").to_csv(SALIDA, index=False)
            print("guardado hasta", f.date(), f"({i + 1}/{len(pendientes)})", flush=True)
    if nuevos:
        d = pd.concat([d, pd.DataFrame(nuevos)], ignore_index=True)
    d = d.sort_values("fecha")
    SALIDA.parent.mkdir(exist_ok=True)
    d.to_csv(SALIDA, index=False)
    hechos = set(d["fecha"])
    restan = [x for x in pd.date_range(eventos.INICIO, ayer) if x not in hechos and str(x.date()) not in faltantes]
    ESTADO.write_text(json.dumps({"completo": not restan, "hasta": str(d["fecha"].max().date()), "dias": int(len(d)),
                                  "dias_sin_archivo": faltantes[-30:]}, indent=2))
    print("titulares:", len(d), "días; faltan", len(restan), flush=True)


if __name__ == "__main__":
    main()
