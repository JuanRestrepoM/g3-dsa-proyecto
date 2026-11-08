"""
Tablero: brecha de desempeño Saber 11 por colegio + simulador del puntaje esperado.

Tiene tres vistas:

  Panorama   Filtros, tarjetas de resumen, listado de colegios, dispersión
             INSE vs. puntaje e histograma de brechas.
  Ficha      Detalle de un colegio: observado vs. esperado, puntajes por área y
             perfil socioeconómico frente a la mediana.
  Simulador  El usuario ingresa el perfil de un colegio y obtiene su puntaje
             esperado con un rango, consultando la API.

De dónde salen los datos (ver docs/guia_api.md):

  - Panorama, listado y ficha leen la tabla precalculada
    data/processed/colegios_con_esperado.parquet   (se trae con `dvc pull`).
  - El simulador llama a la API:  POST {URL_API}/api/v1/predict

Para correrlo, desde la raíz del repositorio y con la API levantada en otra terminal
(uvicorn src.api.main:app --port 8001):

    python src/dashboard/app.py

y abrir http://127.0.0.1:8050

Configuración por variables de entorno (todas opcionales):

    URL_API            Dirección de la API.        Por defecto http://localhost:8001
    COLEGIOS_PARQUET   Ruta de la tabla de colegios.
    HOST, PORT         Dónde escucha el tablero.   Por defecto 127.0.0.1 y 8050
    DASH_DEBUG         "1" para recarga automática al editar el código.
"""

from __future__ import annotations

import math
import os
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

RAIZ = Path(__file__).resolve().parents[2]

# La URL de la API NO va escrita en el código de las vistas: cuando todo pase a Docker
# se cambia aquí (o con la variable de entorno) y nada más.
URL_API = os.getenv("URL_API", "http://localhost:8001").rstrip("/")
RUTA_COLEGIOS = Path(
    os.getenv("COLEGIOS_PARQUET", RAIZ / "data" / "processed" / "colegios_con_esperado.parquet")
)
TIMEOUT_API = 10  # segundos para predecir
TIMEOUT_CORTO = 3  # segundos para /health y /opciones

TODOS = "__todos__"  # valor centinela de los desplegables "Todos"
MAX_FILAS_TABLA = 400

# Las 8 variables numéricas del perfil, en el orden del contrato de la API.
VARIABLES = [
    "inse_prom",
    "pct_madre_superior",
    "pct_padre_superior",
    "pct_estrato_1_2",
    "pct_estrato_5_6",
    "pct_internet",
    "pct_computador",
    "pct_libros_26mas",
]
ETIQUETAS = {
    "inse_prom": "Índice socioeconómico (INSE)",
    "pct_madre_superior": "% con madre de educación superior",
    "pct_padre_superior": "% con padre de educación superior",
    "pct_estrato_1_2": "% de estrato 1 o 2",
    "pct_estrato_5_6": "% de estrato 5 o 6",
    "pct_internet": "% con internet en casa",
    "pct_computador": "% con computador en casa",
    "pct_libros_26mas": "% con más de 25 libros en casa",
}
AREAS = {
    "punt_lectura_prom": "Lectura crítica",
    "punt_matematicas_prom": "Matemáticas",
    "punt_naturales_prom": "Ciencias naturales",
    "punt_sociales_prom": "Sociales",
    "punt_ingles_prom": "Inglés",
}
COLUMNAS_REQUERIDAS = (
    [
        "cole_cod_dane", "cole_nombre", "departamento", "municipio", "naturaleza",
        "zona", "jornada", "n_evaluados", "punt_global_prom",
        "esperado", "inferior", "superior", "residual",
    ]
    + list(AREAS)
    + VARIABLES
)

# Cuadrantes: combinan el contexto (INSE bajo/alto frente a la mediana) con el signo de
# la brecha cuando ésta supera el umbral elegido por el usuario.
CRITICA = "Prioridad crítica"
POTENCIAL = "Desperdicia potencial"
JOYA = "Joya: replicar prácticas"
CONSOLIDADO = "Consolidado"
ESPERADO = "En lo esperado"
ORDEN_CUADRANTES = [CRITICA, POTENCIAL, JOYA, CONSOLIDADO, ESPERADO]
COLOR = {
    CRITICA: "#c0392b",
    POTENCIAL: "#e59a92",
    JOYA: "#1f5fbf",
    CONSOLIDADO: "#8fb4e8",
    ESPERADO: "#c7c4bb",
}
ROJO, AZUL, GRIS = COLOR[CRITICA], COLOR[JOYA], "#8a877f"


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #

def normalizar(texto) -> str:
    """Sin tildes, en mayúsculas y con espacios colapsados: sirve para comparar."""
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", str(texto)) if unicodedata.category(c) != "Mn"
    )
    return " ".join(sin_tildes.upper().split())


def fmt(x, decimales: int = 0, signo: bool = False) -> str:
    """Número con formato colombiano: punto para miles y coma para decimales."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    s = f"{x:+,.{decimales}f}" if signo else f"{x:,.{decimales}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #

def cargar_colegios() -> pd.DataFrame:
    if not RUTA_COLEGIOS.exists():
        raise FileNotFoundError(
            f"No encuentro la tabla de colegios en {RUTA_COLEGIOS}.\n"
            "Corre primero:  dvc remote modify --local aws-remote allow_anonymous_login true"
            "  y luego  dvc pull"
        )
    df = pd.read_parquet(RUTA_COLEGIOS)
    faltan = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltan:
        raise KeyError(
            f"A la tabla le faltan columnas: {faltan}. Probablemente está desactualizada: "
            "corre `git pull` y `dvc pull`."
        )
    df = df.copy()
    for c in ("cole_nombre", "departamento", "municipio", "naturaleza", "zona", "jornada"):
        df[c] = df[c].astype(str).replace({"None": "Sin dato", "nan": "Sin dato"})
    df["brecha"] = df["residual"]

    # Un identificador único por fila para poder abrir la ficha desde la tabla o la gráfica.
    ident = df["cole_cod_dane"].astype(str)
    if ident.duplicated().any():
        ident = ident + "|" + df["jornada"]
    if ident.duplicated().any():
        ident = ident + "|" + ident.groupby(ident).cumcount().astype(str)
    df["id"] = ident
    df["_busqueda"] = (df["cole_nombre"] + " " + df["municipio"]).map(normalizar)
    return df.reset_index(drop=True)


COLEGIOS = cargar_colegios()
INDICE = COLEGIOS.set_index("id", drop=False)
SIGMA = float(COLEGIOS["brecha"].std())  # desviación de la brecha; 1 desv. ≈ 19 puntos
MEDIANA_INSE = float(COLEGIOS["inse_prom"].median())
MEDIANAS = COLEGIOS[VARIABLES].median()
PROMEDIO_AREAS = COLEGIOS[list(AREAS)].mean()
MODELO_TABLA = str(COLEGIOS["modelo"].mode().iloc[0]) if "modelo" in COLEGIOS else None


def asignar_cuadrante(df: pd.DataFrame, k: float) -> np.ndarray:
    """
    Clasifica cada colegio por su contexto y por su brecha.

    `k` es el umbral en desviaciones: una brecha cuenta cuando pasa de ±k·SIGMA.
    """
    bajo = (df["inse_prom"] <= MEDIANA_INSE).to_numpy()
    arriba = (df["brecha"] > k * SIGMA).to_numpy()
    abajo = (df["brecha"] < -k * SIGMA).to_numpy()
    return np.select(
        [abajo & bajo, abajo & ~bajo, arriba & bajo, arriba & ~bajo],
        [CRITICA, POTENCIAL, JOYA, CONSOLIDADO],
        default=ESPERADO,
    )


def seleccionar(depto, muni, natu, zona, jornada, minev, cuad, umbral, busqueda):
    """Aplica los filtros del panorama. Devuelve (tabla filtrada, umbral usado)."""
    k = float(umbral or 1.0)
    d = COLEGIOS.copy()
    d["cuadrante"] = asignar_cuadrante(d, k)

    m = np.ones(len(d), dtype=bool)
    for columna, valor in (
        ("departamento", depto), ("municipio", muni), ("naturaleza", natu),
        ("zona", zona), ("jornada", jornada), ("cuadrante", cuad),
    ):
        if valor not in (None, TODOS):
            m &= (d[columna] == valor).to_numpy()
    m &= (d["n_evaluados"] >= (minev or 0)).to_numpy()
    if busqueda and busqueda.strip():
        m &= d["_busqueda"].str.contains(normalizar(busqueda), regex=False).to_numpy()
    return d[m], k


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

def estado_api() -> dict | None:
    """Respuesta de /health, o None si la API no contesta."""
    try:
        r = requests.get(f"{URL_API}/health", timeout=TIMEOUT_CORTO)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


def opciones_formulario() -> tuple[dict, dict, str]:
    """
    Categorías válidas y rango de cada variable para armar el simulador.

    Se leen de la API (`/api/v1/opciones`) para que formulario y modelo no se
    desincronicen. Si la API no está disponible al arrancar, se usan los datos de la
    tabla: el simulador igual se dibuja y avisa cuando se intenta calcular.
    """
    categorias = {
        "zona": sorted(COLEGIOS["zona"].unique()),
        "departamento": sorted(COLEGIOS["departamento"].unique()),
    }
    rangos = {
        v: {
            "min": float(COLEGIOS[v].min()),
            "mediana": float(COLEGIOS[v].median()),
            "max": float(COLEGIOS[v].max()),
        }
        for v in VARIABLES
    }
    fuente = "datos"
    try:
        r = requests.get(f"{URL_API}/api/v1/opciones", timeout=TIMEOUT_CORTO)
        r.raise_for_status()
        op = r.json()
        categorias.update({k: v for k, v in op.get("categorias", {}).items() if v})
        for v in VARIABLES:
            if v in op.get("rangos", {}):
                rangos[v] = op["rangos"][v]
        fuente = "api"
    except (requests.RequestException, ValueError):
        pass
    return categorias, rangos, fuente


CATEGORIAS, RANGOS, FUENTE_OPCIONES = opciones_formulario()
EQUIVALENCIA_DEPTO = {normalizar(d): d for d in CATEGORIAS["departamento"]}
EQUIVALENCIA_ZONA = {normalizar(z): z for z in CATEGORIAS["zona"]}


def config_slider(variable: str) -> dict:
    """Mínimo, máximo, paso y valor inicial de un deslizador (los % se muestran de 0 a 100)."""
    r = RANGOS[variable]
    if variable == "inse_prom":
        lo, hi, med, paso = round(r["min"], 1), round(r["max"], 1), r["mediana"], 0.1
        return {"min": lo, "max": hi, "step": paso, "value": round(min(max(med, lo), hi), 1)}
    lo = math.floor(r["min"] * 100)
    hi = math.ceil(r["max"] * 100)
    return {"min": lo, "max": hi, "step": 1, "value": int(min(max(round(r["mediana"] * 100), lo), hi))}


CONFIG_SLIDERS = {v: config_slider(v) for v in VARIABLES}


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #

def plantilla(fig: go.Figure, alto: int, **extra) -> go.Figure:
    base = dict(
        template="plotly_white",
        height=alto,
        margin=dict(l=48, r=16, t=16, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#2b2a27"),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
    )
    base.update(extra)  # lo que pase quien llama (p. ej. otro margen) tiene prioridad
    fig.update_layout(**base)
    return fig


def figura_vacia(texto: str, alto: int) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=texto, showarrow=False, font=dict(size=13, color=GRIS))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return plantilla(fig, alto)


def figura_dispersion(d: pd.DataFrame) -> go.Figure:
    if d.empty:
        return figura_vacia("Ningún colegio cumple los filtros.", 380)
    fig = go.Figure()
    for nombre in [ESPERADO] + [c for c in ORDEN_CUADRANTES if c != ESPERADO]:
        s = d[d["cuadrante"] == nombre]
        if s.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=s["inse_prom"], y=s["punt_global_prom"], mode="markers", name=nombre,
                legendrank=ORDEN_CUADRANTES.index(nombre),
                marker=dict(size=6, color=COLOR[nombre], opacity=0.8),
                text=s["cole_nombre"],
                customdata=np.column_stack([s["id"], s["brecha"]]),
                hovertemplate=(
                    "<b>%{text}</b><br>INSE %{x:.1f} · puntaje %{y:.1f}"
                    "<br>Brecha %{customdata[1]:+.1f}<extra></extra>"
                ),
            )
        )
    if d["inse_prom"].nunique() > 2:
        pendiente, corte = np.polyfit(d["inse_prom"], d["esperado"], 1)
        xs = np.array([d["inse_prom"].min(), d["inse_prom"].max()])
        fig.add_trace(
            go.Scatter(
                x=xs, y=pendiente * xs + corte, mode="lines", name="Esperado (tendencia)",
                line=dict(color="#222", width=2), hoverinfo="skip",
            )
        )
    fig.add_vline(x=MEDIANA_INSE, line_dash="dot", line_color=GRIS)
    fig.update_xaxes(title="INSE promedio del colegio")
    fig.update_yaxes(title="Puntaje global promedio (0–500)")
    return plantilla(fig, 380, hovermode="closest")


def figura_brechas(d: pd.DataFrame, k: float) -> go.Figure:
    if d.empty:
        return figura_vacia("Ningún colegio cumple los filtros.", 280)
    inicio = math.floor(d["brecha"].min() / 5) * 5
    fin = math.ceil(d["brecha"].max() / 5) * 5
    bins = dict(start=inicio, end=max(fin, inicio + 5), size=5)
    fig = go.Figure()
    for nombre, serie, color in (
        ("Por debajo de lo esperado", d.loc[d["brecha"] < 0, "brecha"], ROJO),
        ("Por encima de lo esperado", d.loc[d["brecha"] >= 0, "brecha"], AZUL),
    ):
        fig.add_trace(
            go.Histogram(x=serie, xbins=bins, name=nombre, marker_color=color, opacity=0.85)
        )
    for x in (-k * SIGMA, k * SIGMA):
        fig.add_vline(x=x, line_dash="dash", line_color="#444")
    fig.update_xaxes(title=f"Brecha en puntos (líneas: ±{fmt(k, 2).rstrip('0').rstrip(',')} desv. = ±{fmt(k * SIGMA, 1)})")
    fig.update_yaxes(title="Colegios")
    return plantilla(fig, 280, barmode="stack", bargap=0.04)


def figura_rango(esperado, inferior, superior, real=None, alto=170) -> go.Figure:
    """Una franja con el rango esperado, el esperado (rombo) y, si hay, el observado."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[inferior, superior], y=[0, 0], mode="lines", name="Rango esperado",
            line=dict(width=16, color="#d6e3f7"), hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[esperado], y=[0], mode="markers", name="Esperado",
            marker=dict(symbol="diamond", size=16, color="#222"),
            hovertemplate="Esperado %{x:.1f}<extra></extra>",
        )
    )
    extremos = [inferior, superior]
    if real is not None:
        color = AZUL if real >= esperado else ROJO
        fig.add_trace(
            go.Scatter(
                x=[real], y=[0], mode="markers", name="Observado",
                marker=dict(size=16, color=color, line=dict(color="white", width=2)),
                hovertemplate="Observado %{x:.1f}<extra></extra>",
            )
        )
        extremos.append(real)
    fig.update_xaxes(range=[min(extremos) - 15, max(extremos) + 15], title="Puntaje global")
    fig.update_yaxes(visible=False, range=[-1, 1])
    return plantilla(fig, alto, margin=dict(l=16, r=16, t=16, b=44))


def figura_areas(fila: pd.Series) -> go.Figure:
    nombres = list(AREAS.values())
    fig = go.Figure()
    fig.add_trace(go.Bar(x=nombres, y=[fila[c] for c in AREAS], name="Este colegio", marker_color=AZUL))
    fig.add_trace(
        go.Bar(x=nombres, y=[PROMEDIO_AREAS[c] for c in AREAS], name="Promedio de colegios", marker_color="#c7c4bb")
    )
    fig.update_yaxes(title="Puntaje promedio")
    return plantilla(fig, 280, barmode="group")


def figura_simulador(resultado: dict, real: float | None) -> go.Figure:
    """Dónde cae el esperado (y la banda) frente al puntaje real de todos los colegios."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=COLEGIOS["punt_global_prom"], nbinsx=45, marker_color="#d9d6cd",
            name="Colegios reales", hovertemplate="%{x}: %{y} colegios<extra></extra>",
        )
    )
    fig.add_vrect(
        x0=resultado["inferior"], x1=resultado["superior"], fillcolor=AZUL, opacity=0.15,
        line_width=0, annotation_text="Rango esperado", annotation_position="top left",
    )
    fig.add_vline(x=resultado["esperado"], line_color="#222", line_width=3)
    if real is not None:
        fig.add_vline(
            x=real, line_color=AZUL if real >= resultado["esperado"] else ROJO,
            line_dash="dash", line_width=3, annotation_text="Observado",
        )
    fig.update_xaxes(title="Puntaje global promedio del colegio")
    fig.update_yaxes(title="Colegios")
    return plantilla(fig, 300, showlegend=False)


# --------------------------------------------------------------------------- #
# Piezas de interfaz
# --------------------------------------------------------------------------- #

def tarjeta_kpi(titulo, valor, detalle, clase="") -> html.Div:
    return html.Div(
        [html.Div(titulo, className="kpi-titulo"), html.Div(valor, className="kpi-valor"),
         html.Div(detalle, className="kpi-detalle")],
        className=f"kpi {clase}".strip(),
    )


def desplegable(id_, etiqueta, opciones, valor=TODOS, **extra) -> html.Div:
    return html.Div(
        [html.Label(etiqueta, className="filtro-etiqueta"),
         dcc.Dropdown(id=id_, options=opciones, value=valor, clearable=False, **extra)],
        className="filtro",
    )


def opciones_con_todos(serie: pd.Series, plural: str, con_conteo=True) -> list[dict]:
    valores = sorted(serie.dropna().unique())
    etiqueta = f"{plural} ({len(valores)})" if con_conteo else plural
    return [{"label": etiqueta, "value": TODOS}] + [{"label": v, "value": v} for v in valores]


def control_slider(variable: str) -> html.Div:
    cfg = CONFIG_SLIDERS[variable]
    return html.Div(
        [
            html.Label(ETIQUETAS[variable], className="filtro-etiqueta"),
            dcc.Slider(
                id=f"sim-{variable}", min=cfg["min"], max=cfg["max"], step=cfg["step"],
                value=cfg["value"], marks=None, updatemode="mouseup",
            ),
        ],
        className="control",
    )


COLUMNAS_TABLA = [
    {"name": "Colegio", "id": "cole_nombre"},
    {"name": "Municipio", "id": "municipio"},
    {"name": "Naturaleza", "id": "naturaleza"},
    {"name": "Eval.", "id": "n_evaluados", "type": "numeric"},
    {"name": "INSE", "id": "inse_prom", "type": "numeric"},
    {"name": "Observado", "id": "punt_global_prom", "type": "numeric"},
    {"name": "Esperado", "id": "esperado", "type": "numeric"},
    {"name": "Brecha", "id": "brecha", "type": "numeric"},
    {"name": "Cuadrante", "id": "cuadrante"},
]
ESTILO_TABLA = [
    {"if": {"filter_query": "{brecha} > 0", "column_id": "brecha"}, "color": AZUL, "fontWeight": "600"},
    {"if": {"filter_query": "{brecha} < 0", "column_id": "brecha"}, "color": ROJO, "fontWeight": "600"},
] + [
    {"if": {"filter_query": f'{{cuadrante}} = "{nombre}"', "column_id": "cuadrante"},
     "color": COLOR[nombre] if nombre != ESPERADO else GRIS, "fontWeight": "600"}
    for nombre in ORDEN_CUADRANTES
] + [
    {"if": {"state": "active"}, "backgroundColor": "#e8f0fb", "border": f"1px solid {AZUL}"},
]

IDS_FILTROS = [
    "f-depto", "f-muni", "f-natu", "f-zona", "f-jornada", "f-minev", "f-cuad", "f-umbral", "f-busqueda",
]
ENTRADAS_FILTROS = [Input(i, "value") for i in IDS_FILTROS]
ESTADOS_FILTROS = [State(i, "value") for i in IDS_FILTROS]


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

def vista_panorama() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    desplegable("f-depto", "Departamento", opciones_con_todos(COLEGIOS["departamento"], "Todos")),
                    desplegable("f-muni", "Municipio",
                                [{"label": "Elige un departamento", "value": TODOS}], disabled=True),
                    desplegable("f-natu", "Naturaleza", opciones_con_todos(COLEGIOS["naturaleza"], "Todas", False)),
                    desplegable("f-zona", "Zona", opciones_con_todos(COLEGIOS["zona"], "Todas", False)),
                    desplegable("f-jornada", "Jornada", opciones_con_todos(COLEGIOS["jornada"], "Todas", False)),
                    html.Div(
                        [html.Label("Mín. evaluados", className="filtro-etiqueta"),
                         dcc.Input(id="f-minev", type="number", debounce=True,
                                   min=int(COLEGIOS["n_evaluados"].min()),
                                   value=int(COLEGIOS["n_evaluados"].min()), className="entrada")],
                        className="filtro",
                    ),
                    desplegable(
                        "f-cuad", "Cuadrante",
                        [{"label": "Todos", "value": TODOS}] + [{"label": c, "value": c} for c in ORDEN_CUADRANTES],
                    ),
                    desplegable(
                        "f-umbral", "Umbral de brecha",
                        [{"label": "±1 desviación", "value": 1.0},
                         {"label": "±1,5 desviaciones", "value": 1.5},
                         {"label": "±1,96 desviaciones", "value": 1.96}],
                        valor=1.0,
                    ),
                    html.Button("Restablecer", id="btn-reset", className="boton boton-claro"),
                ],
                className="filtros",
            ),
            html.Details(
                [
                    html.Summary("¿Cómo leer la brecha y los cuadrantes?"),
                    html.P(
                        "Brecha = puntaje observado − puntaje esperado según el perfil socioeconómico. "
                        "Positiva: el colegio rinde por encima de su contexto; negativa: por debajo. "
                        "Un colegio entra a un cuadrante cuando su brecha pasa del umbral elegido; "
                        "el contexto es «bajo» si su INSE está en la mitad inferior de los colegios."
                    ),
                    html.Ul(
                        [
                            html.Li([html.B(CRITICA), ": contexto bajo y rinde por debajo de lo esperado. Primer candidato a acompañamiento."]),
                            html.Li([html.B(POTENCIAL), ": contexto alto y rinde por debajo de lo esperado."]),
                            html.Li([html.B(JOYA), ": contexto bajo y rinde por encima de lo esperado. Vale la pena documentar sus prácticas."]),
                            html.Li([html.B(CONSOLIDADO), ": contexto alto y rinde por encima de lo esperado."]),
                        ]
                    ),
                    html.P("La brecha es una alerta, no una calificación: también recoge lo que el modelo no observa.",
                           className="nota"),
                ],
                className="ayuda",
            ),
            html.Div(id="kpis", className="kpis"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [html.H3("Colegios de la selección"),
                                         html.P("Los de mayor brecha en valor absoluto, ordenados por brecha. "
                                                "Haz clic en una fila para abrir su ficha.", className="sub")],
                                    ),
                                    html.Div(
                                        [dcc.Input(id="f-busqueda", type="text", placeholder="Buscar colegio o municipio",
                                                   className="entrada"),
                                         html.Button("Exportar CSV", id="btn-csv", className="boton boton-claro")],
                                        className="acciones",
                                    ),
                                ],
                                className="tarjeta-cabecera",
                            ),
                            dash_table.DataTable(
                                id="tabla-colegios", columns=COLUMNAS_TABLA, data=[],
                                sort_action="native", sort_by=[{"column_id": "brecha", "direction": "desc"}],
                                page_action="none", fixed_rows={"headers": True},
                                style_table={"height": "470px", "overflowY": "auto", "overflowX": "auto"},
                                style_header={"fontWeight": "600", "backgroundColor": "#faf9f6", "whiteSpace": "normal", "height": "auto",
                                              "borderBottom": "1px solid #d9d6cd", "fontSize": "12px"},
                                style_cell={"fontFamily": "Inter, system-ui, sans-serif", "fontSize": "13px",
                                            "padding": "6px 6px", "textAlign": "left", "border": "none",
                                            "borderBottom": "1px solid #efede7", "maxWidth": "260px",
                                            "overflow": "hidden", "textOverflow": "ellipsis"},
                                style_cell_conditional=[
                                    {"if": {"column_id": c}, "textAlign": "right", "width": ancho, "minWidth": ancho, "maxWidth": ancho}
                                    for c, ancho in (("n_evaluados", "52px"), ("inse_prom", "52px"),
                                                     ("punt_global_prom", "74px"), ("esperado", "72px"),
                                                     ("brecha", "62px"))
                                ] + [
                                    {"if": {"column_id": "cole_nombre"}, "width": "165px", "minWidth": "165px", "maxWidth": "165px"},
                                    {"if": {"column_id": "naturaleza"}, "width": "84px", "minWidth": "84px"},
                                    {"if": {"column_id": "cuadrante"}, "width": "175px", "minWidth": "175px", "maxWidth": "175px"},
                                ],
                                style_data_conditional=ESTILO_TABLA,
                            ),
                            html.P(id="tabla-pie", className="sub pie"),
                        ],
                        className="tarjeta",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [html.H3("Puntaje observado frente al contexto socioeconómico"),
                                 html.P("Cada punto es un colegio. La línea es el esperado según el modelo; la distancia "
                                        "vertical a ella es la brecha. Clic en un punto para abrir su ficha.", className="sub"),
                                 dcc.Graph(id="grafico-dispersion", config={"displayModeBar": False})],
                                className="tarjeta",
                            ),
                            html.Div(
                                [html.H3("Distribución de brechas"),
                                 html.P("Colegios por tramo de 5 puntos.", className="sub"),
                                 dcc.Graph(id="grafico-brechas", config={"displayModeBar": False})],
                                className="tarjeta",
                            ),
                        ],
                        className="columna",
                    ),
                ],
                className="rejilla",
            ),
        ],
        id="vista-panorama",
    )


def vista_ficha() -> html.Div:
    opciones = [
        {"label": f"{r.cole_nombre} · {r.municipio} ({r.departamento})", "value": r.id}
        for r in COLEGIOS.sort_values("cole_nombre").itertuples()
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [html.Label("Colegio", className="filtro-etiqueta"),
                         dcc.Dropdown(id="ficha-colegio", options=opciones, value=None, clearable=True,
                                      placeholder="Busca un colegio por nombre…")],
                        className="filtro ancho",
                    ),
                    html.Button("Simular este perfil →", id="btn-simular-perfil", disabled=True, className="boton"),
                ],
                className="filtros",
            ),
            html.Div(id="ficha-contenido"),
        ],
        id="vista-ficha",
        style={"display": "none"},
    )


def vista_simulador() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Perfil del colegio"),
                    html.P("Mueve los controles para describir el colegio. Los porcentajes se muestran de 0 a 100; "
                           "el tablero los convierte a proporciones antes de consultar la API.", className="sub"),
                    html.Div([control_slider(v) for v in VARIABLES], className="controles"),
                    html.Div(
                        [
                            html.Div(
                                [html.Label("Zona", className="filtro-etiqueta"),
                                 dcc.Dropdown(id="sim-zona", options=CATEGORIAS["zona"], value="URBANO"
                                              if "URBANO" in CATEGORIAS["zona"] else CATEGORIAS["zona"][0],
                                              clearable=False)],
                                className="filtro",
                            ),
                            html.Div(
                                [html.Label("Departamento", className="filtro-etiqueta"),
                                 dcc.Dropdown(id="sim-departamento", options=CATEGORIAS["departamento"],
                                              value=EQUIVALENCIA_DEPTO.get("CUNDINAMARCA", CATEGORIAS["departamento"][0]),
                                              clearable=False)],
                                className="filtro",
                            ),
                            html.Div(
                                [html.Label("Puntaje real (opcional)", className="filtro-etiqueta"),
                                 dcc.Input(id="sim-real", type="number", min=0, max=500, step=0.1,
                                           placeholder="Cuando llegue el resultado", className="entrada")],
                                className="filtro",
                            ),
                        ],
                        className="filtros",
                    ),
                    html.Button("Calcular puntaje esperado", id="sim-calcular", className="boton boton-grande"),
                    html.P(f"Opciones del formulario tomadas de: {'la API' if FUENTE_OPCIONES == 'api' else 'la tabla de datos (la API no respondió al arrancar)'}.",
                           className="sub pie"),
                ],
                className="tarjeta",
            ),
            html.Div(
                [html.H3("Puntaje esperado"), html.Div(id="sim-resultado",
                    children=html.P("Describe un colegio y pulsa «Calcular puntaje esperado».", className="sub"))],
                className="tarjeta",
            ),
        ],
        id="vista-simulador",
        className="rejilla",
        style={"display": "none"},
    )


def crear_layout() -> html.Div:
    return html.Div(
        [
            html.Header(
                [
                    html.Div(
                        [html.H1("Brecha de desempeño Saber 11 por colegio"),
                         html.P(id="resumen-texto", className="resumen")],
                        className="cabecera-texto",
                    ),
                    html.Div(
                        [html.Div(id="estado-api", className="estado"),
                         html.P("Periodo 2020-2 · colegios con 20 o más evaluados. "
                                "Brecha = puntaje observado − puntaje esperado según el perfil socioeconómico.",
                                className="sub derecha")],
                        className="cabecera-lado",
                    ),
                ],
                className="cabecera",
            ),
            dcc.Tabs(
                id="tabs", value="panorama", className="pestanas",
                children=[
                    dcc.Tab(label="Panorama", value="panorama", className="pestana", selected_className="pestana-activa"),
                    dcc.Tab(label="Ficha del colegio", value="ficha", className="pestana", selected_className="pestana-activa"),
                    dcc.Tab(label="Simulador", value="simulador", className="pestana", selected_className="pestana-activa"),
                ],
            ),
            vista_panorama(),
            vista_ficha(),
            vista_simulador(),
            dcc.Download(id="descarga"),
            dcc.Interval(id="intervalo-salud", interval=20_000),
        ],
        className="pagina",
    )


app = Dash(__name__, title="Brecha Saber 11", suppress_callback_exceptions=False)
server = app.server  # para desplegar con gunicorn:  gunicorn src.dashboard.app:server
app.layout = crear_layout


# --------------------------------------------------------------------------- #
# Callbacks: navegación y estado
# --------------------------------------------------------------------------- #

@app.callback(
    Output("vista-panorama", "style"),
    Output("vista-ficha", "style"),
    Output("vista-simulador", "style"),
    Output("resumen-texto", "style"),
    Input("tabs", "value"),
)
def mostrar_vista(pestana):
    estilos = [{} if pestana == v else {"display": "none"} for v in ("panorama", "ficha", "simulador")]
    return [*estilos, {} if pestana == "panorama" else {"display": "none"}]  # el resumen es del panorama


@app.callback(Output("estado-api", "children"), Input("intervalo-salud", "n_intervals"))
def mostrar_estado_api(_):
    """Avisa si la API está caída antes de que el usuario pida su primera simulación."""
    salud = estado_api()
    if salud is None or salud.get("estado") != "ok":
        return html.Span(
            f"API sin conexión ({URL_API}). Levántala con: uvicorn src.api.main:app --port 8001",
            className="insignia insignia-mala",
        )
    texto = f"API conectada · modelo {salud.get('modelo')}"
    if salud.get("rmse"):
        texto += f" · RMSE {fmt(salud['rmse'], 1)}"
    hijos = [html.Span(texto, className="insignia insignia-buena")]
    if MODELO_TABLA and salud.get("modelo") and MODELO_TABLA != salud["modelo"]:
        hijos.append(
            html.Span(
                f"La tabla precalculada usa «{MODELO_TABLA}» y la API sirve «{salud['modelo']}»: "
                "hay que volver a correr src/models/predecir_colegios.py.",
                className="insignia insignia-aviso",
            )
        )
    return hijos


# --------------------------------------------------------------------------- #
# Callbacks: panorama
# --------------------------------------------------------------------------- #

@app.callback(
    Output("f-muni", "options"),
    Output("f-muni", "value"),
    Output("f-muni", "disabled"),
    Input("f-depto", "value"),
)
def municipios_del_departamento(depto):
    if depto in (None, TODOS):
        return [{"label": "Elige un departamento", "value": TODOS}], TODOS, True
    munis = COLEGIOS.loc[COLEGIOS["departamento"] == depto, "municipio"]
    return opciones_con_todos(munis, "Todos"), TODOS, False


@app.callback(
    Output("f-depto", "value"), Output("f-natu", "value"), Output("f-zona", "value"),
    Output("f-jornada", "value"), Output("f-minev", "value"), Output("f-cuad", "value"),
    Output("f-umbral", "value"), Output("f-busqueda", "value"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def restablecer_filtros(_):
    return TODOS, TODOS, TODOS, TODOS, int(COLEGIOS["n_evaluados"].min()), TODOS, 1.0, ""


@app.callback(
    Output("resumen-texto", "children"),
    Output("kpis", "children"),
    Output("tabla-colegios", "data"),
    Output("tabla-pie", "children"),
    Output("grafico-dispersion", "figure"),
    Output("grafico-brechas", "figure"),
    *ENTRADAS_FILTROS,
)
def actualizar_panorama(*filtros):
    d, k = seleccionar(*filtros)
    n = len(d)
    k_txt = fmt(k, 2).rstrip("0").rstrip(",")

    n_critica = int((d["cuadrante"] == CRITICA).sum())
    n_joya = int((d["cuadrante"] == JOYA).sum())
    n_arriba = int((d["brecha"] > k * SIGMA).sum())
    n_abajo = int((d["brecha"] < -k * SIGMA).sum())

    # Resumen de una frase, siempre calculada sobre la selección.
    if n == 0:
        resumen = "Ningún colegio cumple los filtros actuales. Prueba con «Restablecer»."
    else:
        resumen = (
            f"{fmt(n_arriba + n_abajo)} colegios se alejan más de {fmt(k * SIGMA, 1)} puntos (±{k_txt} desv.) "
            f"de lo esperado para su contexto; {fmt(n_critica)} están en contexto adverso con prioridad de "
            f"acompañamiento y {fmt(n_joya)} rinden muy por encima pese a un contexto adverso."
        )
        if n_critica:
            top = d.loc[d["cuadrante"] == CRITICA, "departamento"].value_counts()
            resumen += f" Mayor concentración de prioridad crítica: {top.index[0]} ({int(top.iloc[0])})."

    sin_filtros = n == len(COLEGIOS)
    r = d["inse_prom"].corr(d["punt_global_prom"]) if n > 2 else float("nan")
    kpis = [
        tarjeta_kpi("Colegios en la selección", fmt(n),
                    "Todos los colegios del periodo" if sin_filtros else "Los que cumplen los filtros", "kpi-negro"),
        tarjeta_kpi("Por encima de lo esperado", fmt(n_arriba),
                    f"{fmt(100 * n_arriba / n, 1) if n else '—'} % · brecha mayor a +{k_txt} desv.", "kpi-azul"),
        tarjeta_kpi("Por debajo de lo esperado", fmt(n_abajo),
                    f"{fmt(100 * n_abajo / n, 1) if n else '—'} % · brecha menor a −{k_txt} desv.", "kpi-rojo"),
        tarjeta_kpi("Desviación de la brecha", f"{fmt(d['brecha'].std(), 1)} puntos" if n > 1 else "—",
                    f"Correlación INSE–puntaje: {fmt(r, 2)}", "kpi-gris"),
    ]

    # Los de mayor |brecha|, ordenados de mayor a menor brecha.
    top = d.reindex(d["brecha"].abs().sort_values(ascending=False).index).head(MAX_FILAS_TABLA)
    top = top.sort_values("brecha", ascending=False)
    cols = ["id", "cole_nombre", "municipio", "naturaleza", "n_evaluados", "inse_prom",
            "punt_global_prom", "esperado", "brecha", "cuadrante"]
    tabla = top[cols].round({"inse_prom": 1, "punt_global_prom": 1, "esperado": 1, "brecha": 1})
    pie = f"Mostrando {fmt(len(tabla))} de {fmt(n)} colegios. Exporta a CSV para obtener la lista completa."

    return resumen, kpis, tabla.to_dict("records"), pie, figura_dispersion(d), figura_brechas(d, k)


@app.callback(
    Output("descarga", "data"),
    Input("btn-csv", "n_clicks"),
    *ESTADOS_FILTROS,
    prevent_initial_call=True,
)
def exportar_csv(_, *filtros):
    d, _k = seleccionar(*filtros)
    cols = ["cole_cod_dane", "cole_nombre", "departamento", "municipio", "naturaleza", "zona", "jornada",
            "n_evaluados", "inse_prom", "punt_global_prom", "esperado", "inferior", "superior", "brecha", "cuadrante"]
    return dcc.send_data_frame(d[cols].sort_values("brecha", ascending=False).to_csv,
                               "colegios_seleccion.csv", index=False)


# --------------------------------------------------------------------------- #
# Callbacks: ficha
# --------------------------------------------------------------------------- #

@app.callback(
    Output("ficha-colegio", "value"),
    Output("tabs", "value"),
    Input("tabla-colegios", "active_cell"),
    Input("grafico-dispersion", "clickData"),
    prevent_initial_call=True,
)
def abrir_ficha(celda, clic):
    """Un clic en la tabla o en un punto de la gráfica abre la ficha de ese colegio."""
    if ctx.triggered_id == "tabla-colegios" and celda and celda.get("row_id") in INDICE.index:
        return celda["row_id"], "ficha"
    if ctx.triggered_id == "grafico-dispersion" and clic:
        dato = clic["points"][0].get("customdata")
        ident = dato[0] if isinstance(dato, (list, tuple)) else dato
        if ident in INDICE.index:
            return ident, "ficha"
    return no_update, no_update


def fila_perfil(variable: str, fila: pd.Series) -> html.Tr:
    if variable == "inse_prom":
        v, m = fmt(fila[variable], 1), fmt(MEDIANAS[variable], 1)
    else:
        v, m = f"{fmt(100 * fila[variable], 0)} %", f"{fmt(100 * MEDIANAS[variable], 0)} %"
    return html.Tr([html.Td(ETIQUETAS[variable]), html.Td(v, className="num"), html.Td(m, className="num")])


@app.callback(
    Output("ficha-contenido", "children"),
    Output("btn-simular-perfil", "disabled"),
    Input("ficha-colegio", "value"),
    Input("f-umbral", "value"),
)
def mostrar_ficha(ident, umbral):
    if not ident or ident not in INDICE.index:
        return html.Div(
            html.P("Elige un colegio en el buscador, o haz clic en una fila de la tabla del panorama.", className="sub"),
            className="tarjeta",
        ), True

    fila = INDICE.loc[ident]
    k = float(umbral or 1.0)
    cuadrante = asignar_cuadrante(fila.to_frame().T.astype({"inse_prom": float, "brecha": float}), k)[0]
    brecha = float(fila["brecha"])
    clase_brecha = "kpi-azul" if brecha >= 0 else "kpi-rojo"

    kpis = html.Div(
        [
            tarjeta_kpi("Puntaje observado", fmt(fila["punt_global_prom"], 1), "Promedio global del colegio", "kpi-negro"),
            tarjeta_kpi("Puntaje esperado", fmt(fila["esperado"], 1),
                        f"Rango {fmt(fila['inferior'], 1)} – {fmt(fila['superior'], 1)}", "kpi-gris"),
            tarjeta_kpi("Brecha", fmt(brecha, 1, signo=True), "observado − esperado", clase_brecha),
            tarjeta_kpi("Cuadrante", cuadrante, str(fila.get("clasificacion", "")), "kpi-gris"),
        ],
        className="kpis",
    )

    contenido = html.Div(
        [
            html.Div(
                [html.H2(fila["cole_nombre"]),
                 html.P(f"{fila['municipio']}, {fila['departamento']} · {fila['naturaleza']} · "
                        f"zona {str(fila['zona']).lower()} · jornada {str(fila['jornada']).lower()} · "
                        f"{fmt(fila['n_evaluados'])} evaluados · código DANE {fila['cole_cod_dane']}", className="sub")],
                className="tarjeta",
            ),
            kpis,
            html.Div(
                [
                    html.Div(
                        [html.H3("Observado frente al rango esperado"),
                         html.P("Si el punto observado cae fuera de la franja, el colegio se aparta de lo que su contexto haría prever.", className="sub"),
                         dcc.Graph(figure=figura_rango(fila["esperado"], fila["inferior"], fila["superior"],
                                                       float(fila["punt_global_prom"])),
                                   config={"displayModeBar": False})],
                        className="tarjeta",
                    ),
                    html.Div(
                        [html.H3("Puntajes por área"),
                         dcc.Graph(figure=figura_areas(fila), config={"displayModeBar": False})],
                        className="tarjeta",
                    ),
                ],
                className="rejilla",
            ),
            html.Div(
                [html.H3("Perfil socioeconómico"),
                 html.P("Lo que el modelo usa para calcular el esperado, frente a la mediana de todos los colegios.", className="sub"),
                 html.Table(
                     [html.Thead(html.Tr([html.Th("Variable"), html.Th("Este colegio", className="num"),
                                          html.Th("Mediana", className="num")])),
                      html.Tbody([fila_perfil(v, fila) for v in VARIABLES])],
                     className="tabla-simple",
                 )],
                className="tarjeta",
            ),
        ]
    )
    return contenido, False


# --------------------------------------------------------------------------- #
# Callbacks: simulador
# --------------------------------------------------------------------------- #

@app.callback(
    *[Output(f"sim-{v}", "value") for v in VARIABLES],
    Output("sim-zona", "value"),
    Output("sim-departamento", "value"),
    Output("tabs", "value", allow_duplicate=True),
    Input("btn-simular-perfil", "n_clicks"),
    State("ficha-colegio", "value"),
    prevent_initial_call=True,
)
def cargar_perfil_en_simulador(_, ident):
    """Lleva el perfil del colegio abierto en la ficha al simulador."""
    if not ident or ident not in INDICE.index:
        return [no_update] * (len(VARIABLES) + 3)
    fila = INDICE.loc[ident]
    valores = []
    for v in VARIABLES:
        cfg = CONFIG_SLIDERS[v]
        x = round(float(fila[v]), 1) if v == "inse_prom" else round(100 * float(fila[v]))
        valores.append(min(max(x, cfg["min"]), cfg["max"]))
    zona = EQUIVALENCIA_ZONA.get(normalizar(fila["zona"]), no_update)
    depto = EQUIVALENCIA_DEPTO.get(normalizar(fila["departamento"]), no_update)
    return [*valores, zona, depto, "simulador"]


def mensaje_error(titulo: str, detalle: str | None = None) -> html.Div:
    return html.Div([html.B(titulo), html.P(detalle) if detalle else None], className="alerta")


@app.callback(
    Output("sim-resultado", "children"),
    Input("sim-calcular", "n_clicks"),
    *[State(f"sim-{v}", "value") for v in VARIABLES],
    State("sim-zona", "value"),
    State("sim-departamento", "value"),
    State("sim-real", "value"),
    prevent_initial_call=True,
)
def calcular(_, *valores):
    *numericos, zona, depto, real = valores
    perfil = {
        v: (x if v == "inse_prom" else x / 100)  # la API espera proporciones de 0 a 1
        for v, x in zip(VARIABLES, numericos)
    }
    perfil.update({"zona": zona, "departamento": depto})

    try:
        r = requests.post(f"{URL_API}/api/v1/predict", json=perfil, timeout=TIMEOUT_API)
    except requests.RequestException:
        return mensaje_error(
            "No se pudo conectar con la API.",
            f"Busqué en {URL_API}. Levántala en otra terminal con:  uvicorn src.api.main:app --port 8001",
        )

    if r.status_code == 503:
        return mensaje_error("La API no tiene un modelo entrenado.",
                             "Falta correr el entrenamiento: python src/models/entrenar_lineales.py")
    if r.status_code == 422:
        detalle = r.json().get("detail")
        if isinstance(detalle, dict):
            return mensaje_error(detalle.get("mensaje", "Valor no válido."))
        if isinstance(detalle, list) and detalle:
            campo = ".".join(str(p) for p in detalle[0].get("loc", [])[1:])
            return mensaje_error(f"Valor no válido en «{campo}».", detalle[0].get("msg"))
        return mensaje_error("La API rechazó el perfil.")
    if not r.ok:
        return mensaje_error(f"La API respondió con el error {r.status_code}.")

    res = r.json()
    texto_real = None
    if real is not None:
        if real > res["superior"]:
            texto_real = (f"Observado {fmt(real, 1)}: por encima de lo esperado "
                          f"({fmt(real - res['esperado'], 1, signo=True)} puntos sobre el esperado).")
        elif real < res["inferior"]:
            texto_real = (f"Observado {fmt(real, 1)}: por debajo de lo esperado "
                          f"({fmt(real - res['esperado'], 1, signo=True)} puntos frente al esperado).")
        else:
            texto_real = (f"Observado {fmt(real, 1)}: dentro del rango, es decir, en lo esperado "
                          f"({fmt(real - res['esperado'], 1, signo=True)} puntos frente al esperado).")

    return html.Div(
        [
            html.Div(fmt(res["esperado"], 1), className="resultado-valor"),
            html.Div(f"Rango esperado: {fmt(res['inferior'], 1)} – {fmt(res['superior'], 1)} puntos",
                     className="resultado-rango"),
            html.P(texto_real, className="resultado-real") if texto_real else None,
            dcc.Graph(figure=figura_simulador(res, real), config={"displayModeBar": False}),
            html.P(
                f"Modelo: {res['modelo']}. El rango es el esperado ± 1,96 veces la desviación del error del "
                "modelo: es ancho a propósito, porque refleja cuánto del puntaje de un colegio no explica "
                "su contexto socioeconómico. Léelo como una banda, no como un número exacto.",
                className="sub",
            ),
        ]
    )


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8050")),
        debug=os.getenv("DASH_DEBUG", "0") == "1",
    )
