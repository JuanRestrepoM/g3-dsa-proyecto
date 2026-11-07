"""
Protocolo común para guardar y comparar modelos.

Todos los modelos del proyecto —lineales, árboles, lo que venga— se guardan con
`guardar_modelo()` de este módulo. No es burocracia: la función es la que calcula
las métricas, así que nadie puede usar sin darse cuenta una validación cruzada
distinta y terminar comparando números que no son comparables.

Lo que garantiza:

  * Las mismas 5 particiones con random_state=42 para todo el mundo.
  * Las mismas métricas (RMSE, MAE, R²) calculadas igual.
  * Un .json con la misma forma, que la API sabe leer.
  * Un .joblib que lleva su preprocesamiento adentro.

Cómo se usa (ver docs/contrato_modelos.md para el ejemplo completo):

    from src.models import ficha

    X, y = ficha.cargar_datos()
    mi_pipeline = Pipeline([("pre", mi_preprocesamiento), ("modelo", LGBMRegressor(...))])
    ficha.guardar_modelo("lightgbm", mi_pipeline, X, y,
                         autor="Javier Cañon",
                         hiperparametros={"n_estimators": 400, "max_depth": 6})

Para revisar que un modelo quedó bien, desde la raíz del repositorio:

    python src/models/ficha.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, cross_val_predict, cross_validate

RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATOS = RAIZ / "data" / "processed" / "colegios_2020_2.parquet"
CARPETA_MODELOS = RAIZ / "models"

# El protocolo acordado en docs/acuerdos_entrega2.md. Si esto cambia, cambia para
# todos a la vez, y hay que volver a entrenar todos los modelos.
N_PARTICIONES = 5
SEMILLA = 42

OBJETIVO = "punt_global_prom"

PERFIL = [
    "inse_prom",
    "pct_madre_superior",
    "pct_padre_superior",
    "pct_estrato_1_2",
    "pct_estrato_5_6",
    "pct_internet",
    "pct_computador",
    "pct_libros_26mas",
]
CATEGORICAS = ["zona", "departamento"]
PREDICTORAS = PERFIL + CATEGORICAS

# Para el intervalo de la API: esperado ± Z * sigma.
Z = 1.96


def particiones() -> KFold:
    return KFold(n_splits=N_PARTICIONES, shuffle=True, random_state=SEMILLA)


def cargar_datos() -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (X, y) con las columnas acordadas. Úsala en vez de leer el parquet a mano."""
    colegios = pd.read_parquet(RUTA_DATOS)
    faltan = [c for c in PREDICTORAS + [OBJETIVO] if c not in colegios.columns]
    if faltan:
        raise KeyError(f"A la tabla le faltan columnas: {faltan}. ¿Hiciste dvc pull?")
    return colegios[PREDICTORAS], colegios[OBJETIVO]


def evaluar(pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Valida el pipeline completo y devuelve las tres métricas acordadas."""
    puntajes = cross_validate(
        pipeline,
        X,
        y,
        cv=particiones(),
        scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
    )
    return {
        "rmse": float(-puntajes["test_neg_root_mean_squared_error"].mean()),
        "rmse_std": float(puntajes["test_neg_root_mean_squared_error"].std()),
        "mae": float(-puntajes["test_neg_mean_absolute_error"].mean()),
        "r2": float(puntajes["test_r2"].mean()),
        "r2_std": float(puntajes["test_r2"].std()),
    }


def diagnosticar_residuales(pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Mide los residuales fuera de muestra.

    La API arma el intervalo como esperado ± 1.96 * sigma, que es lo correcto solo si
    los residuales se parecen a una normal. `cobertura_1.96_sigma` dice qué proporción
    cae de verdad dentro: si se aleja mucho de 0.95, el intervalo está mintiendo.

    Se usa cross_val_predict y no los residuales de entrenamiento, porque esos
    siempre se ven mejor de lo que son.
    """
    predicho = cross_val_predict(pipeline, X, y, cv=particiones())
    residual = y.to_numpy() - predicho
    sigma = float(residual.std(ddof=1))
    return {
        "sigma_residual": sigma,
        "asimetria": float(stats.skew(residual)),
        "curtosis_exceso": float(stats.kurtosis(residual)),
        "cobertura_1.96_sigma": float(np.mean(np.abs(residual) <= Z * sigma)),
        "amplitud_intervalo": float(2 * Z * sigma),
    }


def guardar_modelo(
    nombre: str,
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    autor: str,
    hiperparametros: dict | None = None,
    carpeta: Path = CARPETA_MODELOS,
) -> Path:
    """
    Evalúa el pipeline, lo reentrena con todos los colegios y lo guarda.

    `pipeline` debe llegar **sin entrenar** y debe incluir su propio preprocesamiento:
    quien lo cargue le pasa un DataFrame con las diez columnas crudas y obtiene la
    predicción, sin repetir ninguna transformación. La API depende de eso.

    `nombre` es el del archivo y el que la API reporta en su respuesta: 'ridge',
    'lightgbm', 'random_forest'. En minúsculas y sin espacios.

    Devuelve la ruta del .joblib.
    """
    faltan = [c for c in PREDICTORAS if c not in X.columns]
    if faltan:
        raise KeyError(f"A X le faltan columnas del acuerdo: {faltan}")

    metricas = evaluar(pipeline, X, y)
    residuales = diagnosticar_residuales(pipeline, X, y)
    pipeline.fit(X, y)

    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_modelo = carpeta / f"{nombre}.joblib"
    joblib.dump(pipeline, ruta_modelo)

    contenido = {
        "modelo": nombre,
        "autor": autor,
        "entrenado": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "hiperparametros": hiperparametros or {},
        "objetivo": OBJETIVO,
        "variables": PREDICTORAS,
        "n_colegios": int(len(X)),
        "evaluacion": {
            "metodo": f"validación cruzada de {N_PARTICIONES} particiones",
            "random_state": SEMILLA,
            "rmse": round(metricas["rmse"], 4),
            "rmse_std": round(metricas["rmse_std"], 4),
            "mae": round(metricas["mae"], 4),
            "r2": round(metricas["r2"], 4),
            "r2_std": round(metricas["r2_std"], 4),
        },
        # La API lee sigma_residual para el ancho del intervalo.
        "residuales": {k: round(v, 4) for k, v in residuales.items()},
        # La API valida contra esto, y el tablero llena su formulario con esto.
        "categorias_validas": {c: sorted(X[c].dropna().unique().tolist()) for c in CATEGORICAS},
        "rangos_observados": {
            c: {
                "min": round(float(X[c].min()), 4),
                "mediana": round(float(X[c].median()), 4),
                "max": round(float(X[c].max()), 4),
            }
            for c in PERFIL
        },
    }
    (carpeta / f"{nombre}.json").write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return ruta_modelo


# ─────────────────────────── revisión ───────────────────────────

CAMPOS = {
    "modelo": str,
    "autor": str,
    "entrenado": str,
    "variables": list,
    "n_colegios": int,
}


def revisar(ruta_json: Path) -> list[str]:
    """
    Revisa que un modelo guardado cumpla el contrato. Devuelve la lista de problemas;
    vacía quiere decir que está bien.

    No se queda en mirar el .json: carga el .joblib y le pide una predicción real,
    que es la única forma de saber que la API va a poder servirlo.
    """
    problemas = []
    try:
        ficha = json.loads(ruta_json.read_text(encoding="utf-8"))
    except Exception as error:
        return [f"no se pudo leer el .json: {error}"]

    for campo, tipo in CAMPOS.items():
        if campo not in ficha:
            problemas.append(f"falta '{campo}'")
        elif not isinstance(ficha[campo], tipo):
            problemas.append(f"'{campo}' debería ser {tipo.__name__}")

    for metrica in ("rmse", "mae", "r2"):
        if not isinstance(ficha.get("evaluacion", {}).get(metrica), (int, float)):
            problemas.append(f"falta 'evaluacion.{metrica}' o no es un número")

    semilla = ficha.get("evaluacion", {}).get("random_state")
    if semilla != SEMILLA:
        problemas.append(f"'evaluacion.random_state' es {semilla}, debería ser {SEMILLA}")

    sigma = ficha.get("residuales", {}).get("sigma_residual")
    if not isinstance(sigma, (int, float)) or sigma <= 0:
        problemas.append("falta 'residuales.sigma_residual' o no es positivo")

    if ficha.get("variables") != PREDICTORAS:
        problemas.append("'variables' no coincide con las diez acordadas")

    categorias = ficha.get("categorias_validas", {})
    for columna in CATEGORICAS:
        valores = categorias.get(columna)
        if not isinstance(valores, list) or not valores:
            problemas.append(f"falta 'categorias_validas.{columna}' o está vacío")

    ruta_modelo = ruta_json.with_suffix(".joblib")
    if not ruta_modelo.exists():
        problemas.append(f"no existe {ruta_modelo.name} al lado del .json")
        return problemas

    # La prueba de fuego: cargarlo y pedirle una predicción como lo haría la API.
    try:
        pipeline = joblib.load(ruta_modelo)
        rangos = ficha.get("rangos_observados", {})
        fila = {c: rangos.get(c, {}).get("mediana", 0.5) for c in PERFIL}
        for columna in CATEGORICAS:
            fila[columna] = categorias[columna][0]
        valor = float(pipeline.predict(pd.DataFrame([fila]))[0])
        if not np.isfinite(valor):
            problemas.append(f"la predicción de prueba dio {valor}")
        elif not 0 <= valor <= 500:
            problemas.append(f"la predicción de prueba ({valor:.1f}) se sale de la escala 0–500")
    except Exception as error:
        problemas.append(f"el .joblib no pudo predecir sobre las diez columnas crudas: {error}")

    return problemas


def main() -> int:
    fichas = sorted(CARPETA_MODELOS.glob("*.json"))
    if not fichas:
        print(f"No hay modelos en {CARPETA_MODELOS}.")
        return 1

    print(f"Revisando {len(fichas)} modelo(s) en {CARPETA_MODELOS.relative_to(RAIZ)}\n")
    con_problemas = 0
    filas = []
    for ruta in fichas:
        problemas = revisar(ruta)
        if problemas:
            con_problemas += 1
            print(f"  [X] {ruta.stem}")
            for p in problemas:
                print(f"        - {p}")
        else:
            ficha = json.loads(ruta.read_text(encoding="utf-8"))
            evaluacion = ficha["evaluacion"]
            filas.append((evaluacion["rmse"], ruta.stem, ficha["autor"], evaluacion))
            print(f"  [OK] {ruta.stem}")

    if filas:
        print(f"\n{'modelo':<16}{'autor':<22}{'RMSE':>8}{'MAE':>8}{'R2':>8}")
        for _, nombre, autor, evaluacion in sorted(filas):
            print(
                f"{nombre:<16}{autor:<22}{evaluacion['rmse']:>8.2f}"
                f"{evaluacion['mae']:>8.2f}{evaluacion['r2']:>8.3f}"
            )
        print(f"\nLa API serviría: {sorted(filas)[0][1]} (el de menor RMSE).")

    if con_problemas:
        print(f"\n{con_problemas} modelo(s) con problemas.")
    return 1 if con_problemas else 0


if __name__ == "__main__":
    sys.exit(main())
