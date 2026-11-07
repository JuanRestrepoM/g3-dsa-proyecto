"""
Entrena los modelos de árboles (Random Forest y LightGBM) que predicen el puntaje
Saber 11 esperado de un colegio a partir del perfil socioeconómico de sus estudiantes.

Entrada:  data/processed/colegios_2020_2.parquet   (una fila por colegio)
Salida:   models/<modelo>.joblib                    (pipeline completo, listo para la API)
          models/<modelo>.json                      (métricas, autor y metadatos)
          reports/figures/07_importancia_<modelo>.png

Sigue el contrato de docs/contrato_modelos.md: el protocolo de evaluación (5
particiones, random_state=42, RMSE / MAE / R²) y el formato del .json los pone
src/models/ficha.py, igual que para los modelos lineales. Aquí solo está lo propio
de los árboles: su preprocesamiento y la búsqueda de hiperparámetros.

Cada combinación de hiperparámetros queda como un run en MLflow, así que el
experimento muestra toda la búsqueda y no solo el ganador.

Uso:
    # local, con el MLflow que escribe en este computador
    python src/models/entrenar_arboles.py

    # contra el servidor de MLflow en EC2
    python src/models/entrenar_arboles.py --tracking-uri http://<IP>:8050
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

# Igual que en entrenar_lineales.py: permite correrlo como script desde la raíz.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.models import ficha

AUTOR = "Javier Cañón"
EXPERIMENTO = "colegios-modelos-arboles"
CARPETA_FIGURAS = ficha.RAIZ / "reports" / "figures"

# Combinaciones a probar. Cada una es un run en MLflow.
# En Random Forest, min_samples_leaf arranca en 5: con hojas de 1 colegio el archivo
# .joblib pasa de 150 MB y GitHub no lo acepta (límite de 100 MB).
GRILLAS = {
    "random_forest": {
        "n_estimators": [200],
        "min_samples_leaf": [5, 10, 20],
        "max_depth": [None, 15],
        "max_features": [0.33, 0.5],
    },
    "lightgbm": {
        "n_estimators": [300, 800],
        "learning_rate": [0.03, 0.1],
        "num_leaves": [15, 31],
        "min_child_samples": [20, 50],
    },
}


def crear_estimador(familia: str, params: dict):
    if familia == "random_forest":
        return RandomForestRegressor(random_state=ficha.SEMILLA, n_jobs=-1, **params)
    if familia == "lightgbm":
        # importance_type="gain": importancia según cuánto reduce el error cada variable
        return LGBMRegressor(random_state=ficha.SEMILLA, n_jobs=-1, verbose=-1,
                             importance_type="gain", **params)
    raise ValueError(familia)


def construir_pipeline(estimador) -> Pipeline:
    """
    Imputa las numéricas y codifica las categóricas como enteros; encima, el estimador.

    Los árboles no necesitan estandarizar (cortan por umbrales, no por distancias) y
    manejan bien las categóricas codificadas como enteros, así que basta con un
    OrdinalEncoder: una columna por variable en vez de 33 columnas para departamento.

    Una categoría desconocida se codifica como -1 en vez de fallar; igual que con los
    lineales, la API valida contra `categorias_validas` antes de predecir.
    """
    preprocesamiento = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), ficha.PERFIL),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         ficha.CATEGORICAS),
    ])
    return Pipeline([("pre", preprocesamiento), ("modelo", estimador)])


def registrar(familia: str, i: int, params: dict, X, y) -> dict:
    """Evalúa una combinación con el protocolo común y la deja como run de MLflow."""
    metricas = ficha.evaluar(construir_pipeline(crear_estimador(familia, params)), X, y)

    with mlflow.start_run(run_name=f"{familia}_{i:02d}"):
        mlflow.set_tag("autor", AUTOR)
        mlflow.set_tag("familia", familia)
        mlflow.log_param("modelo", familia)
        for clave, valor in params.items():
            mlflow.log_param(clave, "None" if valor is None else valor)
        mlflow.log_param("n_variables_perfil", len(ficha.PERFIL))
        mlflow.log_param("categoricas", ",".join(ficha.CATEGORICAS))
        mlflow.log_param("particiones_cv", ficha.N_PARTICIONES)
        mlflow.log_param("random_state", ficha.SEMILLA)
        mlflow.log_param("n_colegios", len(X))
        for clave, valor in metricas.items():
            mlflow.log_metric(clave, valor)

    print(f"  {i:2d}. RMSE {metricas['rmse']:6.2f}   MAE {metricas['mae']:6.2f}   "
          f"R2 {metricas['r2']:.3f}   {params}")
    return {"params": params, **metricas}


def importancia_por_variable(pipeline) -> pd.Series:
    """Importancia relativa de cada una de las 10 variables (suma 1)."""
    nombres = [n.split("__", 1)[1] for n in pipeline.named_steps["pre"].get_feature_names_out()]
    valores = pipeline.named_steps["modelo"].feature_importances_.astype(float)
    return pd.Series(valores / valores.sum(), index=nombres).sort_values(ascending=False)


def graficar_importancia(importancia: pd.Series, familia: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    datos = importancia.sort_values()
    ax.barh(datos.index, datos.values, color="#2f6f9f")
    for i, v in enumerate(datos.values):
        ax.text(v + 0.005, i, f"{v:.0%}", va="center", fontsize=9)
    ax.set(title=f"Importancia de variables – {familia}", xlabel="Importancia relativa")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_FIGURAS / f"07_importancia_{familia}.png"
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta


def main() -> None:
    argumentos = argparse.ArgumentParser(description=__doc__)
    argumentos.add_argument("--tracking-uri", default=None,
                            help="Servidor de MLflow. Sin esto registra en este computador.")
    opciones = argumentos.parse_args()

    if opciones.tracking_uri:
        mlflow.set_tracking_uri(opciones.tracking_uri)
    print(f"MLflow: {mlflow.get_tracking_uri()}")
    mlflow.set_experiment(EXPERIMENTO)

    X, y = ficha.cargar_datos()
    print(f"Datos:  {len(X):,} colegios, {len(ficha.PREDICTORAS)} variables predictoras\n")

    finales = []
    for familia, grilla in GRILLAS.items():
        combinaciones = [dict(zip(grilla, v)) for v in itertools.product(*grilla.values())]
        print(f"{familia} ({len(combinaciones)} combinaciones)")
        resultados = [registrar(familia, i, p, X, y) for i, p in enumerate(combinaciones, 1)]
        mejor = min(resultados, key=lambda r: r["rmse"])

        # ficha.guardar_modelo evalúa, reentrena con todos los colegios y escribe .joblib y .json.
        # Recibe el pipeline SIN entrenar, como pide el contrato.
        ruta = ficha.guardar_modelo(
            familia,
            construir_pipeline(crear_estimador(familia, mejor["params"])),
            X, y,
            autor=AUTOR,
            hiperparametros={k: v for k, v in mejor["params"].items()},
        )

        importancia = importancia_por_variable(joblib.load(ruta))
        figura = graficar_importancia(importancia, familia)
        with mlflow.start_run(run_name=f"{familia}_importancia"):
            mlflow.set_tag("autor", AUTOR)
            mlflow.set_tag("familia", familia)
            mlflow.log_artifact(str(figura))
            for variable, valor in importancia.items():
                mlflow.log_metric(f"importancia_{variable}", float(valor))

        tamano = Path(ruta).stat().st_size / 1e6
        print(f"  Mejor: {mejor['params']}")
        print(f"  Guardado: {Path(ruta).relative_to(ficha.RAIZ)} ({tamano:.1f} MB)")
        print("  Importancia: " + ", ".join(f"{k} {v:.0%}" for k, v in importancia.head(4).items()) + "\n")
        finales.append({"modelo": familia, "rmse": mejor["rmse"], "mae": mejor["mae"], "r2": mejor["r2"]})

    print("Mejor de cada familia (validación cruzada de 5 particiones):")
    print(pd.DataFrame(finales).round(3).sort_values("rmse").to_string(index=False))
    print("\nPara revisar que todos cumplan el contrato y ver la tabla con los lineales:")
    print("  python src/models/ficha.py")


if __name__ == "__main__":
    main()
