"""
Entrena los modelos lineales (OLS, Ridge y Lasso) que predicen el puntaje Saber 11
esperado de un colegio a partir del perfil socioeconómico de sus estudiantes.

Entrada:  data/processed/colegios_2020_2.parquet   (una fila por colegio)
Salida:   models/<modelo>.joblib                    (pipeline completo, listo para la API)
          models/<modelo>.json                      (métricas, autor y metadatos)

El protocolo de evaluación —las 5 particiones, la semilla, las métricas y el formato
del .json— vive en src/models/ficha.py y es el mismo para todos los modelos del
proyecto. Aquí solo está lo propio de los modelos lineales.

Ridge y Lasso se barren sobre una rejilla de alfas, así que el experimento de MLflow
queda con la curva completa y no solo con el ganador.

Uso:
    # local, con el MLflow que escribe en este computador
    python src/models/entrenar_lineales.py

    # contra el servidor de MLflow en EC2
    python src/models/entrenar_lineales.py --tracking-uri http://<IP>:8050
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Al correrlo como `python src/models/entrenar_lineales.py`, Python pone src/models
# en la ruta de importación pero no la raíz del repositorio, y `from src.models...`
# fallaría. Esto hace que funcione igual de las dos formas.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models import ficha

AUTOR = "Juan C Restrepo"
EXPERIMENTO = "colegios-modelos-lineales"

# Alfas en escala logarítmica. Con las variables estandarizadas, alfa=1 casi no
# penaliza: hay que subir varios órdenes de magnitud para ver el efecto.
ALFAS_RIDGE = np.logspace(-2, 3, 11)
ALFAS_LASSO = np.logspace(-3, 1, 9)


def construir_pipeline(estimador) -> Pipeline:
    """
    Estandariza las numéricas, codifica las categóricas y encima pone el estimador.

    La estandarización no es opcional para Ridge y Lasso: penalizan el tamaño de los
    coeficientes, y sin una escala común `inse_prom` (que va de 32 a 75) quedaría
    penalizada muchísimo menos que los porcentajes (que van de 0 a 1).

    `handle_unknown="ignore"` es necesario por los departamentos diminutos: VAUPÉS
    tiene 4 colegios y GUAINÍA 5, así que hay particiones de la validación cruzada
    donde alguno no aparece en el entrenamiento y sí en la prueba.

    Ojo con el efecto secundario: una categoría desconocida se codifica como todo
    ceros, que es exactamente la categoría de referencia. O sea que un departamento
    mal escrito no produce un error, produce una predicción equivocada pero creíble.
    Por eso la API valida contra `categorias_validas` del .json antes de predecir.
    """
    numericas = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )
    preprocesamiento = ColumnTransformer(
        [
            ("num", numericas, ficha.PERFIL),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ficha.CATEGORICAS),
        ]
    )
    return Pipeline([("pre", preprocesamiento), ("modelo", estimador)])


def registrar(nombre: str, estimador, X, y, alfa=None) -> dict:
    """Evalúa una configuración y la deja registrada como un run de MLflow."""
    pipeline = construir_pipeline(estimador)
    metricas = ficha.evaluar(pipeline, X, y)

    with mlflow.start_run(run_name=nombre):
        mlflow.set_tag("autor", AUTOR)
        mlflow.set_tag("familia", nombre.split("_")[0])
        mlflow.log_param("modelo", type(estimador).__name__)
        mlflow.log_param("alpha", "n/a" if alfa is None else f"{alfa:.4g}")
        mlflow.log_param("n_variables_perfil", len(ficha.PERFIL))
        mlflow.log_param("categoricas", ",".join(ficha.CATEGORICAS))
        mlflow.log_param("particiones_cv", ficha.N_PARTICIONES)
        mlflow.log_param("random_state", ficha.SEMILLA)
        mlflow.log_param("n_colegios", len(X))
        for clave, valor in metricas.items():
            mlflow.log_metric(clave, valor)

    etiqueta = "" if alfa is None else f"  alpha={alfa:>8.4g}"
    print(
        f"  {nombre:<18}{etiqueta:<20}"
        f"RMSE {metricas['rmse']:6.2f}   MAE {metricas['mae']:6.2f}   R2 {metricas['r2']:.3f}"
    )
    return {"nombre": nombre, "estimador": estimador, "alpha": alfa, **metricas}


def main() -> None:
    argumentos = argparse.ArgumentParser(description=__doc__)
    argumentos.add_argument(
        "--tracking-uri",
        default=None,
        help="Servidor de MLflow. Sin esto registra en este computador.",
    )
    opciones = argumentos.parse_args()

    if opciones.tracking_uri:
        mlflow.set_tracking_uri(opciones.tracking_uri)
    print(f"MLflow: {mlflow.get_tracking_uri()}")
    mlflow.set_experiment(EXPERIMENTO)

    X, y = ficha.cargar_datos()
    print(f"Datos:  {len(X):,} colegios, {len(ficha.PREDICTORAS)} variables predictoras\n")

    # Referencia sin modelo: predecir siempre el promedio nacional. Cualquier modelo
    # que no la supere con holgura no está aportando nada.
    print(
        f"Referencia (predecir la media):  RMSE {y.std():6.2f}   "
        f"MAE {(y - y.mean()).abs().mean():6.2f}   R2 0.000\n"
    )

    resultados = []

    print("OLS")
    resultados.append(registrar("ols", LinearRegression(), X, y))

    print("\nRidge")
    for alfa in ALFAS_RIDGE:
        resultados.append(registrar(f"ridge_a{alfa:.4g}", Ridge(alpha=alfa), X, y, alfa))

    print("\nLasso")
    for alfa in ALFAS_LASSO:
        resultados.append(
            registrar(f"lasso_a{alfa:.4g}", Lasso(alpha=alfa, max_iter=50_000), X, y, alfa)
        )

    print("\nGanador de cada familia (menor RMSE):")
    for familia in ("ols", "ridge", "lasso"):
        mejor = min((r for r in resultados if r["nombre"].startswith(familia)), key=lambda r: r["rmse"])
        ruta = ficha.guardar_modelo(
            familia,
            construir_pipeline(mejor["estimador"]),
            X,
            y,
            autor=AUTOR,
            hiperparametros={} if mejor["alpha"] is None else {"alpha": float(mejor["alpha"])},
        )
        alfa = "" if mejor["alpha"] is None else f"  (alpha={mejor['alpha']:.4g})"
        print(f"  {ruta.relative_to(ficha.RAIZ)}  y  {familia}.json{alfa}")

    campeon = min(resultados, key=lambda r: r["rmse"])
    print(f"\nMejor modelo lineal: {campeon['nombre']}  (RMSE {campeon['rmse']:.2f}, R2 {campeon['r2']:.3f})")
    print("Para revisar que todos cumplan el contrato:  python src/models/ficha.py")


if __name__ == "__main__":
    main()
