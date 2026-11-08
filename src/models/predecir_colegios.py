"""
Calcula el puntaje esperado de los 6.987 colegios y lo guarda junto a la tabla.

El tablero necesita, para el panorama, el listado y la ficha, comparar el puntaje
real de cada colegio con el esperado según su contexto. Pedirle eso a la API uno por
uno serían casi siete mil peticiones HTTP para pintar una pantalla, así que se
precalcula una sola vez y el tablero solo lee el archivo.

Entrada:  data/processed/colegios_2020_2.parquet
          models/<el de menor RMSE>.joblib
Salida:   data/processed/colegios_con_esperado.parquet

El modelo se carga con la misma función que usa la API (`src.api.modelo.cargar`), y
el intervalo se arma con la misma fórmula. Eso importa: si el simulador y la ficha
del mismo colegio mostraran números distintos, para el usuario sería un error.

Uso, desde la raíz del repositorio:
    python src/models/predecir_colegios.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.api import modelo as modelo_api
from src.models import ficha

RUTA_SALIDA = ficha.RAIZ / "data" / "processed" / "colegios_con_esperado.parquet"


def main() -> None:
    colegios = pd.read_parquet(ficha.RUTA_DATOS)
    print(f"Leídos {len(colegios):,} colegios de {ficha.RUTA_DATOS.name}")

    m = modelo_api.cargar()
    print(f"Modelo: {m.nombre}  (RMSE {m.ficha['evaluacion']['rmse']}, σ {m.sigma:.2f})\n")

    esperado = m.pipeline.predict(colegios[ficha.PREDICTORAS])
    margen = modelo_api.Z * m.sigma

    tabla = colegios.copy()
    tabla["esperado"] = esperado.round(1)
    tabla["inferior"] = (esperado - margen).round(1)
    tabla["superior"] = (esperado + margen).round(1)
    tabla["residual"] = (colegios[ficha.OBJETIVO] - esperado).round(1)

    # La misma regla que publica la API: un colegio "se desvía" cuando su puntaje
    # real cae fuera del intervalo que el modelo le asigna.
    real = colegios[ficha.OBJETIVO]
    tabla["clasificacion"] = "En lo esperado"
    tabla.loc[real > tabla["superior"], "clasificacion"] = "Por encima de lo esperado"
    tabla.loc[real < tabla["inferior"], "clasificacion"] = "Por debajo de lo esperado"
    tabla["modelo"] = m.nombre

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(RUTA_SALIDA, index=False)

    conteo = tabla["clasificacion"].value_counts()
    print("Clasificación de los colegios:")
    for etiqueta in ("Por encima de lo esperado", "En lo esperado", "Por debajo de lo esperado"):
        n = int(conteo.get(etiqueta, 0))
        print(f"  {etiqueta:<28} {n:>6,}  ({n / len(tabla):.1%})")

    fuera = len(tabla) - int(conteo.get("En lo esperado", 0))
    print(f"\n  Se desvían del intervalo: {fuera:,} de {len(tabla):,} ({fuera / len(tabla):.1%})")
    print(f"  Residual: media {tabla['residual'].mean():+.2f}, "
          f"desviación {tabla['residual'].std():.2f}, "
          f"rango [{tabla['residual'].min():.1f}, {tabla['residual'].max():.1f}]")

    print(f"\nGuardado en {RUTA_SALIDA.relative_to(ficha.RAIZ)}")
    print(f"  {tabla.shape[0]:,} filas y {tabla.shape[1]} columnas "
          f"({len(colegios.columns)} originales + 6 nuevas)")
    print("\nPara versionarlo:")
    print("  dvc add data/processed/colegios_con_esperado.parquet")
    print("  dvc push")


if __name__ == "__main__":
    main()
