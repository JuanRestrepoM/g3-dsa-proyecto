"""
Carga el modelo que sirve la API y hace la predicción.

Separado de main.py a propósito: aquí está todo lo que tiene que ver con el modelo
y nada que tenga que ver con HTTP, así que se puede probar sin levantar el servidor.

Qué modelo se usa: por acuerdo del equipo, el de menor RMSE. En vez de dejarlo
escrito a mano, esto lee los .json que acompañan a cada modelo en models/ y escoge
el mejor. Así, cuando entren los modelos de árboles, la API los toma sin tocar
código. La variable de entorno MODELO_API fuerza uno específico si hace falta.
"""

from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_MODELOS = RAIZ / "models"

# El intervalo es esperado ± Z * sigma. Con Z = 1.96 cubre el 95 % si los residuales
# son normales; medido sobre los residuales reales la cobertura dio 94.5 %.
Z = 1.96

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


class ModeloNoDisponible(RuntimeError):
    """No hay ningún modelo entrenado en models/."""


class CategoriaInvalida(ValueError):
    """Llegó un valor que el modelo nunca vio en entrenamiento."""

    def __init__(self, columna: str, valor: str, validos: list[str]):
        self.columna, self.valor, self.validos = columna, valor, validos
        super().__init__(f"'{valor}' no es un valor válido de {columna}")


def normalizar(texto: str) -> str:
    """
    Quita tildes, pasa a mayúsculas y colapsa espacios.

    Sirve para comparar, no para mostrar: así 'bogota', 'Bogotá' y 'BOGOTÁ ' llegan
    todos a la misma clave. Se verifica al cargar que la normalización no junte dos
    categorías distintas; si lo hiciera, la carga falla en vez de confundirlas.
    """
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", str(texto)) if unicodedata.category(c) != "Mn"
    )
    return " ".join(sin_tildes.upper().split())


@dataclass
class Modelo:
    nombre: str
    pipeline: object
    sigma: float
    ficha: dict
    categorias: dict[str, list[str]] = field(default_factory=dict)
    # normalizado -> forma canónica, p. ej. {'BOGOTA': 'BOGOTÁ'}
    equivalencias: dict[str, dict[str, str]] = field(default_factory=dict)

    def resolver(self, columna: str, valor: str) -> str:
        """Devuelve la forma exacta que el modelo conoce, o explica qué se aceptaba."""
        canonico = self.equivalencias[columna].get(normalizar(valor))
        if canonico is None:
            raise CategoriaInvalida(columna, valor, self.categorias[columna])
        return canonico

    def predecir(self, perfil: dict) -> dict:
        """
        Recibe el perfil de un colegio y devuelve el puntaje esperado con su rango.

        Las categóricas se resuelven antes de llegar al pipeline. Esto no es un lujo:
        el codificador trata una categoría desconocida como si fuera la de referencia,
        así que sin esta validación un departamento mal escrito no daría error sino
        una predicción equivocada que parece buena.
        """
        fila = {c: perfil[c] for c in PERFIL}
        for c in CATEGORICAS:
            fila[c] = self.resolver(c, perfil[c])

        esperado = float(self.pipeline.predict(pd.DataFrame([fila]))[0])
        margen = Z * self.sigma
        return {
            "esperado": round(esperado, 1),
            "inferior": round(esperado - margen, 1),
            "superior": round(esperado + margen, 1),
            "modelo": self.nombre,
        }


def _fichas() -> list[tuple[Path, dict]]:
    """Los .json de models/ que tienen un .joblib al lado."""
    encontradas = []
    for ruta in sorted(CARPETA_MODELOS.glob("*.json")):
        if not ruta.with_suffix(".joblib").exists():
            continue
        try:
            encontradas.append((ruta, json.loads(ruta.read_text(encoding="utf-8"))))
        except json.JSONDecodeError:
            continue
    return encontradas


def _rmse(ficha: dict) -> float:
    """El RMSE de la ficha. Un modelo sin RMSE queda de último, no rompe la carga."""
    valor = ficha.get("evaluacion", {}).get("rmse", ficha.get("rmse"))
    return float(valor) if valor is not None else float("inf")


def cargar(nombre: str | None = None) -> Modelo:
    """
    Carga el modelo a servir: el indicado por `nombre`, el de MODELO_API, o el de
    menor RMSE entre los que haya en models/.
    """
    nombre = nombre or os.getenv("MODELO_API")
    fichas = _fichas()
    if not fichas:
        raise ModeloNoDisponible(
            f"No hay modelos en {CARPETA_MODELOS}. "
            "Corre primero: python src/models/entrenar_lineales.py"
        )

    if nombre:
        elegida = next((f for f in fichas if f[0].stem == nombre), None)
        if elegida is None:
            disponibles = ", ".join(f[0].stem for f in fichas)
            raise ModeloNoDisponible(f"No existe el modelo '{nombre}'. Hay: {disponibles}")
    else:
        elegida = min(fichas, key=lambda f: _rmse(f[1]))

    ruta, ficha = elegida
    pipeline = joblib.load(ruta.with_suffix(".joblib"))

    # El ancho del intervalo sale de la desviación de los residuales fuera de muestra.
    # Si la ficha no la trae, el RMSE de validación cruzada es la mejor aproximación.
    sigma = ficha.get("residuales", {}).get("sigma_residual") or _rmse(ficha)

    categorias = ficha.get("categorias_validas", {})
    equivalencias = {}
    for columna in CATEGORICAS:
        valores = categorias.get(columna, [])
        mapa = {normalizar(v): v for v in valores}
        if len(mapa) != len(valores):
            raise ModeloNoDisponible(
                f"Dos categorías de '{columna}' coinciden al quitar tildes. "
                "Habría que comparar con el texto exacto en vez de normalizar."
            )
        equivalencias[columna] = mapa

    return Modelo(
        nombre=ficha.get("modelo", ruta.stem),
        pipeline=pipeline,
        sigma=float(sigma),
        ficha=ficha,
        categorias={c: categorias.get(c, []) for c in CATEGORICAS},
        equivalencias=equivalencias,
    )
