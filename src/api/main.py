"""
API de inferencia: recibe el perfil socioeconómico de un colegio y devuelve el
puntaje Saber 11 que se esperaría de él, con un rango.

El contrato (qué entra y qué sale) está acordado en docs/acuerdos_entrega2.md.

Para levantarla, desde la raíz del repositorio:

    uvicorn src.api.main:app --reload --port 8001

Y la documentación interactiva queda en http://localhost:8001/docs
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api import modelo as modelo_mod

app = FastAPI(
    title="Puntaje Saber 11 esperado por colegio",
    description=(
        "Estima el puntaje global que obtendría un colegio dado el perfil "
        "socioeconómico de sus estudiantes. Sirve para fijar metas ajustadas al "
        "contexto antes de que lleguen los resultados del examen."
    ),
    version="1.0.0",
)

# El modelo se carga en la primera petición, no al importar: así el servidor
# levanta aunque todavía no haya un modelo entrenado, y /health lo reporta.
_modelo: modelo_mod.Modelo | None = None


def obtener_modelo() -> modelo_mod.Modelo:
    global _modelo
    if _modelo is None:
        try:
            _modelo = modelo_mod.cargar()
        except modelo_mod.ModeloNoDisponible as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
    return _modelo


class PerfilColegio(BaseModel):
    """Las diez variables con las que el modelo predice."""

    inse_prom: float = Field(
        ..., ge=0, le=100, description="Índice socioeconómico promedio de los estudiantes"
    )
    pct_madre_superior: float = Field(..., ge=0, le=1, description="Proporción con madre de educación superior")
    pct_padre_superior: float = Field(..., ge=0, le=1, description="Proporción con padre de educación superior")
    pct_estrato_1_2: float = Field(..., ge=0, le=1, description="Proporción de estrato 1 o 2")
    pct_estrato_5_6: float = Field(..., ge=0, le=1, description="Proporción de estrato 5 o 6")
    pct_internet: float = Field(..., ge=0, le=1, description="Proporción con internet en casa")
    pct_computador: float = Field(..., ge=0, le=1, description="Proporción con computador en casa")
    pct_libros_26mas: float = Field(..., ge=0, le=1, description="Proporción con más de 25 libros en casa")
    zona: str = Field(..., description="URBANO o RURAL")
    departamento: str = Field(..., description="Nombre del departamento; las tildes son opcionales")

    model_config = {
        "json_schema_extra": {
            "example": {
                "inse_prom": 52.3,
                "pct_madre_superior": 0.25,
                "pct_padre_superior": 0.20,
                "pct_estrato_1_2": 0.60,
                "pct_estrato_5_6": 0.02,
                "pct_internet": 0.70,
                "pct_computador": 0.55,
                "pct_libros_26mas": 0.15,
                "zona": "URBANO",
                "departamento": "CUNDINAMARCA",
            }
        }
    }


class Prediccion(BaseModel):
    esperado: float = Field(..., description="Puntaje global esperado")
    inferior: float = Field(..., description="Extremo inferior del rango")
    superior: float = Field(..., description="Extremo superior del rango")
    modelo: str = Field(..., description="Modelo que produjo la estimación")


@app.get("/", tags=["estado"])
def raiz():
    return {
        "servicio": app.title,
        "version": app.version,
        "documentacion": "/docs",
        "prediccion": "/api/v1/predict",
        "opciones": "/api/v1/opciones",
    }


@app.get("/health", tags=["estado"])
def salud():
    """Dice si hay un modelo cargado. Útil para que el tablero avise antes de fallar."""
    try:
        m = modelo_mod.cargar() if _modelo is None else _modelo
    except modelo_mod.ModeloNoDisponible as error:
        return {"estado": "sin modelo", "detalle": str(error)}
    return {
        "estado": "ok",
        "modelo": m.nombre,
        "rmse": m.ficha.get("evaluacion", {}).get("rmse"),
        "entrenado": m.ficha.get("entrenado"),
    }


@app.get("/api/v1/opciones", tags=["metadatos"])
def opciones():
    """
    Los valores que acepta el formulario: categorías válidas y el rango observado
    de cada variable numérica.

    El tablero debería leer de aquí en vez de escribir los 33 departamentos a mano:
    así los dos lados nunca se desincronizan.
    """
    m = obtener_modelo()
    return {
        "modelo": m.nombre,
        "categorias": m.categorias,
        "rangos": m.ficha.get("rangos_observados", {}),
        "nota": "En las categóricas no importan las tildes ni las mayúsculas.",
    }


@app.post("/api/v1/predict", response_model=Prediccion, tags=["predicción"])
def predecir(perfil: PerfilColegio):
    """
    Devuelve el puntaje esperado del colegio y el rango dentro del cual debería caer.

    El rango es esperado ± 1,96 veces la desviación del error del modelo. Es ancho a
    propósito: refleja cuánto del puntaje de un colegio no explica su contexto
    socioeconómico, que es justamente lo que el proyecto busca medir.
    """
    m = obtener_modelo()
    try:
        return m.predecir(perfil.model_dump())
    except modelo_mod.CategoriaInvalida as error:
        raise HTTPException(
            status_code=422,
            detail={
                "mensaje": f"'{error.valor}' no es un valor válido de {error.columna}.",
                "campo": error.columna,
                "valores_validos": error.validos,
            },
        ) from error
