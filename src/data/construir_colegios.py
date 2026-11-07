"""
Construye la tabla de colegios a partir de los datos por estudiante de Saber 11.

Entrada:  data/raw/saber_11_2020_1_2.parquet        (una fila por estudiante)
Salida:   data/processed/colegios_2020_2.parquet     (una fila por colegio)

Las reglas de limpieza y las columnas de salida están descritas en
docs/acuerdos_entrega2.md.

Uso (desde la carpeta raíz del repositorio):
    python src/data/construir_colegios.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RUTA_ENTRADA = Path("data/raw/saber_11_2020_1_2.parquet")
RUTA_SALIDA = Path("data/processed/colegios_2020_2.parquet")

PERIODO = "20204"            # 2020-2, calendario A
MIN_EVALUADOS = 20

# Valores tal como vienen en los datos (verificados con value_counts)
EDUCACION_SUPERIOR = [
    "Técnica o tecnológica completa",
    "Educación profesional completa",
    "Postgrado",
]
EDUCACION_SIN_DATO = ["No sabe", "No Aplica"]
ESTRATOS_1_2 = ["Estrato 1", "Estrato 2"]
ESTRATOS_5_6 = ["Estrato 5", "Estrato 6"]
LIBROS_26MAS = ["26 A 100 LIBROS", "MÁS DE 100 LIBROS"]

# Los datos traen la acentuación de los departamentos a medias: BOGOTÁ y NARIÑO sí
# la llevan, pero ATLANTICO, BOLIVAR o CHOCO no. Eso obliga a que el tablero y la API
# escriban cada nombre con la rareza exacta del dato, y un nombre mal escrito no da
# error: el modelo lo toma como categoría desconocida y devuelve una predicción
# equivocada pero creíble. Aquí se unifica a la forma correcta en español.
#
# Se corrige solo el departamento. En municipio NO se puede hacer lo mismo: CHIMÁ
# (Córdoba) y CHIMA (Santander) son municipios distintos que colapsarían en uno.
DEPARTAMENTOS = {
    "ATLANTICO": "ATLÁNTICO",
    "BOLIVAR": "BOLÍVAR",
    "BOYACA": "BOYACÁ",
    "CAQUETA": "CAQUETÁ",
    "CHOCO": "CHOCÓ",
    "CORDOBA": "CÓRDOBA",
    "GUAINIA": "GUAINÍA",
    "NORTE SANTANDER": "NORTE DE SANTANDER",
    "QUINDIO": "QUINDÍO",
    "SAN ANDRES": "SAN ANDRÉS",
    "VALLE": "VALLE DEL CAUCA",
    "VAUPES": "VAUPÉS",
}

PUNTAJES = {
    "punt_global": "punt_global_prom",
    "punt_lectura_critica": "punt_lectura_prom",
    "punt_matematicas": "punt_matematicas_prom",
    "punt_c_naturales": "punt_naturales_prom",
    "punt_sociales_ciudadanas": "punt_sociales_prom",
    "punt_ingles": "punt_ingles_prom",
}


def reportar(paso: str, df: pd.DataFrame) -> None:
    print(f"  {paso:<45} {len(df):>9,} filas")


COLUMNAS_NUMERICAS = list(PUNTAJES) + ["estu_inse_individual"]


def convertir_numericos(df: pd.DataFrame) -> pd.DataFrame:
    """Los datos de datos.gov.co traen los números como texto: los pasa a número."""
    df = df.copy()
    for c in COLUMNAS_NUMERICAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica los filtros acordados sobre los datos por estudiante."""
    print("Limpieza:")
    reportar("Datos originales", df)

    df = convertir_numericos(df)

    df = df[df["periodo"].astype(str) == PERIODO]
    reportar("Solo periodo 2020-2", df)

    df = df.drop_duplicates()
    reportar("Sin filas duplicadas", df)

    df = df[df["estu_estadoinvestigacion"] == "PUBLICAR"]
    reportar("Sin registros bajo investigación", df)

    df = df[df["punt_global"] > 0]
    reportar("Sin puntajes globales en cero", df)

    return df.copy()


def indicador(serie: pd.Series, positivos, sin_dato=()) -> pd.Series:
    """
    Devuelve 1 si el valor está en `positivos`, 0 si es otra respuesta
    y NaN si no hay respuesta (vacío o valores de `sin_dato`).
    Así el promedio por colegio es el porcentaje entre quienes respondieron.
    """
    resultado = serie.isin(positivos).astype(float)
    falta = serie.isna() | serie.isin(sin_dato)
    resultado[falta] = np.nan
    return resultado


def como_codigo(serie: pd.Series) -> pd.Series:
    """Convierte un código (DANE, municipio) a texto sin decimales: 111001.0 -> '111001'."""
    numeros = pd.to_numeric(serie, errors="coerce")
    if numeros.notna().all():
        return numeros.astype("int64").astype(str)
    return serie.astype(str)


def mas_frecuente(serie: pd.Series):
    moda = serie.mode()
    return moda.iat[0] if not moda.empty else np.nan


def agregar(df: pd.DataFrame) -> pd.DataFrame:
    """Pasa de una fila por estudiante a una fila por colegio."""
    df = df.copy()
    df["cole_cod_dane"] = como_codigo(df["cole_cod_dane_establecimiento"])

    # Variables del perfil socioeconómico, por estudiante
    df["madre_superior"] = indicador(df["fami_educacionmadre"], EDUCACION_SUPERIOR, EDUCACION_SIN_DATO)
    df["padre_superior"] = indicador(df["fami_educacionpadre"], EDUCACION_SUPERIOR, EDUCACION_SIN_DATO)
    df["estrato_1_2"] = indicador(df["fami_estratovivienda"], ESTRATOS_1_2)
    df["estrato_5_6"] = indicador(df["fami_estratovivienda"], ESTRATOS_5_6)
    df["internet"] = indicador(df["fami_tieneinternet"], ["Si"])
    df["computador"] = indicador(df["fami_tienecomputador"], ["Si"])
    df["libros_26mas"] = indicador(df["fami_numlibros"], LIBROS_26MAS)

    grupos = df.groupby("cole_cod_dane")

    # Datos del colegio: el valor más frecuente entre sus estudiantes
    identificacion = grupos.agg(
        cole_nombre=("cole_nombre_establecimiento", mas_frecuente),
        departamento=("cole_depto_ubicacion", mas_frecuente),
        municipio=("cole_mcpio_ubicacion", mas_frecuente),
        cod_municipio=("cole_cod_mcpio_ubicacion", mas_frecuente),
        naturaleza=("cole_naturaleza", mas_frecuente),
        zona=("cole_area_ubicacion", mas_frecuente),
        jornada=("cole_jornada", mas_frecuente),
    )

    # Número de evaluados y promedios (los NaN se ignoran)
    promedios = grupos.agg(
        n_evaluados=("punt_global", "size"),
        **{nuevo: (original, "mean") for original, nuevo in PUNTAJES.items()},
        inse_prom=("estu_inse_individual", "mean"),
        pct_madre_superior=("madre_superior", "mean"),
        pct_padre_superior=("padre_superior", "mean"),
        pct_estrato_1_2=("estrato_1_2", "mean"),
        pct_estrato_5_6=("estrato_5_6", "mean"),
        pct_internet=("internet", "mean"),
        pct_computador=("computador", "mean"),
        pct_libros_26mas=("libros_26mas", "mean"),
    )

    colegios = identificacion.join(promedios).reset_index()
    colegios["cod_municipio"] = como_codigo(colegios["cod_municipio"])
    colegios["departamento"] = colegios["departamento"].replace(DEPARTAMENTOS)

    print("\nAgregación:")
    print(f"  Colegios en total                             {len(colegios):>9,}")
    colegios = colegios[colegios["n_evaluados"] >= MIN_EVALUADOS]
    print(f"  Colegios con {MIN_EVALUADOS} o más evaluados                 {len(colegios):>9,}")

    return colegios.reset_index(drop=True)


def main() -> None:
    print(f"Leyendo {RUTA_ENTRADA} ...\n")
    estudiantes = pd.read_parquet(RUTA_ENTRADA)

    limpios = limpiar(estudiantes)
    colegios = agregar(limpios)

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    colegios.to_parquet(RUTA_SALIDA, index=False)

    print(f"\nTabla guardada en {RUTA_SALIDA}")
    print(f"  {colegios.shape[0]:,} colegios y {colegios.shape[1]} columnas")
    print(f"  Puntaje global promedio: {colegios['punt_global_prom'].mean():.1f}")
    print("\nFaltantes por columna (solo las que tienen):")
    faltantes = colegios.isna().sum()
    print(faltantes[faltantes > 0].to_string() if faltantes.any() else "  Ninguno")


if __name__ == "__main__":
    main()
