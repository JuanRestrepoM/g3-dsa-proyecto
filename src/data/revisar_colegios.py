"""
Revisa que la tabla de colegios esté bien construida.

Uso (desde la carpeta raíz del repositorio, después de correr construir_colegios.py):
    python src/data/revisar_colegios.py

Cada revisión imprime OK o REVISAR. Si todo sale OK, la tabla está lista.
"""

import pandas as pd

RUTA_ESTUDIANTES = "data/raw/saber_11_2020_1_2.parquet"
RUTA_COLEGIOS = "data/processed/colegios_2020_2.parquet"

COLUMNAS_ESPERADAS = [
    "cole_cod_dane", "cole_nombre", "departamento", "municipio", "cod_municipio",
    "naturaleza", "zona", "jornada", "n_evaluados",
    "punt_global_prom", "punt_lectura_prom", "punt_matematicas_prom",
    "punt_naturales_prom", "punt_sociales_prom", "punt_ingles_prom",
    "inse_prom", "pct_madre_superior", "pct_padre_superior",
    "pct_estrato_1_2", "pct_estrato_5_6", "pct_internet",
    "pct_computador", "pct_libros_26mas",
]
PORCENTAJES = [c for c in COLUMNAS_ESPERADAS if c.startswith("pct_")]

problemas = 0


def revisar(descripcion: str, cumple: bool, detalle: str = "") -> None:
    global problemas
    estado = "OK      " if cumple else "REVISAR "
    if not cumple:
        problemas += 1
    print(f"[{estado}] {descripcion}" + (f"  ->  {detalle}" if detalle else ""))


col = pd.read_parquet(RUTA_COLEGIOS)
print(f"Tabla de colegios: {col.shape[0]:,} filas, {col.shape[1]} columnas\n")

print("1. Estructura")
faltan = [c for c in COLUMNAS_ESPERADAS if c not in col.columns]
sobran = [c for c in col.columns if c not in COLUMNAS_ESPERADAS]
revisar("Tiene exactamente las columnas acordadas", not faltan and not sobran,
        f"faltan {faltan}, sobran {sobran}" if (faltan or sobran) else "")
revisar("Una sola fila por colegio", col["cole_cod_dane"].is_unique,
        f"{col['cole_cod_dane'].duplicated().sum()} códigos repetidos")
revisar("Códigos DANE sin decimales", not col["cole_cod_dane"].astype(str).str.contains(r"\.").any())

print("\n2. Filtros")
revisar("Todos los colegios tienen 20 o más evaluados", (col["n_evaluados"] >= 20).all(),
        f"mínimo = {col['n_evaluados'].min()}")
revisar("Cantidad de colegios cercana a la Entrega 1 (6.990)",
        6700 <= len(col) <= 7100, f"{len(col):,} colegios")

print("\n3. Valores con sentido")
revisar("Puntaje global entre 0 y 500",
        col["punt_global_prom"].between(0, 500).all(),
        f"rango {col['punt_global_prom'].min():.1f} – {col['punt_global_prom'].max():.1f}")
revisar("Puntaje global promedio cercano a la Entrega 1 (~250)",
        235 <= col["punt_global_prom"].mean() <= 265,
        f"{col['punt_global_prom'].mean():.1f}")
for c in PORCENTAJES:
    serie = col[c].dropna()
    revisar(f"{c} entre 0 y 1", serie.between(0, 1).all(),
            f"rango {serie.min():.2f} – {serie.max():.2f}")
revisar("Zona solo URBANO / RURAL", set(col["zona"].dropna()) <= {"URBANO", "RURAL"},
        str(sorted(col["zona"].dropna().unique())))

print("\n4. Faltantes")
faltantes = col.isna().mean()
revisar("Puntaje global sin faltantes", faltantes["punt_global_prom"] == 0)
altas = faltantes[faltantes > 0.01]
revisar("Ninguna columna con más de 1 % de colegios sin dato", altas.empty,
        ", ".join(f"{c}: {v:.1%}" for c, v in altas.items()))

print("\n5. Coherencia con la Entrega 1")
r = col["inse_prom"].corr(col["punt_global_prom"])
revisar("Correlación INSE–puntaje cercana a 0,80 (R² ≈ 0,64)",
        0.75 <= r <= 0.85, f"r = {r:.2f}, R² = {r**2:.2f}")

print("\n6. Comparación de un colegio contra los datos originales")
est = pd.read_parquet(RUTA_ESTUDIANTES)
est["punt_global"] = pd.to_numeric(est["punt_global"], errors="coerce")
est = est[(est["periodo"].astype(str) == "20204")
          & (est["estu_estadoinvestigacion"] == "PUBLICAR")
          & (est["punt_global"] > 0)].drop_duplicates()
codigo = col.sort_values("n_evaluados", ascending=False)["cole_cod_dane"].iloc[0]
codigos_est = pd.to_numeric(est["cole_cod_dane_establecimiento"], errors="coerce")
de_ese = est[codigos_est == float(codigo)]
fila = col[col["cole_cod_dane"] == codigo].iloc[0]
print(f"   Colegio de prueba: {fila['cole_nombre']} ({codigo})")
revisar("Mismo número de evaluados", len(de_ese) == fila["n_evaluados"],
        f"originales {len(de_ese)} vs tabla {fila['n_evaluados']}")
revisar("Mismo puntaje promedio",
        abs(de_ese["punt_global"].mean() - fila["punt_global_prom"]) < 0.01,
        f"originales {de_ese['punt_global'].mean():.2f} vs tabla {fila['punt_global_prom']:.2f}")
internet = de_ese["fami_tieneinternet"].dropna()
revisar("Mismo % con internet",
        abs((internet == "Si").mean() - fila["pct_internet"]) < 0.001,
        f"originales {(internet == 'Si').mean():.3f} vs tabla {fila['pct_internet']:.3f}")

print("\n" + ("Todo bien: la tabla está lista para compartir."
              if problemas == 0 else
              f"{problemas} revisión(es) para mirar. Pégale el resultado a Claude."))
