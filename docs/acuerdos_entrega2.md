# Acuerdos para la Entrega 2

Este documento reúne lo que los tres necesitamos tener claro para trabajar cada uno por su lado sin pisarnos. Si todos respetamos estos nombres y formatos, al final las piezas encajan sin rehacer nada.

Si alguien necesita cambiar algo de aquí, avisa al grupo antes de hacerlo.

---

## Qué cambia por el feedback de la Entrega 1

**Nueva pregunta de negocio**

> ¿Cuál es el puntaje Saber 11 esperado para un colegio según el perfil socioeconómico de sus estudiantes, para que las secretarías de educación fijen metas ajustadas a su contexto y detecten, cuando lleguen los resultados, qué colegios —y en qué regiones y tipos de establecimiento— se desvían de esa meta?

El profesor señaló que la pregunta anterior solo usaba el modelo con colegios cuyo puntaje ya se conocía. Ahora el modelo sirve para algo antes del examen: fijar la meta de cada colegio. La comparación entre puntaje real y esperado queda como el paso de seguimiento.

**Correcciones que debemos cubrir**

1. **Pregunta de negocio predictiva** (Javier, en el resumen del reporte). Resaltar el cambio frente a la Entrega 1.
2. **Simulador en el tablero** (Laura). Una vista donde el usuario ingresa el perfil de un colegio y obtiene su puntaje esperado con un rango, consultando la API.
3. **Exploración más completa** (Javier). Faltó:
   - Distribución del puntaje promedio por colegio y del número de evaluados
   - Relación de las demás variables de contexto con el puntaje, a nivel de colegio (no solo el INSE)
   - Puntajes por área
   - En qué departamentos, zonas, naturaleza y jornadas se concentran los colegios por encima y por debajo de lo esperado
4. **Aplicar la limpieza y recalcular las cifras** (Javier, con el script de la tabla de colegios). Todas las cifras del reporte deben salir de la tabla limpia.

---

## Quién hace qué

**Javier** — datos, exploración y modelos de árboles
- Script que limpia los datos y arma la tabla por colegio
- Exploración complementaria (lo que pidió el profesor)
- Modelos Random Forest y LightGBM
- En el reporte: resumen con los cambios frente a la Entrega 1, exploración y comparación de modelos

**Juan** — modelos lineales y API
- Modelos OLS, Ridge y Lasso
- API que recibe el perfil de un colegio y devuelve su puntaje esperado
- En el reporte: preparación de datos, modelos lineales, API y MLflow

**Laura** — tablero
- Tablero con las vistas de la maqueta
- Vista nueva de "simulador": el usuario ingresa el perfil de un colegio y obtiene el puntaje esperado
- En el reporte: descripción del tablero y reporte de trabajo en equipo (cada uno escribe su párrafo)

Cada uno trabaja en su propia rama y sube sus cambios con un pull request que revisa otro integrante.

---

## La tabla de colegios

Los datos originales tienen una fila por estudiante. Nosotros trabajamos con una fila por colegio. El script de Javier hace esa transformación y guarda el resultado en `data/processed/colegios_2020_2.parquet`.

**Qué datos se quitan antes de agrupar**
- Registros de 2020-1: solo se deja `periodo == 20204`
- Filas duplicadas
- Registros bajo investigación: solo se deja `estu_estadoinvestigacion == "PUBLICAR"`
- Puntajes globales en cero

Después de agrupar, solo quedan los colegios con 20 o más evaluados.

**Columnas de la tabla**

Para identificar y filtrar el colegio:
`cole_cod_dane`, `cole_nombre`, `departamento`, `municipio`, `cod_municipio`, `naturaleza`, `zona`, `jornada`, `n_evaluados`

Puntajes (promedio del colegio):
`punt_global_prom` (lo que el modelo predice), `punt_lectura_prom`, `punt_matematicas_prom`, `punt_naturales_prom`, `punt_sociales_prom`, `punt_ingles_prom`

Perfil socioeconómico (lo que el modelo usa para predecir):

| Columna | Qué significa | Valores que cuentan (tal como vienen en los datos) |
|---|---|---|
| `inse_prom` | Índice socioeconómico promedio de los estudiantes | Promedio de `estu_inse_individual` |
| `pct_madre_superior` | % de estudiantes cuya mamá tiene educación técnica, profesional o posgrado | `Técnica o tecnológica completa`, `Educación profesional completa`, `Postgrado` |
| `pct_padre_superior` | Lo mismo para el papá | Los mismos tres niveles |
| `pct_estrato_1_2` | % de estudiantes de estrato 1 o 2 | `Estrato 1`, `Estrato 2` |
| `pct_estrato_5_6` | % de estudiantes de estrato 5 o 6 | `Estrato 5`, `Estrato 6` |
| `pct_internet` | % con internet en la casa | `Si` |
| `pct_computador` | % con computador en la casa | `Si` |
| `pct_libros_26mas` | % con más de 25 libros en la casa | `26 A 100 LIBROS`, `MÁS DE 100 LIBROS` |

Los porcentajes van de 0 a 1 y se calculan solo con quienes respondieron la pregunta:

- Los vacíos (`None`) no cuentan en el total.
- En la educación de los padres, `No sabe` y `No Aplica` se tratan igual que un vacío.
- `Sin Estrato` sí es una respuesta: cuenta en el total, pero no en estrato 1–2 ni en 5–6.

Mientras la tabla oficial no esté en el repositorio, cualquiera puede armar una versión aproximada con estos mismos nombres de columna y luego reemplazarla.

---

## Los modelos

**Qué recibe cada modelo:** las 8 variables del perfil socioeconómico, más `zona` y `departamento`.

**Qué no recibe:** naturaleza, jornada ni puntajes por área. Esas variables solo se muestran en el tablero.

**Cómo se guarda:** cada modelo se guarda en `models/` con su nombre (por ejemplo `models/ridge.joblib`). El archivo incluye todo el preprocesamiento, así cualquiera puede cargarlo y usarlo directamente. Junto a él va un archivo `.json` con sus métricas y su autor.

**Cómo se evalúa:** todos usamos validación cruzada de 5 particiones con `random_state=42` y reportamos RMSE, MAE y R². Cada uno registra sus experimentos en su propio MLflow en EC2.

**Cuál se usa al final:** el que tenga menor RMSE.

---

## La API

La API recibe el perfil de un colegio y devuelve su puntaje esperado con un rango.

**Lo que recibe**

```json
{
  "inse_prom": 52.3,
  "pct_madre_superior": 0.25,
  "pct_padre_superior": 0.20,
  "pct_estrato_1_2": 0.60,
  "pct_estrato_5_6": 0.02,
  "pct_internet": 0.70,
  "pct_computador": 0.55,
  "pct_libros_26mas": 0.15,
  "zona": "URBANO",
  "departamento": "CUNDINAMARCA"
}
```

**Lo que devuelve**

```json
{
  "esperado": 251.4,
  "inferior": 221.6,
  "superior": 281.2,
  "modelo": "lightgbm"
}
```

El rango se calcula sumando y restando 1,96 veces la desviación del error del modelo, así funciona igual sin importar qué modelo gane.

Mientras la API no esté lista, el tablero puede usar una respuesta de prueba como la de arriba.

---

## El reporte

Máximo 10 páginas en total. Cada uno escribe sus secciones por separado y luego se unen:

| Sección | Responsable | Extensión |
|---|---|---|
| Resumen del problema y cambios frente a la Entrega 1 | Javier | 1 página |
| Exploración complementaria | Javier | 1,5 páginas |
| Preparación de datos y forma de evaluar | Juan | 1 página |
| Modelos lineales | Juan | 1 página |
| Modelos de árboles y comparación | Javier | 1,5 páginas |
| API y experimentos en MLflow | Juan | 1 página |
| Tablero y simulador | Laura | 2 páginas |

El reporte de trabajo en equipo va aparte (máximo 1 página). Laura lo arma y cada uno aporta su párrafo.
