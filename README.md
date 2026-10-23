# Proyecto DSA — Pruebas Saber 11 (ICFES)

Proyecto del curso **Despliegue de Soluciones Analíticas** (MIAD, Universidad de los Andes).
El objetivo del semestre es construir y desplegar un prototipo funcional compuesto por un
**modelo supervisado empaquetado**, una **API de inferencia** y un **tablero** que consuma
esa API, todo desplegado con **contenedores Docker**.

## Estado

| Componente | Estado |
|---|---|
| Problema y contexto | Pendiente de definición con el equipo |
| Pregunta de negocio y alcance | Pendiente de definición con el equipo |
| Repositorio Git | Configurado |
| Repositorio DVC (datos) | Configurado |
| Exploración de datos | En curso |
| Maqueta del tablero | Pendiente |

## Datos

Resultados individuales de las pruebas **Saber 11**, publicados por el ICFES en el portal
de Datos Abiertos de Colombia (`www.datos.gov.co`) y obtenidos vía API Socrata.

| | |
|---|---|
| Periodos | 2020-1 (`a8xr-en99`) y 2020-2 (`rnvb-vnyh`) |
| Registros | 520.307 |
| Variables | 82 |
| Variable objetivo | `punt_global` (escala 0–500) |

Las variables cubren resultados por área (lectura crítica, matemáticas, ciencias naturales,
sociales e inglés), características del estudiante, del hogar (estrato, educación de los
padres, bienes y servicios) y del establecimiento educativo (naturaleza, calendario,
jornada, ubicación urbana o rural).

Los datos **no se versionan en Git**: se gestionan con DVC contra un bucket de S3.

## Estructura

```
.
├── data/
│   ├── raw/           Datos crudos descargados del API   (DVC)
│   └── processed/     Datos limpios y transformados       (DVC)
├── notebooks/         Exploración y análisis
├── src/
│   └── data/          Scripts de descarga y preparación
├── models/            Modelos entrenados                  (DVC)
├── reports/figures/   Gráficas para el informe
└── docs/              Enunciados y documentación
```

## Puesta en marcha

```bash
git clone <URL-DEL-REPO>
cd Project

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

dvc pull                          # descarga los datos desde S3
```

### Descargar periodos adicionales

```bash
export SOCRATA_APP_TOKEN=<tu-token>     # opcional, solo afecta límites de tasa
python src/data/descargar_datos.py rnvb-vnyh a8xr-en99 -o data/raw/saber11.parquet
```

### Flujo de trabajo

```bash
dvc add data/raw/saber11.parquet        # registra una nueva versión de los datos
dvc push                                 # sube los datos a S3
git add data/raw/saber11.parquet.dvc    # versiona el puntero en Git
git commit -m "Actualiza datos"
git push
```

## Equipo

Pendiente.
