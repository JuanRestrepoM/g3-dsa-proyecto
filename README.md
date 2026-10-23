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
| Exploración de datos | Primera versión en `notebooks/01_exploracion_datos.ipynb` |
| Maqueta del tablero | Pendiente |

---

# Guía para empezar

Sigue estos cinco pasos en orden. No necesitas cuenta de AWS ni credenciales de ningún
tipo: el bucket de datos es de lectura pública.

## 1. Requisitos

- **Python 3.11 o superior** — verifica con `python --version`
- **Git** — verifica con `git --version`

## 2. Clonar el repositorio

```bash
git clone https://github.com/JuanRestrepoM/g3-dsa-proyecto.git
cd g3-dsa-proyecto
```

## 3. Identificarte en Git (importante)

La nota del curso es **individual** y se evidencia con los commits de cada quien. Si
commiteas con el nombre o correo de otra persona, tus aportes quedan atribuidos a ella.
Configura tu identidad **dentro de esta carpeta** antes de tu primer commit:

```bash
git config user.name "Tu Nombre"
git config user.email "tu-correo@ejemplo.com"
```

Usa el mismo correo que tienes registrado en GitHub para que los commits se asocien a tu
perfil. Verifica con `git config user.email`.

## 4. Crear el entorno e instalar dependencias

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 5. Traer los datos

Los datos **no están en Git**: pesan demasiado y los gestiona DVC. El repositorio solo
guarda un puntero de 114 bytes con la huella del archivo; DVC usa esa huella para
descargarlo del bucket de S3.

```bash
# Se corre UNA sola vez. Queda en .dvc/config.local, que no se versiona.
dvc remote modify --local aws-remote allow_anonymous_login true

dvc pull
```

Deberías ver `data/raw/saber_11_2020_1_2.parquet` (unos 30 MB). Comprueba con `dvc status`,
que debe responder `Data and pipelines are up to date.`

> **¿Por qué el paso del modo anónimo?** El bucket permite lectura a cualquiera, pero por
> defecto DVC intenta firmar las peticiones con credenciales de AWS. Esa línea le dice que
> no lo haga. Solo quien **sube** datos (`dvc push`) necesita credenciales con permiso de
> escritura.

## 6. Abrir el notebook

```bash
jupyter lab notebooks/01_exploracion_datos.ipynb
```

También puedes abrirlo directamente en VS Code. Funciona tanto si lo ejecutas desde
`notebooks/` como desde la raíz del proyecto: la primera celda resuelve las rutas sola.

---

# Cómo trabajamos

## Ramas y pull requests

No se commitea directo a `main`. Cada tarea va en su propia rama:

```bash
git checkout main
git pull                                  # parte siempre de lo último
git checkout -b exploracion-colegios      # nombre descriptivo de la tarea

# ... trabajas, y commiteas en pedazos con sentido ...
git add archivo_que_cambiaste.py
git commit -m "Agrega el análisis por departamento"

git push -u origin exploracion-colegios
```

Luego abres un Pull Request en GitHub hacia `main`, y otra persona lo revisa antes de
fusionarlo. Así queda registro de quién hizo qué, que es lo que califican.

## Si cambian los datos

Git versiona el puntero, DVC versiona el archivo. Van juntos, en este orden:

```bash
dvc add data/raw/mi_archivo.parquet      # 1. DVC registra la nueva versión
dvc push                                  # 2. sube el archivo a S3  (requiere credenciales)
git add data/raw/mi_archivo.parquet.dvc  # 3. Git versiona el puntero
git commit -m "Actualiza los datos con el periodo 2023"
git push
```

Y al cambiar de rama o traer cambios de otros:

```bash
git checkout otra-rama
dvc checkout                              # sincroniza los datos con esa rama
```

## Descargar periodos adicionales

```bash
# El token es opcional: sin él la API funciona, solo con límites de tasa más estrictos.
export SOCRATA_APP_TOKEN=<tu-token>       # Windows: $env:SOCRATA_APP_TOKEN="<tu-token>"
python src/data/descargar_datos.py rnvb-vnyh a8xr-en99 -o data/raw/saber11.parquet
```

Los códigos de cada periodo se consultan en `www.datos.gov.co` buscando "Saber 11".

## Notebooks

Los notebooks se versionan **con sus salidas**, porque son parte de la evidencia de la
entrega. Eso hace que los diffs sean grandes y que dos personas editando el mismo notebook
choquen con facilidad. Para evitarlo: **un notebook por persona o por tema**, y si hay que
tocar el mismo, avisar en el chat del equipo antes.

---

# Los datos

Resultados individuales de las pruebas **Saber 11**, publicados por el ICFES en el portal
de Datos Abiertos de Colombia (`www.datos.gov.co`) y obtenidos vía API Socrata.

| | |
|---|---|
| Periodos | 2020-1 (`a8xr-en99`) y 2020-2 (`rnvb-vnyh`) |
| Registros | 520.307 |
| Variables | 82 |
| Variable objetivo | `punt_global` (escala 0–500) |
| Almacenamiento | `s3://g3-dsa-proyecto-data` (DVC) |

Las variables cubren resultados por área (lectura crítica, matemáticas, ciencias naturales,
sociales e inglés), características del estudiante, del hogar (estrato, educación de los
padres, bienes y servicios) y del establecimiento educativo (naturaleza, calendario,
jornada, ubicación urbana o rural).

**Advertencia de alcance:** los dos periodos no son comparables entre sí. 2020-1 son 15.435
registros con media 289,4 (colegios de calendario B, antes de la pandemia); 2020-2 son
504.872 registros con media 248,3. La exploración se concentra en 2020-2 y documenta la
razón.

# Estructura

```
.
├── data/
│   ├── raw/           Datos crudos descargados del API   (DVC)
│   └── processed/     Datos limpios y transformados      (DVC)
├── notebooks/         Exploración y análisis
├── src/
│   └── data/          Scripts de descarga y preparación
├── models/            Modelos entrenados                 (DVC)
├── reports/figures/   Gráficas para el informe
└── docs/              Enunciados y documentación
```

# Si algo falla

| Síntoma | Causa y solución |
|---|---|
| `dvc pull` → `Access Denied` o `403` | Falta el modo anónimo. Corre `dvc remote modify --local aws-remote allow_anonymous_login true` |
| `dvc push` → error de credenciales | El token de sesión de AWS Academy expiró. Reinicia el laboratorio y vuelve a configurar `aws configure` más `aws configure set aws_session_token <TOKEN>` |
| `FileNotFoundError` con el parquet | Falta `dvc pull` |
| `ModuleNotFoundError` | El entorno virtual no está activo, o falta `pip install -r requirements.txt` |
| Tus commits aparecen a nombre de otro | No configuraste `git config user.name` y `user.email` en esta carpeta (paso 3) |

# Equipo

- Javier Cañon

- Laura Dominguez

- Juan C Restrepo