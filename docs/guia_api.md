# Guía para el tablero: datos y API

Todo lo que el tablero necesita, de dónde sale y cómo se consume.

---

## Lo primero: nada corre en la máquina de otro

No necesitas conectarte al computador de nadie, ni a una instancia de AWS, ni que
otro integrante esté disponible. Todo corre en tu máquina: el tablero, la API y el
modelo. Clonas el repositorio, levantas la API y ya tienes el servicio completo en
`localhost`.

---

## 1. Puesta en marcha

```bash
git clone https://github.com/JuanRestrepoM/g3-dsa-proyecto.git
cd g3-dsa-proyecto

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt

# Una sola vez por máquina: permite bajar los datos sin cuenta de AWS
dvc remote modify --local aws-remote allow_anonymous_login true
dvc pull
```

Si ya tenías el repositorio clonado, con `git pull` y `dvc pull` basta.

**Comprueba que quedó bien:**

```bash
dvc status        # debe decir: Data and pipelines are up to date.
```

---

## 2. De dónde sale cada cosa

Esto confunde al principio: el repositorio usa **dos mecanismos** según el tamaño
del archivo.

| Qué necesitas | Dónde vive | Cómo lo traes |
|---|---|---|
| El código del tablero y de la API | Git | `git pull` |
| El modelo entrenado (`models/*.joblib`) | Git — pesa 5 KB | `git pull` |
| La tabla de colegios (30 MB) | DVC / S3 | `dvc pull` |
| La tabla con el puntaje esperado | DVC / S3 | `dvc pull` |

El modelo es pequeño, así que viaja en Git y te llega con el `git pull`. Las tablas
de datos pesan demasiado para Git: Git solo guarda un puntero y DVC trae el archivo
desde un bucket de S3. **Ese bucket es de lectura pública**, por eso el `dvc pull` no
te pide credenciales de AWS — solo hay que correr una vez la línea del modo anónimo.

> Si `dvc pull` te da `Access Denied` o `403`, es que falta esa línea. Y si algún día
> intentas `dvc push`, te va a rechazar: solo el dueño del bucket puede subir. Eso es
> normal, no un error tuyo. Ver [guia_datos_dvc.md](guia_datos_dvc.md).

---

## 3. Las dos fuentes que consume el tablero

**No todo pasa por la API.** El tablero tiene dos necesidades distintas y cada una
tiene su fuente. Usar la equivocada es el error más caro que puedes cometer aquí.

| Vista | Qué necesita | De dónde |
|---|---|---|
| Panorama, listado, ficha | Los 6.987 colegios con su esperado ya calculado | **Archivo** `colegios_con_esperado.parquet` |
| **Simulador** | El esperado de **un** perfil que el usuario inventa | **API** `POST /api/v1/predict` |

La razón es simple: las tres primeras vistas muestran colegios **que ya existen**, y
su puntaje esperado está precalculado. Pedírselo a la API serían 6.987 peticiones
HTTP para pintar una pantalla. El simulador, en cambio, recibe un perfil que no está
en ninguna tabla —el usuario mueve los controles—, así que hay que calcularlo en el
momento, y para eso es la API.

Ambas fuentes usan **el mismo modelo y la misma fórmula**, así que los números
coinciden: el esperado que muestra la ficha de un colegio es idéntico al que
devolvería la API con ese mismo perfil.

---

## 4. La tabla con el puntaje esperado

```python
import pandas as pd

colegios = pd.read_parquet("data/processed/colegios_con_esperado.parquet")
```

Son 6.987 filas y 29 columnas: las 23 originales más seis que agrega el modelo.

| Columna | Qué es |
|---|---|
| `punt_global_prom` | El puntaje **real** del colegio |
| `esperado` | El puntaje que el modelo predice según su perfil |
| `inferior`, `superior` | Los extremos del rango (esperado ± 1,96 σ) |
| `residual` | `real − esperado`. Positivo = rinde por encima de su contexto |
| `clasificacion` | `Por encima de lo esperado` · `En lo esperado` · `Por debajo de lo esperado` |
| `modelo` | Cuál modelo produjo la estimación |

`clasificacion` marca los colegios cuyo puntaje real cae **fuera** del rango. Con el
criterio actual son 398 de 6.987 (5,7 %): 189 por encima y 202 por debajo. Si para
alguna vista necesitas un corte más amplio, filtra por `residual` directamente —por
ejemplo `residual.abs() > 19`— sin recalcular nada.

---

## 5. La API

Se levanta desde la raíz del repositorio, en su **propia terminal**:

```bash
uvicorn src.api.main:app --port 8001
```

Déjala corriendo ahí y abre el tablero en otra terminal. Para comprobar que quedó
bien, entra a **http://localhost:8001/docs**: ahí puedes mandar peticiones de prueba
sin escribir código.

> **No escribas la URL a mano en el tablero.** Ponla en una variable de configuración
> con `http://localhost:8001` por defecto. Cuando todo pase a Docker en la entrega
> final, cambiarla es editar una línea.

### Pedir una predicción

**`POST /api/v1/predict`**

```python
import requests

URL_API = "http://localhost:8001"   # en la configuración, no aquí

perfil = {
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

r = requests.post(f"{URL_API}/api/v1/predict", json=perfil, timeout=10)
r.raise_for_status()
print(r.json())
# {'esperado': 240.3, 'inferior': 209.5, 'superior': 271.0, 'modelo': 'ols'}
```

**Los ocho porcentajes van de 0 a 1, no de 0 a 100.** Si tu control está en
porcentaje, divide entre 100 antes de mandarlo. Es el error más frecuente.

El rango mide unos 61 puntos de ancho. No es un defecto: es cuánto del puntaje de un
colegio **no** explica su contexto socioeconómico, que es justo lo que el proyecto
quiere mostrar. Conviene presentarlo como una banda y no como un número exacto.

### Llenar el formulario sin escribir nada a mano

**`GET /api/v1/opciones`**

```python
op = requests.get(f"{URL_API}/api/v1/opciones", timeout=10).json()

op["categorias"]["departamento"]   # los 33 nombres exactos
op["categorias"]["zona"]           # ['RURAL', 'URBANO']
op["rangos"]["inse_prom"]          # {'min': 32.6, 'mediana': 49.29, 'max': 74.74}
```

Úsalo para llenar el desplegable de departamentos y para acotar los deslizadores. Así
el formulario y el modelo no se pueden desincronizar: si mañana cambian los datos, el
tablero se actualiza solo.

Los `rangos` dicen dónde fue entrenado el modelo. Un `inse_prom` de 85 no existe en
los datos, así que esa predicción sería una extrapolación; acotar el control al rango
observado evita pedirle al modelo algo que no puede responder bien.

### Las tildes no importan

La API normaliza antes de comparar, así que `"BOGOTÁ"`, `"BOGOTA"`, `"bogota"` y
`" Bogotá "` llegan todas al mismo departamento. No te preocupes por la acentuación.

Lo que **sí** importa es que el departamento exista: uno inventado o mal escrito
devuelve un error explícito, no una predicción equivocada.

---

## 6. Errores

| Código | Qué pasó | Qué hacer en el tablero |
|---|---|---|
| `200` | Todo bien | Mostrar `esperado` y la banda |
| `422` con `detail.campo` | Departamento o zona inválidos | `detail.valores_validos` trae la lista correcta |
| `422` con `detail[0].msg` | Un número fuera de rango o un campo faltante | `detail[0].loc` dice cuál campo |
| `503` | No hay modelo entrenado | Avisar que falta correr el entrenamiento |
| No conecta | La API no está levantada | `uvicorn src.api.main:app --port 8001` |

Un error de categoría se ve así:

```json
{
  "detail": {
    "mensaje": "'NARNIA' no es un valor válido de departamento.",
    "campo": "departamento",
    "valores_validos": ["AMAZONAS", "ANTIOQUIA", "..."]
  }
}
```

**`GET /health`** dice si hay un modelo cargado y cuál:

```json
{"estado": "ok", "modelo": "ols", "rmse": 15.6994, "entrenado": "2026-11-07 01:53 UTC"}
```

Vale la pena consultarlo al arrancar el tablero y mostrar un aviso claro si la API no
responde, en vez de fallar cuando el usuario pida la primera simulación.

---

## 7. Si no quieres esperar a nada

El contrato no va a cambiar. Puedes programar el simulador contra esta respuesta de
ejemplo y cambiar al servicio real sin reescribir nada:

```python
{"esperado": 240.3, "inferior": 209.5, "superior": 271.0, "modelo": "ols"}
```

---

## 8. Qué modelo responde

La API sirve **un solo modelo**: el de menor RMSE entre los que haya en `models/`,
como quedó acordado. Los demás quedan en el repositorio pero no contestan. Por eso la
respuesta incluye el campo `modelo`, para que siempre conste cuál produjo el número.

Cuando entren los modelos de árboles y alguno gane, la API empieza a servirlo sin que
nadie toque código. **Pero la tabla precalculada no se actualiza sola:** alguien tiene
que volver a correr `python src/models/predecir_colegios.py` y publicarla. Si ves que
la columna `modelo` de la tabla no coincide con la que devuelve `/health`, avisa — es
señal de que la tabla quedó con un modelo viejo.
