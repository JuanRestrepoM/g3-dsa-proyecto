# Guía: cómo usar la API desde el tablero

La API recibe el perfil socioeconómico de un colegio y devuelve el puntaje Saber 11
que se esperaría de él, con un rango. Es lo que alimenta la vista de simulador.

---

## Lo primero: la API corre en tu máquina, no en la de nadie más

No necesitas conectarte a un servidor, ni que haya una instancia prendida en AWS, ni
que Juan esté disponible. La API es código de este repositorio: la levantas en tu
computador y le pegas a `localhost`.

```bash
git pull
dvc pull
pip install -r requirements.txt

uvicorn src.api.main:app --reload --port 8001
```

Déjala corriendo en una terminal y abre el tablero en otra. Para comprobar que quedó
bien, entra a **http://localhost:8001/docs**: ahí está la documentación interactiva,
donde puedes mandar peticiones de prueba sin escribir una línea de código.

> **La URL nunca se escribe a mano en el tablero.** Ponla en una variable de
> configuración con `http://localhost:8001` por defecto. Cuando todo pase a Docker en
> la entrega final, cambiar de `localhost` al nombre del servicio es editar una línea.

---

## Pedir una predicción

**`POST /api/v1/predict`**

Lo que mandas:

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

Lo que te devuelve:

```json
{
  "esperado": 240.3,
  "inferior": 209.5,
  "superior": 271.0,
  "modelo": "ols"
}
```

Los ocho porcentajes van **de 0 a 1**, no de 0 a 100. Si tu control está en
porcentaje, divide entre 100 antes de mandarlo.

Desde Python:

```python
import requests

URL_API = "http://localhost:8001"   # ponla en la configuración, no aquí

respuesta = requests.post(f"{URL_API}/api/v1/predict", json=perfil, timeout=10)
respuesta.raise_for_status()
datos = respuesta.json()
```

El rango mide unos 61 puntos de ancho. No es un error: es cuánto del puntaje de un
colegio **no** explica su contexto socioeconómico, que es precisamente lo que el
proyecto quiere mostrar. Vale la pena que el tablero lo presente como una banda y no
como un número exacto.

---

## Llenar el formulario sin escribir nada a mano

**`GET /api/v1/opciones`**

Devuelve las categorías válidas y el rango observado de cada variable numérica:

```json
{
  "modelo": "ols",
  "categorias": {
    "zona": ["RURAL", "URBANO"],
    "departamento": ["AMAZONAS", "ANTIOQUIA", "...", "VICHADA"]
  },
  "rangos": {
    "inse_prom": {"min": 32.6046, "mediana": 49.2946, "max": 74.7362},
    "pct_internet": {"min": 0.0, "mediana": 0.7143, "max": 1.0}
  }
}
```

Úsalo para llenar el desplegable de departamentos y para acotar los deslizadores.
Así el formulario y el modelo nunca se desincronizan: si mañana cambian los datos, el
tablero se actualiza solo.

Los `rangos` dicen dónde fue entrenado el modelo. Un `inse_prom` de 85 no existe en
los datos, así que la predicción sería una extrapolación; acotar el control al rango
observado evita que el usuario pida algo que el modelo no puede responder bien.

---

## Las tildes no importan

La API normaliza antes de comparar, así que `"BOGOTÁ"`, `"BOGOTA"`, `"bogota"` y
`" Bogotá "` llegan todos al mismo lugar. No tienes que preocuparte por la
acentuación al construir la petición.

Lo que **sí** importa es que el departamento exista. Uno inventado o mal escrito
devuelve un error explícito, no una predicción equivocada.

---

## Errores

| Código | Qué pasó | Qué hacer en el tablero |
|---|---|---|
| `200` | Todo bien | Mostrar `esperado` y la banda |
| `422` con `detail.campo` | Departamento o zona inválidos | El `detail.valores_validos` trae la lista correcta |
| `422` con `detail[0].msg` | Un número fuera de rango o un campo faltante | `detail[0].loc` dice cuál campo |
| `503` | No hay modelo entrenado | Avisar que falta correr el entrenamiento |
| No conecta | La API no está levantada | Recordar `uvicorn src.api.main:app --port 8001` |

Un ejemplo de error de categoría:

```json
{
  "detail": {
    "mensaje": "'NARNIA' no es un valor válido de departamento.",
    "campo": "departamento",
    "valores_validos": ["AMAZONAS", "ANTIOQUIA", "..."]
  }
}
```

---

## Mientras tanto

**`GET /health`** dice si hay un modelo cargado y cuál:

```json
{"estado": "ok", "modelo": "ols", "rmse": 15.6994, "entrenado": "2026-11-07 01:44 UTC"}
```

Vale la pena que el tablero lo consulte al arrancar y muestre un aviso claro si la
API no responde, en vez de fallar cuando el usuario pida la primera simulación.

Y si necesitas avanzar sin la API levantada, la respuesta de ejemplo de arriba sirve
como dato de prueba: el contrato es el mismo, así que cambiar al servicio real no te
obliga a reescribir nada.

---

## Qué modelo responde

La API sirve **el modelo de menor RMSE** entre los que haya en `models/`, como
quedó acordado. Hoy es el lineal; cuando entren los de árboles, los toma
automáticamente sin tocar código — por eso la respuesta incluye el campo `modelo`,
para que se sepa cuál contestó.

Para forzar uno específico, la variable de entorno `MODELO_API`:

```bash
MODELO_API=ridge uvicorn src.api.main:app --port 8001
```
