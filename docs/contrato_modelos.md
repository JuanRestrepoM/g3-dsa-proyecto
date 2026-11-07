# Contrato de modelos

Para que los modelos de los tres se puedan comparar entre sí y la API pueda servir
cualquiera de ellos, todos se guardan igual. Este documento explica cómo, y son unas
diez líneas de código.

---

## La idea

El problema que esto evita no es de formato, es de comparabilidad. Si cada quien
calcula sus propias métricas, basta con que alguien use `train_test_split` en vez de
validación cruzada, o una semilla distinta, para que la tabla del reporte compare
números que no son comparables — y el error no se ve en ninguna parte.

Por eso **la función es la que calcula las métricas**, no tú. Le pasas tu pipeline
sin entrenar y ella hace el resto con el protocolo acordado: 5 particiones,
`random_state=42`, RMSE / MAE / R².

---

## Cómo guardar tu modelo

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor

from src.models import ficha

# 1. Los datos, con las columnas acordadas
X, y = ficha.cargar_datos()

# 2. Tu pipeline, con TU preprocesamiento. No tiene que parecerse al de nadie:
#    los árboles no necesitan estandarizar y pueden codificar las categóricas
#    como enteros. Lo único obligatorio es que el pipeline reciba las diez
#    columnas crudas y devuelva la predicción.
preprocesamiento = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), ficha.PERFIL),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ficha.CATEGORICAS),
])
modelo = Pipeline([
    ("pre", preprocesamiento),
    ("modelo", LGBMRegressor(n_estimators=400, max_depth=6, random_state=42)),
])

# 3. Guardar. Esto evalúa, reentrena con todos los colegios y escribe los dos archivos.
ficha.guardar_modelo(
    "lightgbm", modelo, X, y,
    autor="Javier Cañon",
    hiperparametros={"n_estimators": 400, "max_depth": 6},
)
```

Quedan `models/lightgbm.joblib` y `models/lightgbm.json`.

El `nombre` es el del archivo y el que la API reporta en su respuesta: en minúsculas,
sin espacios ni tildes (`lightgbm`, `random_forest`, `ridge`).

---

## Las dos reglas que importan

**1. El pipeline llega sin entrenar.** `guardar_modelo` necesita evaluarlo con
validación cruzada, y eso significa entrenarlo cinco veces desde cero. Si le pasas un
pipeline que ya hizo `.fit()`, las métricas salen infladas porque el modelo ya vio
los datos de prueba. La función lo reentrena ella al final.

**2. El preprocesamiento va dentro del pipeline.** Si escalas o codificas por fuera y
guardas solo el estimador, la API recibiría datos crudos y le daría predicciones sin
sentido, sin avisar. Todo lo que transforme datos tiene que ser un paso del
`Pipeline`.

---

## Qué queda en el `.json`

```json
{
  "modelo": "lightgbm",
  "autor": "Javier Cañon",
  "entrenado": "2026-11-07 14:22 UTC",
  "hiperparametros": {"n_estimators": 400, "max_depth": 6},
  "objetivo": "punt_global_prom",
  "variables": ["inse_prom", "...", "departamento"],
  "n_colegios": 6987,
  "evaluacion": {
    "metodo": "validación cruzada de 5 particiones",
    "random_state": 42,
    "rmse": 14.9849, "rmse_std": 0.3021,
    "mae": 11.4218,
    "r2": 0.7798, "r2_std": 0.0094
  },
  "residuales": {
    "sigma_residual": 14.99,
    "asimetria": 0.11,
    "curtosis_exceso": 1.23,
    "cobertura_1.96_sigma": 0.9448,
    "amplitud_intervalo": 58.76
  },
  "categorias_validas": {"zona": ["RURAL", "URBANO"], "departamento": ["AMAZONAS", "..."]},
  "rangos_observados": {"inse_prom": {"min": 32.6, "mediana": 49.29, "max": 74.74}, "...": {}}
}
```

Tres partes se usan en otro lado y por eso no son opcionales:

| Campo | Quién lo usa | Para qué |
|---|---|---|
| `evaluacion.rmse` | la API | escoge el modelo de menor RMSE, como quedó acordado |
| `residuales.sigma_residual` | la API | el ancho del intervalo: esperado ± 1,96 σ |
| `categorias_validas` | la API y el tablero | validar la entrada y llenar el formulario |

Vale la pena mirar **`cobertura_1.96_sigma`**: dice qué proporción de los colegios
cae de verdad dentro del intervalo. Debería andar cerca de 0,95. Si tu modelo da algo
como 0,89, el intervalo que la API publica está mintiendo y hay que decirlo en el
reporte.

---

## Revisar antes de empujar

```bash
python src/models/ficha.py
```

Revisa todos los modelos de `models/`: que la ficha tenga lo que debe, que la semilla
sea la acordada, que las variables sean las diez, **y que el `.joblib` cargue y
prediga de verdad sobre las diez columnas crudas**. Esto último es lo que atrapa el
error de haber dejado el preprocesamiento por fuera.

Al final imprime la tabla comparativa y dice cuál serviría la API:

```
modelo          autor                     RMSE     MAE      R2
random_forest   Javier Cañon             14.98   11.42   0.780
ols             Juan C Restrepo          15.70   11.87   0.758
ridge           Juan C Restrepo          15.70   11.87   0.758

La API serviría: random_forest (el de menor RMSE).
```

Esa tabla sirve tal cual para la sección de comparación del reporte.

---

## Qué pasa después

La API toma **sola** el modelo de menor RMSE entre los que haya en `models/`. No hay
que editar código ni avisarle a nadie: guardas tu modelo, lo empujas, y si es mejor
que los demás empieza a servirse. La respuesta de la API incluye el campo `modelo`
para que siempre se sepa cuál contestó.

Si necesitas probar uno específico sin borrar los otros:

```bash
MODELO_API=lightgbm uvicorn src.api.main:app --port 8001
```

---

## Errores comunes

| Lo que pasa | Por qué | Cómo se arregla |
|---|---|---|
| `revisar()` dice que el `.joblib` no pudo predecir | El preprocesamiento quedó por fuera del pipeline | Métele las transformaciones como pasos del `Pipeline` |
| Tu RMSE sale sospechosamente bajo | Le pasaste un pipeline ya entrenado | Pásalo sin `.fit()` |
| `KeyError: A la tabla le faltan columnas` | La tabla de colegios está desactualizada | `git pull` y `dvc pull` |
| `random_state` distinto de 42 | Lo cambiaste en el estimador y en la evaluación | La semilla de la validación la pone `ficha`; en tu estimador usa `ficha.SEMILLA` |
| Dos modelos con el mismo nombre | Se sobrescriben | Un nombre por modelo |
