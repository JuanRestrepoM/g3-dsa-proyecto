# Guía: cómo traer los datos con DVC

Esta guía resuelve las dudas que ya nos han salido trabajando en equipo. Si solo
necesitas los datos para empezar a trabajar, lee la primera sección y listo.

---

## En 30 segundos

Los datos no están en Git: pesan demasiado. El repositorio guarda un **puntero** de
unas cuantas líneas (`.dvc`) con la huella del archivo, y DVC usa esa huella para
bajarlo de un bucket de S3.

La primera vez, dentro de la carpeta del repositorio:

```bash
dvc remote modify --local aws-remote allow_anonymous_login true
dvc pull
```

De ahí en adelante, cada vez que alguien actualice los datos:

```bash
git pull
dvc pull
```

Siempre en ese orden. `git pull` trae el puntero nuevo; `dvc pull` trae el archivo
al que apunta.

---

## La regla que explica todo lo demás

> **El bucket es de lectura pública y de escritura privada.**

Bajar datos no necesita cuenta de AWS ni credenciales de ningún tipo: por eso la
línea de `allow_anonymous_login`, que le dice a DVC que no intente firmar la
petición. Subir datos sí necesita credenciales, y esas están solo en la cuenta de
AWS Academy donde vive el bucket.

Esto no es un permiso que se nos olvidó dar. Es a propósito: evita que dos personas
suban versiones distintas del mismo archivo y nadie sepa cuál es la buena.

---

## Primera vez

```bash
git clone https://github.com/JuanRestrepoM/g3-dsa-proyecto.git
cd g3-dsa-proyecto

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt

dvc remote modify --local aws-remote allow_anonymous_login true
dvc pull
```

La línea de `allow_anonymous_login` se corre **una sola vez por máquina**. Queda
escrita en `.dvc/config.local`, que no se versiona — por eso cada quien la corre en
la suya y no aparece en los commits de nadie.

Verifica con:

```bash
dvc status
```

Debe responder `Data and pipelines are up to date.`

---

## Cuando alguien cambia los datos

Te avisan por el chat del equipo. Haces:

```bash
git pull        # trae el puntero nuevo
dvc pull        # trae el archivo nuevo
```

> **Ojo:** `dvc pull` reemplaza los archivos de datos de tu carpeta por los que diga
> el puntero. Si habías editado a mano un archivo rastreado por DVC, ese cambio se
> pierde. Los notebooks, los scripts y todo lo que esté en Git no se tocan.

Y si te cambias de rama, los datos no se mueven solos:

```bash
git checkout otra-rama
dvc checkout            # pone los datos que corresponden a esa rama
```

---

## "`dvc push` me da Access Denied"

**Eso es lo esperado, no es un error de configuración tuya.** Solo la cuenta dueña
del bucket puede escribir en él. El mensaje que sale es alguno de estos:

```
ERROR: unexpected error - Access Denied
ERROR: unexpected error - An error occurred (AccessDenied) when calling the PutObject operation
```

No pierdas tiempo configurando credenciales de AWS para arreglarlo: aunque tengas tu
propio laboratorio de AWS Academy, tus credenciales son de **tu** cuenta, no de la
que tiene el bucket, y van a dar el mismo rechazo.

**Si necesitas publicar datos nuevos o corregidos:** sube el archivo al chat del
equipo o a una carpeta compartida y avísale a Juan, que lo registra y lo empuja. Él
devuelve el aviso cuando ya esté arriba para que hagas `git pull` + `dvc pull`.

---

## Errores comunes

| Lo que ves | Qué pasó | Qué hacer |
|---|---|---|
| `dvc pull` → `Access Denied` o `403` | Falta el modo anónimo en esta máquina | `dvc remote modify --local aws-remote allow_anonymous_login true` |
| `dvc push` → `Access Denied` | Normal: no eres el dueño del bucket | Nada. Pásale el archivo a Juan |
| `WARNING: Some of the cache files do not exist neither locally nor on remote` | El puntero apunta a algo que todavía nadie subió | Avísale a quien hizo el cambio: le faltó el `dvc push` |
| `FileNotFoundError` con un `.parquet` | Falta el `dvc pull` | `dvc pull` |
| `dvc status` → `changed outs: modified` | Tu archivo local no coincide con el puntero | Si no lo cambiaste a propósito: `dvc checkout` para volver al bueno |
| `ModuleNotFoundError: No module named 'dvc'` | El entorno virtual no está activo | Actívalo y `pip install -r requirements.txt` |

---

## Para quien publica los datos

Esta parte es solo para quien tiene las credenciales del bucket.

Las credenciales de AWS Academy son **temporales**: expiran a las pocas horas y hay
que volver a ponerlas en cada sesión del laboratorio. Se copian de
**AWS Details → AWS CLI → Show** y se pegan completas en `~/.aws/credentials`
(en Windows, `C:\Users\<tu-usuario>\.aws\credentials`), reemplazando lo que hubiera:

```
[default]
aws_access_key_id=ASIA...
aws_secret_access_key=...
aws_session_token=IQoJ...
```

Pégalas en el archivo, no en el prompt de `aws configure`: ese comando es
interactivo y si le pegas una línea completa la guarda tal cual, dejando el access
key corrupto con un error difícil de leer.

Comprueba antes de seguir:

```bash
aws sts get-caller-identity     # debe responder un JSON con Account y Arn
```

Para publicar un cambio en los datos, el orden importa:

```bash
dvc add data/processed/mi_archivo.parquet    # 1. registra la versión nueva
dvc push                                     # 2. sube el archivo a S3
git add data/processed/mi_archivo.parquet.dvc
git commit -m "Describe qué cambió en los datos"
git push                                     # 3. publica el puntero
```

`dvc add` reescribe el `.dvc`, así que va **antes** del `git add`. Y si copiaste un
archivo a mano y `dvc status` dice `not in cache`, usa `dvc commit <archivo>.dvc`
antes de empujar: sin eso `dvc push` responde `Everything is up to date` sin haber
subido nada.

Cuando termines, avísale al equipo para que hagan `git pull` + `dvc pull`.
