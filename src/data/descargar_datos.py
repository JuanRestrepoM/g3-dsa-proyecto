"""
Descarga datos de las pruebas Saber 11 desde el portal de Datos Abiertos
de Colombia (API Socrata, www.datos.gov.co) y los guarda en formato parquet.

Cada periodo de la prueba es un dataset independiente, identificado por un
código de 9 caracteres (ej. 'rnvb-vnyh'). Los códigos se consultan en
https://www.datos.gov.co buscando "Saber 11".

Uso:
    python src/data/descargar_datos.py rnvb-vnyh a8xr-en99 -o data/raw/saber11.parquet

El app token de Socrata se lee de la variable de entorno SOCRATA_APP_TOKEN.
Sin token la API funciona igual, pero con límites de tasa más estrictos.
"""
import argparse
import os
import sys

import pandas as pd
from sodapy import Socrata

DOMINIO = "www.datos.gov.co"
TAM_PAGINA = 50_000


def descargar_dataset(cliente, dataset_id, tam_pagina=TAM_PAGINA):
    """Descarga un dataset completo paginando sobre la API."""
    paginas = []
    offset = 0

    while True:
        registros = cliente.get(dataset_id, limit=tam_pagina, offset=offset)
        if not registros:
            break

        pagina = pd.DataFrame.from_records(registros)
        pagina["periodo_origen"] = dataset_id
        paginas.append(pagina)

        offset += tam_pagina
        print(f"  {dataset_id}: {offset:,} registros", file=sys.stderr)

    if not paginas:
        raise RuntimeError(f"El dataset {dataset_id} no devolvió registros")

    return pd.concat(paginas, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", help="IDs de dataset en Socrata")
    parser.add_argument("-o", "--salida", default="data/raw/saber11.parquet")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    cliente = Socrata(
        DOMINIO,
        os.environ.get("SOCRATA_APP_TOKEN"),
        timeout=args.timeout,
    )

    partes = [descargar_dataset(cliente, ds) for ds in args.datasets]
    df = pd.concat(partes, ignore_index=True)

    print(f"\nDimensiones finales: {df.shape}", file=sys.stderr)
    df.to_parquet(args.salida, index=False)
    print(f"Guardado en {args.salida}", file=sys.stderr)


if __name__ == "__main__":
    main()
