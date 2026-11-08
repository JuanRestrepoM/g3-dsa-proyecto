"""
Genera la figura del barrido de alfa: RMSE de validación cruzada frente a α para
Ridge y Lasso, con la línea de referencia de OLS.

Lee los resultados del registro local de MLflow (`mlflow.db` en la raíz del
repositorio), así que hay que haber corrido antes:

    python src/models/entrenar_lineales.py

Salida: reports/figures/07_barrido_alpha.png
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
BASE_MLFLOW = RAIZ / "mlflow.db"
SALIDA = RAIZ / "reports" / "figures" / "07_barrido_alpha.png"

# Paleta del sistema de visualización: slots categóricos 1 y 2, más la tinta de
# cromo. La línea de OLS es una anotación, no una serie, así que va en gris.
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUTE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
RIDGE = "#2a78d6"
LASSO = "#eb6834"


def leer_resultados() -> tuple[list, list, float]:
    """Saca (alfa, rmse) de cada run de Ridge y Lasso, y el RMSE de OLS."""
    if not BASE_MLFLOW.exists():
        sys.exit(f"No existe {BASE_MLFLOW}. Corre primero src/models/entrenar_lineales.py")

    con = sqlite3.connect(BASE_MLFLOW)
    filas = con.execute(
        """
        SELECT r.name,
               MAX(CASE WHEN p.key='alpha' THEN p.value END) AS alpha,
               MAX(CASE WHEN m.key='rmse'  THEN m.value END) AS rmse
        FROM runs r
        LEFT JOIN params  p ON p.run_uuid = r.run_uuid
        LEFT JOIN metrics m ON m.run_uuid = r.run_uuid
        GROUP BY r.run_uuid
        """
    ).fetchall()
    con.close()

    # Si el entrenamiento se corrió más de una vez, hay un run por alfa repetido.
    # Son deterministas (misma semilla, mismas particiones): se conserva uno.
    por_alfa: dict[str, dict[float, float]] = {"ridge": {}, "lasso": {}}
    ols = None
    for nombre, alpha, rmse in filas:
        if rmse is None or nombre is None:
            continue
        if nombre == "ols":
            ols = rmse
        else:
            familia = nombre.split("_")[0]
            if familia in por_alfa and alpha not in (None, "n/a"):
                por_alfa[familia][round(float(alpha), 10)] = rmse

    if not por_alfa["ridge"] or not por_alfa["lasso"] or ols is None:
        sys.exit("El registro de MLflow no tiene los runs esperados.")
    return sorted(por_alfa["ridge"].items()), sorted(por_alfa["lasso"].items()), ols


def main() -> None:
    ridge, lasso, ols = leer_resultados()

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=200)
    fig.patch.set_facecolor(SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)

    ax.axhline(ols, color=MUTE, linewidth=1.4, linestyle=(0, (5, 4)), zorder=1)
    # A la izquierda las dos curvas van pegadas a la referencia, así que la franja
    # justo debajo está vacía.
    ax.text(
        8.5e-4, ols - 0.22, f"OLS, sin penalización — RMSE {ols:.2f}",
        color=TINTA_2, fontsize=9, va="top", ha="left",
    )

    # Ridge va debajo y un poco más ancho: en el tramo donde ambas valen lo mismo,
    # el azul asoma como borde del naranja en vez de quedar tapado.
    for datos, color, etiqueta, grosor, capa in (
        (ridge, RIDGE, "Ridge", 3.2, 2),
        (lasso, LASSO, "Lasso", 2.0, 3),
    ):
        x = [a for a, _ in datos]
        y = [r for _, r in datos]
        ax.plot(
            x, y, color=color, linewidth=grosor, marker="o", markersize=5,
            markeredgecolor=SUPERFICIE, markeredgewidth=1.2, zorder=capa,
            solid_capstyle="round",
        )
        # Etiqueta directa al final de la curva: el texto va en tinta y la línea
        # de color pegada a él es la que carga la identidad.
        ax.annotate(
            etiqueta, (x[-1], y[-1]), textcoords="offset points", xytext=(9, 0),
            va="center", ha="left", color=TINTA, fontsize=10, fontweight="semibold", zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlabel("α (penalización, escala logarítmica)", color=TINTA_2, labelpad=8)
    ax.set_ylabel("RMSE en validación cruzada (puntos)", color=TINTA_2, labelpad=8)
    ax.set_ylim(15.15, 21.6)
    ax.set_xlim(7e-4, 2.6e3)

    ax.grid(axis="y", color=REJILLA, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(EJE)
        ax.spines[lado].set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelcolor=TINTA_2, length=4, width=0.8)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:g}" if v >= 1 else f"{v:.4g}".rstrip("0").rstrip("."))
    )

    ax.set_title(
        "La regularización no mejora la predicción: el óptimo está en α ≈ 0",
        color=TINTA, fontsize=12, fontweight="semibold", loc="left", pad=14,
    )

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(SALIDA, facecolor=SUPERFICIE, bbox_inches="tight")
    print(f"Guardada: {SALIDA.relative_to(RAIZ)}")
    print(f"  OLS   RMSE {ols:.4f}")
    print(f"  Ridge {len(ridge)} alfas, mejor {min(ridge, key=lambda t: t[1])[0]:.4g}")
    print(f"  Lasso {len(lasso)} alfas, mejor {min(lasso, key=lambda t: t[1])[0]:.4g}")


if __name__ == "__main__":
    main()
