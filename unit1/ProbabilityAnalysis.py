"""
Task 5 - Probability Analysis

Usa la desigualdad de Chebyshev para acotar cuánto puede alejarse la tasa de bajo stock
estimada con una muestra de la tasa real de toda la bodega.

Si p es la tasa real (fracción de scans con is_low_stock) y p_hat es la fracción en una
muestra uniforme de n scans, entonces E[p_hat] = p y Var(p_hat) <= p(1 - p) / n. Chebyshev da

    Pr[ |p_hat - p| >= eps ]  <=  p(1 - p) / (n * eps^2)

Se calcula con la p real del dataset y con el peor caso p(1 - p) <= 1/4 (cuando no se conoce p).
Despejando n se obtiene la muestra mínima para una confianza 1 - delta:

    n >= p(1 - p) / (eps^2 * delta)

La cota se compara contra la probabilidad empírica: se repite el muestreo muchas veces y se
cuenta cuántas veces el error llega a eps. Como las muestras se toman sin reemplazo, la varianza
real es todavía menor que p(1 - p) / n (corrección por población finita), así que la cota
sigue siendo válida y resulta conservadora.
"""
import json
import math
import random
from pathlib import Path

RUTA_DATASET = Path(__file__).resolve().parent.parent / "data" / "dataset.json"
SEMILLA = 42


def cargar_dataset(ruta=RUTA_DATASET):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def tasa_real(dataset):
    """p: fraccion de scans marcados como bajo stock en todo el dataset."""
    return sum(r["is_low_stock"] for r in dataset) / len(dataset)


def cota_chebyshev(n, eps, p=None):
    """Cota de Pr[|p_hat - p| >= eps] con muestra de tamaño n (peor caso 1/4 si p es None)."""
    varianza_unitaria = p * (1 - p) if p is not None else 0.25
    return min(1.0, varianza_unitaria / (n * eps ** 2))


def muestra_minima(eps, delta, p=None):
    """Menor n con cota <= delta: n >= p(1-p) / (eps^2 * delta)."""
    varianza_unitaria = p * (1 - p) if p is not None else 0.25
    return math.ceil(varianza_unitaria / (eps ** 2 * delta))


def probabilidad_empirica(dataset, n, eps, ensayos=2000, semilla=SEMILLA):
    """Fraccion de ensayos donde |p_hat - p| >= eps (muestreo uniforme de scans)."""
    rng = random.Random(semilla)
    p = tasa_real(dataset)
    flags = [r["is_low_stock"] for r in dataset]
    fallos = 0
    for _ in range(ensayos):
        p_hat = sum(rng.sample(flags, n)) / n
        if abs(p_hat - p) >= eps:
            fallos += 1
    return fallos / ensayos


def analisis(dataset, tamanos=(100, 250, 500, 1000, 2000), eps=0.03, ensayos=2000):
    p = tasa_real(dataset)
    filas = []
    for n in tamanos:
        filas.append({
            "n": n,
            "cota_chebyshev": cota_chebyshev(n, eps, p),
            "cota_peor_caso": cota_chebyshev(n, eps),
            "empirica": probabilidad_empirica(dataset, n, eps, ensayos),
        })
    return {"p_real": p, "eps": eps, "filas": filas}


def main():
    dataset = cargar_dataset()
    r = analisis(dataset)
    print(f"Tasa real de bajo stock p = {r['p_real']:.4f} | eps = {r['eps']}")
    print(f"{'n':>6} {'Chebyshev(p)':>14} {'Chebyshev(1/4)':>16} {'Empirica':>10}")
    for f in r["filas"]:
        print(f"{f['n']:>6} {f['cota_chebyshev']:>14.4f} "
              f"{f['cota_peor_caso']:>16.4f} {f['empirica']:>10.4f}")
    for delta in (0.10, 0.05):
        print(f"n minimo para eps={r['eps']}, confianza {1 - delta:.0%}: "
              f"{muestra_minima(r['eps'], delta, r['p_real'])} "
              f"(peor caso: {muestra_minima(r['eps'], delta)})")


if __name__ == "__main__":
    main()
