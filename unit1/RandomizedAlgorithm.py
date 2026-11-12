"""
Task 2 - Randomized Algorithm

Reservoir sampling (Algoritmo R) sobre el stream de scans para estimar la fracción de
SKUs que están en bajo stock, y comparación contra el conteo exacto.

Se calculan dos estimaciones a partir de la misma muestra:
  - Fracción de scans en bajo stock: es lo que la muestra uniforme estima sin sesgo.
  - Fracción de SKUs en bajo stock: para cada SKU de la muestra se toma su scan más
    reciente. Los SKUs muy escaneados tienen más probabilidad de entrar a la muestra,
    así que esta estimación puede tener sesgo. Se mide contra el valor exacto.
"""
import json
import math
import os
import random
import statistics
import time

RUTA_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.json"
)

TAMANOS_MUESTRA = [100, 250, 500, 1000, 2000]


def cargar_dataset(ruta=RUTA_DATASET):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def reservoir_sample(stream, k, rng=None):
    """Muestra uniforme de k elementos de un stream de largo desconocido (Algoritmo R).

    Se guardan los primeros k elementos; el elemento i (desde 0, i >= k) reemplaza una
    posición al azar del reservorio con probabilidad k / (i + 1). Al terminar, cada
    elemento quedó en el reservorio con probabilidad exacta k / n. Una sola pasada y
    memoria O(k). Si el stream tiene menos de k elementos, devuelve todos.
    """
    if k <= 0:
        raise ValueError("k debe ser positivo")
    rng = rng or random.Random()
    reservorio = []
    for i, elemento in enumerate(stream):
        if i < k:
            reservorio.append(elemento)
        else:
            j = int(rng.random() * (i + 1))  # uniforme en [0, i]
            if j < k:
                reservorio[j] = elemento
    return reservorio


def conteo_exacto(scans):
    """Valores reales sobre todo el dataset (referencia para comparar la estimación)."""
    ultimo_estado = {}
    bajos = 0
    for s in scans:
        bajos += s["is_low_stock"]
        ultimo_estado[s["sku"]] = s["is_low_stock"]  # el último scan de cada SKU es el actual
    return {
        "scans": len(scans),
        "skus": len(ultimo_estado),
        "fraccion_scans": bajos / len(scans),
        "skus_bajo_stock": sum(ultimo_estado.values()),
        "fraccion_skus": sum(ultimo_estado.values()) / len(ultimo_estado),
    }


def estimar(muestra, n_total):
    """Estima las fracciones a partir de la muestra.

    Devuelve un dict con la fracción de scans, su error estándar (con corrección por
    población finita), el intervalo de confianza del 95 % y la fracción de SKUs.
    """
    k = len(muestra)
    if k == 0:
        raise ValueError("muestra vacía")
    p_scans = sum(1 for s in muestra if s["is_low_stock"]) / k

    varianza = p_scans * (1 - p_scans) / k
    if n_total > 1:
        varianza *= (n_total - k) / (n_total - 1)
    error_est = math.sqrt(varianza)

    # Por SKU se conserva el scan más reciente que apareció en la muestra
    reciente = {}
    for s in muestra:
        previo = reciente.get(s["sku"])
        if previo is None or s["stream_seq"] > previo["stream_seq"]:
            reciente[s["sku"]] = s
    p_skus = sum(1 for s in reciente.values() if s["is_low_stock"]) / len(reciente)

    return {
        "fraccion_scans": p_scans,
        "error_estandar": error_est,
        "ic95": (max(0.0, p_scans - 1.96 * error_est), min(1.0, p_scans + 1.96 * error_est)),
        "fraccion_skus": p_skus,
        "skus_en_muestra": len(reciente),
    }


def experimento(scans, tamanos=TAMANOS_MUESTRA, repeticiones=200, semilla=42):
    """Repite el muestreo con distintas semillas para cada tamaño k y resume el error."""
    exacto = conteo_exacto(scans)
    n = len(scans)
    filas = []
    for k in tamanos:
        est_scans, est_skus, err_scans, err_skus, cubierto, tiempos = [], [], [], [], 0, []
        for r in range(repeticiones):
            rng = random.Random(semilla * 1_000_003 + k * 1009 + r)
            inicio = time.perf_counter()
            muestra = reservoir_sample(scans, k, rng)
            tiempos.append(time.perf_counter() - inicio)
            e = estimar(muestra, n)
            est_scans.append(e["fraccion_scans"])
            est_skus.append(e["fraccion_skus"])
            err_scans.append(abs(e["fraccion_scans"] - exacto["fraccion_scans"]))
            err_skus.append(abs(e["fraccion_skus"] - exacto["fraccion_skus"]))
            cubierto += e["ic95"][0] <= exacto["fraccion_scans"] <= e["ic95"][1]
        p = exacto["fraccion_scans"]
        filas.append({
            "k": k,
            "promedio_scans": statistics.mean(est_scans),
            "error_abs_scans": statistics.mean(err_scans),
            "desv_empirica": statistics.pstdev(est_scans),
            "desv_teorica": math.sqrt(p * (1 - p) / k * (n - k) / (n - 1)),
            "cobertura_ic95": cubierto / repeticiones,
            "promedio_skus": statistics.mean(est_skus),
            "error_abs_skus": statistics.mean(err_skus),
            "tiempo_muestreo_s": statistics.mean(tiempos),
        })
    return exacto, filas


def imprimir_resultados(exacto, filas, repeticiones):
    print(f"\n  {exacto['scans']} scans, {exacto['skus']} SKUs distintos, "
          f"{repeticiones} repeticiones por tamaño ")
    print("\nConteo exacto (referencia)")
    print(f"  Fracción de scans en bajo stock             : {exacto['fraccion_scans']:.4f}")
    print(f"  SKUs cuyo último scan es bajo stock         : {exacto['skus_bajo_stock']} "
          f"de {exacto['skus']} ({exacto['fraccion_skus']:.4f})")

    print("\nEstimación de la fracción de scans en bajo stock")
    print(f"  {'k':>5} {'promedio':>9} {'error abs':>10} {'desv emp':>9} {'desv teor':>10} "
          f"{'IC95 cubre':>11}")
    for f in filas:
        print(f"  {f['k']:>5} {f['promedio_scans']:>9.4f} {f['error_abs_scans']:>10.4f} "
              f"{f['desv_empirica']:>9.4f} {f['desv_teorica']:>10.4f} {f['cobertura_ic95']:>10.1%}")

    print("\nEstimación de la fracción de SKUs en bajo stock (último scan de cada SKU en la muestra)")
    print(f"  {'k':>5} {'promedio':>9} {'sesgo':>9} {'error abs':>10}")
    for f in filas:
        sesgo = f["promedio_skus"] - exacto["fraccion_skus"]
        print(f"  {f['k']:>5} {f['promedio_skus']:>9.4f} {sesgo:>+9.4f} {f['error_abs_skus']:>10.4f}")

    print("\nTiempo de una pasada de muestreo")
    for f in filas:
        print(f"  k={f['k']:>5}: {f['tiempo_muestreo_s'] * 1000:.2f} ms")


if __name__ == "__main__":
    scans = cargar_dataset()
    repeticiones = 200
    exacto, filas = experimento(scans, repeticiones=repeticiones)
    imprimir_resultados(exacto, filas, repeticiones)
