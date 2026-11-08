"""
Task 2 - MapReduce Algorithm

S12 - K-Means Clustering: agrupa los SKUs en k=3 clusters sobre tres caracteristicas
-- avg_quantity, low_stock_frac y scan_count -- calculadas con un job de MapReduce
sobre los scans crudos, y despues corre K-Means sobre esas caracteristicas, con cada
iteracion de K-Means tambien escrita como un paso Map + un paso Reduce.

Job 1 (agregar por SKU):
  Map    emite (sku, (quantity, is_low_stock, 1)) por cada scan.
  Reduce junta por sku y calcula avg_quantity (promedio de quantity), low_stock_frac
         (promedio de is_low_stock) y scan_count (cuantos scans tuvo ese SKU).

Job 2, repetido hasta converger (K-Means):
  Map    a cada SKU le asigna el centroide mas cercano (en el espacio de las 3
         caracteristicas, estandarizadas -- ver nota abajo).
  Reduce por cada cluster promedia los puntos asignados para obtener el centroide
         nuevo.
  Converge cuando ningun centroide se movio mas de `tol`, o a las `max_iter`
  iteraciones.

Nota sobre estandarizar: scan_count va de 1 a 188 (sesgado, como el trafico de bins en
la Unidad 3), avg_quantity de 0 a 52 y low_stock_frac de 0 a 1 -- en escalas muy
distintas. Sin estandarizar, la distancia euclidiana la domina casi por completo
scan_count y K-Means termina agrupando solo por eso. Por eso cada caracteristica se
estandariza (z-score: resta la media, divide por la desviacion estandar) antes de
correr K-Means, siguiendo la misma convencion de "2 z-scored metrics" que usan los
templates de datos de las Unidades 4 y 5 del taller.
"""
import json
import os
import random
import statistics
from collections import defaultdict

RUTA_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.json"
)

CARACTERISTICAS = ("avg_quantity", "low_stock_frac", "scan_count")
K = 3


def cargar_scans(ruta=RUTA_DATASET):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Job 1: agregar por SKU (Map + Reduce)
# ---------------------------------------------------------------------------

def mapper_job1(scan):
    """Map: de un scan saca la clave (sku) y el valor a sumar en el Reduce."""
    return scan["sku"], (scan["quantity"], scan["is_low_stock"], 1)


def reducer_job1(valores):
    """Reduce: junta los valores de un mismo sku en avg_quantity, low_stock_frac y scan_count."""
    suma_q = suma_low = n = 0
    for q, low, uno in valores:
        suma_q += q
        suma_low += low
        n += uno
    return {"avg_quantity": suma_q / n, "low_stock_frac": suma_low / n, "scan_count": n}


def agregar_por_sku(scans):
    """
    Corre el Job 1 completo: agrupa (la parte de 'shuffle' de MapReduce) los pares
    Map por clave y aplica el Reduce a cada grupo. Devuelve {sku: {avg_quantity,
    low_stock_frac, scan_count}}.
    """
    agrupado = defaultdict(list)
    for scan in scans:
        clave, valor = mapper_job1(scan)
        agrupado[clave].append(valor)
    return {sku: reducer_job1(valores) for sku, valores in agrupado.items()}


# ---------------------------------------------------------------------------
# Estandarizacion (z-score) de las 3 caracteristicas
# ---------------------------------------------------------------------------

def calcular_estandarizacion(stats_por_sku):
    """Media y desviacion estandar (poblacional) de cada caracteristica, sobre todos los SKUs."""
    parametros = {}
    for c in CARACTERISTICAS:
        valores = [s[c] for s in stats_por_sku.values()]
        media = statistics.mean(valores)
        desv = statistics.pstdev(valores)
        parametros[c] = (media, desv if desv > 0 else 1.0)
    return parametros


def vector_estandarizado(stats_sku, parametros):
    return tuple(
        (stats_sku[c] - parametros[c][0]) / parametros[c][1] for c in CARACTERISTICAS
    )


# ---------------------------------------------------------------------------
# Job 2, repetido: K-Means (Map = asignar, Reduce = promediar)
# ---------------------------------------------------------------------------

def distancia2(a, b):
    """Distancia euclidiana al cuadrado (no hace falta la raiz para comparar distancias)."""
    return sum((x - y) ** 2 for x, y in zip(a, b))


def mapper_asignar(vector, centroides):
    """Map: a un vector le asigna el indice del centroide mas cercano."""
    return min(range(len(centroides)), key=lambda i: distancia2(vector, centroides[i]))


def reducer_promediar(vectores):
    """Reduce: el centroide nuevo de un cluster es el promedio (por coordenada) de sus puntos."""
    n = len(vectores)
    dim = len(vectores[0])
    return tuple(sum(v[d] for v in vectores) / n for d in range(dim))


def inercia(vectores, centroides, asignaciones):
    """Suma de distancias al cuadrado de cada punto a su centroide asignado (WCSS):
    mientras mas baja, mejor agrupados quedan los puntos dentro de su cluster."""
    return sum(distancia2(v, centroides[c]) for v, c in zip(vectores, asignaciones))


def kmeans_una_corrida(vectores, k, max_iter, tol, semilla):
    """
    Una corrida de K-Means como una secuencia de jobs Map (asignar al centroide mas
    cercano) + Reduce (promediar cada cluster), hasta que ningun centroide se mueva
    mas de `tol` entre una iteracion y la siguiente, o se agoten las `max_iter`
    iteraciones.

    Devuelve (centroides finales, asignaciones -- una por vector, en el mismo orden
    que `vectores` --, iteraciones usadas).
    """
    rng = random.Random(semilla)
    centroides = list(rng.sample(vectores, k))  # init: k puntos reales al azar

    asignaciones = [0] * len(vectores)
    for iteracion in range(1, max_iter + 1):
        # Map: asignar cada punto al centroide mas cercano
        asignaciones = [mapper_asignar(v, centroides) for v in vectores]

        # Reduce: promediar los puntos de cada cluster
        grupos = defaultdict(list)
        for v, c in zip(vectores, asignaciones):
            grupos[c].append(v)

        nuevos = list(centroides)  # si un cluster queda vacio, conserva su centroide anterior
        for c, puntos in grupos.items():
            nuevos[c] = reducer_promediar(puntos)

        movimiento = max(distancia2(nuevos[c], centroides[c]) ** 0.5 for c in range(k))
        centroides = nuevos
        if movimiento < tol:
            break

    return centroides, asignaciones, iteracion


def kmeans(vectores, k=K, max_iter=20, tol=1e-3, semilla=42, reinicios=10):
    """
    K-Means es sensible al punto de partida: con centroides iniciales distintos puede
    quedar atascado en óptimos locales distintos (lo medimos mas abajo, en las
    pruebas, comparando varias semillas). Por eso se corre `reinicios` veces con
    semillas derivadas de `semilla`, y se queda con la corrida de menor inercia
    (WCSS) -- la practica estandar para K-Means.

    Devuelve (centroides, asignaciones, iteraciones de la mejor corrida, inercia de
    la mejor corrida).
    """
    mejor = None
    for i in range(reinicios):
        centroides, asignaciones, iteraciones = kmeans_una_corrida(
            vectores, k, max_iter, tol, semilla=semilla * 1000 + i
        )
        wcss = inercia(vectores, centroides, asignaciones)
        if mejor is None or wcss < mejor[3]:
            mejor = (centroides, asignaciones, iteraciones, wcss)
    return mejor


# ---------------------------------------------------------------------------
# Interpretacion de los clusters
# ---------------------------------------------------------------------------

def caracterizar_clusters(skus, stats_por_sku, asignaciones, k=K):
    """
    Para cada cluster calcula el promedio (en unidades reales, no estandarizadas) de
    las 3 caracteristicas y cuantos SKUs tiene.
    """
    por_cluster = defaultdict(list)
    for sku, c in zip(skus, asignaciones):
        por_cluster[c].append(sku)

    resumen = {}
    for c, skus_cluster in por_cluster.items():
        promedios = {
            feat: statistics.mean(stats_por_sku[s][feat] for s in skus_cluster)
            for feat in CARACTERISTICAS
        }
        resumen[c] = {"n_skus": len(skus_cluster), **promedios}
    return resumen


def etiquetar_clusters(resumen):
    """
    Etiqueta de negocio por cluster, a partir de sus promedios (no de un indice fijo,
    porque el orden de los clusters que da K-Means es arbitrario):
      - 'critico': el cluster con mayor low_stock_frac promedio (SKUs que se quedan
        sin stock con mas frecuencia).
      - 'rotacion rapida': de los que quedan, el de mayor scan_count promedio (SKUs
        que los robots escanean mucho -- bins de alto movimiento).
      - 'stock estable': el que queda.
    """
    pendientes = set(resumen)
    etiquetas = {}

    critico = max(pendientes, key=lambda c: resumen[c]["low_stock_frac"])
    etiquetas[critico] = "critico"
    pendientes.discard(critico)

    rapido = max(pendientes, key=lambda c: resumen[c]["scan_count"])
    etiquetas[rapido] = "rotacion rapida"
    pendientes.discard(rapido)

    for c in pendientes:
        etiquetas[c] = "stock estable"

    return etiquetas


def ejecutar(scans=None, k=K, max_iter=20, tol=1e-3, semilla=42, reinicios=10):
    scans = scans if scans is not None else cargar_scans()
    stats_por_sku = agregar_por_sku(scans)
    skus = sorted(stats_por_sku)  # orden fijo, para que asignaciones[i] corresponda a skus[i]

    parametros = calcular_estandarizacion(stats_por_sku)
    vectores = [vector_estandarizado(stats_por_sku[s], parametros) for s in skus]

    centroides, asignaciones, iteraciones, wcss = kmeans(
        vectores, k=k, max_iter=max_iter, tol=tol, semilla=semilla, reinicios=reinicios
    )
    resumen = caracterizar_clusters(skus, stats_por_sku, asignaciones, k=k)
    etiquetas = etiquetar_clusters(resumen)

    return {
        "stats_por_sku": stats_por_sku,
        "skus": skus,
        "parametros_estandarizacion": parametros,
        "centroides": centroides,
        "asignaciones": asignaciones,
        "iteraciones": iteraciones,
        "wcss": wcss,
        "resumen": resumen,
        "etiquetas": etiquetas,
    }


def imprimir_resultados(k=K, reinicios=10):
    r = ejecutar(k=k, reinicios=reinicios)
    print(f"{len(r['skus'])} SKUs agregados con el Job 1 (Map+Reduce sobre los scans crudos).")
    print(f"K-Means (k={k}): mejor de {reinicios} reinicios, {r['iteraciones']} iteraciones en esa "
          f"corrida (movimiento maximo de centroide < 1e-3, o se agotaron las 20 iteraciones), "
          f"inercia (WCSS) = {r['wcss']:.1f}.")

    print("\nClusters (promedios en unidades reales, no estandarizadas):")
    for c in sorted(r["resumen"]):
        s = r["resumen"][c]
        print(f"  Cluster {c} [{r['etiquetas'][c]}]: {s['n_skus']} SKUs -- "
              f"avg_quantity={s['avg_quantity']:.2f}, low_stock_frac={s['low_stock_frac']:.4f}, "
              f"scan_count={s['scan_count']:.2f}")

    print("\nInterpretacion:")
    for c in sorted(r["resumen"]):
        s = r["resumen"][c]
        print(f"  '{r['etiquetas'][c]}' ({s['n_skus']} SKUs, "
              f"{100 * s['n_skus'] / len(r['skus']):.1f}% del total)")


if __name__ == "__main__":
    imprimir_resultados()
