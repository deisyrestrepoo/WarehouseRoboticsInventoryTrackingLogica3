"""
Task 3 - Cost Analysis

Mide el tamaño del shuffle (lo que viaja de los mappers a los reducers) del pipeline de la
Task 2 (S12 - K-Means), con y sin combiner, a 1x, 2x y 4x datos, comprueba si crece de forma
lineal y lo extrapola a 10x. La extrapolación se contrasta con una corrida real a 10x.

Qué se mide:

  - Job de agregación por SKU (Job 1 de la Task 2): el Map emite (sku, (quantity, is_low_stock,
    1)) por cada scan. Sin combiner, cada tupla del Map viaja al reducer. Con combiner, cada map
    task suma localmente por sku antes de mandar: viaja una tupla por cada sku DISTINTO de ese
    map task. El combiner es válido porque el Reduce solo suma (y el promedio se calcula al
    final), y el resultado es idéntico con y sin él.
  - Cada iteración de K-Means (Job 2 de la Task 2, que trabaja sobre la tabla de SKUs): el Map
    asigna cada punto a su centroide más cercano y emite (cluster, (x, y, z, 1)). Sin combiner
    viaja un registro por punto; con combiner, cada map task manda a lo sumo una suma parcial
    por cluster, es decir, a lo sumo k registros por map task.

El shuffle se mide en registros y en bytes. Los bytes son el tamaño de cada par (clave, valor)
serializado como JSON compacto (la misma convención de la Task 1 de la Unit 1); un framework
real usa otra serialización, que cambia las cifras absolutas pero no las razones ni el tipo de
crecimiento.

Cómo se escalan los datos: 1x es el dataset de 10.000 scans; 2x, 4x y 10x son los primeros
20.000, 40.000 y 100.000 scans del mismo generador (misma semilla, mismos 20 robots, 2.000 SKUs
posibles). El generador produce los scans de forma secuencial, así que el 1x es exactamente
data/dataset.json y cada tamaño contiene al anterior. No se repite el dataset (como en la
Task 5): repetirlo haría que un map task viera SKUs repetidos de forma artificial y regalaría
ventaja al combiner.

Supuestos de los map tasks:

  - Agregación por SKU: cada map task procesa un fragmento de TAM_FRAGMENTO scans (1.250 =
    10.000 / 8), así que el número de map tasks crece con los datos (8, 16, 32 y 80), como los
    bloques de un sistema de archivos distribuido. Como variante se repite el análisis con un
    número FIJO de 8 map tasks; ahí el combiner mejora con los datos hasta saturar.
  - K-Means: la entrada es la tabla de SKUs, repartida en SPLITS_KMEANS = 8 map tasks. Esa
    tabla casi no crece con los datos: tiene un SKU por cada SKU distinto, y el catálogo es de
    2.000.

Resultado clave: el shuffle de la agregación por SKU crece linealmente con los datos sin
combiner. El de K-Means NO crece con los scans sino con los SKUs distintos, que saturan en el
catálogo; con combiner es de k registros por map task, sin importar los datos.
"""
import os
import random
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402

from data.generador import generar_dataset  # noqa: E402
from unit4.MapReduceAlgorithm import (  # noqa: E402
    K,
    agregar_por_sku,
    calcular_estandarizacion,
    distancia2,
    kmeans_una_corrida,
    mapper_asignar,
    mapper_job1,
    reducer_job1,
    reducer_promediar,
    vector_estandarizado,
)

N_BASE = 10_000
FACTORES = (1, 2, 4)
FACTOR_VALIDACION = 10
TAM_FRAGMENTO = 1250
N_FRAGMENTOS_FIJOS = 8
SPLITS_KMEANS = 8
MAX_ITER = 20
TOL = 1e-3
SEMILLA_KMEANS = 42 * 1000  # la primera corrida de MapReduceAlgorithm.kmeans(semilla=42)
TOL_LINEAL = 0.05  # desviación máxima respecto a la proporcionalidad para llamarlo "lineal"
TOL_CONSTANTE = 0.10  # "constante": el valor crece menos de 10 % mientras los datos se multiplican


# ---------------------------------------------------------------------------
# Datos y utilidades
# ---------------------------------------------------------------------------

def datasets_por_factor(factores, n_base=N_BASE):
    """
    {factor: scans}, con n_base * factor scans por factor. Se genera una sola vez el tamaño
    más grande y los demás son sus primeros scans (el generador es secuencial).
    """
    todos = generar_dataset(n_base * max(factores))
    return {f: todos[: n_base * f] for f in factores}


def partir(items, tam=None, n=None):
    """Parte `items` en trozos contiguos: de `tam` elementos cada uno, o en `n` trozos casi iguales."""
    if (tam is None) == (n is None):
        raise ValueError("dar tam o n, pero no los dos")
    if tam is not None:
        if tam < 1:
            raise ValueError("tam debe ser >= 1")
        return [items[i:i + tam] for i in range(0, len(items), tam)]
    if n < 1:
        raise ValueError("n debe ser >= 1")
    base, resto = divmod(len(items), n)
    trozos, inicio = [], 0
    for i in range(n):
        fin = inicio + base + (1 if i < resto else 0)
        trozos.append(items[inicio:fin])
        inicio = fin
    return trozos


def bytes_registro(clave, valor):
    """Tamaño en bytes de un par (clave, valor) serializado como JSON compacto (UTF-8)."""
    return len(json.dumps([clave, valor], separators=(",", ":")).encode("utf-8"))


def tamano_shuffle(registros):
    """{registros, bytes} de una lista de pares (clave, valor) que viaja en el shuffle."""
    return {
        "registros": len(registros),
        "bytes": sum(bytes_registro(c, v) for c, v in registros),
    }


def combinar_registros(registros):
    """
    Combiner de UN map task: suma localmente, coordenada a coordenada, los valores de una
    misma clave. Sirve para cualquier valor que sea una tupla de sumas (la agregación por SKU
    y las sumas parciales de K-Means).
    """
    acumulado = {}
    for clave, valor in registros:
        actual = acumulado.get(clave)
        if actual is None:
            acumulado[clave] = list(valor)
        else:
            for i, v in enumerate(valor):
                actual[i] += v
    return [(clave, tuple(valor)) for clave, valor in acumulado.items()]


# ---------------------------------------------------------------------------
# Job de agregación por SKU
# ---------------------------------------------------------------------------

def mapper_sku(scan):
    """Map de la Task 2 (sku, (quantity, is_low_stock, 1)), con el booleano como 0/1."""
    clave, valor = mapper_job1(scan)
    return clave, tuple(int(x) for x in valor)


def agregacion_sku(scans, con_combiner, tam_fragmento=TAM_FRAGMENTO, n_fragmentos=None):
    """
    Corre el job de agregación por SKU y mide su shuffle. Cada fragmento es un map task: aplica
    el Map y, si `con_combiner`, el combiner. Después se agrupa por sku y se aplica el Reduce de
    la Task 2 (`reducer_job1`). Se parte en fragmentos de `tam_fragmento` scans o, si se da
    `n_fragmentos`, en esa cantidad fija de fragmentos.

    Devuelve {"stats": {sku: {...}}, "mappers", "registros_map", "registros", "bytes"}.
    """
    if n_fragmentos is None:
        fragmentos = partir(scans, tam=tam_fragmento)
    else:
        fragmentos = partir(scans, n=n_fragmentos)

    registros_map = 0
    shuffle = []
    for fragmento in fragmentos:
        salida = [mapper_sku(s) for s in fragmento]
        registros_map += len(salida)
        shuffle.extend(combinar_registros(salida) if con_combiner else salida)

    grupos = defaultdict(list)
    for sku, valor in shuffle:
        grupos[sku].append(valor)
    stats = {sku: reducer_job1(valores) for sku, valores in grupos.items()}

    return {
        "stats": stats,
        "mappers": len(fragmentos),
        "registros_map": registros_map,
        **tamano_shuffle(shuffle),
    }


# ---------------------------------------------------------------------------
# K-Means: shuffle de cada iteración
# ---------------------------------------------------------------------------

def kmeans_con_costo(vectores, k=K, max_iter=MAX_ITER, tol=TOL, semilla=SEMILLA_KMEANS,
                     n_splits=SPLITS_KMEANS):
    """
    Una corrida de K-Means idéntica a `MapReduceAlgorithm.kmeans_una_corrida` (mismos
    centroides iniciales, mismas asignaciones, mismo Reduce) que además mide, en cada
    iteración, el shuffle sin y con combiner. Los puntos se reparten en `n_splits` map tasks
    contiguos.

    Sin combiner viaja (cluster, (x, y, z, 1)) por cada punto. Con combiner, cada map task
    manda una suma parcial por cluster. Para comprobar que el combiner no cambia el resultado,
    se calcula cada centroide también desde las sumas parciales y se guarda la mayor diferencia.
    """
    rng = random.Random(semilla)
    centroides = list(rng.sample(vectores, k))
    dim = len(vectores[0])
    splits = partir(list(range(len(vectores))), n=n_splits)

    iteraciones = []
    asignaciones = [0] * len(vectores)
    dif_combiner = 0.0
    for iteracion in range(1, max_iter + 1):
        # Map: asignar cada punto al centroide más cercano
        asignaciones = [mapper_asignar(v, centroides) for v in vectores]
        sin = [(c, (*v, 1)) for v, c in zip(vectores, asignaciones)]
        con = []
        for indices in splits:
            con.extend(combinar_registros([sin[i] for i in indices]))

        # Reduce (igual que MapReduceAlgorithm): promediar los puntos de cada cluster
        grupos = defaultdict(list)
        for v, c in zip(vectores, asignaciones):
            grupos[c].append(v)
        nuevos = list(centroides)
        for c, puntos in grupos.items():
            nuevos[c] = reducer_promediar(puntos)

        # Mismo Reduce, pero desde las sumas parciales del combiner
        sumas = defaultdict(lambda: [0.0] * (dim + 1))
        for c, valor in con:
            for i, x in enumerate(valor):
                sumas[c][i] += x
        for c, s in sumas.items():
            desde_parciales = tuple(s[d] / s[dim] for d in range(dim))
            dif_combiner = max(dif_combiner, distancia2(desde_parciales, nuevos[c]) ** 0.5)

        movimiento = max(distancia2(nuevos[c], centroides[c]) ** 0.5 for c in range(k))
        centroides = nuevos
        iteraciones.append({
            "iteracion": iteracion,
            "sin_combiner": tamano_shuffle(sin),
            "con_combiner": tamano_shuffle(con),
            "movimiento": movimiento,
        })
        if movimiento < tol:
            break

    return {
        "centroides": centroides,
        "asignaciones": asignaciones,
        "iteraciones": iteracion,
        "detalle": iteraciones,
        "dif_combiner": dif_combiner,
        "total_sin": _sumar(iteraciones, "sin_combiner"),
        "total_con": _sumar(iteraciones, "con_combiner"),
    }


def _sumar(iteraciones, clave):
    return {
        "registros": sum(it[clave]["registros"] for it in iteraciones),
        "bytes": sum(it[clave]["bytes"] for it in iteraciones),
    }


def analisis_kmeans(stats_por_sku, semilla=SEMILLA_KMEANS, n_splits=SPLITS_KMEANS):
    """K-Means sobre la tabla de SKUs de la Task 2, con el costo de cada iteración."""
    skus = sorted(stats_por_sku)
    parametros = calcular_estandarizacion(stats_por_sku)
    vectores = [vector_estandarizado(stats_por_sku[s], parametros) for s in skus]
    r = kmeans_con_costo(vectores, semilla=semilla, n_splits=n_splits)

    # la versión instrumentada debe dar exactamente el mismo resultado que la de la Task 2
    centroides, asignaciones, iteraciones = kmeans_una_corrida(vectores, K, MAX_ITER, TOL, semilla)
    r["igual_a_task2"] = (
        asignaciones == r["asignaciones"]
        and iteraciones == r["iteraciones"]
        and centroides == r["centroides"]
    )
    r["puntos"] = len(vectores)
    return r


# ---------------------------------------------------------------------------
# Crecimiento lineal y extrapolación
# ---------------------------------------------------------------------------

def ajuste_lineal(factores, valores):
    """
    Regresión lineal valores = intercepto + pendiente * factor. Devuelve también R² (None si
    todos los valores son iguales), las razones valores/valores[0] y la desviación máxima
    respecto a la proporcionalidad exacta (razón observada / razón esperada - 1).
    """
    pendiente, intercepto = statistics.linear_regression(factores, valores)
    media = sum(valores) / len(valores)
    ss_tot = sum((v - media) ** 2 for v in valores)
    ss_res = sum((v - (intercepto + pendiente * f)) ** 2 for f, v in zip(factores, valores))
    razones = [v / valores[0] for v in valores]
    esperadas = [f / factores[0] for f in factores]
    return {
        "pendiente": pendiente,
        "intercepto": intercepto,
        "r2": None if ss_tot == 0 else 1 - ss_res / ss_tot,
        "razones": razones,
        "desviacion_max": max(abs(r / e - 1) for r, e in zip(razones, esperadas)),
    }


def clasificar_crecimiento(factores, valores, tol=TOL_LINEAL, tol_constante=TOL_CONSTANTE):
    """
    'constante' (el máximo supera al mínimo en menos de `tol_constante`), 'lineal' (desviación
    máxima de la proporcionalidad <= `tol`), 'sublineal' o 'superlineal' (crece menos o más
    que los datos).
    """
    if max(valores) <= min(valores) * (1 + tol_constante):
        return "constante"
    ajuste = ajuste_lineal(factores, valores)
    if ajuste["desviacion_max"] <= tol:
        return "lineal"
    crece_menos = valores[-1] / valores[0] < factores[-1] / factores[0]
    return "sublineal" if crece_menos else "superlineal"


def extrapolar(ajuste, objetivo):
    """Valor que predice la recta ajustada en el factor `objetivo`."""
    return ajuste["intercepto"] + ajuste["pendiente"] * objetivo


# ---------------------------------------------------------------------------
# Análisis completo
# ---------------------------------------------------------------------------

def _media_por_iteracion(km, clave, campo):
    return km[clave][campo] / km["iteraciones"]


def ejecutar(factores=FACTORES, factor_validacion=FACTOR_VALIDACION, tam_fragmento=TAM_FRAGMENTO,
             n_splits=SPLITS_KMEANS, n_fragmentos_fijos=N_FRAGMENTOS_FIJOS):
    """
    Corre el análisis a cada factor de `factores` y, si `factor_validacion` no es None, también
    a ese factor (por ejemplo 10x) para contrastar la extrapolación con una corrida real.
    """
    todos = sorted(set(factores) | ({factor_validacion} if factor_validacion else set()))
    datasets = datasets_por_factor(todos)

    por_factor = {}
    for f in todos:
        scans = datasets[f]
        sin = agregacion_sku(scans, False, tam_fragmento)
        con = agregacion_sku(scans, True, tam_fragmento)
        fijo = agregacion_sku(scans, True, n_fragmentos=n_fragmentos_fijos)
        referencia = agregar_por_sku(scans)
        km = analisis_kmeans(sin["stats"], n_splits=n_splits)
        por_factor[f] = {
            "scans": len(scans),
            "skus": len(sin["stats"]),
            "agregacion": {"sin": _sin_stats(sin), "con": _sin_stats(con), "con_fijo": _sin_stats(fijo)},
            "resultados_iguales": sin["stats"] == con["stats"] == fijo["stats"] == referencia,
            "kmeans": km,
        }
        por_factor[f]["pipeline"] = {
            "sin": sin["registros"] + km["total_sin"]["registros"],
            "con": con["registros"] + km["total_con"]["registros"],
        }

    series = {
        "Agregación por SKU, registros, sin combiner": lambda d: d["agregacion"]["sin"]["registros"],
        "Agregación por SKU, bytes, sin combiner": lambda d: d["agregacion"]["sin"]["bytes"],
        "Agregación por SKU, registros, con combiner": lambda d: d["agregacion"]["con"]["registros"],
        "Agregación por SKU, bytes, con combiner": lambda d: d["agregacion"]["con"]["bytes"],
        "Agregación por SKU, registros, con combiner y 8 map tasks fijos":
            lambda d: d["agregacion"]["con_fijo"]["registros"],
        "K-Means, registros por iteración, sin combiner":
            lambda d: _media_por_iteracion(d["kmeans"], "total_sin", "registros"),
        "K-Means, registros por iteración, con combiner":
            lambda d: _media_por_iteracion(d["kmeans"], "total_con", "registros"),
        "Pipeline completo, registros, sin combiner": lambda d: d["pipeline"]["sin"],
        "Pipeline completo, registros, con combiner": lambda d: d["pipeline"]["con"],
    }
    linealidad = {}
    for nombre, extraer in series.items():
        valores = [extraer(por_factor[f]) for f in factores]
        ajuste = ajuste_lineal(list(factores), valores)
        entrada = {
            "valores": valores,
            "ajuste": ajuste,
            "crecimiento": clasificar_crecimiento(list(factores), valores),
        }
        if factor_validacion:
            prediccion = extrapolar(ajuste, factor_validacion)
            medido = extraer(por_factor[factor_validacion])
            entrada.update({
                "prediccion": prediccion,
                "medido": medido,
                "error_relativo": (prediccion - medido) / medido if medido else None,
            })
        linealidad[nombre] = entrada

    return {
        "factores": list(factores),
        "factor_validacion": factor_validacion,
        "por_factor": por_factor,
        "linealidad": linealidad,
    }


def _sin_stats(r):
    return {k: v for k, v in r.items() if k != "stats"}


# ---------------------------------------------------------------------------
# Salida por consola
# ---------------------------------------------------------------------------

def _kib(b):
    return b / 1024


def imprimir_resultados(factores=FACTORES, factor_validacion=FACTOR_VALIDACION):
    r = ejecutar(factores, factor_validacion)
    todos = sorted(r["por_factor"])

    print(f"Shuffle del pipeline de la Task 2 (agregación por SKU + K-Means, k={K}).")
    print(f"Datos: {N_BASE:,} scans = 1x; map tasks de {TAM_FRAGMENTO:,} scans para la agregación "
          f"y {SPLITS_KMEANS} para K-Means.")
    print("Los resultados (stats por SKU) son idénticos con y sin combiner: "
          + ("sí" if all(d["resultados_iguales"] for d in r["por_factor"].values()) else "NO"))
    print("K-Means instrumentado == K-Means de la Task 2: "
          + ("sí" if all(d["kmeans"]["igual_a_task2"] for d in r["por_factor"].values()) else "NO"))
    print(f"Diferencia máxima de centroides con/sin combiner: "
          f"{max(d['kmeans']['dif_combiner'] for d in r['por_factor'].values()):.2e}")

    print("\n1) Agregación por SKU")
    print(f"  {'factor':>6} {'scans':>8} {'SKUs':>5} {'maps':>5} | {'sin combiner':>20} | "
          f"{'con combiner':>20} | {'registros':>9}")
    print(f"  {'':>6} {'':>8} {'':>5} {'':>5} | {'registros':>10} {'KiB':>9} | "
          f"{'registros':>10} {'KiB':>9} | {'ahorrado':>9}")
    for f in todos:
        d = r["por_factor"][f]
        s, c = d["agregacion"]["sin"], d["agregacion"]["con"]
        marca = "  <- validación" if f not in r["factores"] else ""
        print(f"  {f:>5}x {d['scans']:>8,} {d['skus']:>5,} {s['mappers']:>5} | "
              f"{s['registros']:>10,} {_kib(s['bytes']):>9.1f} | "
              f"{c['registros']:>10,} {_kib(c['bytes']):>9.1f} | "
              f"{100 * (1 - c['registros'] / s['registros']):>8.1f}%{marca}")

    print(f"\n   Variante con {N_FRAGMENTOS_FIJOS} map tasks fijos (el combiner mejora con los datos):")
    print(f"  {'factor':>6} {'scans por map':>14} {'registros con combiner':>23} {'ahorrado':>9}")
    for f in todos:
        d = r["por_factor"][f]
        fijo = d["agregacion"]["con_fijo"]
        print(f"  {f:>5}x {d['scans'] // N_FRAGMENTOS_FIJOS:>14,} {fijo['registros']:>23,} "
              f"{100 * (1 - fijo['registros'] / d['scans']):>8.1f}%")

    print("\n2) K-Means, shuffle de cada iteración")
    print(f"  {'factor':>6} {'puntos':>7} {'iter.':>6} | {'sin combiner':>20} | {'con combiner':>20}")
    print(f"  {'':>6} {'':>7} {'':>6} | {'reg./iter.':>10} {'KiB/iter.':>9} | "
          f"{'reg./iter.':>10} {'KiB/iter.':>9}")
    for f in todos:
        km = r["por_factor"][f]["kmeans"]
        n = km["iteraciones"]
        print(f"  {f:>5}x {km['puntos']:>7,} {n:>6} | "
              f"{km['total_sin']['registros'] / n:>10,.0f} {_kib(km['total_sin']['bytes']) / n:>9.1f} | "
              f"{km['total_con']['registros'] / n:>10,.0f} {_kib(km['total_con']['bytes']) / n:>9.1f}")

    f1 = todos[0]
    print(f"\n   Detalle por iteración a {f1}x (una corrida, semilla {SEMILLA_KMEANS}):")
    print(f"  {'iter.':>6} | {'sin: registros':>14} {'KiB':>8} | {'con: registros':>14} {'KiB':>8} | "
          f"{'movimiento':>10}")
    for it in r["por_factor"][f1]["kmeans"]["detalle"]:
        print(f"  {it['iteracion']:>6} | {it['sin_combiner']['registros']:>14,} "
              f"{_kib(it['sin_combiner']['bytes']):>8.1f} | {it['con_combiner']['registros']:>14,} "
              f"{_kib(it['con_combiner']['bytes']):>8.1f} | {it['movimiento']:>10.4f}")

    print("\n3) Pipeline completo (agregación + todas las iteraciones de K-Means), registros")
    print(f"  {'factor':>6} {'sin combiner':>13} {'con combiner':>13} {'ahorrado':>9}")
    for f in todos:
        p = r["por_factor"][f]["pipeline"]
        print(f"  {f:>5}x {p['sin']:>13,} {p['con']:>13,} {100 * (1 - p['con'] / p['sin']):>8.1f}%")

    print(f"\n4) Crecimiento con los datos ({', '.join(f'{f}x' for f in r['factores'])}) y "
          f"extrapolación a {factor_validacion}x")
    print(f"  {'serie':<62} {'razones':>13} {'crece':>10}", end="")
    print(f" {'predicho':>10} {'medido':>10} {'error':>7}" if factor_validacion else "")
    for nombre, e in r["linealidad"].items():
        razones = " ".join(f"{x:.2f}" for x in e["ajuste"]["razones"][1:])
        linea = f"  {nombre:<62} {razones:>13} {e['crecimiento']:>10}"
        if factor_validacion:
            err = "-" if e["error_relativo"] is None else f"{100 * e['error_relativo']:+.1f}%"
            linea += f" {e['prediccion']:>10,.0f} {e['medido']:>10,.0f} {err:>7}"
        print(linea)
    esperadas = " ".join(f"{f / r['factores'][0]:.2f}" for f in r["factores"][1:])
    print(f"  (razón = valor / valor a {r['factores'][0]}x; si fuera lineal, saldría {esperadas})")


if __name__ == "__main__":
    imprimir_resultados()
