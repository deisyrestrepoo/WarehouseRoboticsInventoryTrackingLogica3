"""
Task 1 - MapReduce Pipeline

Dos jobs de MapReduce convierten los scans crudos en el resumen por bin (nodo) de la Unit 4:

  Job 1 (scans -> aristas)
    Map     emite (robot_id, (stream_seq, bin_location, quantity, is_low_stock)) por cada scan.
    Reduce  junta todos los scans de un robot, los ordena por stream_seq y emite una tupla
            por cada par de bins consecutivos ((from_bin, to_bin), (1, quantity, is_low_stock)),
            con quantity e is_low_stock del scan de LLEGADA (igual que la Unit 3). Esas tuplas
            se suman en un registro por arista: move_count, avg_quantity, low_stock_frac.

  Job 2 (aristas -> nodos)
    Map     emite cada arista dos veces: una al bin de origen (movimiento saliente) y otra
            al bin de destino (movimiento entrante, con las sumas de quantity y low stock).
    Reduce  junta por bin_location y calcula out_move_count, in_move_count,
            net_flow = in_move_count - out_move_count, avg_quantity y low_stock_frac.

Cómo se paraleliza el Job 1:

  1. El dataset se parte en N fragmentos (archivos) contiguos, como los bloques de un sistema
     de archivos distribuido (por defecto 8). Un robot aparece en VARIOS fragmentos: ninguno
     tiene la historia completa de un robot, así que el orden por stream_seq solo se puede
     recuperar después del shuffle.
  2. Los workers corren el Map sobre los fragmentos en paralelo (multiprocessing.Pool). Cada
     worker lee solo sus archivos.
  3. Shuffle: todas las tuplas con el mismo robot_id se mandan al mismo reducer,
     hash(robot_id) % R, con un hash determinista (FNV-1a): el de Python cambia en cada
     ejecución y el reparto no sería reproducible.
  4. Los reducers corren en paralelo, uno por partición. Cada uno emite los pares de bins de
     sus robots y los suma localmente por arista (como un combiner); el paso final suma las
     sumas parciales de los reducers en un registro por arista.

El Job 2 trabaja sobre unas 4.000 aristas, así que corre en el proceso principal; sí está
escrito como Map + shuffle + Reduce.

Validación contra la Unit 3 (`validar_contra_unit3`): las aristas deben ser idénticas a las de
`unit3.DTMCConstruction.construir_aristas` (y a data/edges.json), y el resumen por nodo debe
cuadrar con ellas. Los resultados no dependen del número de fragmentos ni de workers.

Nota sobre el resumen por nodo: avg_quantity y low_stock_frac describen lo que el robot
encontró al LLEGAR al bin (promedio sobre los movimientos entrantes), como en las aristas. Por
eso el primer scan de cada robot, que no es la llegada de ningún movimiento, no cuenta:
con 20 robots son 9.980 movimientos entrantes de 10.000 scans.
"""
import contextlib
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit3.DTMCConstruction import (  # noqa: E402
    cargar_scans,
    construir_aristas,
    obtener_aristas,
    trafico_entrante,
)

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_NODOS = os.path.join(RAIZ, "data", "nodes.json")

N_FRAGMENTOS = 8
WORKERS = 4

# Tolerancias para comparar el promedio por nodo contra el promedio ponderado de las
# aristas de la Unit 3, que ya vienen redondeadas (2 y 4 decimales).
TOL_CANTIDAD = 0.011
TOL_FRACCION = 0.0002


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def hash_clave(clave, r):
    """Hash determinista (FNV-1a) de un string: simula hash(clave) % R del shuffle."""
    h = 2166136261
    for ch in str(clave):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h % r


def _orden_bin(nombre_bin):
    """'bin_12' -> 12, para ordenar numéricamente (igual que la Unit 3)."""
    return int(nombre_bin.split("_")[1])


def escribir_fragmentos(scans, directorio, n_fragmentos=N_FRAGMENTOS):
    """
    Parte los scans en `n_fragmentos` archivos JSON contiguos de tamaño casi igual (difieren
    en a lo sumo 1 scan) y devuelve sus rutas. Con menos scans que fragmentos, los
    fragmentos sobrantes quedan vacíos.
    """
    if n_fragmentos < 1:
        raise ValueError("n_fragmentos debe ser >= 1")
    base, resto = divmod(len(scans), n_fragmentos)
    rutas = []
    inicio = 0
    for i in range(n_fragmentos):
        fin = inicio + base + (1 if i < resto else 0)
        ruta = os.path.join(directorio, f"fragmento_{i:02d}.json")
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(scans[inicio:fin], f)
        rutas.append(ruta)
        inicio = fin
    return rutas


# ---------------------------------------------------------------------------
# Job 1: scans -> aristas
# ---------------------------------------------------------------------------

def mapper_job1(scan):
    """Map: un scan -> (robot_id, (stream_seq, bin_location, quantity, is_low_stock))."""
    return scan["robot_id"], (
        scan["stream_seq"], scan["bin_location"], scan["quantity"], int(scan["is_low_stock"])
    )


def map_fragmento(ruta):
    """Tarea de un worker: lee SOLO su fragmento y le aplica el Map a cada scan."""
    with open(ruta, encoding="utf-8") as f:
        return [mapper_job1(s) for s in json.load(f)]


def shuffle_job1(salidas_map, n_reducers):
    """
    Shuffle: junta las salidas de todos los mappers y reparte las tuplas por
    hash(robot_id) % R. Devuelve R particiones {robot_id: [valores]}; todos los registros
    de un robot caen en la misma partición.
    """
    particiones = [defaultdict(list) for _ in range(n_reducers)]
    for salida in salidas_map:
        for robot, valor in salida:
            particiones[hash_clave(robot, n_reducers)][robot].append(valor)
    return particiones


def reducer_job1(valores):
    """
    Reduce de UN robot: ordena sus scans por stream_seq y emite, por cada par de scans
    consecutivos, ((from_bin, to_bin), (1, quantity, is_low_stock)) con los datos del scan
    de llegada.
    """
    ordenados = sorted(valores, key=lambda v: v[0])
    return [
        ((a[1], b[1]), (1, b[2], b[3]))
        for a, b in zip(ordenados, ordenados[1:])
    ]


def reduce_particion(particion):
    """
    Tarea de un reducer: aplica `reducer_job1` a cada robot de su partición y suma los pares
    por arista. Devuelve {(from_bin, to_bin): [move_count, suma_quantity, suma_low_stock]}.
    """
    parcial = defaultdict(lambda: [0, 0, 0])
    for valores in particion.values():
        for clave, (n, cantidad, bajo) in reducer_job1(valores):
            acum = parcial[clave]
            acum[0] += n
            acum[1] += cantidad
            acum[2] += bajo
    return dict(parcial)


def combinar_parciales(parciales):
    """Paso final del Job 1: suma las sumas parciales de todos los reducers por arista."""
    total = defaultdict(lambda: [0, 0, 0])
    for parcial in parciales:
        for clave, (n, cantidad, bajo) in parcial.items():
            acum = total[clave]
            acum[0] += n
            acum[1] += cantidad
            acum[2] += bajo
    return dict(total)


def registros_de_aristas(acumulado):
    """
    Convierte las sumas por arista en los registros de la Unit 3 (edge_id, from_bin, to_bin,
    move_count, avg_quantity, low_stock_frac), con el mismo orden y redondeo.
    """
    claves = sorted(acumulado, key=lambda c: (_orden_bin(c[0]), _orden_bin(c[1])))
    aristas = []
    for i, (from_bin, to_bin) in enumerate(claves, start=1):
        n, suma_q, suma_bajo = acumulado[(from_bin, to_bin)]
        aristas.append({
            "edge_id": f"e{i:03d}",
            "from_bin": from_bin,
            "to_bin": to_bin,
            "move_count": n,
            "avg_quantity": round(suma_q / n, 2),
            "low_stock_frac": round(suma_bajo / n, 4),
        })
    return aristas


def job1(rutas_fragmentos, workers=WORKERS, n_reducers=None):
    """
    Job 1 completo sobre los fragmentos: Map en paralelo -> shuffle -> Reduce en paralelo
    -> suma por arista. Devuelve (acumulado_por_arista, estadisticas).
    Con workers=1 todo corre en el proceso principal, sin Pool.
    """
    if workers < 1:
        raise ValueError("workers debe ser >= 1")
    n_reducers = n_reducers or workers
    est = {"workers": workers, "fragmentos": len(rutas_fragmentos), "reducers": n_reducers}

    pool = mp.Pool(workers) if workers > 1 else None
    try:
        t0 = time.perf_counter()
        if pool:
            salidas = pool.map(map_fragmento, rutas_fragmentos)
        else:
            salidas = [map_fragmento(r) for r in rutas_fragmentos]
        est["t_map"] = time.perf_counter() - t0
        est["tuplas_map_por_fragmento"] = [len(s) for s in salidas]
        est["robots_por_fragmento"] = [len({robot for robot, _ in s}) for s in salidas]

        t0 = time.perf_counter()
        particiones = shuffle_job1(salidas, n_reducers)
        est["t_shuffle"] = time.perf_counter() - t0
        est["robots_por_reducer"] = [len(p) for p in particiones]
        est["registros_por_reducer"] = [sum(len(v) for v in p.values()) for p in particiones]

        t0 = time.perf_counter()
        if pool:
            parciales = pool.map(reduce_particion, particiones)
        else:
            parciales = [reduce_particion(p) for p in particiones]
        est["t_reduce"] = time.perf_counter() - t0
    finally:
        if pool:
            pool.close()
            pool.join()

    acumulado = combinar_parciales(parciales)
    est["tuplas_map"] = sum(est["tuplas_map_por_fragmento"])
    est["movimientos"] = sum(v[0] for v in acumulado.values())
    est["aristas"] = len(acumulado)
    return acumulado, est


# ---------------------------------------------------------------------------
# Job 2: aristas -> nodos
# ---------------------------------------------------------------------------

def mapper_job2(clave_arista, acumulado):
    """
    Map: una arista -> 2 tuplas. Al bin de origen va el movimiento saliente; al de destino,
    el entrante con las sumas de quantity y low stock de los scans de llegada.
    """
    from_bin, to_bin = clave_arista
    n, suma_q, suma_bajo = acumulado
    return [
        (from_bin, ("out", n, 0, 0)),
        (to_bin, ("in", n, suma_q, suma_bajo)),
    ]


def shuffle_job2(salida_map):
    grupos = defaultdict(list)
    for bin_, valor in salida_map:
        grupos[bin_].append(valor)
    return grupos


def reducer_job2(bin_, valores):
    """Reduce de UN bin: resume sus movimientos salientes y entrantes."""
    salida = sum(n for tipo, n, _, _ in valores if tipo == "out")
    entrada = sum(n for tipo, n, _, _ in valores if tipo == "in")
    suma_q = sum(q for tipo, _, q, _ in valores if tipo == "in")
    suma_bajo = sum(b for tipo, _, _, b in valores if tipo == "in")
    return {
        "bin_location": bin_,
        "out_move_count": salida,
        "in_move_count": entrada,
        "net_flow": entrada - salida,
        "avg_quantity": round(suma_q / entrada, 2) if entrada else 0.0,
        "low_stock_frac": round(suma_bajo / entrada, 4) if entrada else 0.0,
    }


def job2(acumulado):
    """Job 2 completo: devuelve (nodos ordenados por bin, estadisticas)."""
    salida_map = []
    for clave, valores in acumulado.items():
        salida_map.extend(mapper_job2(clave, valores))
    grupos = shuffle_job2(salida_map)
    nodos = [reducer_job2(bin_, valores) for bin_, valores in grupos.items()]
    nodos.sort(key=lambda n: _orden_bin(n["bin_location"]))
    est = {
        "tuplas_entrada": len(acumulado),
        "tuplas_map": len(salida_map),
        "nodos": len(nodos),
    }
    return nodos, est


# ---------------------------------------------------------------------------
# Pipeline completo y validación
# ---------------------------------------------------------------------------

def ejecutar_pipeline(scans, n_fragmentos=N_FRAGMENTOS, workers=WORKERS, n_reducers=None,
                      directorio=None):
    """
    Corre Job 1 + Job 2 sobre `scans`: escribe los fragmentos (en `directorio` o en uno
    temporal que se borra al terminar), corre el Job 1 con `workers` procesos y luego el Job 2.
    Devuelve {"aristas", "nodos", "stats"}.
    """
    with _directorio_de_trabajo(directorio) as d:
        rutas = escribir_fragmentos(scans, d, n_fragmentos)
        acumulado, est1 = job1(rutas, workers, n_reducers)
    aristas = registros_de_aristas(acumulado)
    nodos, est2 = job2(acumulado)
    # el Map emite una tupla por scan, así que las tuplas de un fragmento son sus scans
    est1["scans_por_fragmento"] = est1["tuplas_map_por_fragmento"]
    return {"aristas": aristas, "nodos": nodos, "stats": {"job1": est1, "job2": est2}}


@contextlib.contextmanager
def _directorio_de_trabajo(directorio):
    """Directorio temporal (se borra al salir) o, si se da `directorio`, ese mismo (se conserva)."""
    if directorio is None:
        with tempfile.TemporaryDirectory() as d:
            yield d
    else:
        os.makedirs(directorio, exist_ok=True)
        yield directorio


def validar_contra_unit3(scans, aristas, nodos):
    """
    Compara el resultado del pipeline contra la Unit 3. Devuelve una lista de
    (nombre, ok, detalle).
    """
    checks = []

    ref = construir_aristas(scans)
    checks.append((
        "Aristas == unit3.construir_aristas(scans)",
        aristas == ref,
        f"{len(aristas)} aristas del pipeline, {len(ref)} de la Unit 3",
    ))

    try:
        en_disco = obtener_aristas()
        checks.append((
            "Aristas == data/edges.json",
            aristas == en_disco,
            f"{len(en_disco)} aristas en el archivo",
        ))
    except OSError as exc:  # pragma: no cover - solo si no existe el archivo
        checks.append(("Aristas == data/edges.json", False, str(exc)))

    entrante = trafico_entrante(ref)
    saliente = defaultdict(int)
    suma_q = defaultdict(float)
    suma_bajo = defaultdict(float)
    for e in ref:
        saliente[e["from_bin"]] += e["move_count"]
        suma_q[e["to_bin"]] += e["move_count"] * e["avg_quantity"]
        suma_bajo[e["to_bin"]] += e["move_count"] * e["low_stock_frac"]

    bins_ref = set(entrante) | set(saliente)
    checks.append((
        "Mismos bins que la Unit 3",
        {n["bin_location"] for n in nodos} == bins_ref,
        f"{len(nodos)} nodos, {len(bins_ref)} bins en las aristas",
    ))
    checks.append((
        "out_move_count == suma de move_count de aristas que salen del bin",
        all(n["out_move_count"] == saliente[n["bin_location"]] for n in nodos),
        "",
    ))
    checks.append((
        "in_move_count == unit3.trafico_entrante",
        all(n["in_move_count"] == entrante[n["bin_location"]] for n in nodos),
        "",
    ))
    checks.append((
        "net_flow == in_move_count - out_move_count",
        all(n["net_flow"] == n["in_move_count"] - n["out_move_count"] for n in nodos),
        "",
    ))
    checks.append((
        "Suma de net_flow == 0 (cada movimiento sale de un bin y entra a otro)",
        sum(n["net_flow"] for n in nodos) == 0,
        f"suma = {sum(n['net_flow'] for n in nodos)}",
    ))

    robots = len({s["robot_id"] for s in scans})
    total_entrante = sum(n["in_move_count"] for n in nodos)
    checks.append((
        "Movimientos == scans - robots (el primer scan de cada robot no es una llegada)",
        total_entrante == len(scans) - robots,
        f"{total_entrante} movimientos = {len(scans)} scans - {robots} robots",
    ))

    dif_q = max((abs(n["avg_quantity"] - suma_q[n["bin_location"]] / n["in_move_count"])
                 for n in nodos if n["in_move_count"]), default=0.0)
    dif_f = max((abs(n["low_stock_frac"] - suma_bajo[n["bin_location"]] / n["in_move_count"])
                 for n in nodos if n["in_move_count"]), default=0.0)
    checks.append((
        "avg_quantity coincide con el promedio ponderado de las aristas",
        dif_q <= TOL_CANTIDAD,
        f"diferencia máxima {dif_q:.4f} (por el redondeo de las aristas)",
    ))
    checks.append((
        "low_stock_frac coincide con el promedio ponderado de las aristas",
        dif_f <= TOL_FRACCION,
        f"diferencia máxima {dif_f:.6f} (por el redondeo de las aristas)",
    ))
    return checks


def verificar_independencia(scans, configuraciones=((1, 8), (2, 8), (4, 8), (2, 3), (4, 16))):
    """
    Corre el pipeline con distintas combinaciones (workers, fragmentos) y comprueba que el
    resultado es idéntico en todas. Devuelve una lista de (workers, fragmentos, tiempo, igual).
    """
    base = None
    filas = []
    for workers, fragmentos in configuraciones:
        t0 = time.perf_counter()
        r = ejecutar_pipeline(scans, n_fragmentos=fragmentos, workers=workers)
        t = time.perf_counter() - t0
        resultado = (r["aristas"], r["nodos"])
        if base is None:
            base = resultado
        filas.append((workers, fragmentos, t, resultado == base))
    return filas


def guardar_nodos(nodos, ruta=RUTA_NODOS):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(nodos, f, indent=2)


def obtener_nodos(forzar_reconstruccion=False):
    """Carga data/nodes.json si existe; si no (o si se pide forzar), corre el pipeline."""
    if not forzar_reconstruccion and os.path.exists(RUTA_NODOS):
        with open(RUTA_NODOS, encoding="utf-8") as f:
            return json.load(f)
    nodos = ejecutar_pipeline(cargar_scans())["nodos"]
    guardar_nodos(nodos)
    return nodos


# ---------------------------------------------------------------------------
# Salida por consola
# ---------------------------------------------------------------------------

def imprimir_resultados(n_fragmentos=N_FRAGMENTOS, workers=WORKERS):
    scans = cargar_scans()
    r = ejecutar_pipeline(scans, n_fragmentos=n_fragmentos, workers=workers)
    j1, j2 = r["stats"]["job1"], r["stats"]["job2"]
    robots = len({s["robot_id"] for s in scans})

    print(f"Dataset: {len(scans)} scans de {robots} robots")
    print(f"Job 1 con {j1['workers']} workers, {j1['fragmentos']} fragmentos y "
          f"{j1['reducers']} reducers\n")

    print("Map en paralelo, por fragmento:")
    print(f"  {'fragmento':>9} {'scans':>6} {'robots que aparecen':>20}")
    for i, (n, rb) in enumerate(zip(j1["scans_por_fragmento"], j1["robots_por_fragmento"])):
        print(f"  {i:>9} {n:>6} {rb:>20}")
    print(f"  (cada robot aparece en varios fragmentos: el orden por stream_seq "
          f"solo se recupera tras el shuffle)\n")

    print("Reducers (hash(robot_id) % R):")
    print(f"  {'reducer':>7} {'robots':>7} {'registros':>10}")
    for i, (rb, reg) in enumerate(zip(j1["robots_por_reducer"], j1["registros_por_reducer"])):
        print(f"  {i:>7} {rb:>7} {reg:>10}")

    print(f"\nJob 1: {j1['tuplas_map']} tuplas del Map -> {j1['movimientos']} movimientos "
          f"-> {j1['aristas']} aristas")
    print(f"       tiempo: map {j1['t_map'] * 1000:.1f} ms, shuffle {j1['t_shuffle'] * 1000:.1f} ms, "
          f"reduce {j1['t_reduce'] * 1000:.1f} ms")
    print(f"Job 2: {j2['tuplas_entrada']} aristas -> {j2['tuplas_map']} tuplas del Map "
          f"(2 por arista) -> {j2['nodos']} nodos\n")

    print("Resumen por nodo (los 5 bins con más tráfico entrante):")
    print(f"  {'bin':<8} {'out':>5} {'in':>5} {'net':>5} {'avg_qty':>8} {'low_stock':>10}")
    for n in sorted(r["nodos"], key=lambda x: -x["in_move_count"])[:5]:
        print(f"  {n['bin_location']:<8} {n['out_move_count']:>5} {n['in_move_count']:>5} "
              f"{n['net_flow']:>5} {n['avg_quantity']:>8.2f} {n['low_stock_frac']:>10.4f}")

    print("\nValidación contra la Unit 3:")
    todos_ok = True
    for nombre, ok, detalle in validar_contra_unit3(scans, r["aristas"], r["nodos"]):
        todos_ok &= ok
        extra = f"  ({detalle})" if detalle else ""
        print(f"  [{'OK' if ok else 'FALLA'}] {nombre}{extra}")
    print(f"  -> {'todo coincide' if todos_ok else 'HAY DIFERENCIAS'}")

    print("\nMismo resultado con distinto número de workers y fragmentos:")
    print("  (el tiempo incluye escribir los fragmentos y arrancar el Pool; con solo 10.000 scans")
    print("   ese costo fijo pesa más que el cómputo: el speedup real se mide en la Task 5)")
    print(f"  {'workers':>7} {'fragmentos':>10} {'tiempo s':>9} {'igual al primero':>17}")
    for w, fr, t, igual in verificar_independencia(scans):
        print(f"  {w:>7} {fr:>10} {t:>9.3f} {'sí' if igual else 'NO':>17}")

    guardar_nodos(r["nodos"])
    print(f"\nResumen por nodo guardado en data/nodes.json ({len(r['nodos'])} nodos)")


if __name__ == "__main__":
    imprimir_resultados()
