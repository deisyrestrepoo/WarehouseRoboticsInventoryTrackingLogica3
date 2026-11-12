"""
Task 5 - Parallel Efficiency

S13 - Scaling on Balanced Keys: corre el pipeline completo (el mismo que describe la
Task 1: Map agrupa los scans de cada robot por stream_seq y emite pares consecutivos
de bins; Reduce los suma en conteos de arista) con P = 1, 2, 4, 8 workers, sobre
>=2 tamanos de datos, mide el tiempo real de pared (wall-clock) con cada P, y reporta
el speedup S_P = T_1 / T_P y la eficiencia E_P = S_P / P.

(Esta tarea reimplementa el pipeline de la Task 1 aqui mismo, en vez de importarlo,
para que no dependa de que esa tarea de un compañero ya esté en el repo -- la misma
decisión que tomó la Task 3 de la Unidad 3 con la Task 2.)

Dos decisiones de diseño importantes, encontradas probando esto de la forma mas
simple primero y viendo que los resultados no tenian sentido:

1. Particion "balanced keys" por SHARDS, no por pickling. La primera version
   repartia los scans entre los workers pasandolos como argumento a Pool.map (via
   pickle) o haciendo que cada worker releyera el dataset.json completo y filtrara
   su parte. Las dos dieron tiempos PEORES con mas workers, en cualquier tamano de
   datos: mover/releer los datos cuesta mas que contarlos. La solucion real de
   MapReduce es la localidad de datos: el dataset se particiona una sola vez en
   N_SHARDS archivos chicos (por hash(robot_id), igual de parejo que antes), y cada
   worker solo LEE sus propios shards del disco -- nunca recibe los datos de otro
   proceso ni relee el archivo completo.

2. "Pool tibio" (warm pool), no una Pool nueva por corrida. Crear un
   multiprocessing.Pool en Windows arranca procesos nuevos de Python desde cero
   (spawn), y eso cuesta cientos de milisegundos por worker -- mas que todo el
   trabajo que hay que hacer en este pipeline tan liviano. Un cluster real no se
   apaga y se prende entre cada job; por eso aqui se mide por separado el costo de
   arranque del Pool (una sola vez, informativo) y el tiempo de "estado estable"
   (varias corridas reutilizando el mismo Pool ya arrancado), y el speedup/eficiencia
   se calculan sobre el estado estable -- la comparacion justa entre P=1 y P>1.
"""
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict, Counter

RUTA_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.json"
)
N_SHARDS = 8


def cargar_scans(ruta=RUTA_DATASET):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def escalar_dataset(scans, factor):
    """
    Repite el dataset `factor` veces para simular un almacen `factor` veces mas
    grande (mismos 20 robots, mas scans cada uno), con factor=1 devolviendo el
    dataset tal cual. El stream_seq de cada copia se corre hacia adelante para que
    dentro de cada robot el orden siga siendo creciente.
    """
    if factor == 1:
        return list(scans)
    n = len(scans)
    copia = []
    for i in range(factor):
        desplazamiento = i * n
        for s in scans:
            nuevo = dict(s)
            nuevo["scan_id"] = f"{s['scan_id']}_c{i}"
            nuevo["stream_seq"] = s["stream_seq"] + desplazamiento
            copia.append(nuevo)
    return copia


def escribir_shards(scans, prefijo, n_shards=N_SHARDS):
    """
    Particion "balanced keys" por localidad de datos: reparte los robots (y todos
    sus scans) entre `n_shards` archivos en disco, por hash(robot_id) % n_shards --
    la misma idea de claves balanceadas que `particionar`, pero escrita UNA VEZ al
    disco en vez de pasada en memoria entre procesos. Devuelve la lista de rutas.
    """
    robots = sorted({s["robot_id"] for s in scans})
    asignacion = {r: hash(r) % n_shards for r in robots}
    shards = [[] for _ in range(n_shards)]
    for s in scans:
        shards[asignacion[s["robot_id"]]].append(s)

    rutas = []
    for i, shard in enumerate(shards):
        ruta = f"{prefijo}_shard{i}.json"
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(shard, f)
        rutas.append(ruta)
    return rutas


def mapper_shards(rutas):
    """
    Map de un worker: lee SOLO sus propios archivos de shard (nunca el dataset
    completo ni datos de otro worker), agrupa por robot_id y, para cada robot
    ordenado por stream_seq, emite un conteo por cada movimiento consecutivo de un
    bin a otro. Devuelve un Counter {(from_bin, to_bin): conteo} con lo que vio este
    worker -- el merge final lo hace el Reduce.
    """
    por_robot = defaultdict(list)
    for ruta in rutas:
        with open(ruta, encoding="utf-8") as f:
            for s in json.load(f):
                por_robot[s["robot_id"]].append(s)

    conteo = Counter()
    for scans_robot in por_robot.values():
        scans_robot.sort(key=lambda s: s["stream_seq"])
        for actual, siguiente in zip(scans_robot, scans_robot[1:]):
            conteo[(actual["bin_location"], siguiente["bin_location"])] += 1
    return conteo


def reduce_combinar(conteos_parciales):
    """Reduce final: suma los Counter de cada worker en uno solo."""
    total = Counter()
    for c in conteos_parciales:
        total.update(c)
    return total


def repartir_shards(rutas_shards, p):
    """Reparte los N_SHARDS archivos entre los p workers, lo mas parejo posible."""
    grupos = [[] for _ in range(p)]
    for i, ruta in enumerate(rutas_shards):
        grupos[i % p].append(ruta)
    return grupos


def medir_con_pool(rutas_shards, p, repeticiones=5):
    """
    Mide el pipeline con `p` workers reutilizando el mismo Pool (o, si p=1, sin
    levantar ningun proceso aparte -- la referencia secuencial). La PRIMERA corrida
    paga el costo de arrancar los procesos del Pool y se reporta aparte; el tiempo de
    "estado estable" que se usa para el speedup es el minimo de las corridas
    siguientes (misma idea que tomar el minimo de varias repeticiones: filtra ruido
    del sistema operativo).
    """
    grupos = repartir_shards(rutas_shards, p)

    def una_corrida():
        if p == 1:
            resultado = [mapper_shards(g) for g in grupos]
        else:
            resultado = pool.map(mapper_shards, grupos)
        return reduce_combinar(resultado)

    if p == 1:
        pool = None
        tiempos = []
        for _ in range(repeticiones):
            inicio = time.perf_counter()
            una_corrida()
            tiempos.append(time.perf_counter() - inicio)
        costo_arranque = tiempos[0]
        estado_estable = min(tiempos[1:]) if len(tiempos) > 1 else tiempos[0]
    else:
        with mp.Pool(p) as pool:
            tiempos = []
            for _ in range(repeticiones):
                inicio = time.perf_counter()
                una_corrida()
                tiempos.append(time.perf_counter() - inicio)
        costo_arranque = tiempos[0]
        estado_estable = min(tiempos[1:]) if len(tiempos) > 1 else tiempos[0]

    return {"costo_arranque_s": costo_arranque, "estado_estable_s": estado_estable}


def experimento(tamanos=(1, 20), workers=(1, 2, 4, 8), repeticiones=5, scans_base=None, directorio_shards=None):
    """
    Para cada tamano de dataset (factor de `escalar_dataset`) y cada numero de
    workers, mide el pipeline (con `medir_con_pool`). Devuelve una lista de filas con
    tamano, workers, el costo de arranque (informativo), el tiempo de estado estable,
    speedup S_P = T_1/T_P y eficiencia E_P = S_P/P -- S_P y E_P se calculan sobre el
    tiempo de estado estable, no sobre el costo de arranque.
    """
    scans_base = scans_base if scans_base is not None else cargar_scans()
    directorio_shards = directorio_shards or os.path.dirname(RUTA_DATASET)

    filas = []
    for factor in tamanos:
        scans = escalar_dataset(scans_base, factor)
        prefijo = os.path.join(directorio_shards, f"_shards_tmp_f{factor}")
        rutas_shards = escribir_shards(scans, prefijo)
        try:
            mediciones = {p: medir_con_pool(rutas_shards, p, repeticiones) for p in workers}
        finally:
            for ruta in rutas_shards:
                os.remove(ruta)

        t1 = mediciones[workers[0]]["estado_estable_s"]  # workers[0] debe ser 1
        for p in workers:
            m = mediciones[p]
            speedup = t1 / m["estado_estable_s"]
            filas.append({
                "factor": factor,
                "n_scans": len(scans),
                "p": p,
                "costo_arranque_s": m["costo_arranque_s"],
                "tiempo_s": m["estado_estable_s"],
                "speedup": speedup,
                "eficiencia": speedup / p,
            })
    return filas


def imprimir_resultados(tamanos=(1, 20), workers=(1, 2, 4, 8), repeticiones=5):
    nucleos = mp.cpu_count()
    print(f"Esta maquina reporta {nucleos} nucleos logicos; P > {nucleos} sobresuscribe la CPU.")
    print(f"{repeticiones} corridas por P; la primera (costo de arranque del Pool) se reporta "
          f"aparte y no entra en el speedup/eficiencia.\n")

    filas = experimento(tamanos=tamanos, workers=workers, repeticiones=repeticiones)
    for factor in tamanos:
        del_factor = [f for f in filas if f["factor"] == factor]
        print(f"Tamano {factor}x ({del_factor[0]['n_scans']} scans):")
        print(f"  {'P':>3} {'arranque (s)':>13} {'estable (s)':>12} {'speedup S_P':>12} {'eficiencia E_P':>15}")
        for f in del_factor:
            print(f"  {f['p']:>3} {f['costo_arranque_s']:>13.3f} {f['tiempo_s']:>12.3f} "
                  f"{f['speedup']:>12.2f} {f['eficiencia']:>14.1%}")
        print()


if __name__ == "__main__":
    imprimir_resultados()
