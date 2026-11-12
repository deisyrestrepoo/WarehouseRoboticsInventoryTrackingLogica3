"""
Task 4 - Reducer Size & Replication Rate

Dos jobs de MapReduce construyen el grafo de movimientos (Unit 3/4) a escala:

  Job 1: Map emite (robot_id, (stream_seq, bin_location, quantity, is_low_stock));
         Reduce ordena por stream_seq y emite los pares de bins consecutivos,
         sumados en aristas (from_bin, to_bin, move_count, avg_quantity, low_stock_frac).
  Job 2: a partir de las aristas de Job 1, construye el resumen por bin (nodo):
         out_move_count, in_move_count, net_flow, avg_quantity, low_stock_frac.

Esta tarea mide dos cosas de cada job:

  - replication rate (r): cuántas tuplas de salida produce el Map por cada tupla
    de entrada, en promedio. Job 1 no tiene fan-out (cada scan produce 1 tupla,
    r = 1); Job 2 sí: cada arista se emite dos veces, una por cada bin extremo
    (una para el conteo saliente del from_bin, otra para el entrante del to_bin),
    así que r = 2.
  - hot key y reducer size (q): la clave con más registros (el robot más activo
    en Job 1, el bin con más aristas en Job 2) determina un piso para la carga
    del reducer más cargado: como todos los registros de una misma clave tienen
    que ir al mismo reducer, ningún particionado puede bajar la carga máxima por
    debajo de q, sin importar cuántos reducers (R) se usen.

Para Job 1 se simula el particionado hash(robot_id) % R con R = 1, 2, 4, 8 y se
mide la carga (número de registros) del reducer más cargado en cada caso.
"""
import json
import os
from collections import defaultdict

RUTA_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.json"
)


def cargar_scans(ruta=RUTA_DATASET):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _hash_clave(clave, r):
    """Hash determinista (FNV-1a) sobre un string, para simular hash(key) % R en el shuffle."""
    h = 2166136261
    for ch in str(clave):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h % r


# Job 1: construir aristas (grafo de movimientos) a partir de los scans

def job1_map(scans):
    """Un tuple de salida por cada scan de entrada: sin fan-out, replication rate = 1."""
    return [
        (s["robot_id"], (s["stream_seq"], s["bin_location"], s["quantity"], s["is_low_stock"]))
        for s in scans
    ]


def job1_agrupar(salida_map):
    grupos = defaultdict(list)
    for clave, valor in salida_map:
        grupos[clave].append(valor)
    return grupos


def job1_reduce(grupos):
    """Por cada robot: ordena por stream_seq y emite las aristas (from_bin, to_bin) consecutivas."""
    aristas = defaultdict(lambda: {"move_count": 0, "suma_quantity": 0, "suma_low_stock": 0})
    for registros in grupos.values():
        registros_ordenados = sorted(registros, key=lambda r: r[0])  # stream_seq
        for (_, bin_a, _, _), (_, bin_b, cant_b, bajo_b) in zip(registros_ordenados, registros_ordenados[1:]):
            e = aristas[(bin_a, bin_b)]
            e["move_count"] += 1
            e["suma_quantity"] += cant_b
            e["suma_low_stock"] += bajo_b
    return aristas


def job1_replication_rate(scans, salida_map):
    """r = tuplas de salida del Map / tuplas de entrada."""
    return len(salida_map) / len(scans)


def job1_hot_key(grupos):
    """El robot_id con más registros (hot key) y q = su cantidad de scans."""
    robot, registros = max(grupos.items(), key=lambda kv: len(kv[1]))
    return robot, len(registros)


def job1_carga_por_reducer(grupos, r):
    """Simula el particionado hash(robot_id) % r y devuelve la carga de cada uno de los r reducers."""
    carga = [0] * r
    for clave, registros in grupos.items():
        carga[_hash_clave(clave, r)] += len(registros)
    return carga


# Job 2: resumen por bin (nodo), a partir de las aristas de Job 1

def job2_map(aristas):
    """Cada arista produce 2 tuplas de salida (una por extremo): replication rate = 2."""
    salida = []
    for (from_bin, to_bin), e in aristas.items():
        salida.append((from_bin, ("out", e["move_count"], 0.0, 0.0)))
        avg_q = e["suma_quantity"] / e["move_count"]
        frac_bajo = e["suma_low_stock"] / e["move_count"]
        salida.append((to_bin, ("in", e["move_count"], avg_q, frac_bajo)))
    return salida


def job2_agrupar(salida_map):
    grupos = defaultdict(list)
    for clave, valor in salida_map:
        grupos[clave].append(valor)
    return grupos


def job2_reduce(grupos):
    resumen = {}
    for bin_, registros in grupos.items():
        salientes = [mc for tipo, mc, _, _ in registros if tipo == "out"]
        entrantes = [(mc, aq, fb) for tipo, mc, aq, fb in registros if tipo == "in"]
        out_count = sum(salientes)
        in_count = sum(mc for mc, _, _ in entrantes)
        if in_count > 0:
            avg_q = sum(mc * aq for mc, aq, _ in entrantes) / in_count
            frac_bajo = sum(mc * fb for mc, _, fb in entrantes) / in_count
        else:
            avg_q, frac_bajo = 0.0, 0.0
        resumen[bin_] = {
            "bin_location": bin_,
            "out_move_count": out_count,
            "in_move_count": in_count,
            "net_flow": in_count - out_count,
            "avg_quantity": round(avg_q, 2),
            "low_stock_frac": round(frac_bajo, 4),
        }
    return resumen


def job2_replication_rate(aristas, salida_map):
    return len(salida_map) / len(aristas)


def job2_hot_key(grupos):
    """El bin con más registros (más aristas entrantes + salientes) y q = esa cantidad."""
    bin_, registros = max(grupos.items(), key=lambda kv: len(kv[1]))
    return bin_, len(registros)



def experimento(ruta_dataset=RUTA_DATASET, valores_r=(1, 2, 4, 8)):
    scans = cargar_scans(ruta_dataset)

    # Job 1
    salida1 = job1_map(scans)
    grupos1 = job1_agrupar(salida1)
    aristas = job1_reduce(grupos1)
    r1 = job1_replication_rate(scans, salida1)
    robot_caliente, q1 = job1_hot_key(grupos1)
    cargas_job1 = {r: job1_carga_por_reducer(grupos1, r) for r in valores_r}

    # Job 2
    salida2 = job2_map(aristas)
    grupos2 = job2_agrupar(salida2)
    resumen_nodos = job2_reduce(grupos2)
    r2 = job2_replication_rate(aristas, salida2)
    bin_caliente, q2 = job2_hot_key(grupos2)

    return {
        "n_scans": len(scans),
        "n_aristas": len(aristas),
        "n_bins": len(resumen_nodos),
        "job1": {"r": r1, "hot_key": robot_caliente, "q": q1, "cargas": cargas_job1},
        "job2": {"r": r2, "hot_key": bin_caliente, "q": q2},
        "resumen_nodos": resumen_nodos,
    }


if __name__ == "__main__":
    resultado = experimento()
    print(
        f"Scans: {resultado['n_scans']}  |  Aristas (salida de Job 1): {resultado['n_aristas']}  "
        f"|  Bins (salida de Job 2): {resultado['n_bins']}"
    )

    j1 = resultado["job1"]
    print("\n--- Job 1 (construir aristas) ---")
    print(f"Replication rate r = {j1['r']:.2f}  (cada scan emite exactamente 1 tupla: sin fan-out)")
    print(f"Hot key: {j1['hot_key']}  con q = {j1['q']} scans (el robot más activo)")
    print(f"{'R (reducers)':>14} | {'carga máx. reducer':>20} | {'carga máx. / n':>15} | {'carga máx. / q':>15}")
    for r in sorted(j1["cargas"]):
        carga_max = max(j1["cargas"][r])
        print(
            f"{r:>14} | {carga_max:>20} | {carga_max / resultado['n_scans']:>14.1%} "
            f"| {carga_max / j1['q']:>13.2f}x"
        )

    j2 = resultado["job2"]
    print("\n--- Job 2 (resumen por bin) ---")
    print(f"Replication rate r = {j2['r']:.2f}  (cada arista emite 2 tuplas: una por bin extremo)")
    print(f"Hot key: {j2['hot_key']}  con q = {j2['q']} registros (bin con más aristas entrantes + salientes)")