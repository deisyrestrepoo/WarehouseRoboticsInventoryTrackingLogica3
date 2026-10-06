"""
Task 1 - DTMC Construction & State Classification

El stream de scans de la Unit 2 se agrega por robot_id (en orden de stream_seq) en un
grafo de movimientos: cada vez que un robot pasa de un bin a otro, eso es una arista.
Con esas aristas se arma una cadena de Markov de tiempo discreto (DTMC): los estados
son los bins, y la matriz de transición M sale de normalizar move_count por fila.

Luego se clasifican los estados. Hay dos clasificaciones, con significados distintos:

  - Formal (teoría de cadenas de Markov): un estado es recurrente si, una vez el
    sistema sale de él, siempre puede volver; es transitorio si existe la posibilidad
    de abandonarlo para siempre. Se calcula con componentes fuertemente conexas (SCC).

  - Práctica (para el negocio): "bin hub" = bin con mucho tráfico entrante (por ahí
    pasan los robots todo el tiempo); "bin poco visitado" = bin con tráfico entrante
    bajo. Se calcula ordenando los bins por cuántas veces llegan robots a ellos.

    Como se ve en el resultado de abajo, con este dataset las dos no coinciden: el
    grafo completo es una sola componente fuertemente conexa, así que en el sentido
    formal TODOS los bins son recurrentes (desde cualquier bin se puede volver a
    cualquier otro). La clasificación que pide la práctica ("hub" vs "poco visitado")
    es la práctica, basada en volumen de tráfico, no la formal.
"""
import json
import os
from collections import defaultdict

RUTA_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.json"
)
RUTA_ARISTAS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "edges.json"
)


# ---------------------------------------------------------------------------
# Construcción del dataset de aristas (agregado por robot_id, en orden de stream_seq)
# ---------------------------------------------------------------------------

def _orden_bin(nombre_bin):
    """'bin_12' -> 12, para ordenar numéricamente y no como texto ('bin_10' < 'bin_2')."""
    return int(nombre_bin.split("_")[1])


def construir_aristas(scans):
    """
    Agrupa los scans por robot_id, los ordena por stream_seq, y por cada par de scans
    consecutivos del mismo robot registra un movimiento from_bin -> to_bin. Si el robot
    se queda en el mismo bin en dos scans seguidos, eso es un self-loop (from_bin == to_bin).
    """
    por_robot = defaultdict(list)
    for s in scans:
        por_robot[s["robot_id"]].append(s)
    for robot in por_robot:
        por_robot[robot].sort(key=lambda s: s["stream_seq"])

    acumulado = defaultdict(lambda: {"move_count": 0, "suma_quantity": 0, "suma_low_stock": 0})
    for robot, scans_robot in por_robot.items():
        for actual, siguiente in zip(scans_robot, scans_robot[1:]):
            clave = (actual["bin_location"], siguiente["bin_location"])
            e = acumulado[clave]
            e["move_count"] += 1
            # avg_quantity y low_stock_frac describen lo que el robot encontró al LLEGAR al bin destino
            e["suma_quantity"] += siguiente["quantity"]
            e["suma_low_stock"] += siguiente["is_low_stock"]

    aristas = []
    claves_ordenadas = sorted(acumulado, key=lambda c: (_orden_bin(c[0]), _orden_bin(c[1])))
    for i, (from_bin, to_bin) in enumerate(claves_ordenadas, start=1):
        e = acumulado[(from_bin, to_bin)]
        aristas.append({
            "edge_id": f"e{i:03d}",
            "from_bin": from_bin,
            "to_bin": to_bin,
            "move_count": e["move_count"],
            "avg_quantity": round(e["suma_quantity"] / e["move_count"], 2),
            "low_stock_frac": round(e["suma_low_stock"] / e["move_count"], 4),
        })
    return aristas


def cargar_scans(ruta=RUTA_DATASET):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def obtener_aristas(forzar_reconstruccion=False):
    """Carga data/edges.json si ya existe; si no (o si se pide forzar), lo construye y lo guarda."""
    if not forzar_reconstruccion and os.path.exists(RUTA_ARISTAS):
        with open(RUTA_ARISTAS, encoding="utf-8") as f:
            return json.load(f)
    aristas = construir_aristas(cargar_scans())
    with open(RUTA_ARISTAS, "w", encoding="utf-8") as f:
        json.dump(aristas, f, indent=2)
    return aristas


# ---------------------------------------------------------------------------
# Construcción de la cadena de Markov (estados + matriz de transición)
# ---------------------------------------------------------------------------

def construir_estados(aristas):
    bins = {e["from_bin"] for e in aristas} | {e["to_bin"] for e in aristas}
    return sorted(bins, key=_orden_bin)


def construir_matriz_transicion(aristas, estados):
    """
    M[i][j] = probabilidad de moverse del bin i al bin j, estimada como
    move_count(i->j) / (suma de move_count saliente de i).
    Se representa como diccionario de diccionarios: M[bin_origen][bin_destino] = probabilidad.
    """
    salientes = defaultdict(lambda: defaultdict(int))
    totales = defaultdict(int)
    for e in aristas:
        salientes[e["from_bin"]][e["to_bin"]] += e["move_count"]
        totales[e["from_bin"]] += e["move_count"]

    M = {}
    estados_sin_salida = []
    for bin_ in estados:
        total = totales.get(bin_, 0)
        if total == 0:
            estados_sin_salida.append(bin_)
            M[bin_] = {}
        else:
            M[bin_] = {destino: cuenta / total for destino, cuenta in salientes[bin_].items()}
    return M, estados_sin_salida


# ---------------------------------------------------------------------------
# Clasificación formal: componentes fuertemente conexas (Kosaraju)
# ---------------------------------------------------------------------------

def _adjacencia(aristas):
    adj = defaultdict(set)
    for e in aristas:
        adj.setdefault(e["from_bin"], set())
        adj.setdefault(e["to_bin"], set())
        if e["from_bin"] != e["to_bin"]:
            adj[e["from_bin"]].add(e["to_bin"])
    return adj


def _componentes_fuertemente_conexas(estados, adj):
    visitado = set()
    orden = []

    def dfs1(inicio):
        pila = [(inicio, iter(adj[inicio]))]
        visitado.add(inicio)
        while pila:
            nodo, vecinos = pila[-1]
            avanzo = False
            for v in vecinos:
                if v not in visitado:
                    visitado.add(v)
                    pila.append((v, iter(adj[v])))
                    avanzo = True
                    break
            if not avanzo:
                orden.append(nodo)
                pila.pop()

    for s in estados:
        if s not in visitado:
            dfs1(s)

    radj = defaultdict(set)
    for u in adj:
        for v in adj[u]:
            radj[v].add(u)

    visitado2 = set()
    componentes = []

    def dfs2(inicio):
        pila = [inicio]
        comp = []
        visitado2.add(inicio)
        while pila:
            nodo = pila.pop()
            comp.append(nodo)
            for v in radj[nodo]:
                if v not in visitado2:
                    visitado2.add(v)
                    pila.append(v)
        return comp

    for nodo in reversed(orden):
        if nodo not in visitado2:
            componentes.append(dfs2(nodo))
    return componentes


def clasificar_formal(estados, aristas):
    """Recurrente/transitorio en el sentido estricto de cadenas de Markov, vía SCC."""
    adj = _adjacencia(aristas)
    componentes = _componentes_fuertemente_conexas(estados, adj)

    comp_de = {}
    for i, comp in enumerate(componentes):
        for n in comp:
            comp_de[n] = i

    cerrada = [True] * len(componentes)
    for u in adj:
        for v in adj[u]:
            if comp_de[u] != comp_de[v]:
                cerrada[comp_de[u]] = False

    clasificacion = {
        n: ("recurrente" if cerrada[comp_de[n]] else "transitorio")
        for n in estados
    }
    return clasificacion, componentes


# ---------------------------------------------------------------------------
# Clasificación práctica: hubs vs. bins poco visitados, por tráfico entrante
# ---------------------------------------------------------------------------

def trafico_entrante(aristas):
    entrante = defaultdict(int)
    for e in aristas:
        entrante[e["to_bin"]] += e["move_count"]
    return entrante


def clasificar_hubs(estados, aristas, top_n=10):
    entrante = trafico_entrante(aristas)
    ranking = sorted(estados, key=lambda b: -entrante.get(b, 0))
    hubs = ranking[:top_n]
    poco_visitados = ranking[-top_n:]
    return hubs, poco_visitados, entrante


def cargar_grafo():
    """
    Punto de entrada para las demás tareas de Unit 3: devuelve las aristas, la lista
    de estados (bins, orden fijo) y la matriz de transición M, ya construidos.
    """
    aristas = obtener_aristas()
    estados = construir_estados(aristas)
    M, estados_sin_salida = construir_matriz_transicion(aristas, estados)
    return aristas, estados, M, estados_sin_salida


if __name__ == "__main__":
    aristas, estados, M, estados_sin_salida = cargar_grafo()
    self_loops = sum(1 for e in aristas if e["from_bin"] == e["to_bin"])

    print(f"Estados (bins): {len(estados)}")
    print(f"Aristas (pares from_bin -> to_bin distintos): {len(aristas)}")
    print(f"  de las cuales self-loops: {self_loops}")
    if estados_sin_salida:
        print(f"Bins sin ningún movimiento saliente (quedarían absorbentes): {estados_sin_salida}")
    else:
        print("Todos los bins tienen al menos un movimiento saliente registrado.")

    sumas = {b: round(sum(M[b].values()), 6) for b in estados if M[b]}
    print(f"Filas de M que suman 1 (o 0 si no tenían salida): "
          f"{sum(1 for v in sumas.values() if v == 1.0)}/{len(sumas)}")

    print("\n--- Clasificación formal (componentes fuertemente conexas) ---")
    clasificacion, componentes = clasificar_formal(estados, aristas)
    recurrentes = [b for b in estados if clasificacion[b] == "recurrente"]
    transitorios = [b for b in estados if clasificacion[b] == "transitorio"]
    print(f"Componentes fuertemente conexas encontradas: {len(componentes)}")
    print(f"Bins recurrentes (formal): {len(recurrentes)}")
    print(f"Bins transitorios (formal): {len(transitorios)}")
    if not transitorios:
        print("El grafo es una sola componente fuertemente conexa: TODOS los bins son "
              "recurrentes en el sentido estricto de cadenas de Markov.")

    print(f"\n--- Clasificación práctica (hubs vs. poco visitados, por tráfico entrante) ---")
    hubs, poco_visitados, entrante = clasificar_hubs(estados, aristas, top_n=10)
    print("Top 10 bins 'hub' (más llegadas de robots):")
    for b in hubs:
        print(f"  {b}: {entrante[b]} llegadas")
    print("10 bins 'poco visitados' (menos llegadas de robots):")
    for b in poco_visitados:
        print(f"  {b}: {entrante[b]} llegadas")