"""
Task 4 - Random Walk (caminata aleatoria perezosa)

Simula un robot que se mueve entre bins siguiendo la cadena de Markov de la Task 1 y comprueba que
la fracción de pasos que pasa en cada bin se acerca a la probabilidad estacionaria teórica π_i.

Caminata perezosa: en cada paso el robot se queda donde está con probabilidad 1/2 (self-loop) y,
con probabilidad 1/2, se mueve según M. La matriz es
L = (I + M) / 2.

  - π sigue siendo estacionaria: π·L = (π + π·M)/2 = π.
  - L siempre es aperiódica (tiene self-loop en cada estado), aunque M no lo sea. Esto importa en
    grafos bipartitos: una caminata simple en un grafo bipartito alterna entre los dos lados, así
    que la distribución en el paso t nunca converge (aunque el promedio en el tiempo sí). Con la
    pereza sí converge, y π se puede calcular por iteración de potencias sobre L sin riesgo.
  - Costo: se necesitan más pasos, porque la mitad de ellos no mueven al robot.

Verificación pedida (≥ 50.000 pasos): se compara la frecuencia empírica de visitas con π_i
(error L1, error máximo por bin y error relativo en los bins más visitados), cómo baja el error
con el número de pasos y cómo converge la distribución exacta del paso t (distancia de variación
total contra π) para la caminata simple y para la perezosa.

(La π se calcula aquí por iteración de potencias sobre L, para que esta tarea no dependa de que la
Task 2 de un compañero ya esté en el repo.)
"""
import bisect
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit3.DTMCConstruction import cargar_grafo  # noqa: E402

SEMILLA = 42
PASOS = 50_000
P_QUEDARSE = 0.5


# Cadena perezosa y distribución estacionaria teórica

def hacer_perezosa(estados, M, p_quedarse=P_QUEDARSE):
    """
    L = p*I + (1-p)*M, con M[origen][destino] = probabilidad (diccionario de diccionarios).
    Un bin sin fila en M (sin salidas) queda como absorbente: L[bin][bin] = 1.
    """
    if not 0 <= p_quedarse < 1:
        raise ValueError("p_quedarse debe estar en [0, 1)")
    L = {}
    for origen in estados:
        fila_M = M.get(origen, {})
        if not fila_M:
            L[origen] = {origen: 1.0}
            continue
        fila = {destino: (1 - p_quedarse) * p for destino, p in fila_M.items()}
        fila[origen] = fila.get(origen, 0.0) + p_quedarse
        L[origen] = fila
    return L


def _fila_completa(estados, M, origen):
    fila = M.get(origen, {})
    return fila if fila else {origen: 1.0}


def paso_distribucion(estados, P, dist):
    """Una iteración dist · P (los bins sin salida retienen su masa)."""
    nuevo = {s: 0.0 for s in estados}
    for origen, masa in dist.items():
        if masa == 0.0:
            continue
        for destino, p in _fila_completa(estados, P, origen).items():
            nuevo[destino] += masa * p
    return nuevo


def distribucion_estacionaria(estados, P, tol=1e-12, max_iter=100_000):
    """π por iteración de potencias desde la uniforme. Devuelve (π, iteraciones)."""
    n = len(estados)
    pi = {s: 1.0 / n for s in estados}
    iteraciones = 0
    for iteraciones in range(1, max_iter + 1):
        nuevo = paso_distribucion(estados, P, pi)
        diferencia = sum(abs(nuevo[s] - pi[s]) for s in estados)
        pi = nuevo
        if diferencia < tol:
            break
    return pi, iteraciones


# Periodo de la cadena

def periodo(estados, M):
    """
    Periodo de una cadena irreducible: mcd de (nivel[u] + 1 - nivel[v]) sobre todas las aristas
    u -> v, con niveles de un BFS desde el primer estado. Periodo 1 = aperiódica. Solo tiene
    sentido si la cadena es irreducible (Task 1: una sola componente fuertemente conexa).
    """
    inicio = estados[0]
    nivel = {inicio: 0}
    cola = [inicio]
    for u in cola:
        for v in M.get(u, {}):
            if v not in nivel:
                nivel[v] = nivel[u] + 1
                cola.append(v)
    g = 0
    for u in nivel:
        for v in M.get(u, {}):
            if v in nivel:
                g = math.gcd(g, abs(nivel[u] + 1 - nivel[v]))
    return g


# Simulación de la caminata

def _preparar_muestreo(estados, P):
    """Por estado (por índice): destinos y probabilidades acumuladas, para muestrear con bisect."""
    indice = {s: i for i, s in enumerate(estados)}
    destinos, acumuladas = [], []
    for s in estados:
        fila = _fila_completa(estados, P, s)
        d, acum, total = [], [], 0.0
        for destino, p in fila.items():
            total += p
            d.append(indice[destino])
            acum.append(total)
        destinos.append(d)
        acumuladas.append(acum)
    return destinos, acumuladas


def simular_caminata(estados, P, pasos=PASOS, inicio=None, semilla=SEMILLA, puntos_control=()):
    """
    Camina `pasos` pasos por la cadena P desde `inicio` (por defecto, el primer estado) y cuenta
    en cuántos de esos pasos el robot terminó en cada bin (no cuenta la posición inicial).

    Devuelve (conteos por bin, {paso: conteos hasta ese paso} para cada paso de `puntos_control`).
    """
    rng = random.Random(semilla)
    destinos, acumuladas = _preparar_muestreo(estados, P)
    actual = estados.index(inicio) if inicio is not None else 0
    conteos = [0] * len(estados)
    control = set(puntos_control)
    instantaneas = {}
    for t in range(1, pasos + 1):
        acum = acumuladas[actual]
        j = bisect.bisect_right(acum, rng.random())
        if j >= len(acum):  # por redondeo, la última acumulada puede quedar apenas bajo 1
            j = len(acum) - 1
        actual = destinos[actual][j]
        conteos[actual] += 1
        if t in control:
            instantaneas[t] = list(conteos)
    return (
        {s: c for s, c in zip(estados, conteos)},
        {t: {s: c for s, c in zip(estados, cs)} for t, cs in instantaneas.items()},
    )


# Comparación empírica vs. teórica

def comparar_con_pi(pi, conteos, top_n=10):
    """
    Frecuencia empírica f_i = visitas_i / pasos contra π_i: error L1 (suma de |f_i − π_i|),
    error máximo absoluto por bin, error relativo promedio en los `top_n` bins con mayor π
    y cuántos de esos `top_n` bins coinciden con los `top_n` más visitados de la simulación.
    """
    pasos = sum(conteos.values())
    f = {s: conteos[s] / pasos for s in pi}
    top_teorico = sorted(pi, key=lambda s: -pi[s])[:top_n]
    top_empirico = sorted(f, key=lambda s: -f[s])[:top_n]
    return {
        "pasos": pasos,
        "frecuencias": f,
        "error_l1": sum(abs(f[s] - pi[s]) for s in pi),
        "error_max": max(abs(f[s] - pi[s]) for s in pi),
        "bin_error_max": max(pi, key=lambda s: abs(f[s] - pi[s])),
        "error_relativo_top": statistics.mean(abs(f[s] - pi[s]) / pi[s] for s in top_teorico),
        "coincidencias_top": len(set(top_teorico) & set(top_empirico)),
        "top_teorico": top_teorico,
        "top_empirico": top_empirico,
    }


def error_vs_pasos(estados, P, pi, pasos_lista=(1_000, 5_000, 10_000, 50_000, 200_000), semillas=20,
                   semilla_base=1000):
    """
    Error L1 promedio (sobre varias semillas) de la frecuencia empírica a distintos números de
    pasos. Si el error baja como 1/√T, el producto error·√T se mantiene aproximadamente constante.
    """
    maximo = max(pasos_lista)
    errores = {t: [] for t in pasos_lista}
    for k in range(semillas):
        _, instantaneas = simular_caminata(
            estados, P, maximo, semilla=semilla_base + k, puntos_control=pasos_lista
        )
        for t in pasos_lista:
            errores[t].append(comparar_con_pi(pi, instantaneas[t])["error_l1"])
    return [
        {
            "pasos": t,
            "error_l1_medio": statistics.mean(errores[t]),
            "error_l1_desviacion": statistics.pstdev(errores[t]),
            "error_por_raiz_t": statistics.mean(errores[t]) * math.sqrt(t),
        }
        for t in pasos_lista
    ]


def variacion_total_por_paso(estados, P, pi, inicio, pasos_lista=(1, 2, 5, 10, 20, 50)):
    """
    Distancia de variación total (1/2 de la distancia L1) entre π y la distribución EXACTA del robot
    en el paso t, partiendo con certeza de `inicio`. Muestra si la distribución converge a π
    (la frecuencia de visitas de arriba converge para cualquier cadena irreducible; esto es más fuerte).
    """
    dist = {s: 0.0 for s in estados}
    dist[inicio] = 1.0
    resultado = {}
    for t in range(1, max(pasos_lista) + 1):
        dist = paso_distribucion(estados, P, dist)
        if t in pasos_lista:
            resultado[t] = 0.5 * sum(abs(dist[s] - pi[s]) for s in estados)
    return resultado


def ejemplo_bipartito():
    """
    Ciclo de 4 bins (a-b-c-d-a), un grafo bipartito: cada paso cambia de lado, así que la caminata
    simple tiene periodo 2. Devuelve la distancia de variación total contra π (uniforme) en los pasos
    1..10, partiendo de 'a', para la caminata simple (se queda en 1/2 para siempre) y para la perezosa
    (converge geométricamente).
    """
    estados = ["a", "b", "c", "d"]
    M = {
        "a": {"b": 0.5, "d": 0.5},
        "b": {"a": 0.5, "c": 0.5},
        "c": {"b": 0.5, "d": 0.5},
        "d": {"c": 0.5, "a": 0.5},
    }
    pi = {s: 0.25 for s in estados}
    perezosa = hacer_perezosa(estados, M)
    pasos = tuple(range(1, 11))
    return {
        "periodo_simple": periodo(estados, M),
        "periodo_perezosa": periodo(estados, perezosa),
        "simple": variacion_total_por_paso(estados, M, pi, "a", pasos),
        "perezosa": variacion_total_por_paso(estados, perezosa, pi, "a", pasos),
    }


def main():
    aristas, estados, M, estados_sin_salida = cargar_grafo()
    L = hacer_perezosa(estados, M)
    pi, iteraciones = distribucion_estacionaria(estados, L)
    pi_simple, it_simple = distribucion_estacionaria(estados, M)

    print(f"Cadena con {len(estados)} estados (bins), semilla {SEMILLA}")
    print(f"Periodo de M: {periodo(estados, M)} | periodo de la caminata perezosa: {periodo(estados, L)}")
    print(f"π de la caminata perezosa converge en {iteraciones} iteraciones; "
          f"π de M, en {it_simple}; distancia L1 entre ambas: "
          f"{sum(abs(pi[s] - pi_simple[s]) for s in estados):.2e} (misma π)")

    inicio = min(estados, key=lambda s: pi[s])
    print(f"\n--- Caminata perezosa de {PASOS:,} pasos desde {inicio} (el bin menos probable) ---")
    conteos, _ = simular_caminata(estados, L, PASOS, inicio=inicio)
    r = comparar_con_pi(pi, conteos)
    print(f"Error L1: {r['error_l1']:.4f} | error máximo por bin: {r['error_max']:.4f} ({r['bin_error_max']})")
    print(f"Error relativo promedio en el top 10 teórico: {r['error_relativo_top']:.1%} | "
          f"coinciden {r['coincidencias_top']}/10 bins en el top 10")
    print(f"{'bin':>8} {'π teórica':>10} {'frecuencia':>11} {'diferencia':>11}")
    for b in r["top_teorico"]:
        print(f"{b:>8} {pi[b]:>10.4%} {r['frecuencias'][b]:>11.4%} {r['frecuencias'][b] - pi[b]:>+11.4%}")

    print("\n--- Error L1 según el número de pasos (promedio de 20 semillas) ---")
    print(f"{'pasos':>9} {'error L1':>9} {'desv.':>7} {'error·√T':>9}")
    for f in error_vs_pasos(estados, L, pi):
        print(f"{f['pasos']:>9,} {f['error_l1_medio']:>9.4f} {f['error_l1_desviacion']:>7.4f} "
              f"{f['error_por_raiz_t']:>9.2f}")

    print(f"\n--- Convergencia exacta de la distribución desde {inicio} (variación total vs. π) ---")
    pasos_lista = (1, 2, 5, 10, 20, 50)
    simple = variacion_total_por_paso(estados, M, pi, inicio, pasos_lista)
    perezosa = variacion_total_por_paso(estados, L, pi, inicio, pasos_lista)
    print(f"{'paso t':>7} {'caminata simple':>16} {'caminata perezosa':>18}")
    for t in pasos_lista:
        print(f"{t:>7} {simple[t]:>16.2e} {perezosa[t]:>18.2e}")

    ej = ejemplo_bipartito()
    print("\n--- Grafo bipartito (ciclo de 4 bins): por qué hace falta la pereza ---")
    print(f"Periodo: simple = {ej['periodo_simple']}, perezosa = {ej['periodo_perezosa']}")
    print(f"{'paso t':>7} {'simple':>9} {'perezosa':>10}")
    for t in (1, 2, 5, 10):
        print(f"{t:>7} {ej['simple'][t]:>9.4f} {ej['perezosa'][t]:>10.4f}")


if __name__ == "__main__":
    main()
