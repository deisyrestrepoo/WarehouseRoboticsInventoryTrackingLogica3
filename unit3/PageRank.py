"""
Task 3 - PageRank

PageRank con teletransportación sobre el mismo grafo de movimientos de la Task 1
(estados = bins, M = movimientos de robots entre bins). A diferencia de simplemente
usar M tal cual (la distribución estacionaria "cruda" que calcula la Task 2), aquí se
mezcla M con un salto uniforme a cualquier bin, con un factor de amortiguación
(damping) d:

    P = d * M + (1 - d) * U,   U[i][j] = 1/n para todo i, j,   d = 0.85 (estándar)

Esto hace que la matriz de Google P sea ergódica incluso si M no lo fuera (bins sin
salida, componentes separadas), y da más peso a los bins que reciben tráfico desde
MUCHOS bins distintos, no solo a los que reciben MUCHO tráfico de pocos. Para este
dataset ya sabemos (Task 1) que M es irreducible, así que PageRank debería parecerse
bastante a esa distribución estacionaria "cruda"; lo interesante es ver cuánto cambia el
ranking al introducir la teletransportación, y qué tan rápido converge la iteración.

(La iteración de potencias es la misma idea que usa la Task 2 para la distribución
estacionaria de M; aquí se reimplementa para la matriz P, para que esta tarea no
dependa de que la Task 2 de un compañero ya esté en el repo.)
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit3.DTMCConstruction import cargar_grafo, cargar_scans  # noqa: E402

D_DEFECTO = 0.85


def construir_matriz_google(estados, M, d=D_DEFECTO):
    """
    P = d*M + (1-d)*U, con U la teletransportación uniforme (1/n a cada bin).

    Un bin sin fila en M (sin movimientos salientes) teletransportaría siempre, con
    probabilidad 1 en vez de d: así P queda estocástica por filas sin inventarle a M
    movimientos que nunca se observaron. En este dataset no hay bins así (Task 1), pero
    el código no lo asume.
    """
    n = len(estados)
    salto = (1 - d) / n
    P = {}
    for origen in estados:
        fila_M = M.get(origen, {})
        if not fila_M:
            P[origen] = {destino: 1.0 / n for destino in estados}
        else:
            fila = {destino: salto for destino in estados}
            for destino, p in fila_M.items():
                fila[destino] += d * p
            P[origen] = fila
    return P


def power_iteration_con_trayectoria(estados, P, tol=1e-12, max_iter=10_000):
    """
    Igual que una iteración de potencias normal (pi_{t+1} = pi_t · P desde la uniforme,
    hasta que ||pi(t+1)-pi(t)||_1 < tol), pero además devuelve la trayectoria completa
    (pi en cada iteración t, incluyendo t=0), para poder medir después qué tan lejos
    estaba cada iteración del resultado final.
    """
    n = len(estados)
    pi = {s: 1.0 / n for s in estados}
    trayectoria = [dict(pi)]
    for _ in range(max_iter):
        nuevo = {s: 0.0 for s in estados}
        for origen in estados:
            masa = pi[origen]
            if masa == 0.0:
                continue
            for destino, p in P[origen].items():
                nuevo[destino] += masa * p
        diferencia = sum(abs(nuevo[s] - pi[s]) for s in estados)
        pi = nuevo
        trayectoria.append(dict(pi))
        if diferencia < tol:
            break
    return pi, trayectoria


def pagerank(d=D_DEFECTO, tol=1e-12, max_iter=10_000):
    """Calcula PageRank sobre el grafo de movimientos de la Task 1."""
    aristas, estados, M, estados_sin_salida = cargar_grafo()
    P = construir_matriz_google(estados, M, d)
    pi, trayectoria = power_iteration_con_trayectoria(estados, P, tol=tol, max_iter=max_iter)
    return pi, estados, M, P, trayectoria


def analizar_convergencia(trayectoria, pi_final):
    """
    Para cada iteración t de la trayectoria, calcula ||pi(t) - pi_final||_1: qué tan
    lejos estaba esa iteración del resultado final. Es la curva de convergencia que pide
    la tarea (distinto de la distancia entre iteraciones consecutivas: esta mide contra
    el punto de llegada, no contra el paso anterior).
    """
    return [sum(abs(punto[s] - pi_final[s]) for s in pi_final) for punto in trayectoria]


def _ranking(diccionario):
    return [b for b, _ in sorted(diccionario.items(), key=lambda kv: -kv[1])]


def _spearman(ranking_a, ranking_b):
    """Correlación de Spearman entre dos rankings completos de los mismos elementos."""
    pos_a = {b: i for i, b in enumerate(ranking_a)}
    pos_b = {b: i for i, b in enumerate(ranking_b)}
    n = len(ranking_a)
    d2 = sum((pos_a[b] - pos_b[b]) ** 2 for b in ranking_a)
    return 1 - (6 * d2) / (n * (n ** 2 - 1))


def comparar_rankings(pi, referencia, top_n=10):
    """Coincidencias en el top-n y correlación de Spearman entre dos rankings de bins."""
    ranking_pi = _ranking(pi)
    ranking_ref = _ranking(referencia)
    interseccion = set(ranking_pi[:top_n]) & set(ranking_ref[:top_n])
    return {
        "top_pi": ranking_pi[:top_n],
        "top_referencia": ranking_ref[:top_n],
        "coincidencias_top_n": len(interseccion),
        "spearman": _spearman(ranking_pi, ranking_ref),
    }


def conteo_scans_por_bin(scans=None):
    """Conteo crudo de scans por bin, directamente sobre el dataset (sin modelo de cadena)."""
    scans = scans if scans is not None else cargar_scans()
    return Counter(s["bin_location"] for s in scans)


def distribucion_sin_teletransporte(estados, M, tol=1e-12, max_iter=10_000):
    """
    Distribución estacionaria de M tal cual (d=1, sin teletransportar): sirve de
    referencia para medir cuánto cambia el ranking al introducir PageRank. No es la
    Task 2 en sí (esa es responsabilidad de un compañero); es la misma idea aplicada
    aquí solo para poder comparar.
    """
    pi, _ = power_iteration_con_trayectoria(estados, M, tol=tol, max_iter=max_iter)
    return pi


def imprimir_resultados(d=D_DEFECTO, top_n=10):
    pi, estados, M, P, trayectoria = pagerank(d=d)
    curva = analizar_convergencia(trayectoria, pi)

    print(f"PageRank con damping d={d} sobre {len(estados)} bins. "
          f"Convergio en {len(trayectoria) - 1} iteraciones (||pi(t+1)-pi(t)||_1 < 1e-12).")
    print(f"Suma de PageRank: {sum(pi.values()):.10f} (debe ser 1.0)")

    print(f"\nCurva de convergencia ||pi(t) - pi_final||_1 (primeras 10 iteraciones):")
    for t, valor in enumerate(curva[:10]):
        print(f"  t={t:>2}: {valor:.6f}")
    print(f"  ... t={len(curva) - 1} (final): {curva[-1]:.2e}")
    if len(curva) > 2 and curva[1] > 0:
        razon = curva[2] / curva[1]
        print(f"  Razon curva[2]/curva[1] = {razon:.3f}: la convergencia es geometrica. La "
              f"teoria solo garantiza que no sea mas lenta que d={d} por iteracion (cota "
              f"superior); aqui converge bastante mas rapido que esa cota.")

    print(f"\nTop {top_n} bins por PageRank:")
    for b in _ranking(pi)[:top_n]:
        print(f"  {b}: PageRank = {pi[b]:.4%}")

    crudo = conteo_scans_por_bin()
    print(f"\nPageRank vs. conteo crudo de scans por bin (top {top_n}):")
    cmp_crudo = comparar_rankings(pi, crudo, top_n)
    print(f"  Top por PageRank    : {cmp_crudo['top_pi']}")
    print(f"  Top por conteo crudo: {cmp_crudo['top_referencia']}")
    print(f"  Coinciden {cmp_crudo['coincidencias_top_n']}/{top_n}. "
          f"Spearman (orden completo): {cmp_crudo['spearman']:.3f}")

    pi_sin_tp = distribucion_sin_teletransporte(estados, M)
    print(f"\nPageRank vs. distribucion estacionaria de M sin teletransportar (d=1):")
    cmp_pi = comparar_rankings(pi, pi_sin_tp, top_n)
    print(f"  Top por PageRank         : {cmp_pi['top_pi']}")
    print(f"  Top sin teletransportar  : {cmp_pi['top_referencia']}")
    print(f"  Coinciden {cmp_pi['coincidencias_top_n']}/{top_n}. "
          f"Spearman (orden completo): {cmp_pi['spearman']:.3f}")
    diferencia_max = max(abs(pi[b] - pi_sin_tp[b]) for b in estados)
    print(f"  Mayor diferencia puntual |PageRank - pi(d=1)|: {diferencia_max:.4%} "
          f"(d={d} reparte {1 - d:.0%} de la masa de forma uniforme en cada paso).")


if __name__ == "__main__":
    imprimir_resultados()
