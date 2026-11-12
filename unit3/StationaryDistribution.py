"""
Task 2 - Stationary Distribution

Distribución estacionaria π de la cadena de Markov que construyó la Task 1 (estados =
bins, matriz M = movimientos de robots entre bins, normalizados por fila). π describe
a dónde van a parar los robots a largo plazo, sin importar en qué bin empezaron: π_i es
la fracción del tiempo que, en el largo plazo, un robot pasa (o llega) al bin i.

Se calcula por iteración de potencias: π_{t+1} = π_t · M, empezando desde la distribución
uniforme, hasta que π deja de moverse (||π_{t+1} - π_t||_1 < tol). Como el grafo es una
sola componente fuertemente conexa (Task 1) y no hay bins sin salida en este dataset, la
cadena es irreducible y la iteración converge a una única π > 0, sin importar el punto
de partida.

Luego se interpreta π comparándola contra dos referencias "crudas", para ver qué agrega
pensar en términos de cadena de Markov en vez de solo contar:
  - El conteo crudo de scans por bin (todas las lecturas de robots en ese bin).
  - El tráfico entrante del grafo de movimientos (llegadas, ya calculado en la Task 1).
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit3.DTMCConstruction import cargar_grafo, cargar_scans, trafico_entrante  # noqa: E402


def power_iteration(estados, M, tol=1e-10, max_iter=10_000, pi0=None):
    """
    Itera π_{t+1} = π_t · M hasta que la distancia L1 entre dos iteraciones sucesivas
    baja de `tol`, o se agoten las `max_iter` iteraciones.

    M es un diccionario de diccionarios (como lo devuelve `cargar_grafo`): M[origen][destino]
    = probabilidad. Un bin sin fila (sin movimientos salientes) se trata como absorbente:
    toda la masa que llega ahí se queda ahí (ver DTMCConstruction: en este dataset no ocurre,
    pero el código no asume que nunca vaya a pasar).

    Devuelve (π final, iteraciones usadas, historial de distancias L1 entre iteraciones
    sucesivas — para analizar la velocidad de convergencia).
    """
    n = len(estados)
    pi = dict(pi0) if pi0 else {s: 1.0 / n for s in estados}
    historial = []
    iteraciones = 0
    for iteraciones in range(1, max_iter + 1):
        nuevo = {s: 0.0 for s in estados}
        for origen in estados:
            masa = pi[origen]
            if masa == 0.0:
                continue
            fila = M.get(origen, {})
            if not fila:
                nuevo[origen] += masa  # sin salida: absorbente de facto
            else:
                for destino, p in fila.items():
                    nuevo[destino] += masa * p
        diferencia = sum(abs(nuevo[s] - pi[s]) for s in estados)
        historial.append(diferencia)
        pi = nuevo
        if diferencia < tol:
            break
    return pi, iteraciones, historial


def distribucion_estacionaria(tol=1e-10, max_iter=10_000):
    """Carga el grafo de la Task 1 y calcula su distribución estacionaria π."""
    aristas, estados, M, estados_sin_salida = cargar_grafo()
    pi, iteraciones, historial = power_iteration(estados, M, tol=tol, max_iter=max_iter)
    return pi, estados, iteraciones, historial, estados_sin_salida


def verificar_estacionaria(pi, M, tol_residual=1e-8):
    """
    Comprueba que π sea de verdad un punto fijo: π · M debe ser (casi) igual a π.
    Devuelve el residuo máximo |(π·M)_i − π_i| y si pasa la tolerancia pedida.
    """
    resultado = {s: 0.0 for s in pi}
    for origen, masa in pi.items():
        for destino, p in M.get(origen, {}).items():
            resultado[destino] += masa * p
        if not M.get(origen):
            resultado[origen] += masa
    residuo_max = max(abs(resultado[s] - pi[s]) for s in pi)
    return residuo_max, residuo_max < tol_residual


def conteo_scans_por_bin(scans=None):
    """Conteo crudo: cuántos scans tuvo cada bin en todo el dataset (referencia 'ingenua',
    sin ningún modelo de cadena de Markov detrás)."""
    scans = scans if scans is not None else cargar_scans()
    return Counter(s["bin_location"] for s in scans)


def _ranking(diccionario):
    """Bins ordenados de mayor a menor valor."""
    return [b for b, _ in sorted(diccionario.items(), key=lambda kv: -kv[1])]


def _spearman(ranking_a, ranking_b):
    """
    Correlación de Spearman entre dos rankings completos de los mismos elementos
    (coeficiente de correlación de Pearson sobre las posiciones/rangos, sin usar numpy
    ni scipy). 1.0 = mismo orden exacto; -1.0 = orden exactamente invertido.
    """
    pos_a = {b: i for i, b in enumerate(ranking_a)}
    pos_b = {b: i for i, b in enumerate(ranking_b)}
    n = len(ranking_a)
    d2 = sum((pos_a[b] - pos_b[b]) ** 2 for b in ranking_a)
    return 1 - (6 * d2) / (n * (n ** 2 - 1))


def comparar_rankings(pi, referencia, top_n=10):
    """
    Compara el ranking de bins por π contra el ranking por un diccionario de referencia
    (conteo crudo o tráfico entrante): cuántos bins coinciden en el top-`top_n` de ambos
    y qué tan parecidos son los dos órdenes completos (Spearman).
    """
    ranking_pi = _ranking(pi)
    ranking_ref = _ranking(referencia)
    interseccion = set(ranking_pi[:top_n]) & set(ranking_ref[:top_n])
    return {
        "top_pi": ranking_pi[:top_n],
        "top_referencia": ranking_ref[:top_n],
        "coincidencias_top_n": len(interseccion),
        "spearman": _spearman(ranking_pi, ranking_ref),
    }


def imprimir_resultados(top_n=10):
    pi, estados, iteraciones, historial, estados_sin_salida = distribucion_estacionaria()
    aristas, _, M, _ = cargar_grafo()

    print(f"Cadena con {len(estados)} estados (bins). Convergio en {iteraciones} iteraciones "
          f"(||pi(t+1) - pi(t)||_1 < 1e-10).")
    if estados_sin_salida:
        print(f"Bins sin salida (absorbentes de facto): {estados_sin_salida}")
    else:
        print("Todos los bins tienen salida: la cadena es irreducible y pi > 0 en todos.")

    residuo, ok = verificar_estacionaria(pi, M)
    print(f"Verificacion pi*M = pi: residuo maximo {residuo:.2e} "
          f"({'OK' if ok else 'FUERA DE TOLERANCIA'}).")

    suma = sum(pi.values())
    print(f"Suma de pi: {suma:.10f} (debe ser 1.0)")

    print(f"\nTop {top_n} bins por participacion de largo plazo (pi):")
    for b in _ranking(pi)[:top_n]:
        print(f"  {b}: pi = {pi[b]:.4%}")

    scans = cargar_scans()
    crudo = conteo_scans_por_bin(scans)
    entrante = trafico_entrante(aristas)

    print(f"\nComparación contra el conteo crudo de scans por bin (top {top_n}):")
    cmp_crudo = comparar_rankings(pi, crudo, top_n)
    print(f"  Top por pi           : {cmp_crudo['top_pi']}")
    print(f"  Top por conteo crudo: {cmp_crudo['top_referencia']}")
    print(f"  Coinciden {cmp_crudo['coincidencias_top_n']}/{top_n} bins en ambos top-{top_n}. "
          f"Spearman (orden completo): {cmp_crudo['spearman']:.3f}")

    print(f"\nComparación contra el tráfico entrante del grafo (llegadas, Task 1):")
    cmp_entrante = comparar_rankings(pi, entrante, top_n)
    print(f"  Top por pi               : {cmp_entrante['top_pi']}")
    print(f"  Top por tráfico entrante: {cmp_entrante['top_referencia']}")
    print(f"  Coinciden {cmp_entrante['coincidencias_top_n']}/{top_n} bins en ambos top-{top_n}. "
          f"Spearman (orden completo): {cmp_entrante['spearman']:.3f}")


if __name__ == "__main__":
    imprimir_resultados()
