"""
Task 5 - Randomized 2-SAT (algoritmo de Papadimitriou)

Restricciones de reabastecimiento sobre los bins más transitados del grafo de la Task 1, resueltas
con el algoritmo aleatorizado de Papadimitriou (1991) para 2-SAT y comparadas con un solver exacto.

Variables (dos por bin candidato b, los K bins con mayor probabilidad estacionaria π):
  R_b = "se desvía un robot hacia b"        P_b = "se prioriza el reabastecimiento en b"

Cinco reglas de restricción (cada cláusula tiene a lo sumo dos literales, o la fórmula no sería 2-SAT):
  1. Urgencia: si π_b es alta (≥ factor_pi / N) y la fracción de bajo stock al llegar a b es alta
     (≥ la tasa global de la bodega), entonces R_b ∨ P_b. Si b no cumple la condición, la regla no
     genera cláusula para b (el "si" es falso y la implicación se cumple sola). La condición sale de
     los datos, así que no es una variable de la fórmula.
  2. Un solo recurso por bin: ¬R_b ∨ ¬P_b (no se desvía un robot y se prioriza a la vez).
  3. Un solo desvío por pasillo: ¬R_a ∨ ¬R_b si a y b están en el mismo pasillo.
  4. Vecinos fuertes no se priorizan a la vez: ¬P_a ∨ ¬P_b si entre a y b hay una arista con al menos
     `umbral_arista` movimientos (en cualquiera de los dos sentidos).
  5. La prioridad bloquea el desvío al vecino fuerte: ¬P_a ∨ ¬R_b y ¬P_b ∨ ¬R_a para esas mismas aristas.
Es un modelo de ejemplo: las reglas 2 a 5 son supuestos de diseño, no hechos medidos en el dataset.

Algoritmo de Papadimitriou: se parte de una asignación aleatoria; mientras haya una cláusula
insatisfecha, se elige una al azar y se invierte una de sus dos variables, también al azar. Si la
fórmula es satisfacible, el número de variables que coinciden con una solución fija hace una caminata
aleatoria en {0, ..., n} que sube con probabilidad ≥ 1/2, así que el número esperado de inversiones es
≤ n². Por Markov, con 2n² inversiones se encuentra una solución con probabilidad ≥ 1/2 y, repitiendo
r rondas con asignaciones nuevas, la probabilidad de fallar es ≤ 2^-r. Si la fórmula es
insatisfacible, el algoritmo nunca termina bien y solo puede rendirse tras agotar las rondas.

Convención de literales: entero v > 0 es la variable v y -v su negación (variables numeradas desde 1).

(La π de los bins se calcula aquí por iteración de potencias, para que esta tarea no dependa de
que la Task 2 de un compañero ya esté en el repo.)
"""
import os
import random
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit3.DTMCConstruction import cargar_grafo, cargar_scans  # noqa: E402

SEMILLA = 42
K_BINS = 8            # 8 bins candidatos -> 16 variables (n >= 8)
FACTOR_PI = 2.0       # "alta" = al menos 2 veces la probabilidad uniforme 1/N
UMBRAL_ARISTA = 50    # "arista fuerte" = al menos 50 movimientos
REGLAS = (
    "1 urgencia",
    "2 un recurso por bin",
    "3 un desvío por pasillo",
    "4 vecinos no se priorizan a la vez",
    "5 prioridad bloquea desvío al vecino",
)


# Construcción de la fórmula a partir de los datos

def _pi(estados, M, iteraciones=2000, tol=1e-12):
    """π de M por iteración de potencias sobre (I + M)/2, que converge aunque M fuera periódica."""
    n = len(estados)
    pi = {s: 1.0 / n for s in estados}
    for _ in range(iteraciones):
        nuevo = {s: 0.5 * pi[s] for s in estados}
        for origen in estados:
            fila = M.get(origen, {})
            if not fila:
                nuevo[origen] += 0.5 * pi[origen]
            for destino, p in fila.items():
                nuevo[destino] += 0.5 * pi[origen] * p
        diferencia = sum(abs(nuevo[s] - pi[s]) for s in estados)
        pi = nuevo
        if diferencia < tol:
            break
    return pi


def _bajo_stock_por_bin(aristas):
    """Fracción de bajo stock al llegar a cada bin (ponderada por move_count) y tasa global."""
    suma, total = defaultdict(float), defaultdict(int)
    for e in aristas:
        suma[e["to_bin"]] += e["low_stock_frac"] * e["move_count"]
        total[e["to_bin"]] += e["move_count"]
    por_bin = {b: suma[b] / total[b] for b in total}
    tasa_global = sum(suma.values()) / sum(total.values())
    return por_bin, tasa_global


def _pasillo_por_bin(scans):
    return {s["bin_location"]: s["aisle"] for s in scans}


def construir_formula(k_bins=K_BINS, factor_pi=FACTOR_PI, umbral_arista=UMBRAL_ARISTA):
    """
    Arma la fórmula 2-CNF. Devuelve un diccionario con las variables (nombres, 1..n), las cláusulas
    (pares de literales), la regla que originó cada cláusula y los datos usados (candidatos, urgentes,
    aristas fuertes, π y fracciones de bajo stock).
    """
    aristas, estados, M, _ = cargar_grafo()
    pi = _pi(estados, M)
    bajo_stock, tasa_global = _bajo_stock_por_bin(aristas)
    pasillo = _pasillo_por_bin(cargar_scans())

    candidatos = sorted(estados, key=lambda b: -pi[b])[:k_bins]
    r = {b: 2 * i + 1 for i, b in enumerate(candidatos)}   # variable R_b
    p = {b: 2 * i + 2 for i, b in enumerate(candidatos)}   # variable P_b
    nombres = []
    for b in candidatos:
        nombres += [f"R_{b}", f"P_{b}"]

    urgentes = [
        b for b in candidatos
        if pi[b] >= factor_pi / len(estados) and bajo_stock.get(b, 0.0) >= tasa_global
    ]

    fuertes = set()
    for e in aristas:
        a, b = e["from_bin"], e["to_bin"]
        if a != b and a in r and b in r and e["move_count"] >= umbral_arista:
            fuertes.add(tuple(sorted((a, b), key=candidatos.index)))
    fuertes = sorted(fuertes, key=lambda par: (candidatos.index(par[0]), candidatos.index(par[1])))

    clausulas, regla_de = [], []

    def agregar(regla, a, b):
        clausulas.append((a, b))
        regla_de.append(regla)

    for b in urgentes:                                              # regla 1
        agregar(REGLAS[0], r[b], p[b])
    for b in candidatos:                                            # regla 2
        agregar(REGLAS[1], -r[b], -p[b])
    for i, a in enumerate(candidatos):                              # regla 3
        for b in candidatos[i + 1:]:
            if pasillo[a] == pasillo[b]:
                agregar(REGLAS[2], -r[a], -r[b])
    for a, b in fuertes:                                            # reglas 4 y 5
        agregar(REGLAS[3], -p[a], -p[b])
        agregar(REGLAS[4], -p[a], -r[b])
        agregar(REGLAS[4], -p[b], -r[a])

    return {
        "n": len(nombres),
        "nombres": nombres,
        "clausulas": clausulas,
        "reglas": regla_de,
        "candidatos": candidatos,
        "urgentes": urgentes,
        "fuertes": fuertes,
        "pi": {b: pi[b] for b in candidatos},
        "bajo_stock": {b: bajo_stock.get(b, 0.0) for b in candidatos},
        "tasa_global": tasa_global,
        "pasillo": {b: pasillo[b] for b in candidatos},
    }


def describir_clausula(formula, clausula):
    def lit(x):
        return ("¬" if x < 0 else "") + formula["nombres"][abs(x) - 1]
    a, b = clausula
    return lit(a) if a == b else f"{lit(a)} ∨ {lit(b)}"


# Evaluación de cláusulas

def literal_verdadero(literal, asignacion):
    return asignacion[abs(literal) - 1] == (literal > 0)


def clausulas_insatisfechas(clausulas, asignacion):
    return [c for c in clausulas if not (literal_verdadero(c[0], asignacion) or literal_verdadero(c[1], asignacion))]


def satisface(clausulas, asignacion):
    return not clausulas_insatisfechas(clausulas, asignacion)


# Algoritmo de Papadimitriou

def papadimitriou(clausulas, n, rng, max_pasos=None):
    """
    Una ronda: asignación inicial aleatoria y hasta `max_pasos` inversiones (2n² por defecto).
    Devuelve (asignación, inversiones hechas), o (None, max_pasos) si no la encontró.
    """
    if max_pasos is None:
        max_pasos = 2 * n * n
    asignacion = [rng.random() < 0.5 for _ in range(n)]
    for paso in range(max_pasos + 1):
        pendientes = clausulas_insatisfechas(clausulas, asignacion)
        if not pendientes:
            return asignacion, paso
        if paso == max_pasos:
            break
        literal = rng.choice(rng.choice(pendientes))
        asignacion[abs(literal) - 1] = not asignacion[abs(literal) - 1]
    return None, max_pasos


def resolver(clausulas, n, rondas=20, semilla=SEMILLA):
    """
    Repite rondas de 2n² inversiones con asignaciones nuevas. Si la fórmula es satisfacible, la
    probabilidad de fallar tras `rondas` rondas es ≤ 2^-rondas. Devuelve (asignación o None,
    inversiones totales, rondas usadas).
    """
    rng = random.Random(semilla)
    total = 0
    for ronda in range(1, rondas + 1):
        asignacion, pasos = papadimitriou(clausulas, n, rng)
        total += pasos
        if asignacion is not None:
            return asignacion, total, ronda
    return None, total, rondas


# Solver exacto (grafo de implicaciones + componentes fuertemente conexas)

def _nodo(literal):
    return 2 * (literal - 1) if literal > 0 else 2 * (-literal - 1) + 1


def sat_exacto(clausulas, n):
    """
    2-SAT exacto en tiempo lineal. Cada cláusula (a ∨ b) da las implicaciones ¬a → b y ¬b → a; la
    fórmula es satisfacible si y solo si ninguna variable queda en la misma componente fuertemente
    conexa que su negación. Devuelve (satisfacible, asignación o None).
    """
    adj = [[] for _ in range(2 * n)]
    radj = [[] for _ in range(2 * n)]
    for a, b in clausulas:
        for u, v in ((_nodo(-a), _nodo(b)), (_nodo(-b), _nodo(a))):
            adj[u].append(v)
            radj[v].append(u)

    visitado = [False] * (2 * n)
    orden = []
    for inicio in range(2 * n):
        if visitado[inicio]:
            continue
        visitado[inicio] = True
        pila = [(inicio, iter(adj[inicio]))]
        while pila:
            nodo, vecinos = pila[-1]
            for v in vecinos:
                if not visitado[v]:
                    visitado[v] = True
                    pila.append((v, iter(adj[v])))
                    break
            else:
                orden.append(nodo)
                pila.pop()

    comp = [-1] * (2 * n)
    numero = 0
    for inicio in reversed(orden):
        if comp[inicio] != -1:
            continue
        comp[inicio] = numero
        pila = [inicio]
        while pila:
            nodo = pila.pop()
            for v in radj[nodo]:
                if comp[v] == -1:
                    comp[v] = numero
                    pila.append(v)
        numero += 1

    asignacion = []
    for i in range(n):
        if comp[2 * i] == comp[2 * i + 1]:
            return False, None
        asignacion.append(comp[2 * i] > comp[2 * i + 1])  # componentes numeradas en orden topológico
    return True, asignacion


# Experimentos

def experimento_pasos(clausulas, n, repeticiones=2000, semilla_base=1000, tope_factor=100):
    """
    Corre `repeticiones` rondas independientes (sin reiniciar al llegar a 2n²: tope de tope_factor·n²)
    y resume cuántas inversiones hicieron falta. Compara con la cota teórica n² (esperanza) y con 2n²
    (probabilidad de éxito ≥ 1/2 por Markov).
    """
    pasos = []
    for k in range(repeticiones):
        asignacion, usados = papadimitriou(clausulas, n, random.Random(semilla_base + k), tope_factor * n * n)
        if asignacion is None:
            raise RuntimeError("una ronda no terminó: ¿la fórmula es satisfacible?")
        pasos.append(usados)
    return {
        "n": n,
        "repeticiones": repeticiones,
        "media": statistics.mean(pasos),
        "mediana": statistics.median(pasos),
        "maximo": max(pasos),
        "cota_esperanza": n * n,
        "dentro_de_2n2": sum(p <= 2 * n * n for p in pasos) / repeticiones,
    }


def formula_aleatoria_satisfacible(n, m, rng):
    """m cláusulas de 2 literales, cada una verdadera bajo una asignación 'plantada' (satisfacible)."""
    plantada = [rng.random() < 0.5 for _ in range(n)]
    clausulas = []
    while len(clausulas) < m:
        a = rng.choice([-1, 1]) * rng.randint(1, n)
        b = rng.choice([-1, 1]) * rng.randint(1, n)
        if abs(a) != abs(b) and (literal_verdadero(a, plantada) or literal_verdadero(b, plantada)):
            clausulas.append((a, b))
    return clausulas


def escalamiento(ns=(8, 16, 32, 64), repeticiones=200, razon=3, semilla=SEMILLA):
    """Inversiones medias en fórmulas aleatorias satisfacibles con m = razon·n cláusulas, para ver cómo crece con n."""
    rng = random.Random(semilla)
    filas = []
    for n in ns:
        pasos = []
        for k in range(repeticiones):
            clausulas = formula_aleatoria_satisfacible(n, razon * n, rng)
            asignacion, usados = papadimitriou(clausulas, n, random.Random(1000 + k), 100 * n * n)
            if asignacion is None:
                raise RuntimeError("una ronda no terminó")
            pasos.append(usados)
        filas.append({
            "n": n,
            "clausulas": razon * n,
            "media": statistics.mean(pasos),
            "maximo": max(pasos),
            "cota_esperanza": n * n,
            "media_sobre_n2": statistics.mean(pasos) / (n * n),
        })
    return filas


def variante_insatisfacible(formula, urgentes_forzados=2):
    """
    Agrega a la fórmula "se desvía un robot a b" como cláusula unitaria (R_b ∨ R_b) para varios bins
    urgentes. Como la regla 3 permite un solo desvío por pasillo, queda insatisfacible.
    """
    n = formula["n"]
    extra = [(formula["nombres"].index(f"R_{b}") + 1,) * 2 for b in formula["urgentes"][:urgentes_forzados]]
    return formula["clausulas"] + extra, n


def interpretar(formula, asignacion):
    """Traduce una asignación a texto: bins con desvío, bins con reabastecimiento prioritario."""
    desviados = [b for i, b in enumerate(formula["candidatos"]) if asignacion[2 * i]]
    priorizados = [b for i, b in enumerate(formula["candidatos"]) if asignacion[2 * i + 1]]
    return desviados, priorizados


def main():
    f = construir_formula()
    n, clausulas = f["n"], f["clausulas"]
    print(f"Candidatos (top {len(f['candidatos'])} por π): {f['candidatos']}")
    print(f"Variables: {n} | cláusulas: {len(clausulas)} | 5 reglas")
    print(f"Tasa global de bajo stock: {f['tasa_global']:.4f} | bins urgentes (π alta y bajo stock alto): {f['urgentes']}")
    print(f"Aristas fuertes (≥ {UMBRAL_ARISTA} movimientos) entre candidatos: {f['fuertes']}")
    for regla in REGLAS:
        ejemplos = [describir_clausula(f, c) for c, r in zip(clausulas, f["reglas"]) if r == regla]
        print(f"  Regla {regla}: {len(ejemplos)} cláusulas, p. ej. {ejemplos[0] if ejemplos else '-'}")

    sat, solucion = sat_exacto(clausulas, n)
    print(f"\nSolver exacto (SCC): {'satisfacible' if sat else 'INSATISFACIBLE'}")
    if sat:
        print(f"  Solución exacta -> desvíos: {interpretar(f, solucion)[0]}, prioridades: {interpretar(f, solucion)[1]}")

    asignacion, total, rondas = resolver(clausulas, n)
    print(f"\nPapadimitriou (semilla {SEMILLA}, rondas de 2n² = {2 * n * n} inversiones):")
    print(f"  {'solución' if asignacion else 'sin solución'} tras {total} inversiones en {rondas} ronda(s); "
          f"satisface todas las cláusulas: {satisface(clausulas, asignacion) if asignacion else '-'}")
    if asignacion:
        print(f"  desvíos: {interpretar(f, asignacion)[0]}, prioridades: {interpretar(f, asignacion)[1]}")

    r = experimento_pasos(clausulas, n)
    print(f"\nInversiones en {r['repeticiones']} rondas independientes (cota teórica: esperanza ≤ n² = {r['cota_esperanza']}):")
    print(f"  media {r['media']:.1f} | mediana {r['mediana']:.0f} | máximo {r['maximo']} | "
          f"rondas que terminan en ≤ 2n² inversiones: {r['dentro_de_2n2']:.1%} (garantía: ≥ 50 %)")

    print("\nCrecimiento con n en fórmulas aleatorias satisfacibles (m = 3n cláusulas, 200 rondas por n):")
    print(f"{'n':>4} {'m':>4} {'media':>8} {'máximo':>7} {'n²':>6} {'media/n²':>9}")
    for fila in escalamiento():
        print(f"{fila['n']:>4} {fila['clausulas']:>4} {fila['media']:>8.1f} {fila['maximo']:>7} "
              f"{fila['cota_esperanza']:>6} {fila['media_sobre_n2']:>9.3f}")

    cl_ins, n_ins = variante_insatisfacible(f)
    sat_i, _ = sat_exacto(cl_ins, n_ins)
    asign_i, total_i, rondas_i = resolver(cl_ins, n_ins)
    print("\nVariante insatisfacible (se fuerza el desvío a dos bins urgentes del mismo pasillo):")
    print(f"  Solver exacto: {'satisfacible' if sat_i else 'INSATISFACIBLE'} | Papadimitriou: "
          f"{'encontró solución' if asign_i else f'sin solución tras {total_i} inversiones en {rondas_i} rondas'}")


if __name__ == "__main__":
    main()
