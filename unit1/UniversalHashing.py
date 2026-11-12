"""
Task 3 - Universal Hashing

Hash universal sobre sku para el índice de búsqueda rápida del sistema de reabastecimiento,
y medición de colisiones.

La familia es la de Carter-Wegman:

    h_{a,b}(x) = ((a * x + b) mod P) mod m,   P primo, a en [1, P-1], b en [0, P-1]

Para dos claves distintas x != y y (a, b) elegidos al azar se cumple Pr[h(x) = h(y)] <= 1/m.
Por eso, con n SKUs distintos en m buckets, el número esperado de pares en colisión es a lo
sumo C(n, 2) / m, sin importar cómo sean los SKUs.

Se compara contra dos funciones hash ingenuas (suma de caracteres y clave mod m) y se mide
cuánto más rápido es buscar un SKU con el índice que recorriendo todos los scans.

Nota sobre la lectura de los resultados: con estos SKUs, "clave mod m" (sobre la clave
polinómica de clave_a_entero) reparte incluso mejor que el promedio de la familia universal.
Eso depende de la forma de los datos y no está garantizado; la familia universal da la cota
C(n, 2) / m para cualquier conjunto de SKUs. La suma de caracteres sí es mala: solo produce
menos de 100 valores distintos (82 en este dataset), así que el tamaño de la tabla no ayuda.
"""
import json
import os
import random
import statistics
import time
from math import comb

# Primo de Mersenne para la familia ((a*x + b) mod P) mod m
P = 2 ** 61 - 1

RUTA_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.json"
)

TAMANOS_TABLA = [512, 1024, 2003, 2048, 4096, 4099]


def cargar_dataset(ruta=RUTA_DATASET):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def clave_a_entero(clave):
    """Convierte un texto a entero de forma determinista (hash() de Python cambia en cada ejecución)."""
    valor = 0
    for caracter in clave:
        valor = (valor * 131 + ord(caracter)) % P
    return valor


class FuncionHashUniversal:
    """Un miembro de la familia Carter-Wegman con a y b aleatorios."""

    def __init__(self, num_buckets, rng=None):
        if num_buckets < 1:
            raise ValueError("num_buckets debe ser positivo")
        rng = rng or random.Random()
        self.m = num_buckets
        self.a = rng.randint(1, P - 1)
        self.b = rng.randint(0, P - 1)

    def de_entero(self, x):
        return ((self.a * x + self.b) % P) % self.m

    def __call__(self, sku):
        return self.de_entero(clave_a_entero(sku))


class HashSumaCaracteres:
    """Referencia ingenua: suma de los códigos de los caracteres mod m."""

    def __init__(self, num_buckets):
        self.m = num_buckets

    def __call__(self, sku):
        return sum(ord(c) for c in sku) % self.m


class HashModulo:
    """Referencia ingenua: la clave entera mod m."""

    def __init__(self, num_buckets):
        self.m = num_buckets

    def de_entero(self, x):
        return x % self.m

    def __call__(self, sku):
        return clave_a_entero(sku) % self.m


class IndiceHash:
    """Índice con chaining: sku -> posiciones de sus scans en el dataset."""

    def __init__(self, funcion_hash):
        self.hash = funcion_hash
        self.buckets = [[] for _ in range(funcion_hash.m)]

    def insertar(self, sku, posicion):
        cadena = self.buckets[self.hash(sku)]
        for entrada in cadena:
            if entrada[0] == sku:
                entrada[1].append(posicion)
                return
        cadena.append((sku, [posicion]))

    def buscar(self, sku):
        """Devuelve (posiciones de ese SKU, SKUs revisados dentro del bucket)."""
        cadena = self.buckets[self.hash(sku)]
        for clave, posiciones in cadena:
            if clave == sku:
                return posiciones, len(cadena)
        return [], len(cadena)

    def cargas(self):
        return [len(b) for b in self.buckets]


def metricas_colision(cargas):
    """Métricas de colisión a partir de la cantidad de SKUs distintos en cada bucket."""
    n = sum(cargas)
    return {
        "skus": n,
        "buckets": len(cargas),
        "factor_carga": n / len(cargas),
        "pares_en_colision": sum(comb(c, 2) for c in cargas),
        "skus_en_colision": sum(c for c in cargas if c > 1),
        "buckets_vacios": cargas.count(0),
        "carga_max": max(cargas),
    }


def medir_colisiones(claves_enteras, funcion_hash):
    """Aplica la función a cada SKU distinto (ya convertido a entero) y mide las colisiones."""
    cargas = [0] * funcion_hash.m
    for x in claves_enteras:
        cargas[funcion_hash.de_entero(x)] += 1
    return metricas_colision(cargas)


def tabla_colisiones(skus, tamanos=TAMANOS_TABLA, repeticiones=300, semilla=42):
    """Para cada tamaño m compara la familia universal (promedio sobre varias elecciones de a, b)
    con las dos funciones ingenuas y con la cota teórica C(n, 2) / m."""
    claves = [clave_a_entero(s) for s in skus]
    if len(set(claves)) != len(skus):
        raise ValueError("clave_a_entero produjo claves duplicadas para SKUs distintos")
    n = len(skus)

    filas = []
    for m in tamanos:
        universales = [
            medir_colisiones(claves, FuncionHashUniversal(m, random.Random(semilla * 7919 + m * 31 + r)))
            for r in range(repeticiones)
        ]
        pares = [u["pares_en_colision"] for u in universales]
        suma = metricas_colision_suma(skus, m)
        modulo = medir_colisiones(claves, HashModulo(m))
        filas.append({
            "m": m,
            "factor_carga": n / m,
            "cota_pares": comb(n, 2) / m,
            "universal_pares_prom": statistics.mean(pares),
            "universal_pares_max": max(pares),
            "universal_carga_max_prom": statistics.mean(u["carga_max"] for u in universales),
            "suma_pares": suma["pares_en_colision"],
            "suma_carga_max": suma["carga_max"],
            "modulo_pares": modulo["pares_en_colision"],
            "modulo_carga_max": modulo["carga_max"],
        })
    return filas


def metricas_colision_suma(skus, m):
    cargas = [0] * m
    h = HashSumaCaracteres(m)
    for s in skus:
        cargas[h(s)] += 1
    return metricas_colision(cargas)


def benchmark_busqueda(scans, num_buckets=2003, semilla=42, consultas_indice=1900, consultas_lineal=100):
    """Construye el índice sobre todos los scans y lo compara con recorrer la lista completa."""
    rng = random.Random(semilla)
    indice = IndiceHash(FuncionHashUniversal(num_buckets, rng))

    inicio = time.perf_counter()
    for posicion, scan in enumerate(scans):
        indice.insertar(scan["sku"], posicion)
    t_construccion = time.perf_counter() - inicio

    skus = sorted({s["sku"] for s in scans})
    consultas = [rng.choice(skus) for _ in range(consultas_indice)]

    inicio = time.perf_counter()
    revisados = 0
    for q in consultas:
        _, r = indice.buscar(q)
        revisados += r
    t_indice = (time.perf_counter() - inicio) / len(consultas)

    subconjunto = consultas[:consultas_lineal]
    inicio = time.perf_counter()
    lineales = [[i for i, s in enumerate(scans) if s["sku"] == q] for q in subconjunto]
    t_lineal = (time.perf_counter() - inicio) / len(subconjunto)

    coinciden = all(indice.buscar(q)[0] == esperado for q, esperado in zip(subconjunto, lineales))
    return {
        "buckets": num_buckets,
        "scans": len(scans),
        "tiempo_construccion_s": t_construccion,
        "colisiones": metricas_colision(indice.cargas()),
        "revisados_por_busqueda": revisados / len(consultas),
        "tiempo_indice_us": t_indice * 1e6,
        "tiempo_lineal_us": t_lineal * 1e6,
        "aceleracion": t_lineal / t_indice,
        "coincide_con_busqueda_lineal": coinciden,
    }


def imprimir_resultados(n_skus, n_scans, repeticiones, filas, bench):
    print(f"\n  {n_scans} scans, {n_skus} SKUs distintos, {repeticiones} elecciones de (a, b) por tamaño ")
    print("\nPares de SKUs distintos que caen en el mismo bucket")
    print(f"  {'m':>5} {'carga':>6} {'cota':>7} {'univ prom':>10} {'univ máx':>9} "
          f"{'carga máx':>10} {'suma chars':>11} {'clave mod m':>12}")
    for f in filas:
        print(f"  {f['m']:>5} {f['factor_carga']:>6.2f} {f['cota_pares']:>7.0f} "
              f"{f['universal_pares_prom']:>10.0f} {f['universal_pares_max']:>9} "
              f"{f['universal_carga_max_prom']:>10.1f} "
              f"{f['suma_pares']:>6} ({f['suma_carga_max']:>3}) "
              f"{f['modulo_pares']:>7} ({f['modulo_carga_max']:>3})")
    print("  (cota = C(n,2)/m; carga máx = SKUs en el bucket más lleno; entre paréntesis, la de las ingenuas)")

    c = bench["colisiones"]
    print(f"\nÍndice hash sobre todos los scans ({bench['buckets']} buckets)")
    print(f"  SKUs distintos indexados      : {c['skus']} (factor de carga {c['factor_carga']:.2f})")
    print(f"  Pares en colisión / cadena máx: {c['pares_en_colision']} / {c['carga_max']}")
    print(f"  SKUs revisados por búsqueda   : {bench['revisados_por_busqueda']:.2f} (promedio)")
    print(f"  Construcción                  : {bench['tiempo_construccion_s'] * 1000:.1f} ms")
    print(f"  Búsqueda con índice           : {bench['tiempo_indice_us']:.2f} us por consulta")
    print(f"  Búsqueda lineal               : {bench['tiempo_lineal_us']:.0f} us por consulta")
    print(f"  Aceleración                   : {bench['aceleracion']:.0f}x")
    print(f"  Resultados iguales a la lineal: {bench['coincide_con_busqueda_lineal']}")


if __name__ == "__main__":
    scans = cargar_dataset()
    skus = sorted({s["sku"] for s in scans})
    repeticiones = 300

    filas = tabla_colisiones(skus, repeticiones=repeticiones)
    bench = benchmark_busqueda(scans)
    imprimir_resultados(len(skus), len(scans), repeticiones, filas, bench)
