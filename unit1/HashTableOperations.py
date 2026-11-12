"""
Task 4 - Hash Table Operations

Tabla hash con clave bin_location que agrupa (bucketing) los registros de scan.
Se comparan dos estrategias de inserción:
  - Chaining: cada scan va al bucket h1(bin_location).
  - Power of Two Choices: cada scan puede ir a h1(bin_location) o h2(bin_location)
    y se elige el bucket que en ese momento tenga menos registros.
"""
import json
import os
import random
import statistics
import time

# Primo de Mersenne para la familia de hash universal ((a*x + b) mod P) mod m
P = 2 ** 61 - 1

RUTA_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.json"
)


def cargar_dataset(ruta=RUTA_DATASET):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def clave_a_entero(clave):
    """Convierte un texto a entero de forma determinista (hash() de Python cambia en cada ejecución)."""
    valor = 0
    for caracter in clave:
        valor = (valor * 131 + ord(caracter)) % P
    return valor


class FuncionHash:
    """Hash universal: h(x) = ((a*x + b) mod P) mod m, con a y b aleatorios."""

    def __init__(self, num_buckets, rng):
        self.m = num_buckets
        self.a = rng.randint(1, P - 1)
        self.b = rng.randint(0, P - 1)

    def __call__(self, clave):
        return ((self.a * clave_a_entero(clave) + self.b) % P) % self.m


class TablaChaining:
    """Cada registro se guarda en el bucket h1(bin_location)."""

    nombre = "Chaining"

    def __init__(self, num_buckets, rng):
        self.buckets = [[] for _ in range(num_buckets)]
        self.h1 = FuncionHash(num_buckets, rng)
        self.tiempo_insercion_s = 0.0

    def insertar(self, scan):
        self.buckets[self.h1(scan["bin_location"])].append(scan)

    def buscar(self, clave):
        """Devuelve (registros de esa clave, registros revisados)."""
        bucket = self.buckets[self.h1(clave)]
        return [s for s in bucket if s["bin_location"] == clave], len(bucket)


class TablaDosOpciones:
    """Cada registro va al menos cargado entre los buckets h1(clave) y h2(clave)."""

    nombre = "Power of Two Choices"

    def __init__(self, num_buckets, rng):
        self.buckets = [[] for _ in range(num_buckets)]
        self.h1 = FuncionHash(num_buckets, rng)
        self.h2 = FuncionHash(num_buckets, rng)
        self.tiempo_insercion_s = 0.0

    def insertar(self, scan):
        clave = scan["bin_location"]
        i, j = self.h1(clave), self.h2(clave)
        destino = i if len(self.buckets[i]) <= len(self.buckets[j]) else j
        self.buckets[destino].append(scan)

    def buscar(self, clave):
        """Los registros de una clave pueden estar en cualquiera de sus dos buckets."""
        i, j = self.h1(clave), self.h2(clave)
        indices = {i, j}
        encontrados, revisados = [], 0
        for idx in indices:
            revisados += len(self.buckets[idx])
            encontrados += [s for s in self.buckets[idx] if s["bin_location"] == clave]
        return encontrados, revisados


def cargas(tabla):
    return [len(b) for b in tabla.buckets]


def metricas(tabla, scans):
    """Métricas de balance de carga, costo de búsqueda y tiempos de la tabla ya construida."""
    c = cargas(tabla)
    claves = sorted({s["bin_location"] for s in scans})

    inicio = time.perf_counter()
    revisados_total = 0
    for clave in claves:
        _, revisados = tabla.buscar(clave)
        revisados_total += revisados
    tiempo_busqueda = time.perf_counter() - inicio

    return {
        "carga_max": max(c),
        "carga_min": min(c),
        "carga_promedio": statistics.mean(c),
        "desviacion": statistics.pstdev(c),
        "buckets_vacios": c.count(0),
        "revisados_por_busqueda": revisados_total / len(claves),
        "tiempo_busqueda_s": tiempo_busqueda,
    }


def construir(clase_tabla, scans, num_buckets, semilla):
    rng = random.Random(semilla)
    tabla = clase_tabla(num_buckets, rng)
    inicio = time.perf_counter()
    for scan in scans:
        tabla.insertar(scan)
    tabla.tiempo_insercion_s = time.perf_counter() - inicio
    return tabla


def comparar(scans, num_buckets=100, semilla=42):
    """Construye ambas tablas con los mismos datos y devuelve cargas y métricas (para el dashboard)."""
    resultado = {}
    for clase in (TablaChaining, TablaDosOpciones):
        tabla = construir(clase, scans, num_buckets, semilla)
        m = metricas(tabla, scans)
        m["tiempo_insercion_s"] = tabla.tiempo_insercion_s
        resultado[clase.nombre] = {"cargas": cargas(tabla), "metricas": m}
    return resultado


def repetir(scans, num_buckets=100, repeticiones=30):
    """Repite el experimento con distintas funciones hash y promedia la carga máxima."""
    maximos = {TablaChaining.nombre: [], TablaDosOpciones.nombre: []}
    for r in range(repeticiones):
        for clase in (TablaChaining, TablaDosOpciones):
            tabla = construir(clase, scans, num_buckets, semilla=1000 + r)
            maximos[clase.nombre].append(max(cargas(tabla)))
    return maximos


def imprimir_resultados(titulo, resultado):
    print(f"\n  {titulo} ")
    for nombre, datos in resultado.items():
        m = datos["metricas"]
        print(f"\n{nombre}")
        print(f"  Carga máxima / mínima : {m['carga_max']} / {m['carga_min']}")
        print(f"  Carga promedio        : {m['carga_promedio']:.1f}")
        print(f"  Desviación estándar   : {m['desviacion']:.1f}")
        print(f"  Buckets vacíos        : {m['buckets_vacios']}")
        print(f"  Registros revisados por búsqueda (promedio): {m['revisados_por_busqueda']:.1f}")
        print(f"  Tiempo de inserción   : {m['tiempo_insercion_s'] * 1000:.2f} ms")
        print(f"  Tiempo de búsqueda    : {m['tiempo_busqueda_s'] * 1000:.2f} ms")


if __name__ == "__main__":
    scans = cargar_dataset()
    num_buckets = 100

    imprimir_resultados(
        f"{len(scans)} scans, {num_buckets} buckets", comparar(scans, num_buckets)
    )

    maximos = repetir(scans, num_buckets)
    print("\n  Carga máxima en 30 repeticiones (distintas funciones hash) ")
    for nombre, valores in maximos.items():
        print(f"  {nombre}: promedio {statistics.mean(valores):.1f}, "
              f"mín {min(valores)}, máx {max(valores)}")