"""
Task 3 - Stream Sampling

Estimación de frecuencias por muestreo: qué fracción de los SKUs está *actualmente* en bajo
stock, usando solo una muestra de tamaño fijo del stream, y comparación contra la fracción
real (la que se obtendría guardando el estado de todos los SKUs).

"Actualmente" significa el estado del último scan de cada SKU visto hasta ese momento.

Se comparan dos formas de muestrear, ambas en una sola pasada y con memoria O(k):

  1. Reservoir sampling de scans (Algoritmo R): cada scan del stream queda en la muestra con
     la misma probabilidad k/n. Los SKUs que más se escanean aparecen más veces en la muestra,
     así que la fracción *por SKU* puede salir sesgada si la frecuencia de escaneo se relaciona
     con el estado de bajo stock; además, la muestra contiene menos de k SKUs distintos.
  2. Muestreo de SKUs por hash (bottom-k): se guardan los k SKUs distintos con el menor valor de
     hash y se actualiza su estado con cada scan que llega. Como el hash no depende de cuántas
     veces se escanea un SKU, la muestra es uniforme sobre los SKUs distintos y el estimador
     p̂ = (SKUs en bajo stock de la muestra) / k es insesgado, con error estándar
     sqrt( p(1-p)/k * (N-k)/(N-1) ), donde N es el número de SKUs distintos (corrección por
     población finita).

Los tamaños de muestra por defecto son 100, 300 y 900.
"""
import heapq
import json
import math
import os
import random
import statistics

RUTA_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.json"
)

P = 2 ** 61 - 1  # mismo primo de Mersenne usado en unit2/BloomFilter.py

TAMANOS_MUESTRA = (100, 300, 900)

_CACHE_CLAVES = {}


def cargar_stream(ruta=RUTA_DATASET):
    """Carga el dataset ordenado por stream_seq, como llegaría un stream real."""
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
    return sorted(datos, key=lambda s: s["stream_seq"])


def clave_a_entero(clave):
    """Convierte un texto a entero de forma determinista (hash() de Python cambia en cada ejecución)."""
    valor = _CACHE_CLAVES.get(clave)
    if valor is None:
        valor = 0
        for caracter in clave:
            valor = (valor * 131 + ord(caracter)) % P
        _CACHE_CLAVES[clave] = valor
    return valor


class HashSKU:
    """Hash universal sin reducir por m: h(x) = (a*x + b) mod P. Menor valor = más prioridad."""

    def __init__(self, rng):
        self.a = rng.randint(1, P - 1)
        self.b = rng.randint(0, P - 1)

    def __call__(self, sku):
        return (self.a * clave_a_entero(sku) + self.b) % P


class MuestraReservorioScans:
    """Reservoir sampling (Algoritmo R) sobre scans: muestra uniforme de k scans del stream."""

    def __init__(self, k, semilla=42):
        if k < 1:
            raise ValueError("k debe ser positivo")
        self.k = k
        self.rng = random.Random(semilla)
        self.vistos = 0
        self.reservorio = []

    def procesar(self, scan):
        self.vistos += 1
        if len(self.reservorio) < self.k:
            self.reservorio.append(scan)
        else:
            j = self.rng.randrange(self.vistos)
            if j < self.k:
                self.reservorio[j] = scan

    def _ultimo_por_sku(self):
        ultimo = {}
        for scan in self.reservorio:
            previo = ultimo.get(scan["sku"])
            if previo is None or scan["stream_seq"] > previo["stream_seq"]:
                ultimo[scan["sku"]] = scan
        return ultimo

    def skus_en_muestra(self):
        return len({s["sku"] for s in self.reservorio})

    def estimar(self):
        """Fracción de SKUs en bajo stock: último scan muestreado de cada SKU que cayó en la muestra."""
        ultimo = self._ultimo_por_sku()
        if not ultimo:
            return 0.0
        return sum(s["is_low_stock"] for s in ultimo.values()) / len(ultimo)


class MuestraSKUsPorHash:
    """
    Muestra uniforme de k SKUs distintos (bottom-k por hash), con su estado actual.

    Se conservan los k SKUs de menor valor de hash entre todos los vistos. Un SKU solo entra
    la primera vez que aparece (si su hash es menor que el mayor de la muestra) y, mientras
    esté en la muestra, cada scan suyo actualiza su estado. Un SKU descartado no puede volver:
    el umbral solo baja con el tiempo.
    """

    def __init__(self, k, semilla=42):
        if k < 1:
            raise ValueError("k debe ser positivo")
        self.k = k
        self.hash = HashSKU(random.Random(semilla))
        self.estado = {}   # sku -> is_low_stock del último scan visto
        self.heap = []     # max-heap por hash: (-hash, sku)

    def procesar(self, scan):
        sku = scan["sku"]
        if sku in self.estado:
            self.estado[sku] = scan["is_low_stock"]
            return
        h = self.hash(sku)
        if len(self.estado) < self.k:
            heapq.heappush(self.heap, (-h, sku))
        elif h < -self.heap[0][0]:
            _, descartado = heapq.heapreplace(self.heap, (-h, sku))
            del self.estado[descartado]
        else:
            return
        self.estado[sku] = scan["is_low_stock"]

    def tamano(self):
        return len(self.estado)

    def estimar(self):
        """Fracción de SKUs en bajo stock dentro de la muestra."""
        if not self.estado:
            return 0.0
        return sum(self.estado.values()) / len(self.estado)


def fraccion_real(scans):
    """Verdad exacta: fracción de SKUs cuyo último scan está en bajo stock, y número de SKUs."""
    ultimo = {}
    for scan in scans:
        ultimo[scan["sku"]] = scan["is_low_stock"]
    if not ultimo:
        return 0.0, 0
    return sum(ultimo.values()) / len(ultimo), len(ultimo)


def error_estandar(p, k, n_skus):
    """Error estándar teórico de p̂ con k SKUs de N, con corrección por población finita."""
    k = min(k, n_skus)
    if k >= n_skus or n_skus < 2:
        return 0.0
    return math.sqrt(p * (1 - p) / k * (n_skus - k) / (n_skus - 1))


def evolucion(scans, k, cada=500, semilla=42):
    """
    Recorre el stream una vez y, cada `cada` scans, compara la fracción real de SKUs en bajo
    stock con las dos estimaciones. Sirve para graficar cómo siguen la verdad a lo largo del turno.
    """
    por_hash = MuestraSKUsPorHash(k, semilla)
    por_reservorio = MuestraReservorioScans(k, semilla)
    ultimo = {}
    bajos = 0
    puntos = []
    for i, scan in enumerate(scans, start=1):
        por_hash.procesar(scan)
        por_reservorio.procesar(scan)
        previo = ultimo.get(scan["sku"])
        if previo is None:
            bajos += scan["is_low_stock"]
        elif previo != scan["is_low_stock"]:
            bajos += 1 if scan["is_low_stock"] else -1
        ultimo[scan["sku"]] = scan["is_low_stock"]
        if i % cada == 0 or i == len(scans):
            puntos.append({
                "scans_procesados": i,
                "skus_vistos": len(ultimo),
                "real": bajos / len(ultimo),
                "muestreo_skus": por_hash.estimar(),
                "reservorio_scans": por_reservorio.estimar(),
            })
    return puntos


def experimento(scans, tamanos=TAMANOS_MUESTRA, repeticiones=100, semilla_base=1000):
    """
    Repite el muestreo con semillas distintas y compara con la fracción real al final del stream.

    Para cada tamaño k y cada método devuelve el promedio de la estimación, el sesgo, la desviación
    empírica y el error absoluto promedio. Para el muestreo de SKUs agrega el error estándar
    teórico y el porcentaje de veces que el intervalo de confianza del 95 % contiene el valor real.
    """
    p_real, n_skus = fraccion_real(scans)
    estimaciones = {k: {"skus": [], "scans": [], "skus_en_reservorio": [], "cubre": 0} for k in tamanos}

    for rep in range(repeticiones):
        semilla = semilla_base + rep
        muestras = {k: (MuestraSKUsPorHash(k, semilla), MuestraReservorioScans(k, semilla)) for k in tamanos}
        for scan in scans:
            for por_hash, por_reservorio in muestras.values():
                por_hash.procesar(scan)
                por_reservorio.procesar(scan)
        for k, (por_hash, por_reservorio) in muestras.items():
            p_hat = por_hash.estimar()
            estimaciones[k]["skus"].append(p_hat)
            estimaciones[k]["scans"].append(por_reservorio.estimar())
            estimaciones[k]["skus_en_reservorio"].append(por_reservorio.skus_en_muestra())
            se_hat = error_estandar(p_hat, k, n_skus)
            if abs(p_hat - p_real) <= 1.96 * se_hat:
                estimaciones[k]["cubre"] += 1

    def resumen(valores):
        media = statistics.mean(valores)
        return {
            "media": media,
            "sesgo": media - p_real,
            "desviacion": statistics.pstdev(valores),
            "error_abs": statistics.mean(abs(v - p_real) for v in valores),
        }

    filas = []
    for k in tamanos:
        e = estimaciones[k]
        filas.append({
            "k": k,
            "muestreo_skus": resumen(e["skus"]),
            "reservorio_scans": resumen(e["scans"]),
            "skus_distintos_en_reservorio": statistics.mean(e["skus_en_reservorio"]),
            "error_estandar_teorico": error_estandar(p_real, k, n_skus),
            "cobertura_ic95": e["cubre"] / repeticiones,
        })
    return {"p_real": p_real, "n_skus": n_skus, "repeticiones": repeticiones, "filas": filas}


def main():
    scans = cargar_stream()
    r = experimento(scans)
    print(f"Stream de {len(scans)} scans, {r['n_skus']} SKUs distintos")
    print(f"Fracción real de SKUs en bajo stock (último scan de cada SKU): {r['p_real']:.4f}")
    print(f"Repeticiones por tamaño: {r['repeticiones']} (semillas distintas)\n")

    print("Muestreo de SKUs por hash (uniforme sobre SKUs):")
    print(f"{'k':>5} {'media':>8} {'sesgo':>9} {'desv.emp':>9} {'desv.teo':>9} {'err.abs':>8} {'IC95 cubre':>11}")
    for f in r["filas"]:
        m = f["muestreo_skus"]
        print(f"{f['k']:>5} {m['media']:>8.4f} {m['sesgo']:>+9.4f} {m['desviacion']:>9.4f} "
              f"{f['error_estandar_teorico']:>9.4f} {m['error_abs']:>8.4f} {f['cobertura_ic95']:>11.1%}")

    print("\nReservoir sampling de scans (uniforme sobre scans):")
    print(f"{'k':>5} {'media':>8} {'sesgo':>9} {'desv.emp':>9} {'err.abs':>8} {'SKUs en muestra':>16}")
    for f in r["filas"]:
        m = f["reservorio_scans"]
        print(f"{f['k']:>5} {m['media']:>8.4f} {m['sesgo']:>+9.4f} {m['desviacion']:>9.4f} "
              f"{m['error_abs']:>8.4f} {f['skus_distintos_en_reservorio']:>16.0f}")

    k = TAMANOS_MUESTRA[1]
    print(f"\nEvolución durante el stream (k = {k}, un punto cada 2000 scans):")
    print(f"{'scans':>7} {'SKUs':>6} {'real':>8} {'por SKU':>9} {'por scan':>9}")
    for p in evolucion(scans, k, cada=2000):
        print(f"{p['scans_procesados']:>7} {p['skus_vistos']:>6} {p['real']:>8.4f} "
              f"{p['muestreo_skus']:>9.4f} {p['reservorio_scans']:>9.4f}")


if __name__ == "__main__":
    main()
