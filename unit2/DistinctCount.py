"""
Task 4 - Distinct Count (Flajolet-Martin)

Estima cuántos SKUs distintos se han escaneado "hoy" sin guardar los SKUs: solo se conserva un
bitmap pequeño por función de hash (10 funciones por defecto).

Idea de Flajolet-Martin: si h es un hash uniforme, la probabilidad de que h(x) termine en al
menos r ceros binarios es 2^-r. Entre n elementos distintos, el mayor número de ceros finales
ronda log2(n). Para cada función de hash se marca en un bitmap cada cantidad de ceros finales
observada; R es la posición del primer cero del bitmap (la menor cantidad de ceros finales que
no apareció) y 2^R / φ, con φ ≈ 0,77351, estima n. Repetir con m funciones de hash independientes
y promediar los R reduce el error: la desviación relativa esperada es ≈ 0,78 / √m
(Flajolet y Martin, 1985), es decir, ≈ 25 % con m = 10.

Los SKUs repetidos no cambian los bitmaps, así que el estimador cuenta SKUs distintos aunque
lleguen muchos scans del mismo SKU. Se compara con el conteo exacto (un conjunto de SKUs), que
necesita memoria proporcional al número de SKUs, mientras que FM usa 10 enteros de 64 bits.
"""
import json
import math
import os
import random
import statistics

RUTA_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.json"
)

P = 2 ** 61 - 1  # mismo primo de Mersenne usado en unit2/BloomFilter.py
PHI = 0.77351    # constante de corrección de Flajolet-Martin
BITS = 61        # el hash vive en [0, P): como máximo 61 ceros finales
NUM_HASHES = 10

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


def ceros_finales(valor):
    """Cantidad de ceros binarios al final de `valor` (BITS si valor es 0)."""
    if valor == 0:
        return BITS
    return (valor & -valor).bit_length() - 1


def primer_cero(bitmap):
    """Posición del bit menos significativo que está en 0 (R en la notación de Flajolet-Martin)."""
    return ((~bitmap) & (bitmap + 1)).bit_length() - 1


class FuncionHashFM:
    """Hash universal sin reducir por m: h(x) = (a*x + b) mod P."""

    def __init__(self, rng):
        self.a = rng.randint(1, P - 1)
        self.b = rng.randint(0, P - 1)

    def __call__(self, entero):
        return (self.a * entero + self.b) % P


class ContadorFlajoletMartin:
    """Estimador de elementos distintos con `num_hashes` funciones de hash y un bitmap por función."""

    def __init__(self, num_hashes=NUM_HASHES, semilla=42):
        if num_hashes < 1:
            raise ValueError("num_hashes debe ser positivo")
        rng = random.Random(semilla)
        self.num_hashes = num_hashes
        self.funciones = [FuncionHashFM(rng) for _ in range(num_hashes)]
        self.bitmaps = [0] * num_hashes
        self.n_agregados = 0

    def agregar(self, clave):
        x = clave_a_entero(clave)
        for i, h in enumerate(self.funciones):
            self.bitmaps[i] |= 1 << ceros_finales(h(x))
        self.n_agregados += 1

    def posiciones(self):
        """R de cada función de hash."""
        return [primer_cero(b) for b in self.bitmaps]

    def estimar(self):
        """2^(promedio de R) / φ. Vale 0 si todavía no se agregó nada."""
        if self.n_agregados == 0:
            return 0.0
        r_promedio = sum(self.posiciones()) / self.num_hashes
        return 2 ** r_promedio / PHI

    def estimar_mediana(self):
        """Variante: mediana de las estimaciones individuales 2^R_i / φ (más robusta a un R extremo)."""
        if self.n_agregados == 0:
            return 0.0
        return statistics.median(2 ** r / PHI for r in self.posiciones())

    def bytes_de_estado(self):
        """Memoria usada por los bitmaps (un entero de 64 bits por función de hash)."""
        return 8 * self.num_hashes


class ConsultaSKUsDistintosFM:
    """
    Consulta puntual "¿cuántos SKUs distintos se escanearon hoy?" con Flajolet-Martin.
    Al cambiar el día del timestamp se reinician los bitmaps, igual que el conteo exacto de la Task 1.
    """

    def __init__(self, num_hashes=NUM_HASHES, semilla=42):
        self.num_hashes = num_hashes
        self.semilla = semilla
        self.dia_actual = None
        self.contador = ContadorFlajoletMartin(num_hashes, semilla)

    def procesar(self, scan):
        dia = scan["timestamp"][:10]
        if dia != self.dia_actual:
            self.dia_actual = dia
            self.contador = ContadorFlajoletMartin(self.num_hashes, self.semilla)
        self.contador.agregar(scan["sku"])

    def estimar(self):
        return self.contador.estimar()


def evolucion(scans, num_hashes=NUM_HASHES, cada=500, semilla=42):
    """
    Recorre el stream una vez y, cada `cada` scans, compara el conteo exacto de SKUs distintos
    del día con la estimación de Flajolet-Martin. Sirve para graficar ambos a lo largo del turno.
    """
    fm = ConsultaSKUsDistintosFM(num_hashes, semilla)
    dia = None
    skus = set()
    puntos = []
    for i, scan in enumerate(scans, start=1):
        fm.procesar(scan)
        if scan["timestamp"][:10] != dia:
            dia = scan["timestamp"][:10]
            skus = set()
        skus.add(scan["sku"])
        if i % cada == 0 or i == len(scans):
            real = len(skus)
            estimado = fm.estimar()
            puntos.append({
                "scans_procesados": i,
                "real": real,
                "estimado": estimado,
                "error_relativo": (estimado - real) / real,
            })
    return puntos


def experimento(scans, hashes=(1, 5, 10, 20, 50), repeticiones=100, semilla_base=1000):
    """
    Repite la estimación final con semillas distintas para cada cantidad de funciones de hash.

    Como los SKUs repetidos no cambian los bitmaps, basta recorrer los SKUs distintos del último
    día (el resultado es idéntico al de procesar todos los scans). Devuelve, por cada m, el sesgo
    relativo, la desviación relativa empírica y la teórica ≈ 0,78 / √m.
    """
    ultimo_dia = scans[-1]["timestamp"][:10]
    skus = sorted({s["sku"] for s in scans if s["timestamp"][:10] == ultimo_dia})
    real = len(skus)
    filas = []
    for m in hashes:
        errores = []
        for rep in range(repeticiones):
            contador = ContadorFlajoletMartin(m, semilla_base + rep)
            for sku in skus:
                contador.agregar(sku)
            errores.append((contador.estimar() - real) / real)
        filas.append({
            "m": m,
            "sesgo": statistics.mean(errores),
            "desviacion": statistics.pstdev(errores),
            "desviacion_teorica": 0.78 / math.sqrt(m),
            "error_abs": statistics.mean(abs(e) for e in errores),
            "dentro_del_25": sum(abs(e) <= 0.25 for e in errores) / repeticiones,
        })
    return {"real": real, "repeticiones": repeticiones, "filas": filas}


def main():
    scans = cargar_stream()
    print(f"Stream de {len(scans)} scans, {NUM_HASHES} funciones de hash (semilla 42)\n")

    print("Evolución durante el día (conteo exacto vs. Flajolet-Martin):")
    print(f"{'scans':>7} {'real':>6} {'estimado':>9} {'error':>8}")
    puntos = evolucion(scans, cada=1000)
    for p in puntos:
        print(f"{p['scans_procesados']:>7} {p['real']:>6} {p['estimado']:>9.0f} {p['error_relativo']:>+8.1%}")

    fm = ConsultaSKUsDistintosFM()
    for scan in scans:
        fm.procesar(scan)
    real = puntos[-1]["real"]
    print(f"\nFinal: real = {real}, FM (promedio de R) = {fm.estimar():.0f}, "
          f"FM (mediana) = {fm.contador.estimar_mediana():.0f}")
    print(f"R por función de hash: {fm.contador.posiciones()}")
    print(f"Memoria: FM = {fm.contador.bytes_de_estado()} bytes de estado; "
          f"exacto = {real} SKUs guardados en un conjunto")

    r = experimento(scans)
    print(f"\nExperimento: {r['repeticiones']} semillas distintas, conteo final real = {r['real']}")
    print(f"{'m':>4} {'sesgo':>8} {'desv.emp':>9} {'desv.teo':>9} {'err.abs':>8} {'|error|<=25%':>13}")
    for f in r["filas"]:
        print(f"{f['m']:>4} {f['sesgo']:>+8.1%} {f['desviacion']:>9.1%} {f['desviacion_teorica']:>9.1%} "
              f"{f['error_abs']:>8.1%} {f['dentro_del_25']:>13.0%}")


if __name__ == "__main__":
    main()
