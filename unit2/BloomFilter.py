"""
Task 2 - Bloom Filter

Filtro de Bloom con los SKUs que ya se marcaron en bajo stock en este turno (shift),
para no generar un ticket de reabastecimiento duplicado. Se compara la tasa de falsos
positivos empírica (medida al correr el stream) contra la teórica (fórmula del filtro).

Un filtro de Bloom nunca da falso negativo: si dice "no está", es verdad que no está.
Cuando dice "sí está", puede equivocarse (falso positivo) y ahí es donde se pierde un
ticket que sí correspondía generar.
"""
import json
import math
import os
import random

RUTA_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.json"
)

P = 2 ** 61 - 1  # mismo primo de Mersenne usado en unit1/HashTableOperations.py


def clave_a_entero(clave):
    valor = 0
    for caracter in clave:
        valor = (valor * 131 + ord(caracter)) % P
    return valor


class FuncionHash:
    """Hash universal: h(x) = ((a*x + b) mod P) mod m."""

    def __init__(self, m, rng):
        self.m = m
        self.a = rng.randint(1, P - 1)
        self.b = rng.randint(0, P - 1)

    def __call__(self, clave):
        return ((self.a * clave_a_entero(clave) + self.b) % P) % self.m


class FiltroBloom:
    def __init__(self, tamano_bits, num_hashes, semilla=42):
        rng = random.Random(semilla)
        self.tamano_bits = tamano_bits
        self.num_hashes = num_hashes
        self.funciones = [FuncionHash(tamano_bits, rng) for _ in range(num_hashes)]
        self.bits = bytearray((tamano_bits + 7) // 8)
        self.n_insertados = 0

    def _set_bit(self, i):
        self.bits[i // 8] |= 1 << (i % 8)

    def _get_bit(self, i):
        return (self.bits[i // 8] >> (i % 8)) & 1

    def agregar(self, clave):
        for h in self.funciones:
            self._set_bit(h(clave))
        self.n_insertados += 1

    def posible_miembro(self, clave):
        return all(self._get_bit(h(clave)) for h in self.funciones)

    def tasa_falsos_positivos_teorica(self, n=None):
        """p = (1 - e^(-k*n/m))^k. Con n=None usa lo insertado hasta ahora."""
        n = self.n_insertados if n is None else n
        if n == 0:
            return 0.0
        return (1 - math.exp(-self.num_hashes * n / self.tamano_bits)) ** self.num_hashes


def calcular_parametros_optimos(n_esperado, p_deseado):
    """Tamaño de bits (m) y número de hashes (k) óptimos para n_esperado elementos
    y una tasa de falsos positivos objetivo p_deseado."""
    m = math.ceil(-(n_esperado * math.log(p_deseado)) / (math.log(2) ** 2))
    k = max(1, round((m / n_esperado) * math.log(2)))
    return m, k


def cargar_stream(ruta=RUTA_DATASET):
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
    return sorted(datos, key=lambda s: s["stream_seq"])


def simular(scans, filtro):
    """
    Recorre el stream y, por cada scan en bajo stock, decide si generar ticket:
    solo si el filtro dice que el SKU NO estaba ya alertado. Se compara contra un
    conjunto exacto (la verdad) para medir cuándo el filtro se equivoca.
    """
    ya_alertados_real = set()  # verdad exacta, solo para medir el error del filtro
    tickets_generados = 0
    negativos_reales = 0   # veces que el SKU en verdad era nuevo (aún no alertado)
    falsos_positivos = 0   # el filtro dijo "ya alertado" siendo falso -> se perdió un ticket

    historial = []  # (n_insertados, tasa_empirica_acumulada, tasa_teorica_en_ese_momento)

    for scan in scans:
        if not scan["is_low_stock"]:
            continue

        sku = scan["sku"]
        es_nuevo_real = sku not in ya_alertados_real
        filtro_dice_nuevo = not filtro.posible_miembro(sku)

        if es_nuevo_real:
            negativos_reales += 1
            if not filtro_dice_nuevo:
                falsos_positivos += 1  # el filtro lo confundió con uno ya visto
            else:
                tickets_generados += 1
                ya_alertados_real.add(sku)
                filtro.agregar(sku)

            tasa_empirica = falsos_positivos / negativos_reales
            historial.append((filtro.n_insertados, tasa_empirica,
                               filtro.tasa_falsos_positivos_teorica()))
        # si no era nuevo, el filtro también debe decir "ya alertado" (nunca hay falso negativo)

    return {
        "tickets_generados": tickets_generados,
        "negativos_reales": negativos_reales,
        "falsos_positivos": falsos_positivos,
        "tasa_empirica": falsos_positivos / negativos_reales if negativos_reales else 0.0,
        "tasa_teorica_final": filtro.tasa_falsos_positivos_teorica(),
        "historial": historial,
    }


def barrer_tamanos(scans, n_esperado, p_deseado=0.01, factores=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0)):
    """
    Corre el mismo experimento con varios tamaños de filtro (fracciones del tamaño
    óptimo), para ver cómo la tasa empírica se acerca a la teórica: un filtro
    subdimensionado (factor bajo) tiene más colisiones, y uno sobredimensionado
    (factor alto) casi no falla.
    """
    m_optimo, k_optimo = calcular_parametros_optimos(n_esperado, p_deseado)
    resultados = []
    for factor in factores:
        m = max(8, int(m_optimo * factor))
        k = max(1, round((m / n_esperado) * math.log(2)))
        filtro = FiltroBloom(tamano_bits=m, num_hashes=k)
        resultado = simular(scans, filtro)
        resultados.append({"factor": factor, "m": m, "k": k, **resultado})
    return resultados


if __name__ == "__main__":
    scans = cargar_stream()

    # n_esperado: SKUs DISTINTOS que se espera ver en bajo stock en el turno (no el total
    # de scans en bajo stock, que cuenta duplicados). Se estima con turnos anteriores; aquí
    # se documenta como 300. El filtro se dimensiona con esta estimación, antes de correr el
    # stream, tal como se haría en un caso real.
    n_esperado = 300
    p_deseado = 0.01
    m, k = calcular_parametros_optimos(n_esperado, p_deseado)
    n_real = len({s["sku"] for s in scans if s["is_low_stock"]})
    print(f"SKUs distintos que realmente entraron en bajo stock (n real): {n_real}")
    print(f"n_esperado usado para dimensionar (estimado antes de correr el stream): {n_esperado}")
    print(f"Parámetros óptimos para p={p_deseado}: m={m} bits, k={k} hashes\n")

    filtro = FiltroBloom(tamano_bits=m, num_hashes=k)
    resultado = simular(scans, filtro)

    print(f"Tickets de reabastecimiento generados : {resultado['tickets_generados']}")
    print(f"SKUs realmente nuevos evaluados        : {resultado['negativos_reales']}")
    print(f"Falsos positivos (tickets perdidos)    : {resultado['falsos_positivos']}")
    print(f"Tasa de falsos positivos empírica       : {resultado['tasa_empirica']:.4%}")
    print(f"Tasa de falsos positivos teórica (final): {resultado['tasa_teorica_final']:.4%}")

    print(f"\nBarrido de tamaños (n_esperado={n_esperado}, p_deseado={p_deseado}):")
    print(f"{'factor':>7} {'m (bits)':>9} {'k':>3} {'empírica':>10} {'teórica':>10}")
    for r in barrer_tamanos(scans, n_esperado, p_deseado):
        print(f"{r['factor']:>7} {r['m']:>9} {r['k']:>3} "
              f"{r['tasa_empirica']:>10.3%} {r['tasa_teorica_final']:>10.3%}")