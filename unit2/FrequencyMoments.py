"""
Task 5 - Frequency Moments

F0, F1 y F2 EXACTOS sobre los conteos de scans por bin_location. A diferencia del
Flajolet-Martin de la Task 4 (que ESTIMA F0 sobre sku sin guardar todo el stream),
aquí se cuenta todo con precisión: el número de bins es pequeño (100) y un solo
diccionario de conteos cabe sin problema en memoria mientras se recorre el stream.

  F0 = número de bins distintos que recibieron al menos un scan.
  F1 = total de scans del stream (= longitud del stream).
  F2 = suma de (conteo por bin)^2.

F2 mide qué tan concentrado está el tráfico entre los bins: con tráfico perfectamente
parejo entre los F0 bins, F2 es el mínimo posible (F1²/F0); mientras más se concentra el
tráfico en pocos bins, más crece F2 por encima de ese mínimo. De ahí sale el "número
efectivo" de bins, F1²/F2: vale F0 cuando el tráfico es parejo, y baja cuanto más
concentrado está (es el inverso del índice de Herfindahl, una medida de concentración
usada en economía para lo mismo: cuántos "jugadores efectivos" hay en un mercado).

También se interpreta qué bins y qué pasillos reciben más tráfico de robots.
"""
import json
import os
from collections import Counter

RUTA_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.json"
)


def cargar_stream(ruta=RUTA_DATASET):
    """Carga el dataset y lo devuelve ordenado por stream_seq, como llegaría un stream real."""
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
    return sorted(datos, key=lambda s: s["stream_seq"])


def calcular_momentos(scans, clave="bin_location"):
    """
    Pasada única sobre el stream: cuenta cuántos scans caen en cada valor de `clave`
    y, con esos conteos, calcula F0, F1 y F2 de forma exacta.
    """
    conteos = Counter(s[clave] for s in scans)
    f0 = len(conteos)
    f1 = sum(conteos.values())
    f2 = sum(c * c for c in conteos.values())
    return {"F0": f0, "F1": f1, "F2": f2, "conteos": conteos}


def numero_efectivo(f0, f1, f2):
    """
    'Número efectivo' de bins que concentran el tráfico: F1²/F2. Vale F0 cuando el
    tráfico está perfectamente parejo entre los F0 bins, y es menor que F0 cuanto más
    se concentra en pocos.
    """
    return f1 * f1 / f2 if f2 else 0.0


def top_bins(conteos, top_n=10):
    return conteos.most_common(top_n)


def trafico_por_pasillo(scans):
    """Agrega el conteo de scans por pasillo (aisle), para interpretar a nivel de pasillo."""
    return Counter(s["aisle"] for s in scans)


def imprimir_resultados(scans=None, top_n=10):
    scans = scans if scans is not None else cargar_stream()
    momentos = calcular_momentos(scans, clave="bin_location")
    f0, f1, f2 = momentos["F0"], momentos["F1"], momentos["F2"]
    n_efectivo = numero_efectivo(f0, f1, f2)

    print(f"Stream de {f1} scans sobre {f0} bins distintos")
    print(f"\nF0 (bins distintos)              : {f0}")
    print(f"F1 (total de scans)              : {f1}")
    print(f"F2 (suma de conteo^2 por bin)     : {f2}")
    print(f"Numero efectivo de bins (F1^2/F2): {n_efectivo:.1f} de {f0}")
    if n_efectivo < f0 * 0.5:
        print("  -> el trafico esta bastante concentrado en pocos bins (numero efectivo "
              "muy por debajo del total de bins).")

    print(f"\nTop {top_n} bins con mas trafico:")
    for bin_, c in top_bins(momentos["conteos"], top_n):
        print(f"  {bin_}: {c} scans")

    print(f"\nTrafico por pasillo:")
    por_pasillo = trafico_por_pasillo(scans)
    for aisle, c in por_pasillo.most_common():
        print(f"  {aisle}: {c} scans")


if __name__ == "__main__":
    imprimir_resultados()
