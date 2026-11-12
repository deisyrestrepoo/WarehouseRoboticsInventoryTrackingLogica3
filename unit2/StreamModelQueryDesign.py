"""
Task 1 - Stream Model & Query Design

Modela los scans como un stream: llegan uno por uno, en orden de stream_seq,
y no se guardan todos (salvo lo que cada consulta necesite recordar).

Dos consultas:
  - Continua: SKUs en bajo stock por pasillo, sobre los últimos 300 scans (ventana deslizante).
  - Puntual: cuántos SKUs distintos se han visto "hoy", respondible en cualquier momento.
"""
import json
import os
from collections import deque

TAMANO_VENTANA = 300

RUTA_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.json"
)


def cargar_stream(ruta=RUTA_DATASET):
    """Carga el dataset y lo devuelve ordenado por stream_seq, como llegaría un stream real."""
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
    return sorted(datos, key=lambda s: s["stream_seq"])


class ConsultaBajoStockPorPasillo:
    """
    Consulta continua: SKUs en bajo stock por pasillo, entre los últimos `tamano_ventana` scans.

    Se mantiene una ventana deslizante (deque) con los scans más recientes y, junto a ella,
    un conteo de "en bajo stock" por (pasillo, sku). Al entrar un scan nuevo y al salir el más
    viejo, solo se actualiza ese conteo — nunca se recorre toda la ventana.
    """

    def __init__(self, tamano_ventana=TAMANO_VENTANA):
        self.tamano_ventana = tamano_ventana
        self.ventana = deque()
        self.conteo_bajo_stock = {}  # (aisle, sku) -> cuántos scans en la ventana lo marcan bajo stock

    def _sumar(self, scan, delta):
        if not scan["is_low_stock"]:
            return
        clave = (scan["aisle"], scan["sku"])
        nuevo = self.conteo_bajo_stock.get(clave, 0) + delta
        if nuevo <= 0:
            self.conteo_bajo_stock.pop(clave, None)
        else:
            self.conteo_bajo_stock[clave] = nuevo

    def procesar(self, scan):
        self.ventana.append(scan)
        self._sumar(scan, +1)
        if len(self.ventana) > self.tamano_ventana:
            viejo = self.ventana.popleft()
            self._sumar(viejo, -1)

    def resultado(self):
        """SKUs actualmente en bajo stock dentro de la ventana, agrupados por pasillo."""
        por_pasillo = {}
        for (aisle, sku) in self.conteo_bajo_stock:
            por_pasillo.setdefault(aisle, set()).add(sku)
        return por_pasillo


class ConsultaSKUsDistintosHoy:
    """
    Consulta puntual: cuántos SKUs distintos se han escaneado "hoy" (según el timestamp del scan).

    Guarda el conjunto exacto de SKUs vistos en el día. Se puede preguntar en cualquier
    momento con `total()`, sin esperar a que termine el stream. Es exacta a propósito: sirve
    de referencia para comparar contra la estimación de Flajolet-Martin más adelante.
    """

    def __init__(self):
        self.dia_actual = None
        self.skus_hoy = set()

    def procesar(self, scan):
        dia = scan["timestamp"][:10]  # "YYYY-MM-DD"
        if dia != self.dia_actual:
            self.dia_actual = dia
            self.skus_hoy = set()
        self.skus_hoy.add(scan["sku"])

    def total(self):
        return len(self.skus_hoy)


def simular_stream(scans, tamano_ventana=TAMANO_VENTANA):
    """
    Recorre el stream una sola vez, alimentando las dos consultas, y guarda una foto del
    estado cada `tamano_ventana` scans (para ver cómo evolucionan en el tiempo).
    """
    consulta_ventana = ConsultaBajoStockPorPasillo(tamano_ventana)
    consulta_distintos = ConsultaSKUsDistintosHoy()
    historial = []

    for i, scan in enumerate(scans, start=1):
        consulta_ventana.procesar(scan)
        consulta_distintos.procesar(scan)

        if i % tamano_ventana == 0:
            por_pasillo = consulta_ventana.resultado()
            historial.append({
                "scans_procesados": i,
                "alertas_por_pasillo": {a: len(skus) for a, skus in por_pasillo.items()},
                "skus_distintos_hoy": consulta_distintos.total(),
            })

    return consulta_ventana, consulta_distintos, historial


def imprimir_resumen(scans, tamano_ventana=TAMANO_VENTANA):
    consulta_ventana, consulta_distintos, historial = simular_stream(scans, tamano_ventana)

    print(f"Stream de {len(scans)} scans, ventana de {tamano_ventana}")
    print(f"\nSKUs distintos vistos hoy (consulta puntual): {consulta_distintos.total()}")

    print(f"\nAlertas de bajo stock por pasillo, en los últimos {tamano_ventana} scans:")
    por_pasillo = consulta_ventana.resultado()
    for aisle in sorted(por_pasillo, key=lambda a: -len(por_pasillo[a])):
        print(f"  {aisle}: {len(por_pasillo[aisle])} SKUs en bajo stock")

    print("\nEvolución cada", tamano_ventana, "scans (primeros 5 puntos):")
    for punto in historial[:5]:
        top = sorted(punto["alertas_por_pasillo"].items(), key=lambda kv: -kv[1])[:3]
        print(f"  scan {punto['scans_procesados']:>6}: "
              f"{punto['skus_distintos_hoy']} SKUs distintos hoy, "
              f"top pasillos {top}")


if __name__ == "__main__":
    imprimir_resumen(cargar_stream())