import os
import random
import json
from datetime import datetime, timedelta

SEMILLA = 42
FECHA_BASE = datetime(2026, 9, 23, 6, 0, 0)
UMBRAL_BAJO_STOCK = 5


def generar_dataset(n=10000, num_skus=2000, num_bins=100, num_aisles=10, num_robots=20):
    random.seed(SEMILLA)

    bins_por_pasillo = num_bins // num_aisles
    ids_bins = list(range(1, num_bins + 1))
    ids_skus = list(range(1, num_skus + 1))

    # Tráfico sesgado: pocos bins/SKUs concentran muchos escaneos
    pesos_bins = [1 / (i ** 0.8) for i in ids_bins]
    pesos_skus = [1 / (i ** 0.6) for i in ids_skus]

    # Cada SKU tiene un nivel de stock base
    stock_base = {s: random.randint(0, 50) for s in ids_skus}

    dataset = []
    for i in range(n):
        sku = random.choices(ids_skus, weights=pesos_skus)[0]
        bin_id = random.choices(ids_bins, weights=pesos_bins)[0]
        aisle = (bin_id - 1) // bins_por_pasillo + 1
        quantity = max(0, stock_base[sku] + random.randint(-3, 3))

        dataset.append({
            "scan_id": f"scan_{i}",
            "timestamp": (FECHA_BASE + timedelta(seconds=i)).isoformat(),
            "robot_id": f"robot_{random.randint(1, num_robots)}",
            "sku": f"sku_{sku}",
            "bin_location": f"bin_{bin_id}",
            "quantity": quantity,
            "is_low_stock": quantity <= UMBRAL_BAJO_STOCK,
            "stream_seq": i,
            "aisle": f"aisle_{aisle}",
        })
    return dataset


def guardar_dataset(dataset, ruta=None):
    if ruta is None:
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset guardado en {ruta} ({len(dataset)} registros)")


if __name__ == "__main__":
    guardar_dataset(generar_dataset())