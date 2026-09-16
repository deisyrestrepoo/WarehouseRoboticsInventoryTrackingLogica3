import random
import json
from datetime import datetime, timedelta

def generar_scan(scan_seq, tiempo_base, num_skus=500, num_bins=100, num_aisles=10, num_robots=20):
    sku_elegido = random.randint(1, num_skus)
    quantity = random.randint(0, 50)
 
    return {
        "scan_id": f"scan_{scan_seq}",
        "timestamp": (tiempo_base + timedelta(seconds=scan_seq)).isoformat(),
        "robot_id": f"robot_{random.randint(1, num_robots)}",
        "sku": f"sku_{sku_elegido}",
        "bin_location": f"bin_{random.randint(1, num_bins)}",
        "quantity": quantity,
        "is_low_stock": quantity <= 5,
        "stream_seq": scan_seq,
        "aisle": f"aisle_{random.randint(1, num_aisles)}"
    }

def generar_dataset(n=10000, num_skus=500, num_bins=100, num_aisles=10, num_robots=20):
    tiempo_base = datetime.now()
    return [
        generar_scan(i, tiempo_base, num_skus, num_bins, num_aisles, num_robots)
        for i in range(n)
    ]

def guardar_dataset(dataset, ruta="data/dataset.json"):
    with open(ruta, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset guardado en {ruta} ({len(dataset)} registros)")

if __name__ == "__main__":
    dataset = generar_dataset(n=10000)
    guardar_dataset(dataset)