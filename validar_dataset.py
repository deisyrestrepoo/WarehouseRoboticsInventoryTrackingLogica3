import json
from collections import Counter

datos = json.load(open("data/dataset.json", encoding="utf-8"))
print("Registros:", len(datos))
print("% bajo stock:", round(100 * sum(d["is_low_stock"] for d in datos) / len(datos), 1))
print("SKUs distintos:", len({d["sku"] for d in datos}))
print("Top 5 bins:", Counter(d["bin_location"] for d in datos).most_common(5))