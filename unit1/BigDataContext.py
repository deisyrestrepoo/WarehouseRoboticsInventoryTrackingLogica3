"""
Task 1 - Big Data Context

Estima cuántos scans por segundo genera la flota de robots y cuánto almacenamiento
necesita el sistema, hoy y con un crecimiento de 10x de la red de bodegas.

Método:
  1. Throughput observado = (scans - 1) / duración del dataset (n scans delimitan n-1 intervalos).
  2. Tamaño por registro = promedio de bytes del scan serializado como JSON compacto (UTF-8).
  3. Proyección = throughput * segundos * bytes por registro, a un día y a un año.
  4. Crecimiento 10x: se multiplican por 10 los robots y, con ellos, el throughput y el
     almacenamiento (se asume que cada robot conserva su ritmo de scans).

Notas para leer los resultados:
  - El throughput sale de los timestamps del dataset sintético, que el generador espacia
    un scan por segundo; por eso la tasa total es 1,0 scans/seg por construcción y no una
    medición de una flota real.
  - Los bytes cuentan los 9 campos que guarda el dataset (los 7 del escenario más
    stream_seq y aisle) y no incluyen índices, réplicas ni compresión.
  - GB = 2^30 bytes.
"""
import json
from datetime import datetime
from pathlib import Path

RUTA_DATASET = Path(__file__).resolve().parent.parent / "data" / "dataset.json"
SEGUNDOS_DIA = 24 * 60 * 60
SEGUNDOS_ANIO = 365 * SEGUNDOS_DIA
GB = 1024 ** 3


def cargar_dataset(ruta=RUTA_DATASET):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def tasa_scans_por_segundo(dataset):
    """Scans por segundo de toda la flota, a partir de los timestamps del dataset."""
    tiempos = [datetime.fromisoformat(r["timestamp"]) for r in dataset]
    duracion = (max(tiempos) - min(tiempos)).total_seconds()
    if duracion <= 0:
        raise ValueError("El dataset no cubre un intervalo de tiempo positivo")
    return (len(dataset) - 1) / duracion


def bytes_por_registro(dataset):
    """Promedio de bytes de un scan serializado como JSON compacto (sin espacios)."""
    total = sum(
        len(json.dumps(r, separators=(",", ":")).encode("utf-8")) for r in dataset
    )
    return total / len(dataset)


def proyectar(scans_por_seg, bytes_registro, factor=1):
    """Scans y GB por día y por año para un throughput multiplicado por `factor`."""
    tasa = scans_por_seg * factor
    return {
        "factor": factor,
        "scans_por_seg": tasa,
        "scans_por_dia": tasa * SEGUNDOS_DIA,
        "scans_por_anio": tasa * SEGUNDOS_ANIO,
        "gb_por_dia": tasa * SEGUNDOS_DIA * bytes_registro / GB,
        "gb_por_anio": tasa * SEGUNDOS_ANIO * bytes_registro / GB,
    }


def analisis(dataset, factor_crecimiento=10):
    """Resumen completo: flota actual y flota con crecimiento de `factor_crecimiento`x."""
    robots = len({r["robot_id"] for r in dataset})
    tasa = tasa_scans_por_segundo(dataset)
    b = bytes_por_registro(dataset)
    return {
        "n_scans": len(dataset),
        "robots": robots,
        "bytes_por_registro": b,
        "scans_por_seg_por_robot": tasa / robots,
        "robots_con_crecimiento": robots * factor_crecimiento,
        "actual": proyectar(tasa, b, 1),
        "crecimiento": proyectar(tasa, b, factor_crecimiento),
    }


def main():
    r = analisis(cargar_dataset())
    print(f"Scans: {r['n_scans']} | robots: {r['robots']} | "
          f"{r['bytes_por_registro']:.0f} B/registro (JSON compacto)")
    print(f"Por robot: {r['scans_por_seg_por_robot']:.3f} scans/seg")
    for clave in ("actual", "crecimiento"):
        p = r[clave]
        print(f"\n--- {p['factor']}x ---")
        print(f"Throughput:   {p['scans_por_seg']:.2f} scans/seg")
        print(f"Por dia:      {p['scans_por_dia']:,.0f} scans | {p['gb_por_dia']:.2f} GB")
        print(f"Por anio:     {p['scans_por_anio']:,.0f} scans | {p['gb_por_anio']:.1f} GB")


if __name__ == "__main__":
    main()
