"""
Task 3 (Unit 5) - Shingling: S22 - Anchor-based Shingles

Compara dos formas de convertir la bin_sequence de un chunk en un conjunto de shingles:

  - Consecutivos (referencia): un shingle empieza en CADA posición y toma k tokens seguidos.
  - Anclados: un shingle empieza SOLO en una posición cuyo token es un ancla (los 5 bins con
    mayor PageRank de la Unit 3) y toma ese token más los k-1 siguientes.

Como un shingle anclado es también un shingle consecutivo, el conjunto anclado es siempre un
subconjunto del consecutivo con el mismo k. Lo que se compara es:

  - tamaño del conjunto (shingles distintos por chunk, chunks que quedan vacíos);
  - separación entre clusters: Jaccard medio entre chunks del mismo cluster plantado vs. de
    clusters distintos, AUC (probabilidad de que un par del mismo cluster tenga más Jaccard que
    uno de clusters distintos) y precisión@5 de los vecinos por Jaccard.

Los chunks son sintéticos: cada cluster sigue una plantilla de barrido por pasillos (aisle-sweep)
con ruido, como pide la Unit 5. La semilla es fija para que los resultados sean reproducibles.
"""
import os
import random
import sys
from bisect import bisect_left, bisect_right

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit3.PageRank import pagerank  # noqa: E402
from unit5.CosineLSH import construir_chunks, cargar_scans  # noqa: E402

SEMILLA = 42
NUM_BINS = 100
BINS_POR_PASILLO = 10
NUM_ANCLAS = 5
DAMPING = 0.85


# Anclas: top-5 por PageRank (unit3/PageRank.py)

def _orden_bin(nombre):
    return int(nombre.split("_")[1])


def obtener_anclas(n_anclas=NUM_ANCLAS):
    pr, _, _, _, _ = pagerank(d=DAMPING)
    ranking = sorted(pr, key=lambda b: (-pr[b], _orden_bin(b)))
    return ranking[:n_anclas], pr


# Chunks sintéticos con clusters plantados (plantillas de barrido por pasillos)

def generar_chunks(n_chunks=1000, n_clusters=10, pasillos_por_plantilla=6, bins_por_pasillo=5,
                   p_ruido=0.10, p_omision=0.05, semilla=SEMILLA):
    """
    Cada cluster tiene una plantilla: recorre `pasillos_por_plantilla` pasillos en un orden propio
    y en cada uno visita `bins_por_pasillo` bins en orden ascendente. Cada chunk copia su plantilla,
    omite cada token con prob. p_omision y reemplaza cada token con prob. p_ruido por un bin al
    azar con el mismo tráfico sesgado que el generador de datos (1 / i^0.8).
    """
    rng = random.Random(semilla)
    pesos = [1 / (i ** 0.8) for i in range(1, NUM_BINS + 1)]
    pasillos = list(range(NUM_BINS // BINS_POR_PASILLO))

    plantillas = []
    for _ in range(n_clusters):
        orden = rng.sample(pasillos, pasillos_por_plantilla)
        plantilla = []
        for p in orden:
            ids = sorted(rng.sample(range(p * BINS_POR_PASILLO + 1, (p + 1) * BINS_POR_PASILLO + 1),
                                    bins_por_pasillo))
            plantilla.extend(f"bin_{i}" for i in ids)
        plantillas.append(plantilla)

    chunks = []
    for i in range(n_chunks):
        c = i % n_clusters
        secuencia = []
        for token in plantillas[c]:
            if rng.random() < p_omision:
                continue
            if rng.random() < p_ruido:
                token = f"bin_{rng.choices(range(1, NUM_BINS + 1), weights=pesos)[0]}"
            secuencia.append(token)
        chunks.append({"robot_id": f"robot_{rng.randint(1, 20)}", "cluster": c,
                       "bin_sequence": secuencia, "seq_length": len(secuencia)})
    return chunks


# Shingles

def shingles_consecutivos(secuencia, k):
    return {"|".join(secuencia[i:i + k]) for i in range(len(secuencia) - k + 1)}


def shingles_anclados(secuencia, k, anclas):
    anclas = set(anclas)
    return {"|".join(secuencia[i:i + k]) for i in range(len(secuencia) - k + 1)
            if secuencia[i] in anclas}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# Métricas

def tamanos(conjuntos):
    t = [len(s) for s in conjuntos]
    return {"media": sum(t) / len(t), "min": min(t), "max": max(t),
            "vacios": sum(1 for x in t if x == 0),
            "vocabulario": len(set().union(*conjuntos))}


def separacion(conjuntos, etiquetas, top=5):
    """Jaccard medio intra/inter cluster, AUC y precisión@top de los vecinos por Jaccard."""
    n = len(conjuntos)
    intra, inter = [], []
    vecinos = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = jaccard(conjuntos[i], conjuntos[j])
            (intra if etiquetas[i] == etiquetas[j] else inter).append(s)
            vecinos[i].append((s, j))
            vecinos[j].append((s, i))

    inter_ord = sorted(inter)
    ganados = sum(bisect_left(inter_ord, s) + 0.5 * (bisect_right(inter_ord, s) - bisect_left(inter_ord, s))
                  for s in intra)
    auc = ganados / (len(intra) * len(inter))

    aciertos = 0
    for i in range(n):
        mejores = sorted(vecinos[i], key=lambda t: (-t[0], t[1]))[:top]
        aciertos += sum(1 for s, j in mejores if s > 0 and etiquetas[j] == etiquetas[i])
    return {"jaccard_intra": sum(intra) / len(intra), "jaccard_inter": sum(inter) / len(inter),
            "auc": auc, "precision_at_5": aciertos / (n * top)}


def experimento(valores_k=(2, 3, 4, 5), n_chunks=1000):
    anclas, pr = obtener_anclas()
    chunks = generar_chunks(n_chunks=n_chunks)
    etiquetas = [c["cluster"] for c in chunks]
    filas = []
    for k in valores_k:
        cons = [shingles_consecutivos(c["bin_sequence"], k) for c in chunks]
        anc = [shingles_anclados(c["bin_sequence"], k, anclas) for c in chunks]
        filas.append({"k": k,
                      "consecutivo": {"tam": tamanos(cons), **separacion(cons, etiquetas)},
                      "anclado": {"tam": tamanos(anc), **separacion(anc, etiquetas)}})
    return {"anclas": anclas, "pagerank": {a: pr[a] for a in anclas}, "n_chunks": len(chunks),
            "chunks_con_ancla": sum(1 for c in chunks if set(c["bin_sequence"]) & set(anclas)),
            "filas": filas}


def tamanos_chunks_reales(valores_k=(2, 3, 4, 5)):
    """
    Tamaño del conjunto sobre los chunks REALES del dataset (los de unit5/CosineLSH.py:
    ventanas de 8 scans consecutivos de un robot). Ahí no hay clusters plantados, así que
    solo se puede comparar el tamaño, no la separación.
    """
    anclas, _ = obtener_anclas()
    secuencias = [c["bins"] for c in construir_chunks(cargar_scans())]
    filas = []
    for k in valores_k:
        cons = [shingles_consecutivos(s, k) for s in secuencias]
        anc = [shingles_anclados(s, k, anclas) for s in secuencias]
        filas.append({"k": k, "n_chunks": len(secuencias),
                      "consecutivo": tamanos(cons), "anclado": tamanos(anc)})
    return filas


if __name__ == "__main__":
    r = experimento()
    print(f"Anclas (top-{NUM_ANCLAS} PageRank, d={DAMPING}): "
          + ", ".join(f"{a} ({r['pagerank'][a]:.4f})" for a in r["anclas"]))
    print(f"Chunks: {r['n_chunks']} en 10 clusters plantados; "
          f"{r['chunks_con_ancla']} contienen al menos un ancla\n")
    cab = (f"{'k':>2} | {'método':<12} | {'shingles/chunk':>14} | {'vacíos':>6} | {'vocab.':>6} | "
           f"{'J intra':>7} | {'J inter':>7} | {'AUC':>5} | {'prec@5':>6}")
    print(cab)
    print("-" * len(cab))
    for f in r["filas"]:
        for nombre in ("consecutivo", "anclado"):
            m = f[nombre]
            print(f"{f['k']:>2} | {nombre:<12} | {m['tam']['media']:>14.1f} | {m['tam']['vacios']:>6} | "
                  f"{m['tam']['vocabulario']:>6} | {m['jaccard_intra']:>7.3f} | {m['jaccard_inter']:>7.3f} | "
                  f"{m['auc']:>5.3f} | {m['precision_at_5']:>6.3f}")

    print("\n--- Tamaño del conjunto sobre los chunks reales del dataset (8 scans por chunk) ---")
    print(f"{'k':>2} | {'chunks':>6} | {'consec. shingles/chunk':>22} | {'anclado shingles/chunk':>22} | {'anclado vacíos':>14}")
    for f in tamanos_chunks_reales():
        print(f"{f['k']:>2} | {f['n_chunks']:>6} | {f['consecutivo']['media']:>22.2f} | "
              f"{f['anclado']['media']:>22.2f} | {f['anclado']['vacios']:>14}")