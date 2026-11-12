"""
Task 1 (Unit 5) - Distance Measures

Representa 1.000 tramos de robot (robot chunks) de dos formas y compara cuatro medidas de
similitud o distancia sobre 5.000 pares al azar:

  - Conjunto: los shingles de k = 3 bins consecutivos de la bin_sequence. Se compara con el
    coeficiente de Jaccard.
  - Vector: la participación de cada bin en el tramo (cuenta del bin / largo de la secuencia;
    un bin por coordenada, 100 en total) más avg_quantity y low_stock_frac del tramo
    estandarizados (z-score) sobre todos los chunks: 102 dimensiones, la misma representación de
    la Task 2 (unit5/CosineLSH.py). Se compara con la similitud coseno y las distancias L1 y L2.

Los chunks son sintéticos, con 10 clusters plantados: cada cluster sigue una plantilla de barrido
por pasillos (aisle-sweep) y cada chunk la copia con ruido y omisiones (`generar_chunks` de la
Task 3, unit5/AnchorShingles.py). Como el generador no da avg_quantity ni low_stock_frac, se
miden sobre scans reales: por cada bin que visita el chunk se toma al azar un scan real de ese
bin, y el chunk guarda el promedio de quantity y la fracción de bajo stock de esos scans. En este
dataset el stock depende del SKU y no del bin, así que las dos métricas no traen información
sobre el cluster: son ruido.

Además de las cuatro medidas pedidas se calcula la similitud LCS (longitud de la subsecuencia
común más larga de las dos bin_sequence, dividida entre la longitud de la más larga), que a
diferencia de Jaccard sí respeta el orden de los bins.

Qué se reporta:

  1. Las distribuciones de cada medida (media, desviación) y su capacidad de separar los pares
     del mismo cluster de los de clusters distintos (AUC: probabilidad de que un par del mismo
     cluster salga más parecido que uno de clusters distintos). Gráfico en data/distance_measures.png.
  2. La correlación (Pearson y Spearman) entre Jaccard, coseno y LCS en 500 pares.
  3. La concentración de las distancias: coeficiente de variación (desviación / media) y
     contraste relativo ((máx - mín) / mín) de las distancias sobre los 5.000 pares, y la razón
     entre el vecino más cercano y el más lejano de varios chunks. Si las distancias se
     concentran, todos los pares quedan casi a la misma distancia y ninguna distingue vecinos de
     no vecinos.

Aviso sobre la escala: la parte de bins de un vector tiene norma ≈ 0,2 (30 bins con participación
≈ 0,035), mientras que la parte de las 2 métricas z-score tiene norma ≈ 1,4. Las métricas, que son
ruido, dominan el coseno y L2. Por eso se calculan también las tres medidas sobre el vector solo
de bins (sufijo `_bins`), para ver qué pasa sin ellas.
"""
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit5.AnchorShingles import generar_chunks, jaccard, shingles_consecutivos  # noqa: E402
from unit5.CosineLSH import cargar_scans, similitud_coseno  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_CHUNKS = os.path.join(RAIZ, "data", "chunks.json")
RUTA_GRAFICO = os.path.join(RAIZ, "data", "distance_measures.png")

SEMILLA = 42
K_SHINGLE = 3
N_CHUNKS = 1000
N_CLUSTERS = 10
NUM_BINS = 100
N_PARES = 5000
N_PARES_CORRELACION = 500
N_CONSULTAS = 50

# clave -> (nombre, tipo). 'similitud': más alto = más parecido; 'distancia': más bajo = más parecido
MEDIDAS = {
    "jaccard": ("Jaccard (shingles k=3)", "similitud"),
    "lcs": ("LCS (bin_sequence)", "similitud"),
    "coseno": ("Coseno (102 dim)", "similitud"),
    "l1": ("L1 (102 dim)", "distancia"),
    "l2": ("L2 (102 dim)", "distancia"),
    "coseno_bins": ("Coseno (solo bins)", "similitud"),
    "l1_bins": ("L1 (solo bins)", "distancia"),
    "l2_bins": ("L2 (solo bins)", "distancia"),
}


# ---------------------------------------------------------------------------
# Chunks, conjuntos y vectores
# ---------------------------------------------------------------------------

def nombres_bins(num_bins=NUM_BINS):
    return [f"bin_{i}" for i in range(1, num_bins + 1)]


def scans_por_bin(scans):
    """{bin_location: [(quantity, is_low_stock), ...]} sobre los scans reales."""
    por_bin = defaultdict(list)
    for s in scans:
        por_bin[s["bin_location"]].append((s["quantity"], int(s["is_low_stock"])))
    return por_bin


def asignar_metricas(chunks, scans, semilla):
    """
    Agrega a cada chunk avg_quantity y low_stock_frac: por cada token de su bin_sequence se toma
    al azar un scan real de ese bin y se promedian quantity e is_low_stock de esos scans.
    """
    por_bin = scans_por_bin(scans)
    rng = random.Random(semilla)
    for c in chunks:
        muestra = [rng.choice(por_bin[b]) for b in c["bin_sequence"]]
        c["avg_quantity"] = sum(q for q, _ in muestra) / len(muestra)
        c["low_stock_frac"] = sum(bajo for _, bajo in muestra) / len(muestra)
    return chunks


def construir_chunks_plantados(scans=None, n_chunks=N_CHUNKS, n_clusters=N_CLUSTERS, semilla=SEMILLA):
    """Chunks sintéticos con clusters plantados (aisle-sweep) y sus dos métricas de stock."""
    scans = scans if scans is not None else cargar_scans()
    chunks = generar_chunks(n_chunks=n_chunks, n_clusters=n_clusters, semilla=semilla)
    return asignar_metricas(chunks, scans, semilla + 1)


def _zscore(valores):
    media = sum(valores) / len(valores)
    desv = math.sqrt(sum((v - media) ** 2 for v in valores) / len(valores)) or 1.0
    return [(v - media) / desv for v in valores]


def construir_representaciones(chunks, k=K_SHINGLE, bins=None):
    """
    Devuelve una copia de cada chunk con su representación como conjunto y como vector:

      shingles          conjunto de shingles de k bins consecutivos
      shingle_count     cuántos shingles distintos tiene
      vector_bins       participación de cada bin (cuenta / largo de la secuencia), 100 dimensiones
      occupancy_vector  vector_bins + avg_quantity y low_stock_frac en z-score (102 dimensiones)
    """
    bins = bins or nombres_bins()
    z_cantidad = _zscore([c["avg_quantity"] for c in chunks])
    z_bajo = _zscore([c["low_stock_frac"] for c in chunks])

    salida = []
    for i, c in enumerate(chunks):
        conteo = defaultdict(int)
        for b in c["bin_sequence"]:
            conteo[b] += 1
        n = len(c["bin_sequence"])
        participacion = tuple(conteo.get(b, 0) / n for b in bins)
        conjunto = shingles_consecutivos(c["bin_sequence"], k)
        salida.append({
            **c,
            "chunk_id": f"chunk_{i}",
            "shingles": conjunto,
            "shingle_count": len(conjunto),
            "vector_bins": participacion,
            "occupancy_vector": participacion + (z_cantidad[i], z_bajo[i]),
        })
    return salida


# ---------------------------------------------------------------------------
# Medidas
# ---------------------------------------------------------------------------

def distancia_l1(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def distancia_l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def distancia_coseno(a, b):
    return 1.0 - similitud_coseno(a, b)


def longitud_lcs(a, b):
    """Longitud de la subsecuencia común más larga de dos secuencias (programación dinámica)."""
    if len(a) < len(b):
        a, b = b, a
    anterior = [0] * (len(b) + 1)
    for x in a:
        actual = [0]
        for j, y in enumerate(b, start=1):
            actual.append(anterior[j - 1] + 1 if x == y else max(anterior[j], actual[j - 1]))
        anterior = actual
    return anterior[-1]


def similitud_lcs(a, b):
    """LCS(a, b) / max(|a|, |b|): 1 si una secuencia contiene a la otra en orden, 0 si no comparten nada."""
    mayor = max(len(a), len(b))
    return longitud_lcs(a, b) / mayor if mayor else 0.0


def pares_al_azar(n_items, n_pares, semilla=SEMILLA):
    """`n_pares` pares (i, j), i < j, distintos entre sí, elegidos al azar entre `n_items` elementos."""
    total = n_items * (n_items - 1) // 2
    if n_pares > total:
        raise ValueError(f"solo hay {total} pares distintos")
    rng = random.Random(semilla)
    elegidos, lista = set(), []
    while len(lista) < n_pares:
        i, j = sorted(rng.sample(range(n_items), 2))
        if (i, j) not in elegidos:
            elegidos.add((i, j))
            lista.append((i, j))
    return lista


def medir_pares(chunks, pares):
    """Calcula todas las medidas de cada par. Devuelve una lista de diccionarios."""
    filas = []
    for i, j in pares:
        a, b = chunks[i], chunks[j]
        va, vb = a["occupancy_vector"], b["occupancy_vector"]
        ba, bb = a["vector_bins"], b["vector_bins"]
        filas.append({
            "i": i,
            "j": j,
            "mismo_cluster": a["cluster"] == b["cluster"],
            "jaccard": jaccard(a["shingles"], b["shingles"]),
            "lcs": similitud_lcs(a["bin_sequence"], b["bin_sequence"]),
            "coseno": similitud_coseno(va, vb),
            "l1": distancia_l1(va, vb),
            "l2": distancia_l2(va, vb),
            "coseno_bins": similitud_coseno(ba, bb),
            "l1_bins": distancia_l1(ba, bb),
            "l2_bins": distancia_l2(ba, bb),
        })
    return filas


# ---------------------------------------------------------------------------
# Estadística
# ---------------------------------------------------------------------------

def rangos(valores):
    """Rangos 1..n de cada valor (promedio en los empates)."""
    orden = sorted(range(len(valores)), key=lambda i: valores[i])
    r = [0.0] * len(valores)
    i = 0
    while i < len(orden):
        j = i
        while j + 1 < len(orden) and valores[orden[j + 1]] == valores[orden[i]]:
            j += 1
        promedio = (i + j) / 2 + 1
        for t in range(i, j + 1):
            r[orden[t]] = promedio
        i = j + 1
    return r


def pearson(x, y):
    """Correlación de Pearson (None si alguna serie es constante)."""
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return statistics.correlation(x, y)


def spearman(x, y):
    """Correlación de Spearman: Pearson sobre los rangos."""
    return pearson(rangos(x), rangos(y))


def auc(mejores, peores):
    """
    Probabilidad de que un valor de `mejores` sea mayor que uno de `peores` (empates cuentan
    1/2). Vale 0,5 si las dos listas no se distinguen y 1 si `mejores` siempre gana.
    """
    n1, n2 = len(mejores), len(peores)
    if not n1 or not n2:
        return None
    r = rangos(list(mejores) + list(peores))
    return (sum(r[:n1]) - n1 * (n1 + 1) / 2) / (n1 * n2)


def auc_de_medida(filas, clave):
    """AUC de una medida para separar pares del mismo cluster (más parecidos) de los de otros."""
    signo = 1 if MEDIDAS[clave][1] == "similitud" else -1
    mismo = [signo * f[clave] for f in filas if f["mismo_cluster"]]
    otro = [signo * f[clave] for f in filas if not f["mismo_cluster"]]
    return auc(mismo, otro)


def resumen_medidas(filas):
    """Media, desviación, media dentro y fuera del cluster y AUC de cada medida."""
    resumen = {}
    for clave in MEDIDAS:
        todos = [f[clave] for f in filas]
        mismo = [f[clave] for f in filas if f["mismo_cluster"]]
        otro = [f[clave] for f in filas if not f["mismo_cluster"]]
        resumen[clave] = {
            "media": statistics.mean(todos),
            "desv": statistics.pstdev(todos),
            "media_mismo": statistics.mean(mismo) if mismo else None,
            "media_otro": statistics.mean(otro) if otro else None,
            "auc": auc_de_medida(filas, clave),
        }
    return resumen


# ---------------------------------------------------------------------------
# Correlación entre medidas de similitud
# ---------------------------------------------------------------------------

PARES_CORRELACION = (
    ("jaccard", "lcs"),
    ("jaccard", "coseno"),
    ("lcs", "coseno"),
    ("jaccard", "coseno_bins"),
    ("lcs", "coseno_bins"),
)


def correlaciones(filas, pares=PARES_CORRELACION):
    """{(a, b): {'pearson', 'spearman'}} entre pares de medidas, sobre las filas dadas."""
    salida = {}
    for a, b in pares:
        x, y = [f[a] for f in filas], [f[b] for f in filas]
        salida[(a, b)] = {"pearson": pearson(x, y), "spearman": spearman(x, y)}
    return salida


# ---------------------------------------------------------------------------
# Concentración de las distancias
# ---------------------------------------------------------------------------

def concentracion(valores):
    """
    Coeficiente de variación (desviación / media) y contraste relativo ((máx - mín) / mín) de una
    lista de distancias. Si son pequeños, todas las distancias son casi iguales (concentración).
    """
    media = statistics.mean(valores)
    minimo, maximo = min(valores), max(valores)
    return {
        "cv": statistics.pstdev(valores) / media if media else None,
        "contraste": (maximo - minimo) / minimo if minimo > 0 else None,
        "min": minimo,
        "max": maximo,
    }


DISTANCIAS = {
    "l1": ("occupancy_vector", distancia_l1),
    "l2": ("occupancy_vector", distancia_l2),
    "coseno": ("occupancy_vector", distancia_coseno),
    "l1_bins": ("vector_bins", distancia_l1),
    "l2_bins": ("vector_bins", distancia_l2),
    "coseno_bins": ("vector_bins", distancia_coseno),
}


def concentracion_en_pares(filas):
    """Concentración de cada distancia sobre los pares medidos (coseno como 1 - coseno)."""
    salida = {}
    for clave in DISTANCIAS:
        valores = [(1 - f[clave]) if clave.startswith("coseno") else f[clave] for f in filas]
        salida[clave] = concentracion(valores)
    return salida


def vecinos_extremos(chunks, n_consultas=N_CONSULTAS, semilla=SEMILLA):
    """
    Para `n_consultas` chunks al azar mide la distancia a todos los demás y guarda, en promedio:
    la razón entre el vecino más cercano y el más lejano (cerca de 1 = concentración), el
    contraste relativo y la fracción de consultas cuyo vecino más cercano es del mismo cluster.
    """
    rng = random.Random(semilla)
    consultas = rng.sample(range(len(chunks)), n_consultas)
    salida = {}
    for clave, (campo, funcion) in DISTANCIAS.items():
        razones, contrastes, aciertos = [], [], 0
        for q in consultas:
            vq = chunks[q][campo]
            distancias = [(funcion(vq, c[campo]), j) for j, c in enumerate(chunks) if j != q]
            cercana, j_cercano = min(distancias)
            lejana = max(d for d, _ in distancias)
            razones.append(cercana / lejana)
            contrastes.append((lejana - cercana) / cercana if cercana > 0 else float("inf"))
            aciertos += chunks[j_cercano]["cluster"] == chunks[q]["cluster"]
        finitos = [c for c in contrastes if c != float("inf")]
        salida[clave] = {
            "razon_min_max": statistics.mean(razones),
            "contraste": statistics.mean(finitos) if finitos else None,
            "nn_mismo_cluster": aciertos / n_consultas,
        }
    return salida


# ---------------------------------------------------------------------------
# Experimento completo
# ---------------------------------------------------------------------------

def normas_por_parte(chunks):
    """Norma media de la parte de bins y de la parte de las 2 métricas z-score de los vectores."""
    n_bins = len(chunks[0]["vector_bins"])
    return {
        "bins": statistics.mean(math.sqrt(sum(x * x for x in c["vector_bins"])) for c in chunks),
        "metricas": statistics.mean(
            math.sqrt(sum(x * x for x in c["occupancy_vector"][n_bins:])) for c in chunks
        ),
    }


def ejecutar(scans=None, n_chunks=N_CHUNKS, n_pares=N_PARES, n_pares_correlacion=N_PARES_CORRELACION,
             n_consultas=N_CONSULTAS, k=K_SHINGLE, semilla=SEMILLA):
    chunks = construir_representaciones(
        construir_chunks_plantados(scans, n_chunks=n_chunks, semilla=semilla), k=k
    )
    pares = pares_al_azar(len(chunks), n_pares, semilla)
    filas = medir_pares(chunks, pares)
    muestra = filas[:n_pares_correlacion]
    mismo_cluster = [f for f in filas if f["mismo_cluster"]]

    return {
        "chunks": chunks,
        "filas": filas,
        "resumen": resumen_medidas(filas),
        "correlacion_al_azar": correlaciones(muestra),
        "correlacion_mismo_cluster": correlaciones(mismo_cluster),
        "pares_correlacion": {"n": len(muestra), "mismo_cluster": sum(f["mismo_cluster"] for f in muestra)},
        "concentracion_pares": concentracion_en_pares(filas),
        "vecinos": vecinos_extremos(chunks, n_consultas, semilla),
        "normas": normas_por_parte(chunks),
    }


def guardar_chunks(chunks, ruta=RUTA_CHUNKS):
    """Guarda los chunks con los campos del template de datos de la Unit 5 (sin los conjuntos)."""
    registros = [{
        "chunk_id": c["chunk_id"],
        "robot_id": c["robot_id"],
        "cluster": c["cluster"],
        "bin_sequence": c["bin_sequence"],
        "seq_length": c["seq_length"],
        "avg_quantity": round(c["avg_quantity"], 4),
        "low_stock_frac": round(c["low_stock_frac"], 4),
        "shingle_count": c["shingle_count"],
        "occupancy_vector": [round(x, 6) for x in c["occupancy_vector"]],
    } for c in chunks]
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(registros, f)


def graficar_distribuciones(resultado, ruta=RUTA_GRAFICO, n_bins=40):
    """
    Guarda una figura con el histograma de cada medida (pares del mismo cluster contra pares de
    clusters distintos) y la dispersión Jaccard contra LCS de los pares de la correlación.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filas = resultado["filas"]
    fig, ejes = plt.subplots(2, 3, figsize=(14, 7.5))
    paneles = ["jaccard", "lcs", "coseno", "l1", "l2"]
    for eje, clave in zip(ejes.flat, paneles):
        mismo = [f[clave] for f in filas if f["mismo_cluster"]]
        otro = [f[clave] for f in filas if not f["mismo_cluster"]]
        todos = mismo + otro
        bordes = [min(todos) + (max(todos) - min(todos)) * i / n_bins for i in range(n_bins + 1)]
        eje.hist(otro, bins=bordes, density=True, alpha=0.6, color="#4c78a8",
                 label=f"clusters distintos ({len(otro):,})")
        eje.hist(mismo, bins=bordes, density=True, alpha=0.7, color="#f58518",
                 label=f"mismo cluster ({len(mismo):,})")
        r = resultado["resumen"][clave]
        eje.set_title(f"{MEDIDAS[clave][0]}\nAUC = {r['auc']:.3f}", fontsize=10)
        eje.set_xlabel("similitud" if MEDIDAS[clave][1] == "similitud" else "distancia")
        eje.set_ylabel("densidad")
        eje.legend(fontsize=8)

    eje = ejes.flat[5]
    muestra = filas[:resultado["pares_correlacion"]["n"]]
    for flag, color, nombre in ((False, "#4c78a8", "clusters distintos"), (True, "#f58518", "mismo cluster")):
        puntos = [f for f in muestra if f["mismo_cluster"] == flag]
        eje.scatter([f["jaccard"] for f in puntos], [f["lcs"] for f in puntos], s=14, alpha=0.7,
                    color=color, label=nombre)
    corr = resultado["correlacion_al_azar"][("jaccard", "lcs")]
    eje.set_title(f"Jaccard vs LCS en {len(muestra)} pares\nPearson = {corr['pearson']:.3f}", fontsize=10)
    eje.set_xlabel("Jaccard")
    eje.set_ylabel("LCS")
    eje.legend(fontsize=8)

    fig.suptitle(f"Distribución de las medidas sobre {len(filas):,} pares al azar de chunks "
                 f"con clusters plantados", fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------------------
# Salida por consola
# ---------------------------------------------------------------------------

def _num(x, decimales=3):
    return "-" if x is None else f"{x:.{decimales}f}"


def imprimir_resultados(guardar=True):
    r = ejecutar()
    chunks = r["chunks"]
    pares_mismo = sum(f["mismo_cluster"] for f in r["filas"])

    print(f"Chunks: {len(chunks):,} en {N_CLUSTERS} clusters plantados (aisle-sweep), semilla {SEMILLA}")
    print(f"  largo de la bin_sequence: {statistics.mean(c['seq_length'] for c in chunks):.1f} bins en promedio")
    print(f"  shingles (k={K_SHINGLE}) distintos por chunk: "
          f"{statistics.mean(c['shingle_count'] for c in chunks):.1f} en promedio, "
          f"vocabulario de {len(set().union(*(c['shingles'] for c in chunks))):,} shingles")
    print(f"  vector: {len(chunks[0]['occupancy_vector'])} dimensiones "
          f"({len(chunks[0]['vector_bins'])} bins + 2 métricas z-score)")
    print(f"  norma media de la parte de bins: {r['normas']['bins']:.2f}; de las 2 métricas: "
          f"{r['normas']['metricas']:.2f}")
    print(f"Pares al azar: {len(r['filas']):,} ({pares_mismo:,} del mismo cluster, "
          f"{len(r['filas']) - pares_mismo:,} de clusters distintos)")

    print("\n1) Distribución de cada medida")
    print(f"  {'medida':<24} {'tipo':<9} {'media':>7} {'desv.':>7} {'mismo cl.':>10} {'otro cl.':>9} {'AUC':>6}")
    for clave, (nombre, tipo) in MEDIDAS.items():
        s = r["resumen"][clave]
        print(f"  {nombre:<24} {tipo:<9} {_num(s['media']):>7} {_num(s['desv']):>7} "
              f"{_num(s['media_mismo']):>10} {_num(s['media_otro']):>9} {_num(s['auc']):>6}")
    print("  (AUC: prob. de que un par del mismo cluster salga más parecido que uno de clusters distintos)")

    n = r["pares_correlacion"]
    print(f"\n2) Correlación entre medidas de similitud")
    for titulo, datos in ((f"{n['n']} pares al azar ({n['mismo_cluster']} del mismo cluster)", r["correlacion_al_azar"]),
                          (f"{pares_mismo} pares del mismo cluster (de los {len(r['filas']):,})",
                           r["correlacion_mismo_cluster"])):
        print(f"  {titulo}")
        print(f"    {'par':<28} {'Pearson':>8} {'Spearman':>9}")
        for (a, b), c in datos.items():
            print(f"    {MEDIDAS[a][0] + ' / ' + MEDIDAS[b][0].split(' (')[0]:<28} "
                  f"{_num(c['pearson']):>8} {_num(c['spearman']):>9}")

    print(f"\n3) Concentración de las distancias")
    print(f"  {'distancia':<24} {'CV 5.000 pares':>15} {'contraste':>10} | "
          f"{'min/max vecinos':>16} {'contraste':>10} {'NN mismo cl.':>13}")
    for clave in DISTANCIAS:
        nombre = MEDIDAS[clave][0].replace("Coseno", "1 - coseno")
        c, v = r["concentracion_pares"][clave], r["vecinos"][clave]
        print(f"  {nombre:<24} {_num(c['cv']):>15} {_num(c['contraste'], 2):>10} | "
              f"{_num(v['razon_min_max']):>16} {_num(v['contraste'], 2):>10} {_num(v['nn_mismo_cluster'], 2):>13}")
    print(f"  (CV = desviación / media; contraste = (máx - mín) / mín; vecinos: promedio de {N_CONSULTAS} consultas)")

    if guardar:
        guardar_chunks(chunks)
        ruta = graficar_distribuciones(r)
        print(f"\nChunks guardados en data/chunks.json ({len(chunks):,} chunks)")
        print(f"Gráfico guardado en data/{os.path.basename(ruta)}")


if __name__ == "__main__":
    imprimir_resultados()
