"""
Task 2 - Cosine LSH with Random Hyperplanes

Firmas de hiperplano aleatorio (bit = signo(v . x)) con n = 16, 32, 64, 128; verifica
que la fraccion de bits coincidentes entre dos firmas se acerca a 1 - theta/180 (theta
= angulo real entre los vectores, en grados), y arma un indice con bandas para buscar
vecinos aproximados: recall@5 contra el top-5 por coseno exacto (fuerza bruta), para
50 consultas, y cuantas comparaciones se ahorran.

Los vectores sobre los que se corre LSH son los "tramos de robot" (robot chunks) que
describe la Task 1 de esta unidad: ventanas de `LONGITUD_CHUNK` scans consecutivos de
un mismo robot (en orden de stream_seq), representados como (participacion por bin +
2 metricas z-score). Esta tarea construye esos vectores aqui mismo -- no depende de
que la Task 1 de un companero ya este en el repo, la misma decision que tomaron ya
otras tareas de las Unidades 3 y 4 -- y se concentra en el algoritmo de LSH en si.

Que es un hiperplano aleatorio como funcion hash: para un vector v elegido al azar
(cada coordenada ~ N(0,1)), h(x) = signo(v . x) parte el espacio en dos mitades con un
plano que pasa por el origen, perpendicular a v. La probabilidad de que dos vectores x
e y caigan en lados distintos es exactamente theta/180 (theta = angulo entre ellos, en
grados) -- por eso, promediando sobre n hiperplanos independientes, la fraccion de
bits donde coinciden las firmas estima 1 - theta/180: vectores casi paralelos (theta
chico) casi siempre coinciden; vectores casi opuestos casi nunca.
"""
import json
import math
import os
import random
from collections import defaultdict

RUTA_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.json"
)

LONGITUD_CHUNK = 8     # scans consecutivos por tramo -> 1.239 chunks con el dataset actual (se pide >= 1.000)
VALORES_N = (16, 32, 64, 128)
N_PRODUCCION = 128      # el indice de banding se arma con el mas largo de los 4
BANDAS, FILAS_POR_BANDA = 16, 8  # 16 * 8 = 128 = N_PRODUCCION


def cargar_scans(ruta=RUTA_DATASET):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _orden_bin(nombre_bin):
    """'bin_12' -> 12, para ordenar numericamente (igual que en unit3/DTMCConstruction.py)."""
    return int(nombre_bin.split("_")[1])


# ---------------------------------------------------------------------------
# Construccion de los tramos (chunks) y sus vectores
# ---------------------------------------------------------------------------

def construir_chunks(scans, longitud=LONGITUD_CHUNK):
    """
    Agrupa los scans por robot_id, ordena por stream_seq, y corta cada secuencia en
    ventanas NO superpuestas de `longitud` scans consecutivos (el resto que no
    alcanza a llenar una ventana completa se descarta). Cada chunk guarda los bins
    visitados, en orden, y el promedio de quantity/is_low_stock dentro del tramo.
    """
    por_robot = defaultdict(list)
    for s in scans:
        por_robot[s["robot_id"]].append(s)
    for robot in por_robot:
        por_robot[robot].sort(key=lambda s: s["stream_seq"])

    chunks = []
    for robot_id, scans_robot in sorted(por_robot.items()):
        n_chunks = len(scans_robot) // longitud
        for i in range(n_chunks):
            ventana = scans_robot[i * longitud:(i + 1) * longitud]
            chunks.append({
                "chunk_id": f"{robot_id}_{i}",
                "robot_id": robot_id,
                "bins": [s["bin_location"] for s in ventana],
                "avg_quantity": sum(s["quantity"] for s in ventana) / longitud,
                "low_stock_frac": sum(s["is_low_stock"] for s in ventana) / longitud,
            })
    return chunks


def construir_vectores(chunks, bins_ordenados):
    """
    Vector de un chunk: participacion de cada bin dentro del tramo (cuenta del bin /
    longitud del tramo, un bin por coordenada) + avg_quantity y low_stock_frac
    estandarizados (z-score) sobre TODOS los chunks. Devuelve {chunk_id: vector}.
    """
    longitud = len(chunks[0]["bins"])
    medias = {
        "avg_quantity": sum(c["avg_quantity"] for c in chunks) / len(chunks),
        "low_stock_frac": sum(c["low_stock_frac"] for c in chunks) / len(chunks),
    }
    varianzas = {
        clave: sum((c[clave] - medias[clave]) ** 2 for c in chunks) / len(chunks)
        for clave in medias
    }
    desviaciones = {clave: (v ** 0.5 or 1.0) for clave, v in varianzas.items()}

    vectores = {}
    for c in chunks:
        conteo_bins = defaultdict(int)
        for b in c["bins"]:
            conteo_bins[b] += 1
        participacion = tuple(conteo_bins.get(b, 0) / longitud for b in bins_ordenados)
        extra = tuple(
            (c[clave] - medias[clave]) / desviaciones[clave] for clave in ("avg_quantity", "low_stock_frac")
        )
        vectores[c["chunk_id"]] = participacion + extra
    return vectores


def cargar_tramos_y_vectores(scans=None, longitud=LONGITUD_CHUNK):
    scans = scans if scans is not None else cargar_scans()
    chunks = construir_chunks(scans, longitud)
    bins_ordenados = sorted({b for c in chunks for b in c["bins"]}, key=_orden_bin)
    vectores = construir_vectores(chunks, bins_ordenados)
    return chunks, vectores


# ---------------------------------------------------------------------------
# Similitud coseno y angulo real (para verificar la formula de LSH)
# ---------------------------------------------------------------------------

def similitud_coseno(a, b):
    punto = sum(x * y for x, y in zip(a, b))
    norma_a = math.sqrt(sum(x * x for x in a))
    norma_b = math.sqrt(sum(y * y for y in b))
    if norma_a == 0 or norma_b == 0:
        return 0.0
    return punto / (norma_a * norma_b)


def angulo_grados(a, b):
    """Angulo real entre dos vectores, en grados (0 = paralelos, 180 = opuestos)."""
    coseno = max(-1.0, min(1.0, similitud_coseno(a, b)))
    return math.degrees(math.acos(coseno))


# ---------------------------------------------------------------------------
# Firmas de hiperplano aleatorio
# ---------------------------------------------------------------------------

def generar_hiperplanos(dim, n, semilla):
    """n vectores normales aleatorios de dimension `dim` (cada coordenada ~ N(0,1))."""
    rng = random.Random(semilla)
    return [tuple(rng.gauss(0, 1) for _ in range(dim)) for _ in range(n)]


def firma(vector, hiperplanos):
    """bit_i = 1 si v . hiperplano_i >= 0, si no 0. Devuelve una tupla de 0/1 de largo n."""
    return tuple(1 if sum(v * h for v, h in zip(vector, hp)) >= 0 else 0 for hp in hiperplanos)


def fraccion_bits_coincidentes(firma_a, firma_b):
    iguales = sum(1 for x, y in zip(firma_a, firma_b) if x == y)
    return iguales / len(firma_a)


# ---------------------------------------------------------------------------
# Verificacion: fraccion de bits coincidentes vs. 1 - theta/180
# ---------------------------------------------------------------------------

def verificar_formula(vectores, valores_n=VALORES_N, n_pares=1000, semilla=42):
    """
    Para cada n en `valores_n`: toma `n_pares` pares de chunks al azar, calcula el
    angulo real entre cada par, su firma con n hiperplanos y la fraccion de bits
    coincidentes, y compara esa fraccion contra la teorica (1 - angulo/180).
    Devuelve, por cada n, el error absoluto promedio (y maximo) entre la fraccion
    empirica y la teorica sobre los `n_pares` pares.
    """
    rng = random.Random(semilla)
    ids = list(vectores)
    dim = len(next(iter(vectores.values())))
    pares = [(rng.choice(ids), rng.choice(ids)) for _ in range(n_pares)]
    pares = [(a, b) for a, b in pares if a != b]

    angulos = {(a, b): angulo_grados(vectores[a], vectores[b]) for a, b in pares}

    resultados = {}
    for n in valores_n:
        hiperplanos = generar_hiperplanos(dim, n, semilla=semilla * 1000 + n)
        firmas = {cid: firma(v, hiperplanos) for cid, v in vectores.items()}
        errores = []
        for a, b in pares:
            teorica = 1 - angulos[(a, b)] / 180
            empirica = fraccion_bits_coincidentes(firmas[a], firmas[b])
            errores.append(abs(empirica - teorica))
        resultados[n] = {
            "error_promedio": sum(errores) / len(errores),
            "error_maximo": max(errores),
        }
    return resultados


# ---------------------------------------------------------------------------
# Banding: candidatos para busqueda aproximada de vecinos
# ---------------------------------------------------------------------------

def construir_indice_bandas(firmas, bandas=BANDAS, filas=FILAS_POR_BANDA):
    """
    Parte cada firma de n=bandas*filas bits en `bandas` trozos de `filas` bits, y
    arma, por cada posicion de banda, un diccionario {valor_de_la_banda: [chunk_ids]}.
    Dos chunks son candidatos si coinciden EXACTO en al menos una banda.
    """
    indice = [defaultdict(list) for _ in range(bandas)]
    for cid, sig in firmas.items():
        for b in range(bandas):
            trozo = sig[b * filas:(b + 1) * filas]
            indice[b][trozo].append(cid)
    return indice


def candidatos(cid, firmas, indice, bandas=BANDAS, filas=FILAS_POR_BANDA):
    """Todos los chunks (menos el propio) que comparten al menos una banda con `cid`."""
    sig = firmas[cid]
    vistos = set()
    for b in range(bandas):
        trozo = sig[b * filas:(b + 1) * filas]
        vistos.update(indice[b][trozo])
    vistos.discard(cid)
    return vistos


# ---------------------------------------------------------------------------
# recall@5 vs. fuerza bruta
# ---------------------------------------------------------------------------

def top5_fuerza_bruta(cid, vectores):
    otros = [c for c in vectores if c != cid]
    otros.sort(key=lambda c: -similitud_coseno(vectores[cid], vectores[c]))
    return otros[:5]


def top5_lsh(cid, vectores, firmas, indice, bandas=BANDAS, filas=FILAS_POR_BANDA):
    cands = candidatos(cid, firmas, indice, bandas, filas)
    cands = sorted(cands, key=lambda c: -similitud_coseno(vectores[cid], vectores[c]))
    return cands[:5], len(cands)


def experimento_recall(vectores, n_consultas=50, semilla=42, bandas=BANDAS, filas=FILAS_POR_BANDA):
    """
    Arma el indice de bandas con n=N_PRODUCCION, elige `n_consultas` chunks al azar, y
    para cada uno compara su top-5 por LSH (coseno exacto solo sobre los candidatos)
    contra el top-5 por fuerza bruta (coseno exacto sobre TODOS los demas chunks).
    Devuelve recall@5 promedio y fraccion promedio de comparaciones ahorradas.
    """
    rng = random.Random(semilla)
    dim = len(next(iter(vectores.values())))
    hiperplanos = generar_hiperplanos(dim, N_PRODUCCION, semilla=semilla)
    firmas = {cid: firma(v, hiperplanos) for cid, v in vectores.items()}
    indice = construir_indice_bandas(firmas, bandas, filas)

    total_otros = len(vectores) - 1
    consultas = rng.sample(list(vectores), n_consultas)

    recalls, ahorros, n_candidatos = [], [], []
    for cid in consultas:
        real = set(top5_fuerza_bruta(cid, vectores))
        aproximado, n_cand = top5_lsh(cid, vectores, firmas, indice, bandas, filas)
        recalls.append(len(real & set(aproximado)) / 5)
        ahorros.append(1 - n_cand / total_otros)
        n_candidatos.append(n_cand)

    return {
        "recall_at_5_promedio": sum(recalls) / len(recalls),
        "comparaciones_ahorradas_promedio": sum(ahorros) / len(ahorros),
        "candidatos_promedio": sum(n_candidatos) / len(n_candidatos),
        "total_otros_chunks": total_otros,
    }


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def imprimir_resultados():
    chunks, vectores = cargar_tramos_y_vectores()
    print(f"{len(chunks)} tramos de robot (chunks de {LONGITUD_CHUNK} scans consecutivos, "
          f"{len(next(iter(vectores.values())))} dimensiones: participacion por bin + 2 z-scores).")

    print("\nFraccion de bits coincidentes vs. teorica (1 - angulo/180), 1000 pares al azar:")
    print(f"  {'n':>4} {'error promedio':>15} {'error maximo':>13}")
    resultados = verificar_formula(vectores)
    for n in VALORES_N:
        r = resultados[n]
        print(f"  {n:>4} {r['error_promedio']:>15.4f} {r['error_maximo']:>13.4f}")

    print(f"\nIndice de bandas: n={N_PRODUCCION}, {BANDAS} bandas de {FILAS_POR_BANDA} bits cada una.")
    r = experimento_recall(vectores)
    print(f"  recall@5 promedio (50 consultas)      : {r['recall_at_5_promedio']:.1%}")
    print(f"  candidatos examinados en promedio     : {r['candidatos_promedio']:.1f} "
          f"de {r['total_otros_chunks']} chunks")
    print(f"  comparaciones ahorradas en promedio   : {r['comparaciones_ahorradas_promedio']:.1%}")


if __name__ == "__main__":
    imprimir_resultados()
