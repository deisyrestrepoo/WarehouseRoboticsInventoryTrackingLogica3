"""
Task 4 (Unit 5) - MinHash + Banding: S25 - Threshold Design

Encuentra pares de chunks parecidos sin comparar los ~500.000 pares:

  Map     Para cada chunk: shingles (k=3, consecutivos) -> firma MinHash de n=100 valores ->
          se parte la firma en b bandas de r filas (b*r = 100) y se emite, por cada banda,
          ((numero de banda, valores de la banda), chunk).
  Reduce  Agrupa por esa clave: los chunks que caen en el mismo cubo de alguna banda son un
          par candidato. Cada reducer procesa solo las claves que le tocan
          (hash(clave) % R), y el resultado final es la union de pares sin repetir.

El Map y el Reduce se corren con un Pool de >=2 procesos, igual que el runner de
unit4/ParallelEfficiency.py (un Pool, el trabajo repartido en grupos).

Diseno de (b, r). Un par con similitud de Jaccard s coincide en una fila de la firma con
probabilidad s, asi que es candidato con probabilidad

    P(s) = 1 - (1 - s^r)^b        (curva en S, con umbral aproximado (1/b)^(1/r))

El criterio que se usa aqui, con b*r = n = 100:
  1. Tasa de falsos negativos en s = 0,8:  1 - P(0,8) <= 1 %.
  2. "Los pares con s >= 0,6 se encuentran": P(0,6) >= 95 %. Como P crece con s, esto
     garantiza que todo par con s >= 0,6 se encuentra con al menos 95 % de probabilidad.
  3. Entre los que cumplen 1 y 2 se toma el de r mas grande: menos candidatos falsos.

Los chunks son sinteticos, con clusters plantados (plantillas de barrido por pasillos), y
cada chunk tiene su propio nivel de ruido para que haya pares con todas las similitudes
(los chunks reales del dataset no tienen estructura de clusters). Semilla fija.
"""
import multiprocessing as mp
import random
import time
from itertools import combinations

SEMILLA = 42
N_FIRMA = 100
K_SHINGLE = 3
NUM_BINS = 100
BINS_POR_PASILLO = 10
PRIMO = (1 << 61) - 1       # primo de Mersenne para las funciones hash (a*x + b) mod p
S_UMBRAL = 0.6
S_ALTA = 0.8
FN_MAX = 0.01
P_UMBRAL_MIN = 0.95
WORKERS = 2
N_GRUPOS = 8                # el trabajo del Map se reparte en 8 grupos (fragmentos)


# Chunks con clusters plantados

def generar_chunks(n_chunks=1000, n_clusters=10, ruido_max=0.08, semilla=SEMILLA):
    """
    Cada cluster tiene una plantilla de 30 bins (6 pasillos x 5 bins, en orden). Cada chunk copia
    su plantilla con SU PROPIO nivel de ruido q ~ U(0, ruido_max): cada token se reemplaza por un
    bin al azar con prob. q (con el trafico sesgado 1/i^0.8 del generador) y se omite con q/2.
    """
    rng = random.Random(semilla)
    pesos = [1 / (i ** 0.8) for i in range(1, NUM_BINS + 1)]
    pasillos = list(range(NUM_BINS // BINS_POR_PASILLO))

    plantillas = []
    for _ in range(n_clusters):
        plantilla = []
        for p in rng.sample(pasillos, 6):
            ids = sorted(rng.sample(range(p * BINS_POR_PASILLO + 1, (p + 1) * BINS_POR_PASILLO + 1), 5))
            plantilla.extend(f"bin_{i}" for i in ids)
        plantillas.append(plantilla)

    chunks = []
    for i in range(n_chunks):
        c = i % n_clusters
        q = rng.uniform(0, ruido_max)
        secuencia = []
        for token in plantillas[c]:
            if rng.random() < q / 2:
                continue
            if rng.random() < q:
                token = f"bin_{rng.choices(range(1, NUM_BINS + 1), weights=pesos)[0]}"
            secuencia.append(token)
        chunks.append({"robot_id": f"robot_{rng.randint(1, 20)}", "cluster": c, "bin_sequence": secuencia})
    return chunks


# Shingles, Jaccard y MinHash

def shingles(secuencia, k=K_SHINGLE):
    return {"|".join(secuencia[i:i + k]) for i in range(len(secuencia) - k + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _hash32(texto):
    """FNV-1a de 32 bits: determinista (el hash() de Python cambia en cada ejecucion)."""
    h = 2166136261
    for ch in texto:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def coeficientes(n=N_FIRMA, semilla=SEMILLA):
    rng = random.Random(semilla)
    return [(rng.randrange(1, PRIMO), rng.randrange(0, PRIMO)) for _ in range(n)]


def firma_minhash(conjunto, coefs):
    """Para cada funcion hash (a*x+b) mod p, el minimo sobre los shingles del chunk."""
    if not conjunto:
        return [PRIMO] * len(coefs)
    xs = [_hash32(s) for s in conjunto]
    return [min((a * x + b) % PRIMO for x in xs) for a, b in coefs]


# Probabilidad de ser candidato y eleccion de (b, r)

def prob_candidato(s, b, r):
    return 1 - (1 - s ** r) ** b


def tabla_opciones(n=N_FIRMA):
    filas = []
    for r in range(1, n + 1):
        if n % r == 0:
            b = n // r
            filas.append({"b": b, "r": r, "umbral": (1 / b) ** (1 / r),
                          "p_06": prob_candidato(S_UMBRAL, b, r),
                          "fn_08": 1 - prob_candidato(S_ALTA, b, r)})
    return filas


def elegir_br(n=N_FIRMA, fn_max=FN_MAX, p_umbral_min=P_UMBRAL_MIN):
    validas = [f for f in tabla_opciones(n) if f["fn_08"] <= fn_max and f["p_06"] >= p_umbral_min]
    return max(validas, key=lambda f: f["r"])


# Map y Reduce (funciones de nivel superior: el Pool necesita poder importarlas)

def _hash_clave(clave, r):
    return _hash32(str(clave)) % r


def mapper_minhash(args):
    """Map de un grupo: devuelve ({indice: firma}, [(clave_de_banda, indice), ...])."""
    grupo, b, r, coefs = args
    firmas, emitidos = {}, []
    for idx, secuencia in grupo:
        firma = firma_minhash(shingles(secuencia), coefs)
        firmas[idx] = firma
        for banda in range(b):
            emitidos.append(((banda, tuple(firma[banda * r:(banda + 1) * r])), idx))
    return firmas, emitidos


def reducer_bandas(particion):
    """Reduce de un reducer: agrupa por clave de banda y devuelve los pares de cada cubo."""
    cubos = {}
    for clave, idx in particion:
        cubos.setdefault(clave, []).append(idx)
    pares = set()
    for miembros in cubos.values():
        if len(miembros) > 1:
            pares.update(combinations(sorted(miembros), 2))
    return pares, len(cubos)


def candidatos_secuencial(firmas, b, r):
    """Misma logica sin Pool (para barrer varios (b, r) y para comparar en los tests)."""
    pares = set()
    for banda in range(b):
        cubos = {}
        for idx, firma in enumerate(firmas):
            cubos.setdefault(tuple(firma[banda * r:(banda + 1) * r]), []).append(idx)
        for miembros in cubos.values():
            if len(miembros) > 1:
                pares.update(combinations(miembros, 2))
    return pares


def ejecutar_pipeline(chunks, b, r, workers=WORKERS, reducers=None, n_grupos=N_GRUPOS):
    """Map (firmas + bandas) y Reduce (cubos -> pares) con un Pool de `workers` procesos."""
    reducers = reducers or workers
    coefs = coeficientes()
    grupos = [[] for _ in range(n_grupos)]
    for i, c in enumerate(chunks):
        grupos[i % n_grupos].append((i, c["bin_sequence"]))

    inicio = time.perf_counter()
    with mp.Pool(workers) as pool:
        salida_map = pool.map(mapper_minhash, [(g, b, r, coefs) for g in grupos])
        firmas_por_idx, emitidos = {}, []
        for f, e in salida_map:
            firmas_por_idx.update(f)
            emitidos.extend(e)
        particiones = [[] for _ in range(reducers)]       # shuffle: hash(clave) % R
        for clave, idx in emitidos:
            particiones[_hash_clave(clave, reducers)].append((clave, idx))
        salida_reduce = pool.map(reducer_bandas, particiones)
    pares = set()
    for p, _ in salida_reduce:
        pares.update(p)
    return {"firmas": [firmas_por_idx[i] for i in range(len(chunks))],
            "candidatos": pares, "tuplas_map": len(emitidos),
            "cubos_por_reducer": [n for _, n in salida_reduce],
            "tuplas_por_reducer": [len(p) for p in particiones],
            "tiempo_s": time.perf_counter() - inicio}


# Verdad de referencia (fuerza bruta) y evaluacion

def jaccard_exacto(chunks):
    """Jaccard de TODOS los pares (fuerza bruta). Solo guarda los pares con s > 0."""
    conjuntos = [shingles(c["bin_sequence"]) for c in chunks]
    exacto = {}
    for i in range(len(conjuntos)):
        for j in range(i + 1, len(conjuntos)):
            s = jaccard(conjuntos[i], conjuntos[j])
            if s > 0:
                exacto[(i, j)] = s
    return exacto


def evaluar(candidatos, exacto, n_chunks):
    total = n_chunks * (n_chunks - 1) // 2
    verdaderos = {p for p, s in exacto.items() if s >= S_UMBRAL}
    altos = {p for p, s in exacto.items() if s >= S_ALTA}
    tp = len(candidatos & verdaderos)
    return {"total_pares": total, "candidatos": len(candidatos),
            "fraccion_candidatos": len(candidatos) / total,
            "pares_verdaderos": len(verdaderos), "tp": tp,
            "precision": tp / len(candidatos) if candidatos else 0.0,
            "recall": tp / len(verdaderos) if verdaderos else 1.0,
            "pares_altos": len(altos),
            "fn_empirico_08": 1 - len(candidatos & altos) / len(altos) if altos else 0.0}


def tabla_por_similitud(candidatos, exacto, b, r, bordes=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)):
    """Por rango de s: pares, cuantos fueron candidatos y P(s) teorica media del rango."""
    filas = []
    for lo, hi in zip(bordes, bordes[1:]):
        pares = [p for p, s in exacto.items() if lo <= s < hi]
        if not pares:
            continue
        hallados = sum(1 for p in pares if p in candidatos)
        teorica = sum(prob_candidato(exacto[p], b, r) for p in pares) / len(pares)
        filas.append({"rango": (lo, min(hi, 1.0)), "pares": len(pares),
                      "candidatos": hallados, "empirica": hallados / len(pares), "teorica": teorica})
    return filas


def experimento(n_chunks=1000, workers=WORKERS):
    chunks = generar_chunks(n_chunks=n_chunks)
    eleccion = elegir_br()
    b, r = eleccion["b"], eleccion["r"]
    corrida = ejecutar_pipeline(chunks, b, r, workers=workers)
    exacto = jaccard_exacto(chunks)

    barrido = []
    for f in tabla_opciones():
        cand = candidatos_secuencial(corrida["firmas"], f["b"], f["r"])
        barrido.append({**f, **evaluar(cand, exacto, n_chunks)})
    return {"eleccion": eleccion, "corrida": corrida, "barrido": barrido,
            "evaluacion": evaluar(corrida["candidatos"], exacto, n_chunks),
            "por_similitud": tabla_por_similitud(corrida["candidatos"], exacto, b, r),
            "n_chunks": n_chunks, "workers": workers}


if __name__ == "__main__":
    res = experimento()
    e, ev, c = res["eleccion"], res["evaluacion"], res["corrida"]
    print(f"Chunks: {res['n_chunks']}  |  pares totales: {ev['total_pares']:,}  |  "
          f"pares con s >= {S_UMBRAL}: {ev['pares_verdaderos']:,}  |  con s >= {S_ALTA}: {ev['pares_altos']:,}")

    print("\n--- Opciones de (b, r) con b*r = 100 (teoria) ---")
    print(f"{'b':>4} {'r':>4} | {'umbral':>6} | {'P(s=0.6)':>9} | {'FN(s=0.8)':>10}")
    for f in tabla_opciones():
        marca = "  <-- elegida" if (f["b"], f["r"]) == (e["b"], e["r"]) else ""
        print(f"{f['b']:>4} {f['r']:>4} | {f['umbral']:>6.3f} | {f['p_06']:>9.4f} | {f['fn_08']:>10.2e}{marca}")

    print(f"\n--- Elegido: b = {e['b']}, r = {e['r']}  (umbral aprox. {e['umbral']:.3f}) ---")
    print(f"Pipeline con {res['workers']} workers: {c['tuplas_map']:,} tuplas emitidas por el Map "
          f"({c['tuplas_map'] // res['n_chunks']} por chunk = b), "
          f"tuplas por reducer {c['tuplas_por_reducer']}, tiempo {c['tiempo_s']:.2f} s")
    print(f"Pares candidatos: {ev['candidatos']:,} de {ev['total_pares']:,} "
          f"({ev['fraccion_candidatos']:.2%} de las comparaciones)")
    print(f"Precision (candidatos con s >= {S_UMBRAL}): {ev['precision']:.3f}   "
          f"Recall (pares con s >= {S_UMBRAL} hallados): {ev['recall']:.3f}")
    print(f"Falsos negativos en s >= {S_ALTA}: {ev['fn_empirico_08']:.2%} empirico "
          f"(teorico en s = {S_ALTA}: {e['fn_08']:.2e})")

    print("\n--- Probabilidad de ser candidato por rango de similitud (empirica vs teorica) ---")
    print(f"{'rango s':>10} | {'pares':>8} | {'candidatos':>10} | {'empirica':>8} | {'teorica':>8}")
    for f in res["por_similitud"]:
        print(f"{f['rango'][0]:.1f}-{f['rango'][1]:.1f}".rjust(10)
              + f" | {f['pares']:>8,} | {f['candidatos']:>10,} | {f['empirica']:>8.3f} | {f['teorica']:>8.3f}")

    print("\n--- Mismas firmas con otras opciones de (b, r) ---")
    print(f"{'b':>4} {'r':>4} | {'candidatos':>10} | {'% de pares':>10} | {'precision':>9} | {'recall':>6} | {'FN s>=0.8':>9}")
    for f in res["barrido"]:
        print(f"{f['b']:>4} {f['r']:>4} | {f['candidatos']:>10,} | {f['fraccion_candidatos']:>10.2%} | "
              f"{f['precision']:>9.3f} | {f['recall']:>6.3f} | {f['fn_empirico_08']:>9.2%}")