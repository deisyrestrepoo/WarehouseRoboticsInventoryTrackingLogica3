# Warehouse Robotics Inventory Tracking

Práctica de Lógica y Representación 3 — Grupo 9.

Sistema simulado de robots que escanean bins de inventario en un almacén.
Incluye la Unidad 1 (análisis batch: algoritmos aleatorizados, hashing y
probabilidad), la Unidad 2 (procesamiento de streams: Bloom filter,
Flajolet-Martin y momentos de frecuencia) y la Unidad 3 (cadenas de Markov
sobre el grafo de movimientos de los robots: clasificación de estados,
distribución estacionaria, PageRank, random walk y 2-SAT).

## Integrantes

- Osorio Jiménez Diego Alejandro
- Restrepo Higuita Alejandro
- Restrepo Ospina Deisy Tatiana
- Sánchez Santa Samuel

## Requisitos

- Python 3.11 o superior
- pip
- Git

## Instalación

1. Clonar el repositorio y entrar a la carpeta del proyecto:

   ```bash
   git clone https://github.com/deisyrestrepoo/WarehouseRoboticsInventoryTrackingLogica3.git
   cd WarehouseRoboticsInventoryTrackingLogica3
   ```

2. Crear y activar el entorno virtual:

   ```bash
   python -m venv venv
   ```

   En Windows (PowerShell):

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   En Mac/Linux:

   ```bash
   source venv/bin/activate
   ```

3. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

   Las dependencias principales son `streamlit`, `pandas` y `plotly`.

> Todos los comandos siguientes se ejecutan desde la raíz del proyecto
> (la carpeta `WarehouseRoboticsInventoryTrackingLogica3`).

## Generar el dataset

```bash
python data/generador.py
```

Genera 10.000 scans simulados y los guarda en `data/dataset.json`. El
generador usa una semilla fija (`42`), así que el dataset es idéntico en cada
ejecución y los resultados del informe se pueden reproducir.

Cada registro representa un scan de un robot y tiene estos campos:

| Campo | Tipo | Descripción |
| --- | --- | --- |
| `scan_id` | string | Identificador único del scan |
| `timestamp` | date | Momento del scan (ISO 8601) |
| `robot_id` | string | Robot que hizo el scan |
| `sku` | string | Código del producto |
| `bin_location` | string | Ubicación del bin en la bodega |
| `quantity` | int | Cantidad observada |
| `is_low_stock` | boolean | Alerta de bajo stock (cantidad ≤ 5) |
| `stream_seq` | int | Número de secuencia en el stream (Unit 2) |
| `aisle` | string | Pasillo del bin (Unit 2) |

Decisiones del generador:

- 2.000 SKUs, 100 bins, 10 pasillos y 20 robots.
- El pasillo se deriva del bin, así que cada bin siempre está en el mismo pasillo.
- El tráfico es sesgado: unos pocos bins y SKUs se escanean mucho más que el resto.
- Cada SKU tiene un nivel de stock base que varía poco entre scans.
- Alrededor del 12 % de los scans quedan marcados como bajo stock.

## Estructura del proyecto

```text
WarehouseRoboticsInventoryTrackingLogica3/
├── data/                  # generador de datos, dataset y grafo de movimientos (edges.json)
├── unit1/                 # tareas batch (Big Data, hashing, tabla hash, Chebyshev)
├── unit2/                 # tareas de streaming (Bloom Filter, Flajolet-Martin, F_k)
├── unit3/                 # tareas de cadenas de Markov (DTMC, estacionaria, PageRank, random walk, 2-SAT)
├── dashboard_unit1.py     # dashboard de la Unidad 1
├── dashboard_unit2.py     # dashboard de la Unidad 2
├── dashboard_unit3.py     # dashboard de la Unidad 3
├── main.py                # selector de dashboards
├── tests/                 # pruebas unitarias (python -m unittest discover -s tests)
├── requirements.txt
└── README.md
```

## Ejecutar las tareas

Cada tarea se puede correr por separado desde la raíz del proyecto.

**Unit 1**

| Tarea | Archivo | Comando |
| --- | --- | --- |
| Task 1 — Big Data Context | `unit1/BigDataContext.py` | `python unit1/BigDataContext.py` |
| Task 2 — Randomized Algorithm | `unit1/RandomizedAlgorithm.py` | `python unit1/RandomizedAlgorithm.py` |
| Task 3 — Universal Hashing | `unit1/UniversalHashing.py` | `python unit1/UniversalHashing.py` |
| Task 4 — Hash Table Operations | `unit1/HashTableOperations.py` | `python unit1/HashTableOperations.py` |
| Task 5 — Probability Analysis | `unit1/ProbabilityAnalysis.py` | `python unit1/ProbabilityAnalysis.py` |

La Task 1 pone el escenario en contexto de Big Data: estima cuántos scans por segundo genera la
flota de robots y cuánto almacenamiento necesita el sistema, hoy y con un crecimiento de 10x
de la red de bodegas.

- **Throughput:** `(scans − 1) / duración del dataset`, porque `n` scans delimitan `n − 1`
  intervalos. También lo reparte por robot.
- **Almacenamiento:** mide los bytes promedio de un scan serializado como JSON compacto (UTF-8) y
  proyecta `throughput × segundos × bytes por registro` a un día y a un año (GB = 2^30 bytes).
- **Crecimiento 10x:** se multiplican por 10 los robots y, con ellos, el throughput y el
  almacenamiento (cada robot conserva su ritmo de scans). La proyección es lineal.
- **Resultado con el dataset actual** (10.000 scans, 20 robots): 1,00 scans/seg en total
  (0,05 por robot) y 189 bytes por registro. Hoy son 86.400 scans y unos 0,015 GB por día
  (31.536.000 scans y 5,5 GB por año). Con 10x son 10 scans/seg con 200 robots, 864.000 scans y
  unos 0,15 GB por día (315.360.000 scans y 55,4 GB por año).
- **Cómo leer el resultado:** el generador espacia los scans un segundo, así que la tasa total de
  1,0 scans/seg sale de los timestamps sintéticos y no es una medición de una flota real. Los
  bytes cuentan los 9 campos del dataset (los 7 del escenario más `stream_seq` y `aisle`) y no
  incluyen índices, réplicas ni compresión, por lo que el almacenamiento real sería mayor.
- **Funciones reutilizables** (por ejemplo, para el dashboard): `tasa_scans_por_segundo`,
  `bytes_por_registro`, `proyectar` y `analisis`.
- **Pruebas:** `python -m unittest tests.test_big_data_context -v`

La Task 2 usa **reservoir sampling** (Algoritmo R) para estimar qué fracción de los SKUs está
en bajo stock sin guardar todo el stream de scans, y compara la estimación contra el conteo
exacto. Recorre los scans una sola vez y mantiene un reservorio de `k` scans; al final cada
scan quedó en la muestra con la misma probabilidad `k/n`, así que la muestra es uniforme.

- **Qué estima:** (1) la fracción de *scans* en bajo stock, con su error estándar (con
  corrección por población finita) y un intervalo de confianza del 95 %; (2) la fracción de
  *SKUs* en bajo stock, tomando el scan más reciente de cada SKU que apareció en la muestra.
  La segunda puede tener sesgo, porque los SKUs que más se escanean tienen más probabilidad
  de entrar a la muestra.
- **Valor exacto de referencia:** se recorre todo el dataset y el último scan de cada SKU
  se toma como su estado actual.
- **Experimento:** tamaños de muestra `k` = 100, 250, 500, 1000 y 2000, con 200 repeticiones
  cada uno (semillas distintas). Imprime el conteo exacto, el error absoluto promedio, la
  desviación empírica contra la teórica, el porcentaje de veces que el intervalo del 95 %
  contiene el valor real, el sesgo de la estimación por SKU y el tiempo de una pasada.
- **Resultado con el dataset actual** (10.000 scans, 1.900 SKUs distintos): el conteo exacto
  es 12,41 % de scans en bajo stock y 220 de 1.900 SKUs (11,58 %). Con `k` = 1000 el error
  absoluto promedio es 0,008 y la desviación empírica (0,0100) coincide con la teórica
  (0,0099). El sesgo de la estimación por SKU baja de +0,010 con `k` = 100 a casi 0 con
  `k` = 2000. Muestrear no es más rápido que el conteo exacto en Python (unos 6 ms contra
  4 ms); la ventaja es usar memoria O(k) y poder trabajar sobre un stream que no se puede
  guardar.
- **Funciones reutilizables** (por ejemplo, para el dashboard): `reservoir_sample`,
  `conteo_exacto`, `estimar` y `experimento`.
- **Pruebas:** `python -m unittest tests.test_randomized_algorithm -v`

La Task 3 aplica **hash universal** sobre `sku` para armar el índice de búsqueda rápida del
sistema de reabastecimiento y mide cuántas colisiones produce. La función es de la familia
Carter-Wegman, `h(x) = ((a·x + b) mod P) mod m`, con `P = 2^61 − 1` y `a`, `b` aleatorios (la
misma familia que usa `FuncionHash` en la Task 4). Para dos SKUs distintos la probabilidad de
colisión es a lo sumo `1/m`, así que con `n` SKUs distintos se esperan como máximo `C(n,2)/m`
pares en colisión, sin importar cómo sean los SKUs.

- **Índice:** `IndiceHash` guarda, con chaining, la lista de posiciones de los scans de cada
  SKU. Buscar un SKU solo revisa los SKUs de su bucket.
- **Estudio de colisiones:** para tablas de 512, 1024, 2003, 2048, 4096 y 4099 buckets mide
  los pares de SKUs distintos que caen en el mismo bucket y la carga del bucket más lleno. La
  familia universal se promedia sobre 300 elecciones de `(a, b)` y se compara contra la cota
  teórica `C(n,2)/m` y contra dos funciones ingenuas: suma de los códigos de los caracteres y
  clave entera mod `m`.
- **Benchmark de búsqueda:** construye el índice con los 10.000 scans (2003 buckets) y compara
  buscar cada SKU con el índice contra recorrer todos los scans. Verifica que ambos devuelvan
  las mismas posiciones.
- **Resultado con el dataset actual** (10.000 scans, 1.900 SKUs distintos): el promedio de
  pares en colisión de la familia universal sigue la cota (por ejemplo 894 contra 901 con
  `m` = 2003, y 3.493 contra 3.524 con `m` = 512). La suma de caracteres es inutilizable:
  produce 44.706 pares y un bucket con 70 SKUs sin importar `m`, porque solo genera 82 valores
  distintos. Clave mod `m` reparte mejor que el promedio universal en estos datos (729 pares con
  `m` = 2003), pero eso depende de la forma de los SKUs y no está garantizado; la ventaja de la
  familia universal es la cota para cualquier conjunto de SKUs. El índice con 2003 buckets
  queda con factor de carga 0,95, cadena máxima de 4 y unos 2 SKUs revisados por búsqueda;
  busca en unos 3 µs contra unos 1.700 µs de la búsqueda lineal (unas 650 veces más rápido; los
  tiempos varían según el equipo).
- **Funciones reutilizables** (por ejemplo, para el dashboard): `FuncionHashUniversal`,
  `IndiceHash`, `tabla_colisiones` y `benchmark_busqueda`.
- **Pruebas:** `python -m unittest tests.test_universal_hashing -v`

La Task 4 construye una tabla hash con clave `bin_location` (100 buckets por defecto) y
compara chaining contra Power of Two Choices. Imprime carga máxima y mínima, desviación
estándar, buckets vacíos y registros revisados por búsqueda, y repite el experimento 30 veces
con distintas funciones hash para promediar la carga máxima.

La Task 5 usa la **desigualdad de Chebyshev** para acotar cuánto puede alejarse la tasa de bajo
stock estimada con una muestra de la tasa real de toda la bodega. Si `p` es la fracción real de
scans con `is_low_stock` y `p̂` es la fracción en una muestra uniforme de `n` scans, entonces
`E[p̂] = p`, `Var(p̂) ≤ p(1 − p)/n` y

`Pr[ |p̂ − p| ≥ ε ] ≤ p(1 − p) / (n · ε²)`

- **Dos cotas:** una con la `p` real del dataset y otra de peor caso (`p(1 − p) ≤ 1/4`), útil
  cuando `p` no se conoce de antemano.
- **Muestra mínima:** despejando `n`, para una confianza `1 − δ` basta `n ≥ p(1 − p) / (ε² · δ)`.
- **Comparación empírica:** con `ε` = 0,03 repite 2.000 veces el muestreo (semilla fija) para
  `n` = 100, 250, 500, 1000 y 2000 y cuenta cuántas veces el error llega a `ε`.
- **Resultado con el dataset actual** (10.000 scans): la tasa real de bajo stock es
  `p` = 0,1241. La cota de Chebyshev con `p` es 1,00 / 0,483 / 0,242 / 0,121 / 0,060 y la
  frecuencia empírica es 0,360 / 0,151 / 0,039 / 0,003 / 0,000 (para `n` = 100 / 250 / 500 /
  1000 / 2000). La cota se cumple siempre, pero es conservadora: con `n` = 1000 acota en 12 %
  un evento que ocurre el 0,3 % de las veces. El peor caso exige `n` = 2.778 para 90 % de
  confianza y 5.556 para 95 %; con la `p` real bastan 1.208 y 2.416.
- **Por qué sobra margen:** Chebyshev solo usa la varianza, sin suponer la forma de la
  distribución. Además, las muestras se toman sin reemplazo, y eso reduce la varianza real
  (corrección por población finita) por debajo de `p(1 − p)/n`, así que la cota sigue siendo
  válida.
- **Funciones reutilizables** (por ejemplo, para el dashboard): `tasa_real`, `cota_chebyshev`,
  `muestra_minima`, `probabilidad_empirica` y `analisis`.
- **Pruebas:** `python -m unittest tests.test_probability_analysis -v`

**Unit 2**

| Tarea | Archivo | Comando |
| --- | --- | --- |
| Task 1 — Stream Model & Query Design | `unit2/StreamModelQueryDesign.py` | `python unit2/StreamModelQueryDesign.py` |
| Task 2 — Bloom Filter | `unit2/BloomFilter.py` | `python unit2/BloomFilter.py` |
| Task 4 — Distinct Count | `unit2/DistinctCount.py` | `python unit2/DistinctCount.py` |
| Task 5 — Frequency Moments | `unit2/FrequencyMoments.py` | `python unit2/FrequencyMoments.py` |

La Task 1 simula el stream de scans y responde dos consultas: SKUs en bajo stock por
pasillo en los últimos 300 scans (ventana deslizante) y SKUs distintos vistos en el día
(conteo exacto, usado luego como referencia para Flajolet-Martin).

La Task 2 arma un filtro de Bloom con los SKUs ya alertados en bajo stock, para no generar
tickets de reabastecimiento duplicados. Imprime la tasa de falsos positivos empírica contra
la teórica, y un barrido con distintos tamaños de filtro para ver cómo se acercan ambas.

La Task 4 estima cuántos SKUs distintos se han escaneado hoy con **Flajolet-Martin** y 10 funciones
de hash, sin guardar los SKUs: por cada función se conserva un bitmap con las cantidades de ceros
finales que aparecieron en los hashes, `R` es la posición de su primer cero y el conteo se
estima como `2^(promedio de R) / φ`, con `φ ≈ 0,77351`. Los SKUs repetidos no cambian los bitmaps. La
desviación relativa esperada es ≈ `0,78 / √m` (Flajolet y Martin, 1985), es decir, ≈ 25 % con `m` = 10.

- **Resultado con el dataset actual** (semilla 42): al final del stream hay 1.900 SKUs distintos y
  la estimación es 1.521 (−20,0 %); la variante con la mediana de las 10 estimaciones da 1.324. El
  estado ocupa 80 bytes (10 enteros de 64 bits), frente a guardar los 1.900 SKUs.
- **Teórico vs. empírico** (100 semillas distintas): con `m` = 10 la desviación relativa empírica
  es 25,6 % contra 24,7 % teórica y el sesgo es −3,1 %. Con `m` = 1 / 5 / 20 / 50 la desviación
  empírica es 101,5 / 35,0 / 18,2 / 10,8 % contra 78,0 / 34,9 / 17,4 / 11,0 % teórica.
- **Cómo leerlo:** la estimación sube en escalones, porque depende de potencias de 2 promediadas, y
  por eso a lo largo del día aparecen tramos planos. Con solo 10 funciones el error típico ronda
  el 25 %; no es un fallo, es el costo de usar tan poca memoria. Más funciones de hash lo reducen
  como `1/√m`.
- **Funciones reutilizables** (por ejemplo, para el Panel B del dashboard): `ContadorFlajoletMartin`,
  `ConsultaSKUsDistintosFM`, `evolucion` y `experimento`.
- **Pruebas:** `python -m unittest tests.test_distinct_count -v`

La Task 5 calcula **F0, F1 y F2 exactos** sobre los conteos de scans por `bin_location`
(a diferencia del Flajolet-Martin de la Task 4, que *estima* F0 sobre `sku` sin guardar
todo el stream; aquí se cuenta todo con precisión porque solo hay 100 bins).

- **F0** = bins distintos que recibieron algún scan. **F1** = total de scans. **F2** =
  suma de (conteo por bin)². Una sola pasada sobre el stream, con un diccionario de
  conteos (cabe en memoria sin problema).
- **Número efectivo de bins:** F1²/F2. Vale F0 si el tráfico está parejo entre todos los
  bins, y baja cuanto más se concentra en pocos (es el inverso del índice de Herfindahl,
  una medida de concentración).
- **Interpretación:** además de los momentos, se rankean los bins y los pasillos por
  tráfico, para ver dónde se concentran los robots.
- **Resultado con el dataset actual** (10.000 scans, 100 bins): F0 = 100, F1 = 10.000,
  F2 = 3.347.136, número efectivo ≈ 29,9 de 100 — el tráfico está bastante concentrado.
  `bin_1` es el más visitado con 1.260 scans y `aisle_1` el pasillo con más tráfico
  (4.362 scans), coherente con el sesgo que trae el generador de datos.
- **Funciones reutilizables** (por ejemplo, para el dashboard): `calcular_momentos`,
  `numero_efectivo`, `top_bins` y `trafico_por_pasillo`.
- **Pruebas:** `python -m unittest tests.test_frequency_moments -v`

*(pendiente — comandos de las demás tareas de `unit2/`)*

**Unit 3**

| Tarea | Archivo | Comando |
| --- | --- | --- |
| Task 1 — DTMC Construction & State Classification | `unit3/DTMCConstruction.py` | `python unit3/DTMCConstruction.py` |
| Task 2 — Stationary Distribution | `unit3/StationaryDistribution.py` | `python unit3/StationaryDistribution.py` |
| Task 3 — PageRank | `unit3/PageRank.py` | `python unit3/PageRank.py` |


La Task 1 agrega el stream de scans por `robot_id` (en orden de `stream_seq`) en un grafo de
movimientos entre bins, y arma con él una cadena de Markov de tiempo discreto: los estados son
los bins y la matriz de transición M sale de normalizar `move_count` por fila. El grafo de
aristas se guarda en `data/edges.json`, para que las demás tareas de Unit 3 lo reutilicen.

- **Clasificación formal:** vía componentes fuertemente conexas (algoritmo de Kosaraju, sin
  dependencias externas). Con este dataset el grafo es una sola componente, así que los 100
  bins son recurrentes en el sentido estricto de cadenas de Markov; no hay transitorios.
- **Clasificación práctica:** ordena los bins por tráfico entrante (llegadas de robots) y
  marca los 10 con más llegadas como "hub" y los 10 con menos como "poco visitados" — la
  etiqueta de negocio que pide la tarea, distinta de la clasificación formal.
- **Resultado con el dataset actual:** `bin_1` es el hub principal con 1.256 llegadas; los
  bins poco visitados rondan las 26 a 31 llegadas.
- **Funciones reutilizables** (por ejemplo, para las demás tareas y el dashboard):
  `cargar_grafo`, `construir_matriz_transicion`, `clasificar_formal` y `clasificar_hubs`.
  
La Task 2 calcula la **distribución estacionaria** π de la cadena de Markov (estados =
bins, M de la Task 1): π_i es la fracción del tiempo que, a largo plazo, un robot pasa (o
llega) al bin i, sin importar en qué bin empezó.

- **Método:** iteración de potencias, π_{t+1} = π_t · M desde la distribución uniforme,
  hasta que ||π(t+1) − π(t)||₁ < 1e-10. Como la cadena es irreducible (Task 1: una sola
  componente fuertemente conexa), converge a una única π > 0 sin importar el punto de
  partida.
- **Verificación:** se comprueba que π · M ≈ π (residuo < 1e-8) y que π suma 1.
- **Interpretación:** se compara el ranking de bins por π contra dos referencias crudas
  (sin ningún modelo de cadena detrás): el conteo de scans por bin y el tráfico entrante
  del grafo de movimientos (de la Task 1).
- **Resultado con el dataset actual** (100 bins): converge en 11 iteraciones, residuo
  3,57e-13. `bin_1` concentra el 12,58 % del tiempo de largo plazo, seguido de `bin_2`
  (7,20 %) y `bin_3` (4,75 %). El ranking por π coincide 10/10 en el top-10 con el conteo
  crudo de scans y con el tráfico entrante (Spearman 0,999 en ambos casos): el sesgo de
  tráfico que trae el generador de datos domina tanto el conteo directo como la dinámica
  de largo plazo de la cadena, así que aquí pensar en términos de cadena de Markov no
  cambia mucho el ranking (sí lo haría si el grafo tuviera ciclos o cuellos de botella
  menos triviales).
- **Funciones reutilizables** (por ejemplo, para el dashboard, Panel D): `power_iteration`,
  `distribucion_estacionaria`, `verificar_estacionaria` y `comparar_rankings`.
- **Pruebas:** `python -m unittest tests.test_stationary_distribution -v`

La Task 3 calcula **PageRank con teletransportación** sobre el mismo grafo de la Task 1:
P = d·M + (1−d)·U, con U la teletransportación uniforme (1/n a cualquier bin) y
d = 0,85 (valor estándar). Es la misma idea de iteración de potencias que usa la Task 2,
pero sobre P en vez de M tal cual (y se reimplementa aquí mismo, para que esta tarea no
dependa de que la Task 2 de un compañero ya esté en el repo).

- **Por qué teletransportar:** hace que P sea ergódica aunque M no lo fuera (bins sin
  salida, componentes separadas), y pondera más a los bins a los que llega tráfico desde
  MUCHOS bins distintos, no solo a los que reciben MUCHO tráfico de pocos.
- **Convergencia:** se guarda la trayectoria completa de π(t) (no solo la distancia entre
  iteraciones sucesivas) para medir ||π(t) − π_final||₁ en cada paso. La teoría solo
  garantiza que la convergencia no sea más lenta que una tasa d por iteración (cota
  superior); en este dataset converge mucho más rápido que esa cota.
- **Interpretación:** se compara el ranking de bins por PageRank contra el conteo crudo de
  scans por bin y contra la distribución estacionaria de M sin teletransportar (d=1).
- **Resultado con el dataset actual** (100 bins, d=0,85): converge en 12 iteraciones
  (||π(t+1)−π(t)||₁ < 1e-12); a la iteración 3 el error contra el resultado final ya bajó
  de 0,64 a 0,0004. `bin_1` lidera con 10,83 % de PageRank (bajó de 12,58 % sin
  teletransportar: el 15 % de masa que ahora se reparte parejo entre los 100 bins le resta
  peso a los más concentrados). El ranking top-10 coincide 10/10 tanto con el conteo crudo
  como con la distribución sin teletransportar (Spearman ≥ 0,998 en ambos casos); la
  diferencia puntual más grande es 1,75 puntos porcentuales en algún bin.
- **Funciones reutilizables** (por ejemplo, para el dashboard, Panel D): `pagerank`,
  `construir_matriz_google`, `analizar_convergencia` y `comparar_rankings`.
- **Pruebas:** `python -m unittest tests.test_pagerank -v`


*(pendiente — comandos de las demás tareas de `unit3/`)*

## Ejecutar el dashboard

Dashboard de la Unit 1 (por ahora con el Panel A):

```bash
streamlit run dashboard_unit1.py
```

Se abre en el navegador. El Panel A muestra la cantidad de scans por bucket con chaining y con
Power of Two Choices, y un deslizador permite cambiar el número de buckets. El Panel B y el
Panel C se agregan cuando estén listas sus tareas.

Dashboard de la Unit 2 (Panel A y Panel C completos; en el Panel B falta superponer la
estimación de Flajolet-Martin, Task 4):

```bash
streamlit run dashboard_unit2.py
```

El Panel A muestra la tasa de falsos positivos del Bloom filter a medida que avanza el turno,
con controles para ajustar el tamaño del filtro. El Panel B grafica el conteo real de SKUs
distintos (falta superponer la estimación de Flajolet-Martin). El Panel C es un tablero de
alertas de bajo stock en vivo, ordenado por urgencia, con un slider para moverse por el stream,
más las métricas F0/F1/F2 (Task 5) y el top 10 de bins con más tráfico, calculados sobre los mismos scans que muestra el slider.

Dashboard de la Unit 3 (Panel A completo; Panel D pendiente de la Task 2 y la Task 3):

```bash
streamlit run dashboard_unit3.py
```

El Panel A muestra el grafo de movimientos (bins, movimientos distintos, self-loops,
componentes fuertemente conexas), el ranking de bins "hub" contra "poco visitados", y un
explorador que muestra los destinos más probables desde cualquier bin elegido.

Selector que reúne los tres dashboards en un solo lugar, con un menú en la barra lateral:

```bash
streamlit run main.py
```

## Informe

*(pendiente — informe de máximo 3 páginas con resumen del escenario, qué se midió,
comparación teórica vs. empírica, capturas del dashboard y conclusiones)*

## Estado del proyecto

- [x] Generador de datos
- [ ] Unit 1 — tareas
  - [x] Task 1 — Big Data Context
  - [x] Task 2 — Randomized Algorithm
  - [x] Task 3 — Universal Hashing
  - [x] Task 4 — Hash Table Operations
  - [x] Task 5 — Probability Analysis
  - [ ] Demás tareas
- [ ] Unit 2 — tareas
  - [x] Task 1 — Stream Model & Query Design
  - [x] Task 2 — Bloom Filter
  - [x] Task 4 — Distinct Count
  - [x] Task 5 — Frequency Moments
  - [ ] Demás tareas
- [ ] Unit 3 — tareas
  - [x] Task 1 — DTMC Construction & State Classification
  - [x] Task 2 — Stationary Distribution
  - [x] Task 3 — PageRank
  - [ ] Demás tareas
- [ ] Dashboards
  - [x] Unit 1 — Panel A
  - [ ] Unit 1 — Panel B y Panel C
  - [x] Unit 2 — Panel A y Panel B
  - [x] Unit 2 — Panel C
  - [x] Unit 3 — Panel A
  - [ ] Unit 3 — Panel D (falta PageRank y distribución estacionaria)
  - [x] Selector `main.py`
- [ ] Informe