# Warehouse Robotics Inventory Tracking

Práctica de Lógica y Representación 3 — Grupo 9.

Sistema simulado de robots que escanean bins de inventario en un almacén.
Incluye la Unidad 1 (análisis batch: algoritmos aleatorizados, hashing y
probabilidad) y la Unidad 2 (procesamiento de streams: Bloom filter,
Flajolet-Martin y momentos de frecuencia).

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
├── data/                  # generador de datos y dataset generado
├── unit1/                 # tareas batch (Big Data, hashing, tabla hash, Chebyshev)
├── unit2/                 # tareas de streaming (Bloom Filter, Flajolet-Martin, F_k)
├── dashboard_unit1.py     # dashboard de la Unidad 1
├── dashboard_unit2.py     # dashboard de la Unidad 2
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
| Task 2 — Randomized Algorithm | `unit1/RandomizedAlgorithm.py` | `python unit1/RandomizedAlgorithm.py` |
| Task 4 — Hash Table Operations | `unit1/HashTableOperations.py` | `python unit1/HashTableOperations.py` |

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

La Task 4 construye una tabla hash con clave `bin_location` (100 buckets por defecto) y
compara chaining contra Power of Two Choices. Imprime carga máxima y mínima, desviación
estándar, buckets vacíos y registros revisados por búsqueda, y repite el experimento 30 veces
con distintas funciones hash para promediar la carga máxima.

*(pendiente — comandos de las demás tareas de `unit1/`)*

**Unit 2**

| Tarea | Archivo | Comando |
| --- | --- | --- |
| Task 1 — Stream Model & Query Design | `unit2/StreamModelQueryDesign.py` | `python unit2/StreamModelQueryDesign.py` |
| Task 2 — Bloom Filter | `unit2/BloomFilter.py` | `python unit2/BloomFilter.py` |

La Task 1 simula el stream de scans y responde dos consultas: SKUs en bajo stock por
pasillo en los últimos 300 scans (ventana deslizante) y SKUs distintos vistos en el día
(conteo exacto, usado luego como referencia para Flajolet-Martin).

La Task 2 arma un filtro de Bloom con los SKUs ya alertados en bajo stock, para no generar
tickets de reabastecimiento duplicados. Imprime la tasa de falsos positivos empírica contra
la teórica, y un barrido con distintos tamaños de filtro para ver cómo se acercan ambas.

*(pendiente — comandos de las demás tareas de `unit2/`)*

## Ejecutar el dashboard

Dashboard de la Unit 1 (por ahora con el Panel A):

```bash
streamlit run dashboard_unit1.py
```

Se abre en el navegador. El Panel A muestra la cantidad de scans por bucket con chaining y con
Power of Two Choices, y un deslizador permite cambiar el número de buckets. El Panel B y el
Panel C se agregan cuando estén listas sus tareas.

Dashboard de la Unit 2 (Panel A y Panel B completos, Panel C con la parte de la Task 1;
falta lo de Flajolet-Martin y F_k):

```bash
streamlit run dashboard_unit2.py
```

El Panel A muestra la tasa de falsos positivos del Bloom filter a medida que avanza el turno,
con controles para ajustar el tamaño del filtro. El Panel B grafica el conteo real de SKUs
distintos (falta superponer la estimación de Flajolet-Martin). El Panel C es un tablero de
alertas de bajo stock en vivo, ordenado por urgencia, con un slider para moverse por el stream
(falta agregar la tabla de F_k).

Selector que reúne los dos dashboards en un solo lugar, con un menú en la barra lateral:

```bash
streamlit run main.py
```

## Informe

*(pendiente — informe de máximo 3 páginas con resumen del escenario, qué se midió,
comparación teórica vs. empírica, capturas del dashboard y conclusiones)*

## Estado del proyecto

- [x] Generador de datos
- [ ] Unit 1 — tareas
  - [x] Task 2 — Randomized Algorithm
  - [x] Task 4 — Hash Table Operations
  - [ ] Demás tareas
- [ ] Unit 2 — tareas
  - [x] Task 1 — Stream Model & Query Design
  - [x] Task 2 — Bloom Filter
  - [ ] Demás tareas
- [ ] Dashboards
  - [x] Unit 1 — Panel A
  - [ ] Unit 1 — Panel B y Panel C
  - [x] Unit 2 — Panel A y Panel B
  - [ ] Unit 2 — Panel C (falta F_k)
  - [x] Selector `main.py`
- [ ] Informe