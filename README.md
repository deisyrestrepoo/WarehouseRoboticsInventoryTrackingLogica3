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
| Task 1 — Big Data Context | `unit1/BigDataContext.py` | `python unit1/BigDataContext.py` |
| Task 4 — Hash Table Operations | `unit1/HashTableOperations.py` | `python unit1/HashTableOperations.py` |

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
*(pendiente — se agrega el comando de cada tarea de `unit1/` y `unit2/` cuando estén listas)*

## Ejecutar el dashboard

*(pendiente — se confirma cuando `dashboard_unit1.py` y `dashboard_unit2.py` estén listos)*

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
- [ ] Unit 2 — tareas
- [ ] Dashboards
- [ ] Informe