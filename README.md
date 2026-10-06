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
| Task 4 — Hash Table Operations | `unit1/HashTableOperations.py` | `python unit1/HashTableOperations.py` |
| Task 5 — Probability Analysis | `unit1/ProbabilityAnalysis.py` | `python unit1/ProbabilityAnalysis.py` |

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
  - [x] Task 4 — Hash Table Operations
  - [x] Task 5 — Probability Analysis
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