# Warehouse Robotics Inventory Tracking

Proyecto de la práctica de Lógica y Representación 3 — Grupo 9.
Sistema simulado de robots que escanean bins de inventario en un 
almacén, para Unit 1 (análisis batch) y Unit 2 (procesamiento de streams).

## Integrantes
OSORIO JIMENEZ DIEGO ALEJANDRO
RESTREPO HIGUITA ALEJANDRO
RESTREPO OSPINA DEISY TATIANA
SANCHEZ SANTA SAMUEL


## Requisitos
- Python 3.11+
- pip

## Instalación

1. Clonar el repositorio:
https://github.com/deisyrestrepoo/WarehouseRoboticsInventoryTrackingLogica3.git

cd Practica1_logica3


2. Crear y activar el entorno virtual:

python -m venv venv

   En Windows (PowerShell):

.\venv\Scripts\Activate.ps1

   En Mac/Linux:

source venv/bin/activate


3. Instalar dependencias:

pip install -r requirements.txt


## Generar el dataset

python data/generator.py


Esto genera 10.000 registros simulados y los guarda en `data/dataset.json`.
Cada registro representa un scan de un robot, con campos como `sku`, 
`bin_location`, `quantity`, `is_low_stock`, `stream_seq` y `aisle`.

## Estructura del proyecto

Practica1_logica3/
├── data/ # generador de datos y dataset generado
├── unit1/ # tareas de análisis batch (hashing, sampling, Chebyshev)
├── unit2/ # tareas de streaming (Bloom Filter, Flajolet-Martin, F_k)
├── dashboard_unit1.py
├── dashboard_unit2.py
└── main.py # selector de dashboards


## Ejecutar el dashboard

*(pendiente — se agrega cuando dashboard_unit1.py / dashboard_unit2.py estén listos)*

streamlit run main.py


## Estado del proyecto

- [x] Generador de datos
- [ ] Unit 1 — tareas
- [ ] Unit 2 — tareas
- [ ] Dashboards