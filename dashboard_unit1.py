"""
Dashboard Unit 1.
Panel A: distribución de carga por bucket, chaining vs. Power of Two Choices (Task 4).
Panel B: tiempo del algoritmo aleatorizado vs. determinista (Task 2) - pendiente.
"""
import statistics

import pandas as pd
import plotly.express as px
import streamlit as st

from unit1.HashTableOperations import cargar_dataset, comparar, repetir


@st.cache_data
def _cargar_scans():
    return cargar_dataset()


@st.cache_data
def _repetir(num_buckets):
    return repetir(_cargar_scans(), num_buckets)


def panel_a():
    st.subheader("Panel A — Distribución de carga por bucket")
    st.caption(
        "Cuántos scans queda guardando cada bucket de la tabla hash (clave: bin_location), "
        "con chaining y con Power of Two Choices."
    )

    scans = _cargar_scans()
    num_buckets = st.slider("Número de buckets", min_value=20, max_value=200, value=100, step=10)
    resultado = comparar(scans, num_buckets)

    chaining = resultado["Chaining"]["metricas"]
    dos_opciones = resultado["Power of Two Choices"]["metricas"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Carga máxima — Chaining", chaining["carga_max"])
    reduccion = (dos_opciones["carga_max"] - chaining["carga_max"]) / chaining["carga_max"] * 100
    col2.metric(
        "Carga máxima — Two Choices",
        dos_opciones["carga_max"],
        delta=f"{reduccion:.0f}% vs chaining",
        delta_color="inverse",
    )
    col3.metric(
        "Desviación estándar (Chaining → Two Choices)",
        f"{chaining['desviacion']:.0f} → {dos_opciones['desviacion']:.0f}",
    )

    filas = [
        {"Bucket": i, "Scans": carga, "Estrategia": estrategia}
        for estrategia, datos in resultado.items()
        for i, carga in enumerate(datos["cargas"])
    ]
    df = pd.DataFrame(filas)

    fig = px.bar(
        df,
        x="Bucket",
        y="Scans",
        color="Estrategia",
        barmode="group",
        title="Scans por bucket",
    )
    fig.update_layout(xaxis_title="Número de bucket", yaxis_title="Cantidad de scans")
    st.plotly_chart(fig, width="stretch")

    total = f"{len(scans):,}".replace(",", ".")

    with st.expander(f"Ver métricas completas — {total} scans, {num_buckets} buckets"):
        filas_tabla = {
            "Carga máxima / mínima": lambda m: f"{m['carga_max']} / {m['carga_min']}",
            "Carga promedio": lambda m: f"{m['carga_promedio']:.1f}",
            "Desviación estándar": lambda m: f"{m['desviacion']:.1f}",
            "Buckets vacíos": lambda m: f"{m['buckets_vacios']}",
            "Registros revisados por búsqueda (promedio)": lambda m: f"{m['revisados_por_busqueda']:.1f}",
            "Tiempo de inserción": lambda m: f"{m['tiempo_insercion_s'] * 1000:.2f} ms",
            "Tiempo de búsqueda": lambda m: f"{m['tiempo_busqueda_s'] * 1000:.2f} ms",
        }
        tabla = pd.DataFrame(
            {
                nombre: {etiqueta: f(datos["metricas"]) for etiqueta, f in filas_tabla.items()}
                for nombre, datos in resultado.items()
            }
        )
        st.dataframe(tabla, width="stretch")

    with st.expander("Ver carga máxima en 30 repeticiones (distintas funciones hash)"):
        maximos = _repetir(num_buckets)
        repeticiones = pd.DataFrame(
            {
                nombre: {
                    "Promedio": f"{statistics.mean(v):.1f}",
                    "Mínimo": f"{min(v)}",
                    "Máximo": f"{max(v)}",
                }
                for nombre, v in maximos.items()
            }
        )
        st.dataframe(repeticiones, width="stretch")


def panel_b():
    st.subheader("Panel B — Algoritmo aleatorizado vs. determinista")
    st.info("Pendiente: se conecta cuando la Task 2 (RandomizedAlgorithm.py) esté lista.")


def main():
    st.set_page_config(page_title="Dashboard Unit 1", layout="wide")
    st.title("Warehouse Robotics — Dashboard Unit 1")
    panel_a()
    panel_b()


if __name__ == "__main__":
    main()