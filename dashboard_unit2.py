"""
Dashboard Unit 2.
Panel A: tasa de falsos positivos del Bloom filter "hasta ahora" (Task 2).
Panel B: SKUs distintos, conteo real (Task 1) vs. estimación de Flajolet-Martin (Task 4) - pendiente.
Panel C: tablero de alertas de bajo stock en vivo (Task 1) + resultado de F_k (Task 5) - pendiente.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from unit2.BloomFilter import FiltroBloom, barrer_tamanos, calcular_parametros_optimos, simular
from unit2.StreamModelQueryDesign import (
    TAMANO_VENTANA,
    ConsultaBajoStockPorPasillo,
    cargar_stream,
    simular_stream,
)


@st.cache_data
def _cargar_scans():
    return cargar_stream()


@st.cache_data
def _simular_bloom(n_esperado, p_deseado):
    scans = _cargar_scans()
    m, k = calcular_parametros_optimos(n_esperado, p_deseado)
    filtro = FiltroBloom(tamano_bits=m, num_hashes=k)
    return m, k, simular(scans, filtro)


@st.cache_data
def _barrido_bloom(n_esperado, p_deseado):
    return barrer_tamanos(_cargar_scans(), n_esperado, p_deseado)


@st.cache_data
def _evolucion_skus_distintos():
    _, _, historial = simular_stream(_cargar_scans())
    return historial


@st.cache_data
def _estado_ventana(posicion, tamano_ventana):
    consulta = ConsultaBajoStockPorPasillo(tamano_ventana)
    for scan in _cargar_scans()[:posicion]:
        consulta.procesar(scan)
    return list(consulta.ventana)


def panel_a():
    st.subheader("Panel A — Tasa de falsos positivos del Bloom filter, hasta ahora")
    st.caption(
        "Filtro de Bloom con los SKUs ya marcados en bajo stock en el turno, para no "
        "duplicar tickets de reabastecimiento. Se compara la tasa de falsos positivos "
        "empírica (medida) contra la teórica (fórmula), a medida que avanza el turno."
    )

    col1, col2 = st.columns(2)
    n_esperado = col1.number_input(
        "SKUs distintos esperados en bajo stock (n)", min_value=10, value=300, step=10
    )
    p_deseado = col2.select_slider(
        "Tasa de falsos positivos objetivo (p)",
        options=[0.001, 0.005, 0.01, 0.05, 0.1],
        value=0.01,
        format_func=lambda p: f"{p:.1%}",
    )

    m, k, resultado = _simular_bloom(n_esperado, p_deseado)

    col1, col2, col3 = st.columns(3)
    col1.metric("Tamaño del filtro", f"{m} bits, {k} hashes")
    col2.metric("Tickets generados", resultado["tickets_generados"])
    col3.metric(
        "Tasa empírica final",
        f"{resultado['tasa_empirica']:.3%}",
        delta=f"teórica: {resultado['tasa_teorica_final']:.3%}",
        delta_color="off",
    )

    historial = resultado["historial"]
    if historial:
        df = pd.DataFrame(historial, columns=["SKUs insertados", "Empírica", "Teórica"])
        df = df.melt(
            id_vars="SKUs insertados", var_name="Tasa", value_name="Falsos positivos"
        )
        fig = px.line(
            df,
            x="SKUs insertados",
            y="Falsos positivos",
            color="Tasa",
            title="Tasa de falsos positivos acumulada durante el turno",
        )
        fig.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No hubo SKUs en bajo stock que procesar con estos parámetros.")

    with st.expander("Comparar distintos tamaños de filtro"):
        barrido = _barrido_bloom(n_esperado, p_deseado)
        tabla = pd.DataFrame(
            [
                {
                    "Factor del tamaño óptimo": f"{r['factor']:.0%}",
                    "Bits (m)": r["m"],
                    "Hashes (k)": r["k"],
                    "Empírica": f"{r['tasa_empirica']:.3%}",
                    "Teórica": f"{r['tasa_teorica_final']:.3%}",
                }
                for r in barrido
            ]
        )
        st.dataframe(tabla, width="stretch", hide_index=True)


def panel_b():
    st.subheader("Panel B — SKUs distintos: conteo real vs. estimación")
    st.caption(
        "Conteo exacto de SKUs distintos vistos en el turno (Task 1), a medida que avanza "
        "el stream. La estimación de Flajolet-Martin se agrega cuando esa tarea esté lista."
    )

    historial = _evolucion_skus_distintos()
    df = pd.DataFrame(historial)[["scans_procesados", "skus_distintos_hoy"]]
    df = df.rename(columns={"scans_procesados": "Scans procesados", "skus_distintos_hoy": "SKUs distintos (real)"})

    fig = px.line(df, x="Scans procesados", y="SKUs distintos (real)", markers=True)
    st.plotly_chart(fig, width="stretch")

    st.metric("SKUs distintos vistos hasta ahora (conteo real)", df["SKUs distintos (real)"].iloc[-1])
    st.info("Pendiente: superponer la estimación de Flajolet-Martin (Task 4) en este mismo gráfico.")


def panel_c():
    st.subheader("Panel C — Tablero de alertas de bajo stock, en vivo")
    st.caption(
        f"SKUs en bajo stock dentro de los últimos {TAMANO_VENTANA} scans, ordenados por "
        "urgencia (menor cantidad primero)."
    )

    scans = _cargar_scans()
    posicion = st.slider(
        "Scans procesados (posición en el stream)",
        min_value=TAMANO_VENTANA,
        max_value=len(scans),
        value=len(scans),
        step=100,
    )
    ventana = _estado_ventana(posicion, TAMANO_VENTANA)

    ultimos = {}
    for scan in ventana:
        if scan["is_low_stock"]:
            ultimos[scan["sku"]] = scan
    filas = sorted(ultimos.values(), key=lambda s: s["quantity"])

    if filas:
        tabla = pd.DataFrame(
            [
                {
                    "SKU": s["sku"],
                    "Bin": s["bin_location"],
                    "Pasillo": s["aisle"],
                    "Cantidad": s["quantity"],
                }
                for s in filas
            ]
        )
        st.dataframe(tabla, width="stretch", hide_index=True)
        st.caption(f"{len(filas)} SKUs en alerta en este momento del stream.")
    else:
        st.info("No hay SKUs en bajo stock en esta ventana.")

    st.markdown("**Momentos F₀, F₁, F₂ (Task 5)**")
    st.info("Pendiente: se agrega la tabla con el resultado de F_k cuando esa tarea esté lista.")


def main():
    st.set_page_config(page_title="Dashboard Unit 2", layout="wide")
    st.title("Warehouse Robotics — Dashboard Unit 2")
    panel_a()
    st.divider()
    panel_b()
    st.divider()
    panel_c()


if __name__ == "__main__":
    main()