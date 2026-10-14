"""
Dashboard Unit 2.
Panel A: tasa de falsos positivos del Bloom filter "hasta ahora" (Task 2).
Panel B: SKUs distintos, conteo real (Task 1) vs. estimación de Flajolet-Martin (Task 4).
Panel C: tablero de alertas de bajo stock en vivo (Task 1) + resultado de F_k (Task 5).
"""
import math

import pandas as pd
import plotly.express as px
import streamlit as st

from unit2.BloomFilter import FiltroBloom, barrer_tamanos, calcular_parametros_optimos, simular
from unit2.DistinctCount import evolucion as evolucion_fm
from unit2.FrequencyMoments import calcular_momentos, numero_efectivo, top_bins
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
def _estimacion_fm(num_hashes, semilla):
    return evolucion_fm(_cargar_scans(), num_hashes=num_hashes, cada=TAMANO_VENTANA, semilla=semilla)


@st.cache_data
def _estado_ventana(posicion, tamano_ventana):
    consulta = ConsultaBajoStockPorPasillo(tamano_ventana)
    for scan in _cargar_scans()[:posicion]:
        consulta.procesar(scan)
    return list(consulta.ventana)


@st.cache_data
def _momentos_hasta(posicion):
    """F0/F1/F2 (Task 5) sobre los primeros `posicion` scans del stream, por bin_location."""
    return calcular_momentos(_cargar_scans()[:posicion], clave="bin_location")


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
        "Conteo exacto de SKUs distintos vistos en el turno (Task 1) contra la estimación de "
        "Flajolet-Martin (Task 4), a medida que avanza el stream."
    )

    col1, col2 = st.columns(2)
    num_hashes = col1.slider("Funciones de hash de Flajolet-Martin", min_value=1, max_value=50, value=10)
    semilla = int(col2.number_input("Semilla de las funciones de hash", min_value=0, value=42, step=1))

    real = pd.DataFrame(_evolucion_skus_distintos())[["scans_procesados", "skus_distintos_hoy"]]
    estimado = pd.DataFrame(_estimacion_fm(num_hashes, semilla))[["scans_procesados", "estimado"]]
    df = real.merge(estimado, on="scans_procesados")
    df.columns = ["Scans procesados", "Real (exacto)", "Estimado (Flajolet-Martin)"]

    fig = px.line(
        df.melt(id_vars="Scans procesados", var_name="Conteo", value_name="SKUs distintos"),
        x="Scans procesados",
        y="SKUs distintos",
        color="Conteo",
        markers=True,
    )
    st.plotly_chart(fig, width="stretch")

    final_real = int(df["Real (exacto)"].iloc[-1])
    final_estimado = float(df["Estimado (Flajolet-Martin)"].iloc[-1])
    col1, col2, col3 = st.columns(3)
    col1.metric("SKUs distintos vistos (conteo real)", final_real)
    col2.metric(
        "Estimación de Flajolet-Martin",
        f"{final_estimado:.0f}",
        delta=f"{(final_estimado - final_real) / final_real:+.1%} vs. real",
        delta_color="off",
    )
    col3.metric("Memoria de Flajolet-Martin", f"{8 * num_hashes} bytes")
    st.caption(
        f"Con {num_hashes} funciones de hash el error relativo típico esperado es de "
        f"≈ {0.78 / math.sqrt(num_hashes):.0%} (0,78/√m). La estimación sube en escalones porque "
        "depende de potencias de 2 promediadas, y por eso la curva tiene tramos planos."
    )

    with st.expander("Ver tabla de valores"):
        tabla = df.copy()
        tabla["Error relativo"] = (
            (tabla["Estimado (Flajolet-Martin)"] - tabla["Real (exacto)"]) / tabla["Real (exacto)"]
        ).map("{:+.1%}".format)
        tabla["Estimado (Flajolet-Martin)"] = tabla["Estimado (Flajolet-Martin)"].round(0).astype(int)
        st.dataframe(tabla.set_index("Scans procesados"), width="stretch")


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

    st.markdown("**Momentos F₀, F₁, F₂ sobre bin_location (Task 5)**")
    st.caption(
        "Calculados de forma exacta (no estimados) sobre los mismos scans procesados "
        "hasta la posición elegida arriba: F0 = bins distintos, F1 = total de scans, "
        "F2 = suma de (conteo por bin)²."
    )

    momentos = _momentos_hasta(posicion)
    f0, f1, f2 = momentos["F0"], momentos["F1"], momentos["F2"]
    n_efectivo = numero_efectivo(f0, f1, f2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("F0 (bins distintos)", f0)
    col2.metric("F1 (total scans)", f1)
    col3.metric("F2 (suma conteo²)", f"{f2:,}")
    col4.metric("Número efectivo de bins (F1²/F2)", f"{n_efectivo:.1f}", delta=f"de {f0}", delta_color="off")

    tabla_top = pd.DataFrame(top_bins(momentos["conteos"], top_n=10), columns=["Bin", "Scans"])
    fig_top = px.bar(tabla_top, x="Bin", y="Scans", title="Top 10 bins con más tráfico")
    st.plotly_chart(fig_top, width="stretch")


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