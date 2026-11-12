"""
Dashboard Unit 1.
Panel A: distribución de carga por bucket, chaining vs. Power of Two Choices (Task 4).
Panel B: algoritmo aleatorizado (reservoir sampling) vs. determinista (conteo exacto), Task 2.
Panel C: tablero de alertas de bajo stock en vivo (SKU, bin, cantidad), ordenado por urgencia.
         La búsqueda de historial por SKU usa el índice de hash universal (Task 3).
"""
import random
import statistics
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from unit1.HashTableOperations import cargar_dataset, comparar, repetir
from unit1.UniversalHashing import FuncionHashUniversal, IndiceHash
from unit1.RandomizedAlgorithm import (
    TAMANOS_MUESTRA,
    conteo_exacto,
    estimar,
    experimento,
    reservoir_sample,
)


@st.cache_data
def _cargar_scans():
    return cargar_dataset()


@st.cache_data
def _repetir(num_buckets):
    return repetir(_cargar_scans(), num_buckets)


@st.cache_data
def _experimento(repeticiones):
    return experimento(_cargar_scans(), repeticiones=repeticiones)


@st.cache_data
def _medir_tiempos(repeticiones=30):
    """Tiempo promedio del conteo exacto y de muestreo + estimación para cada tamaño k."""
    scans = _cargar_scans()
    n = len(scans)

    inicio = time.perf_counter()
    for _ in range(repeticiones):
        conteo_exacto(scans)
    exacto = (time.perf_counter() - inicio) / repeticiones

    aleatorizado = {}
    for k in TAMANOS_MUESTRA:
        inicio = time.perf_counter()
        for r in range(repeticiones):
            muestra = reservoir_sample(scans, k, random.Random(r))
            estimar(muestra, n)
        aleatorizado[k] = (time.perf_counter() - inicio) / repeticiones
    return exacto, aleatorizado


@st.cache_resource
def _indice_skus(num_buckets=2003, semilla=42):
    """Índice SKU -> posiciones de sus scans (hash universal, Task 3)."""
    indice = IndiceHash(FuncionHashUniversal(num_buckets, random.Random(semilla)))
    for posicion, scan in enumerate(_cargar_scans()):
        indice.insertar(scan["sku"], posicion)
    return indice


def _nivel_urgencia(cantidad):
    if cantidad == 0:
        return "Agotado"
    if cantidad <= 2:
        return "Crítico"
    return "Bajo"


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
    st.caption(
        "Estimar la fracción de scans en bajo stock: reservoir sampling (aleatorizado, guarda "
        "solo k scans) contra el conteo exacto que recorre y guarda todo el dataset (determinista)."
    )

    scans = _cargar_scans()
    n = len(scans)
    exacto_ref = conteo_exacto(scans)

    col_k, col_semilla = st.columns(2)
    k = col_k.select_slider("Tamaño de la muestra (k)", options=TAMANOS_MUESTRA, value=1000)
    semilla = col_semilla.number_input("Semilla de la muestra", min_value=0, value=42, step=1)

    # Una corrida concreta con el k y la semilla elegidos
    muestra = reservoir_sample(scans, k, random.Random(semilla))
    est = estimar(muestra, n)
    error = abs(est["fraccion_scans"] - exacto_ref["fraccion_scans"])
    ic_bajo, ic_alto = est["ic95"]
    dentro_ic = ic_bajo <= exacto_ref["fraccion_scans"] <= ic_alto

    tiempo_exacto, tiempos_aleatorizado = _medir_tiempos()
    tiempo_k = tiempos_aleatorizado[k]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fracción exacta (determinista)", f"{exacto_ref['fraccion_scans']:.2%}")
    col2.metric(
        f"Estimación con k={k}",
        f"{est['fraccion_scans']:.2%}",
        delta=f"error {error:.2%}",
        delta_color="off",
    )
    col3.metric(
        "IC 95 %",
        f"{ic_bajo:.2%} – {ic_alto:.2%}",
        delta="contiene el valor real" if dentro_ic else "no contiene el valor real",
        delta_color="normal" if dentro_ic else "inverse",
    )
    col4.metric(
        "Scans en memoria",
        f"{k:,}".replace(",", "."),
        delta=f"{k / n:.0%} de {n:,}".replace(",", "."),
        delta_color="off",
    )

    # --- Gráfico 1: tiempo ---
    filas_tiempo = [{"Método": f"Aleatorizado k={k_}", "ms": t * 1000, "k": k_}
                    for k_, t in tiempos_aleatorizado.items()]
    df_tiempo = pd.DataFrame(filas_tiempo)
    fig_tiempo = px.bar(
        df_tiempo, x="Método", y="ms", title="Tiempo de ejecución (muestreo + estimación)"
    )
    fig_tiempo.add_hline(
        y=tiempo_exacto * 1000,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Determinista (conteo exacto): {tiempo_exacto * 1000:.2f} ms",
        annotation_position="top left",
    )
    fig_tiempo.update_layout(xaxis_title="", yaxis_title="Milisegundos")

    # --- Gráfico 2: error vs. k ---
    repeticiones = 200
    _, filas = _experimento(repeticiones)
    df_error = pd.DataFrame(filas)
    fig_error = go.Figure()
    fig_error.add_trace(go.Scatter(
        x=df_error["k"], y=df_error["error_abs_scans"], mode="lines+markers",
        name="Error absoluto promedio"))
    fig_error.add_trace(go.Scatter(
        x=df_error["k"], y=df_error["desv_empirica"], mode="lines+markers",
        name="Desviación empírica"))
    fig_error.add_trace(go.Scatter(
        x=df_error["k"], y=df_error["desv_teorica"], mode="lines",
        line={"dash": "dot"}, name="Desviación teórica"))
    fig_error.add_hline(
        y=0, line_color="red", annotation_text="Determinista: error 0",
        annotation_position="top right")
    fig_error.update_layout(
        title=f"Error de la estimación según k ({repeticiones} repeticiones)",
        xaxis_title="Tamaño de la muestra (k)",
        yaxis_title="Error (fracción de scans)",
        xaxis={"type": "log", "tickvals": list(df_error["k"])},
    )

    izq, der = st.columns(2)
    izq.plotly_chart(fig_tiempo, width="stretch")
    der.plotly_chart(fig_error, width="stretch")

    st.info(
        f"Con el dataset actual el muestreo no es más rápido que el conteo exacto en Python "
        f"(k={k}: {tiempo_k * 1000:.2f} ms contra {tiempo_exacto * 1000:.2f} ms). La ventaja del "
        f"algoritmo aleatorizado es la memoria: usa O(k) y una sola pasada, así que sirve cuando "
        f"el stream no se puede guardar completo. A cambio acepta un error que baja al subir k."
    )

    with st.expander("Ver resultados completos del experimento"):
        tabla = pd.DataFrame(
            {
                "k": df_error["k"],
                "Promedio estimado": df_error["promedio_scans"].map("{:.4f}".format),
                "Error absoluto": df_error["error_abs_scans"].map("{:.4f}".format),
                "Desv. empírica": df_error["desv_empirica"].map("{:.4f}".format),
                "Desv. teórica": df_error["desv_teorica"].map("{:.4f}".format),
                "Cobertura IC 95 %": df_error["cobertura_ic95"].map("{:.1%}".format),
                "Error abs. (por SKU)": df_error["error_abs_skus"].map("{:.4f}".format),
                "Tiempo (ms)": [f"{tiempos_aleatorizado[k_] * 1000:.2f}" for k_ in df_error["k"]],
            }
        ).set_index("k")
        st.dataframe(tabla, width="stretch")
        st.caption(
            f"Referencia exacta: {exacto_ref['fraccion_scans']:.4f} de scans en bajo stock; "
            f"{exacto_ref['skus_bajo_stock']} de {exacto_ref['skus']} SKUs "
            f"({exacto_ref['fraccion_skus']:.4f}) tienen su último scan en bajo stock."
        )


def panel_c():
    st.subheader("Panel C — Tablero de alertas de bajo stock, en vivo")
    st.caption(
        "SKUs cuyo último scan, hasta la posición elegida del stream, está en bajo stock "
        "(cantidad ≤ 5). Ordenados por urgencia: primero la menor cantidad y, a igual cantidad, "
        "el scan más reciente."
    )

    scans = _cargar_scans()
    posicion = st.slider(
        "Scans procesados (posición en el stream)",
        min_value=100,
        max_value=len(scans),
        value=len(scans),
        step=100,
        key="panel_c_posicion",
    )

    # Estado actual de cada SKU = su scan más reciente hasta `posicion`
    actual = {}
    for scan in scans[:posicion]:
        actual[scan["sku"]] = scan
    alertas = sorted(
        (s for s in actual.values() if s["is_low_stock"]),
        key=lambda s: (s["quantity"], -s["stream_seq"]),
    )
    ultimo = scans[posicion - 1]
    st.caption(f"Último scan procesado: {ultimo['scan_id']} · {ultimo['timestamp'].replace('T', ' ')}")

    agotados = sum(1 for s in alertas if s["quantity"] == 0)
    col1, col2, col3 = st.columns(3)
    col1.metric("SKUs en alerta", len(alertas))
    col2.metric("Agotados (cantidad 0)", agotados)
    col3.metric(
        "% de SKUs vistos en alerta",
        f"{len(alertas) / len(actual):.1%}",
        delta=f"{len(alertas)} de {len(actual)}",
        delta_color="off",
    )

    if alertas:
        tabla = pd.DataFrame(
            [
                {
                    "Urgencia": _nivel_urgencia(s["quantity"]),
                    "SKU": s["sku"],
                    "Bin": s["bin_location"],
                    "Cantidad": s["quantity"],
                }
                for s in alertas
            ]
        )
        st.dataframe(
            tabla,
            width="stretch",
            hide_index=True,
            height=380,
            column_config={
                "Cantidad": st.column_config.ProgressColumn(
                    "Cantidad", min_value=0, max_value=5, format="%d"
                ),
            },
        )
    else:
        st.info("No hay SKUs en bajo stock hasta este punto del stream.")

    st.markdown("**Historial de un SKU**")
    st.caption(
        "Búsqueda con el índice de hash universal (Task 3): solo revisa los SKUs del bucket "
        "en lugar de recorrer todos los scans."
    )
    sku = st.text_input("SKU a consultar", value=alertas[0]["sku"] if alertas else "sku_1", key="panel_c_sku")
    sku = sku.strip()
    if sku:
        posiciones, revisados = _indice_skus().buscar(sku)
        posiciones = [i for i in posiciones if i < posicion]
        if not posiciones:
            st.warning(f"El SKU «{sku}» no tiene scans hasta este punto del stream.")
        else:
            historial = pd.DataFrame(
                [
                    {
                        "Scan": scans[i]["scan_id"],
                        "Fecha y hora": scans[i]["timestamp"].replace("T", " "),
                        "Bin": scans[i]["bin_location"],
                        "Cantidad": scans[i]["quantity"],
                        "Bajo stock": "Sí" if scans[i]["is_low_stock"] else "No",
                    }
                    for i in posiciones
                ]
            )
            st.dataframe(historial, width="stretch", hide_index=True)
            st.caption(
                f"{len(posiciones)} scans encontrados revisando {revisados} SKU(s) del bucket."
            )
            if len(historial) > 1:
                fig = px.line(
                    historial, x="Fecha y hora", y="Cantidad", markers=True,
                    title=f"Cantidad escaneada de {sku}",
                )
                fig.add_hline(y=5, line_dash="dash", line_color="red",
                              annotation_text="Umbral de bajo stock (5)")
                st.plotly_chart(fig, width="stretch")


def main():
    st.set_page_config(page_title="Dashboard Unit 1", layout="wide")
    st.title("Warehouse Robotics — Dashboard Unit 1")
    panel_a()
    panel_b()
    panel_c()


if __name__ == "__main__":
    main()