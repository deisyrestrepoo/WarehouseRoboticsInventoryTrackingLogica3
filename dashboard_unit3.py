"""
Dashboard Unit 3.
Panel A: grafo de movimientos y clasificación de bins (Task 1).
Panel D: PageRank (Task 3) vs. ranking por conteo crudo de scans por bin, con overlay de la
tasa de bajo stock de cada bin.
"""
import statistics
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from unit3.DTMCConstruction import (
    cargar_grafo,
    cargar_scans,
    clasificar_formal,
    clasificar_hubs,
    trafico_entrante,
)
from unit3.PageRank import comparar_rankings, pagerank


@st.cache_data
def _cargar():
    aristas, estados, M, estados_sin_salida = cargar_grafo()
    return aristas, estados, M


def panel_a():
    st.subheader("Panel A — Grafo de movimientos y clasificación de bins")
    st.caption(
        "El stream de scans se agrega por robot en un grafo de movimientos entre bins. "
        "Con ese grafo se arma la cadena de Markov (Task 1) y se clasifican los bins."
    )

    aristas, estados, M = _cargar()
    self_loops = sum(1 for e in aristas if e["from_bin"] == e["to_bin"])
    clasificacion, componentes = clasificar_formal(estados, aristas)
    recurrentes = sum(1 for c in clasificacion.values() if c == "recurrente")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bins (estados)", len(estados))
    col2.metric("Movimientos distintos", len(aristas))
    col3.metric("Self-loops", self_loops)
    col4.metric("Componentes fuertemente conexas", len(componentes))

    if len(componentes) == 1:
        st.info(
            "El grafo es una sola componente fuertemente conexa: los "
            f"{recurrentes} bins son recurrentes en el sentido formal de cadenas de Markov. "
            "La clasificación de abajo (hub / poco visitado) es una etiqueta de negocio por "
            "volumen de tráfico, no la clasificación formal."
        )

    entrante = trafico_entrante(aristas)
    top_n = st.slider("Cuántos bins mostrar en cada extremo", min_value=5, max_value=25, value=10)
    hubs, poco_visitados, _ = clasificar_hubs(estados, aristas, top_n=top_n)

    filas = (
        [{"Bin": b, "Llegadas": entrante[b], "Categoría": "Hub"} for b in hubs]
        + [{"Bin": b, "Llegadas": entrante[b], "Categoría": "Poco visitado"} for b in poco_visitados]
    )
    df = pd.DataFrame(filas)
    df["Bin"] = pd.Categorical(df["Bin"], categories=[f["Bin"] for f in filas], ordered=True)

    fig = px.bar(
        df,
        x="Bin",
        y="Llegadas",
        color="Categoría",
        title=f"Top {top_n} bins 'hub' y {top_n} bins 'poco visitados', por llegadas de robots",
    )
    fig.update_layout(xaxis_title="Bin", yaxis_title="Llegadas (tráfico entrante)")
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Explorar una fila de la matriz de transición M**")
    bin_elegido = st.selectbox("Bin de origen", estados, index=0)
    destinos = M.get(bin_elegido, {})
    if destinos:
        df_fila = (
            pd.DataFrame(
                [{"Bin destino": d, "Probabilidad": p} for d, p in destinos.items()]
            )
            .sort_values("Probabilidad", ascending=False)
            .head(10)
        )
        st.caption(
            f"A dónde va un robot que está en **{bin_elegido}** (probabilidades de M, "
            "top 10 destinos más probables)."
        )
        fig2 = px.bar(df_fila, x="Bin destino", y="Probabilidad")
        fig2.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig2, width="stretch")
    else:
        st.info(f"{bin_elegido} no tiene movimientos salientes registrados.")


@st.cache_data
def _datos_panel_d(d):
    """PageRank con damping d, conteo crudo de scans por bin y scans en bajo stock por bin."""
    pi, estados, _, _, trayectoria = pagerank(d)
    scans = cargar_scans()
    total = Counter(sc["bin_location"] for sc in scans)
    bajo = Counter(sc["bin_location"] for sc in scans if sc["is_low_stock"])
    return pi, estados, len(trayectoria) - 1, total, bajo, len(scans)


def panel_d():
    st.subheader("Panel D — PageRank vs. ranking por conteo crudo de scans, con overlay de bajo stock")
    st.caption(
        "Dos formas de decir qué bins son importantes: cuántos scans recibe cada bin (conteo crudo) "
        "o cuánto pesa en la cadena de Markov con teletransportación (PageRank, Task 3). "
        "Encima se dibuja la tasa de scans en bajo stock de cada bin."
    )

    col_d, col_n = st.columns(2)
    d = col_d.slider(
        "Factor de amortiguación d (PageRank)", min_value=0.50, max_value=0.99, value=0.85, step=0.01,
        help="Con probabilidad d el robot sigue un movimiento real; con 1 − d salta a un bin al azar.",
    )
    top_n = col_n.slider("Bins a mostrar en el gráfico de barras", min_value=5, max_value=25, value=10, key="panel_d_top_n")

    pi, estados, iteraciones, total, bajo, n_scans = _datos_panel_d(d)
    tasa_global = sum(bajo.values()) / n_scans
    comparacion = comparar_rankings(pi, total, top_n=top_n)

    ranking_scans = [b for b, _ in sorted(total.items(), key=lambda kv: -kv[1])]
    ranking_pr = [b for b, _ in sorted(pi.items(), key=lambda kv: -kv[1])]
    pos_scans = {b: i + 1 for i, b in enumerate(ranking_scans)}
    pos_pr = {b: i + 1 for i, b in enumerate(ranking_pr)}

    df = pd.DataFrame(
        [
            {
                "Bin": b,
                "Scans": total[b],
                "% de scans": total[b] / n_scans * 100,
                "PageRank %": pi[b] * 100,
                "Posición por scans": pos_scans[b],
                "Posición por PageRank": pos_pr[b],
                "Cambio de posición": pos_scans[b] - pos_pr[b],
                "Scans en bajo stock": bajo.get(b, 0),
                "Bajo stock %": bajo.get(b, 0) / total[b] * 100,
            }
            for b in estados
        ]
    )
    df["Sobre la tasa global"] = df["Bajo stock %"] > tasa_global * 100

    mayor_cambio = df.loc[df["Cambio de posición"].abs().idxmax()]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Iteraciones hasta converger", iteraciones)
    col2.metric(f"Coincidencias en el top-{top_n}", f"{comparacion['coincidencias_top_n']} / {top_n}")
    col3.metric("Correlación de Spearman", f"{comparacion['spearman']:.4f}")
    col4.metric(
        "Mayor cambio de posición",
        mayor_cambio["Bin"],
        delta=f"{int(mayor_cambio['Cambio de posición']):+d} posiciones con PageRank",
        delta_color="off",
    )

    # --- Gráfico 1: barras PageRank vs. scans + overlay de bajo stock ---
    top = df.sort_values("PageRank %", ascending=False).head(top_n)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=top["Bin"], y=top["PageRank %"], name="PageRank (%)"), secondary_y=False)
    fig.add_trace(go.Bar(x=top["Bin"], y=top["% de scans"], name="Conteo crudo (% de scans)"), secondary_y=False)
    fig.add_trace(
        go.Scatter(
            x=top["Bin"], y=top["Bajo stock %"], mode="lines+markers", name="Bajo stock (% de sus scans)",
            line={"color": "red"}, marker={"symbol": "diamond", "size": 9},
            customdata=top[["Scans", "Scans en bajo stock"]],
            hovertemplate="%{x}<br>Bajo stock: %{y:.1f}%<br>%{customdata[1]} de %{customdata[0]} scans<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_hline(
        y=tasa_global * 100, line_dash="dot", line_color="red", secondary_y=True,
        annotation_text=f"Tasa global {tasa_global:.1%}", annotation_position="top right",
    )
    fig.update_layout(
        title=f"Top {top_n} bins por PageRank (d = {d:.2f}): PageRank, conteo crudo y bajo stock",
        barmode="group", xaxis_title="Bin",
    )
    fig.update_yaxes(title_text="% del total", secondary_y=False)
    fig.update_yaxes(title_text="% de scans en bajo stock", secondary_y=True, rangemode="tozero")
    st.plotly_chart(fig, width="stretch")

    # --- Gráfico 2: todos los bins, % de scans vs. PageRank ---
    lim_bajo = min(df["% de scans"].min(), df["PageRank %"].min()) * 0.9
    lim_alto = max(df["% de scans"].max(), df["PageRank %"].max()) * 1.1
    fig2 = px.scatter(
        df, x="% de scans", y="PageRank %", color="Bajo stock %",
        color_continuous_scale="YlOrRd", hover_name="Bin",
        hover_data={"Scans": True, "Posición por scans": True, "Posición por PageRank": True,
                    "% de scans": ":.2f", "PageRank %": ":.2f", "Bajo stock %": ":.1f"},
        log_x=True, log_y=True,
        title="Todos los bins: peso por conteo crudo contra peso por PageRank (color = bajo stock)",
    )
    fig2.add_trace(go.Scatter(
        x=[lim_bajo, lim_alto], y=[lim_bajo, lim_alto], mode="lines",
        line={"dash": "dash", "color": "gray"}, name="PageRank = conteo crudo", hoverinfo="skip"))
    fig2.update_layout(coloraxis_colorbar_title="Bajo stock %")
    st.plotly_chart(fig2, width="stretch")

    correlacion = statistics.correlation(df["PageRank %"].tolist(), df["Bajo stock %"].tolist())
    fuerza = "débil" if abs(correlacion) < 0.3 else "moderada" if abs(correlacion) < 0.6 else "fuerte"
    iguales = "el top coincide" if comparacion["coincidencias_top_n"] == top_n else "el top difiere"
    st.info(
        f"Con d = {d:.2f}, {iguales} entre PageRank y conteo crudo ({comparacion['coincidencias_top_n']} de {top_n}) "
        f"y la correlación de Spearman es {comparacion['spearman']:.3f}: en este dataset ambos rankings casi "
        "no se distinguen. Lo que sí cambia es el peso: la teletransportación le quita masa a los bins "
        "con más tráfico y se la reparte a los de menos, por eso los bins de más tráfico (a la derecha) "
        "quedan bajo la diagonal y los de menos tráfico (a la izquierda), sobre ella. "
        f"La tasa de bajo stock tiene una relación {fuerza} con el PageRank (r = {correlacion:.2f}): "
        "un bin importante no es necesariamente un bin que se quede sin stock. Los bins con pocos "
        "scans tienen tasas muy variables, así que conviene mirar también la cantidad de scans."
    )

    with st.expander("Ver tabla completa de bins"):
        solo_alerta = st.checkbox("Solo bins con bajo stock sobre la tasa global", key="panel_d_solo_alerta")
        tabla = df.sort_values("PageRank %", ascending=False)
        if solo_alerta:
            tabla = tabla[tabla["Sobre la tasa global"]]
        st.dataframe(
            tabla.drop(columns=["Sobre la tasa global"]),
            width="stretch",
            hide_index=True,
            column_config={
                "% de scans": st.column_config.NumberColumn(format="%.2f"),
                "PageRank %": st.column_config.NumberColumn(format="%.2f"),
                "Bajo stock %": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption(
            f"{len(tabla)} de {len(df)} bins. Cambio de posición positivo: el bin sube al pasar del conteo "
            f"crudo a PageRank. Tasa global de bajo stock: {tasa_global:.2%}."
        )


def main():
    st.set_page_config(page_title="Dashboard Unit 3", layout="wide")
    st.title("Warehouse Robotics — Dashboard Unit 3")
    panel_a()
    st.divider()
    panel_d()


if __name__ == "__main__":
    main()