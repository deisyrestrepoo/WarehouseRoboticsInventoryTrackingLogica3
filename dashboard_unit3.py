"""
Dashboard Unit 3.
Panel A: grafo de movimientos y clasificación de bins (Task 1).
Panel D (pendiente): PageRank vs. tráfico crudo por bin, con overlay de bajo stock (Task 6),
necesita la Task 3 (PageRank) y la Task 2 (distribución estacionaria) de compañeros.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from unit3.DTMCConstruction import cargar_grafo, clasificar_formal, clasificar_hubs, trafico_entrante


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


def panel_d():
    st.subheader("Panel D — PageRank vs. tráfico crudo por bin (con overlay de bajo stock)")
    st.info(
        "Pendiente: necesita el PageRank de la Task 3 y la distribución estacionaria de la "
        "Task 2, hechas por compañeros de equipo. Se arma cuando esas tareas estén listas."
    )


def main():
    st.set_page_config(page_title="Dashboard Unit 3", layout="wide")
    st.title("Warehouse Robotics — Dashboard Unit 3")
    panel_a()
    st.divider()
    panel_d()


if __name__ == "__main__":
    main()