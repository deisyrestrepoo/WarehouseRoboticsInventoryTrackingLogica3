"""
Selector de dashboards.
Ejecutar con: streamlit run main.py
"""
import streamlit as st

import dashboard_unit1
import dashboard_unit2

st.set_page_config(page_title="Warehouse Robotics — Dashboards", layout="wide")

st.sidebar.title("Warehouse Robotics")
unidad = st.sidebar.radio("Elegir dashboard", ["Unit 1", "Unit 2"])

if unidad == "Unit 1":
    dashboard_unit1.panel_a()
    dashboard_unit1.panel_b()
else:
    dashboard_unit2.panel_a()
    st.divider()
    dashboard_unit2.panel_b()
    st.divider()
    dashboard_unit2.panel_c()
# menú/selector para elegir qué dashboard mostrar en la expo
