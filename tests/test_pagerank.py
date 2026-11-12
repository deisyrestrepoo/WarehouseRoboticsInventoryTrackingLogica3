import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit3.PageRank import (  # noqa: E402
    analizar_convergencia,
    comparar_rankings,
    construir_matriz_google,
    distribucion_sin_teletransporte,
    pagerank,
    power_iteration_con_trayectoria,
)


class TestConstruirMatrizGoogle(unittest.TestCase):
    def test_filas_suman_uno(self):
        estados = ["a", "b", "c"]
        M = {"a": {"b": 1.0}, "b": {"a": 0.5, "c": 0.5}, "c": {"a": 1.0}}
        P = construir_matriz_google(estados, M, d=0.85)
        for origen in estados:
            self.assertAlmostEqual(sum(P[origen].values()), 1.0, places=10)

    def test_bin_sin_salida_teletransporta_siempre(self):
        estados = ["a", "b"]
        M = {"a": {"b": 1.0}, "b": {}}
        P = construir_matriz_google(estados, M, d=0.85)
        self.assertAlmostEqual(P["b"]["a"], 0.5)
        self.assertAlmostEqual(P["b"]["b"], 0.5)

    def test_d_igual_a_cero_es_teletransporte_puro(self):
        estados = ["a", "b"]
        M = {"a": {"b": 1.0}, "b": {"a": 1.0}}
        P = construir_matriz_google(estados, M, d=0.0)
        for origen in estados:
            for destino in estados:
                self.assertAlmostEqual(P[origen][destino], 0.5)

    def test_d_igual_a_uno_es_M_tal_cual(self):
        estados = ["a", "b"]
        M = {"a": {"b": 1.0}, "b": {"a": 1.0}}
        P = construir_matriz_google(estados, M, d=1.0)
        self.assertAlmostEqual(P["a"]["b"], 1.0)
        self.assertAlmostEqual(P["a"]["a"], 0.0)


class TestPowerIterationConTrayectoria(unittest.TestCase):
    def test_trayectoria_empieza_en_uniforme_y_termina_en_pi(self):
        estados = ["a", "b"]
        P = {"a": {"a": 0.7, "b": 0.3}, "b": {"a": 0.6, "b": 0.4}}
        pi, trayectoria = power_iteration_con_trayectoria(estados, P, tol=1e-12)
        self.assertAlmostEqual(trayectoria[0]["a"], 0.5)
        self.assertAlmostEqual(trayectoria[-1]["a"], pi["a"])
        # pi debe ser punto fijo: pi * P = pi
        self.assertAlmostEqual(
            pi["a"] * P["a"]["a"] + pi["b"] * P["b"]["a"], pi["a"], places=8
        )


class TestAnalizarConvergencia(unittest.TestCase):
    def test_distancia_al_final_decrece_a_cero(self):
        # asimétrica a propósito: pi no es la uniforme, así que el punto de partida
        # (uniforme) sí está lejos del final.
        estados = ["a", "b"]
        P = {"a": {"a": 0.9, "b": 0.1}, "b": {"a": 0.3, "b": 0.7}}
        pi, trayectoria = power_iteration_con_trayectoria(estados, P, tol=1e-14)
        curva = analizar_convergencia(trayectoria, pi)
        self.assertEqual(len(curva), len(trayectoria))
        self.assertAlmostEqual(curva[-1], 0.0, places=8)
        self.assertGreater(curva[0], curva[-1])


class TestPagerank(unittest.TestCase):
    def test_pagerank_es_una_distribucion_valida(self):
        pi, estados, M, P, trayectoria = pagerank()
        self.assertAlmostEqual(sum(pi.values()), 1.0, places=8)
        self.assertTrue(all(v > 0 for v in pi.values()))  # la teletransportación da masa > 0 a todos
        self.assertGreater(len(estados), 8)

    def test_d_mas_bajo_empuja_hacia_la_uniforme(self):
        pi_alto, *_ = pagerank(d=0.85)
        pi_bajo, estados, *_ = pagerank(d=0.1)
        uniforme = 1 / len(estados)
        dispersión_alta = sum((v - uniforme) ** 2 for v in pi_alto.values())
        dispersión_baja = sum((v - uniforme) ** 2 for v in pi_bajo.values())
        self.assertLess(dispersión_baja, dispersión_alta)


class TestComparacionesYReferencia(unittest.TestCase):
    def test_comparar_rankings(self):
        pi = {"a": 0.5, "b": 0.3, "c": 0.2}
        referencia = {"a": 9, "b": 5, "c": 1}
        r = comparar_rankings(pi, referencia, top_n=2)
        self.assertEqual(r["coincidencias_top_n"], 2)
        self.assertAlmostEqual(r["spearman"], 1.0)

    def test_distribucion_sin_teletransporte_coincide_con_d_uno(self):
        pi_d1, estados, M, _, _ = pagerank(d=1.0)
        pi_directo = distribucion_sin_teletransporte(estados, M)
        for b in estados:
            self.assertAlmostEqual(pi_d1[b], pi_directo[b], places=6)


if __name__ == "__main__":
    unittest.main()
