import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit3.DTMCConstruction import cargar_grafo  # noqa: E402
from unit3.RandomWalk import (  # noqa: E402
    PASOS,
    comparar_con_pi,
    distribucion_estacionaria,
    ejemplo_bipartito,
    error_vs_pasos,
    hacer_perezosa,
    paso_distribucion,
    periodo,
    simular_caminata,
    variacion_total_por_paso,
)

ESTADOS = ["a", "b", "c"]
M = {"a": {"b": 1.0}, "b": {"a": 0.5, "c": 0.5}, "c": {"a": 1.0}}


class TestHacerPerezosa(unittest.TestCase):
    def test_filas_suman_uno_y_diagonal_al_menos_la_mitad(self):
        L = hacer_perezosa(ESTADOS, M)
        for s in ESTADOS:
            self.assertAlmostEqual(sum(L[s].values()), 1.0)
            self.assertGreaterEqual(L[s][s], 0.5)

    def test_mezcla_con_m(self):
        L = hacer_perezosa(ESTADOS, M)
        self.assertAlmostEqual(L["b"]["a"], 0.25)
        self.assertAlmostEqual(L["b"]["c"], 0.25)
        self.assertAlmostEqual(L["b"]["b"], 0.5)

    def test_self_loop_previo_se_suma_a_la_pereza(self):
        L = hacer_perezosa(["x", "y"], {"x": {"x": 0.4, "y": 0.6}, "y": {"x": 1.0}})
        self.assertAlmostEqual(L["x"]["x"], 0.5 + 0.5 * 0.4)

    def test_bin_sin_salida_es_absorbente(self):
        L = hacer_perezosa(["x", "y"], {"x": {"y": 1.0}})
        self.assertEqual(L["y"], {"y": 1.0})

    def test_p_invalida(self):
        with self.assertRaises(ValueError):
            hacer_perezosa(ESTADOS, M, 1.0)
        with self.assertRaises(ValueError):
            hacer_perezosa(ESTADOS, M, -0.1)


class TestEstacionaria(unittest.TestCase):
    def test_es_punto_fijo_y_suma_uno(self):
        L = hacer_perezosa(ESTADOS, M)
        pi, _ = distribucion_estacionaria(ESTADOS, L)
        self.assertAlmostEqual(sum(pi.values()), 1.0)
        despues = paso_distribucion(ESTADOS, L, pi)
        for s in ESTADOS:
            self.assertAlmostEqual(despues[s], pi[s], places=9)

    def test_la_pereza_no_cambia_pi(self):
        pi_m, _ = distribucion_estacionaria(ESTADOS, hacer_perezosa(ESTADOS, M, 0.0))
        pi_l, _ = distribucion_estacionaria(ESTADOS, hacer_perezosa(ESTADOS, M, 0.5))
        for s in ESTADOS:
            self.assertAlmostEqual(pi_m[s], pi_l[s], places=8)

    def test_valor_conocido(self):
        # a = 2 * b/... : pi_a = pi_b/2 + pi_c, pi_b = pi_a, pi_c = pi_b/2  ->  (0.4, 0.4, 0.2)
        pi, _ = distribucion_estacionaria(ESTADOS, hacer_perezosa(ESTADOS, M))
        self.assertAlmostEqual(pi["a"], 0.4, places=8)
        self.assertAlmostEqual(pi["b"], 0.4, places=8)
        self.assertAlmostEqual(pi["c"], 0.2, places=8)


class TestPeriodo(unittest.TestCase):
    def test_bipartito_tiene_periodo_2_y_la_pereza_lo_rompe(self):
        ej = ejemplo_bipartito()
        self.assertEqual(ej["periodo_simple"], 2)
        self.assertEqual(ej["periodo_perezosa"], 1)

    def test_ciclo_de_tres(self):
        self.assertEqual(periodo(["a", "b", "c"], {"a": {"b": 1}, "b": {"c": 1}, "c": {"a": 1}}), 3)

    def test_grafo_del_proyecto_es_aperiodico(self):
        _, estados, Mp, _ = cargar_grafo()
        self.assertEqual(periodo(estados, Mp), 1)


class TestSimulacion(unittest.TestCase):
    def test_conteos_suman_los_pasos_y_son_reproducibles(self):
        L = hacer_perezosa(ESTADOS, M)
        a, _ = simular_caminata(ESTADOS, L, 1000, semilla=3)
        b, _ = simular_caminata(ESTADOS, L, 1000, semilla=3)
        self.assertEqual(sum(a.values()), 1000)
        self.assertEqual(a, b)

    def test_puntos_de_control_son_acumulados(self):
        L = hacer_perezosa(ESTADOS, M)
        total, inst = simular_caminata(ESTADOS, L, 500, puntos_control=(100, 500))
        self.assertEqual(sum(inst[100].values()), 100)
        self.assertEqual(inst[500], total)

    def test_la_frecuencia_se_acerca_a_pi(self):
        L = hacer_perezosa(ESTADOS, M)
        pi, _ = distribucion_estacionaria(ESTADOS, L)
        conteos, _ = simular_caminata(ESTADOS, L, 60_000, inicio="c", semilla=5)
        r = comparar_con_pi(pi, conteos, top_n=2)
        self.assertLess(r["error_max"], 0.01)
        self.assertLess(r["error_l1"], 0.02)

    def test_no_depende_del_bin_inicial(self):
        L = hacer_perezosa(ESTADOS, M)
        pi, _ = distribucion_estacionaria(ESTADOS, L)
        for inicio in ESTADOS:
            conteos, _ = simular_caminata(ESTADOS, L, 40_000, inicio=inicio, semilla=9)
            self.assertLess(comparar_con_pi(pi, conteos, 2)["error_l1"], 0.03)


class TestComparar(unittest.TestCase):
    def test_conteos_proporcionales_a_pi_dan_error_cero(self):
        pi = {"a": 0.5, "b": 0.3, "c": 0.2}
        r = comparar_con_pi(pi, {"a": 500, "b": 300, "c": 200}, top_n=2)
        self.assertAlmostEqual(r["error_l1"], 0.0)
        self.assertAlmostEqual(r["error_max"], 0.0)
        self.assertEqual(r["coincidencias_top"], 2)

    def test_error_l1_conocido(self):
        pi = {"a": 0.5, "b": 0.5}
        r = comparar_con_pi(pi, {"a": 700, "b": 300}, top_n=1)
        self.assertAlmostEqual(r["error_l1"], 0.4)
        self.assertAlmostEqual(r["error_max"], 0.2)


class TestConvergencia(unittest.TestCase):
    def test_variacion_total_bipartito(self):
        ej = ejemplo_bipartito()
        for t in (1, 2, 5, 10):
            self.assertAlmostEqual(ej["simple"][t], 0.5)          # nunca converge
        self.assertAlmostEqual(ej["perezosa"][1], 0.25)
        self.assertLess(ej["perezosa"][10], 1e-3)
        self.assertLess(ej["perezosa"][10], ej["perezosa"][5])

    def test_distribucion_exacta_converge_en_el_grafo_real(self):
        _, estados, Mp, _ = cargar_grafo()
        L = hacer_perezosa(estados, Mp)
        pi, _ = distribucion_estacionaria(estados, L)
        tv = variacion_total_por_paso(estados, L, pi, estados[-1], (1, 10, 40))
        self.assertGreater(tv[1], tv[10])
        self.assertGreater(tv[10], tv[40])
        self.assertLess(tv[40], 1e-6)


class TestConDatasetReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.estados, cls.M, _ = cargar_grafo()
        cls.L = hacer_perezosa(cls.estados, cls.M)
        cls.pi, _ = distribucion_estacionaria(cls.estados, cls.L)

    def test_50000_pasos_se_acercan_a_pi(self):
        self.assertGreaterEqual(PASOS, 50_000)
        conteos, _ = simular_caminata(self.estados, self.L, PASOS, inicio=self.estados[-1])
        r = comparar_con_pi(self.pi, conteos)
        self.assertEqual(sum(conteos.values()), PASOS)
        self.assertLess(r["error_l1"], 0.1)
        self.assertLess(r["error_max"], 0.01)
        self.assertEqual(r["coincidencias_top"], 10)

    def test_el_error_baja_como_uno_sobre_raiz_t(self):
        filas = error_vs_pasos(self.estados, self.L, self.pi, pasos_lista=(1_000, 16_000), semillas=4)
        chico, grande = filas
        self.assertLess(grande["error_l1_medio"], chico["error_l1_medio"])
        self.assertLess(abs(grande["error_por_raiz_t"] / chico["error_por_raiz_t"] - 1), 0.35)


if __name__ == "__main__":
    unittest.main()
