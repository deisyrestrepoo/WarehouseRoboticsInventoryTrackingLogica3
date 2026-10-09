import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit1.ProbabilityAnalysis import (  # noqa: E402
    analisis,
    cargar_dataset,
    cota_chebyshev,
    muestra_minima,
    probabilidad_empirica,
    tasa_real,
)


def scans_con_tasa(total, bajo_stock):
    return [{"is_low_stock": i < bajo_stock} for i in range(total)]


class TestTasaReal(unittest.TestCase):
    def test_fraccion(self):
        self.assertAlmostEqual(tasa_real(scans_con_tasa(200, 50)), 0.25)

    def test_extremos(self):
        self.assertEqual(tasa_real(scans_con_tasa(10, 0)), 0.0)
        self.assertEqual(tasa_real(scans_con_tasa(10, 10)), 1.0)


class TestCotaChebyshev(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(cota_chebyshev(1000, 0.05, 0.2), 0.2 * 0.8 / (1000 * 0.05 ** 2))

    def test_peor_caso_es_un_cuarto(self):
        self.assertAlmostEqual(cota_chebyshev(1000, 0.05), 0.25 / (1000 * 0.05 ** 2))

    def test_peor_caso_domina_a_la_cota_con_p(self):
        for p in (0.01, 0.12, 0.5, 0.9):
            self.assertLessEqual(cota_chebyshev(500, 0.03, p), cota_chebyshev(500, 0.03))

    def test_se_limita_a_uno(self):
        self.assertEqual(cota_chebyshev(1, 0.01, 0.5), 1.0)

    def test_baja_al_aumentar_n_y_sube_al_achicar_eps(self):
        self.assertLess(cota_chebyshev(2000, 0.03, 0.12), cota_chebyshev(500, 0.03, 0.12))
        self.assertGreater(cota_chebyshev(2000, 0.01, 0.12), cota_chebyshev(2000, 0.03, 0.12))


class TestMuestraMinima(unittest.TestCase):
    def test_cumple_la_cota_y_es_minima(self):
        for eps, delta, p in [(0.03, 0.10, 0.1241), (0.05, 0.05, 0.3), (0.02, 0.10, None)]:
            n = muestra_minima(eps, delta, p)
            self.assertLessEqual(cota_chebyshev(n, eps, p), delta + 1e-12)
            self.assertGreater(cota_chebyshev(n - 1, eps, p), delta)

    def test_valor_conocido(self):
        self.assertEqual(muestra_minima(0.05, 0.05), math.ceil(0.25 / (0.05 ** 2 * 0.05)))

    def test_mas_confianza_pide_mas_muestra(self):
        self.assertGreater(muestra_minima(0.03, 0.05, 0.12), muestra_minima(0.03, 0.10, 0.12))


class TestProbabilidadEmpirica(unittest.TestCase):
    def test_es_reproducible_con_la_misma_semilla(self):
        datos = scans_con_tasa(1000, 120)
        a = probabilidad_empirica(datos, 100, 0.03, ensayos=200, semilla=7)
        b = probabilidad_empirica(datos, 100, 0.03, ensayos=200, semilla=7)
        self.assertEqual(a, b)
        self.assertTrue(0.0 <= a <= 1.0)

    def test_muestra_igual_a_la_poblacion_no_tiene_error(self):
        datos = scans_con_tasa(500, 60)
        self.assertEqual(probabilidad_empirica(datos, 500, 0.01, ensayos=20), 0.0)

    def test_la_cota_se_cumple_en_el_dataset_real(self):
        dataset = cargar_dataset()
        p = tasa_real(dataset)
        for n in (250, 500, 1000):
            empirica = probabilidad_empirica(dataset, n, 0.03, ensayos=1000)
            self.assertLessEqual(empirica, cota_chebyshev(n, 0.03, p))


class TestAnalisis(unittest.TestCase):
    def test_estructura_y_orden(self):
        r = analisis(cargar_dataset(), tamanos=(100, 500), ensayos=100)
        self.assertEqual([f["n"] for f in r["filas"]], [100, 500])
        self.assertAlmostEqual(r["p_real"], tasa_real(cargar_dataset()))
        for f in r["filas"]:
            self.assertLessEqual(f["cota_chebyshev"], f["cota_peor_caso"])
        self.assertGreaterEqual(r["filas"][0]["cota_chebyshev"], r["filas"][1]["cota_chebyshev"])


if __name__ == "__main__":
    unittest.main()
