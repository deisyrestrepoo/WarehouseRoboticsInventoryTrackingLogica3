import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit1.RandomizedAlgorithm import (  # noqa: E402
    cargar_dataset,
    conteo_exacto,
    estimar,
    reservoir_sample,
)


class TestReservoirSample(unittest.TestCase):
    def test_devuelve_k_elementos_distintos_del_stream(self):
        muestra = reservoir_sample(iter(range(1000)), 50, random.Random(1))
        self.assertEqual(len(muestra), 50)
        self.assertEqual(len(set(muestra)), 50)
        self.assertTrue(all(0 <= x < 1000 for x in muestra))

    def test_stream_corto_devuelve_todo(self):
        self.assertEqual(sorted(reservoir_sample(range(5), 10, random.Random(1))), [0, 1, 2, 3, 4])

    def test_k_invalido(self):
        with self.assertRaises(ValueError):
            reservoir_sample(range(5), 0)

    def test_todos_los_elementos_igual_de_probables(self):
        # n=20, k=5: cada elemento debe quedar con probabilidad 5/20 = 0.25
        n, k, pruebas = 20, 5, 20_000
        cuentas = [0] * n
        rng = random.Random(123)
        for _ in range(pruebas):
            for x in reservoir_sample(range(n), k, rng):
                cuentas[x] += 1
        for c in cuentas:
            self.assertAlmostEqual(c / pruebas, k / n, delta=0.02)


class TestEstimacion(unittest.TestCase):
    def setUp(self):
        self.scans = [
            {"sku": f"sku_{i % 10}", "is_low_stock": i % 4 == 0, "stream_seq": i}
            for i in range(400)
        ]

    def test_muestra_completa_es_exacta(self):
        e = estimar(self.scans, len(self.scans))
        self.assertAlmostEqual(e["fraccion_scans"], 0.25)
        self.assertEqual(e["error_estandar"], 0.0)  # k = N: no queda incertidumbre

    def test_muestra_vacia(self):
        with self.assertRaises(ValueError):
            estimar([], 100)

    def test_conteo_exacto_usa_el_ultimo_scan_de_cada_sku(self):
        scans = [
            {"sku": "a", "is_low_stock": True, "stream_seq": 0},
            {"sku": "a", "is_low_stock": False, "stream_seq": 1},
            {"sku": "b", "is_low_stock": True, "stream_seq": 2},
        ]
        r = conteo_exacto(scans)
        self.assertEqual(r["skus"], 2)
        self.assertEqual(r["skus_bajo_stock"], 1)  # a ya se repuso, b sigue bajo
        self.assertAlmostEqual(r["fraccion_scans"], 2 / 3)


class TestDatasetReal(unittest.TestCase):
    def test_estimacion_cerca_del_valor_exacto(self):
        scans = cargar_dataset()
        exacto = conteo_exacto(scans)
        muestra = reservoir_sample(scans, 2000, random.Random(7))
        e = estimar(muestra, len(scans))
        self.assertAlmostEqual(e["fraccion_scans"], exacto["fraccion_scans"], delta=0.03)


if __name__ == "__main__":
    unittest.main()
