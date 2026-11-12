import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit3.StationaryDistribution import (  # noqa: E402
    _spearman,
    comparar_rankings,
    conteo_scans_por_bin,
    distribucion_estacionaria,
    power_iteration,
    verificar_estacionaria,
)


def cadena_dos_estados(p=0.3, q=0.6):
    """Cadena de 2 estados con solución exacta conocida: pi_a = q/(p+q), pi_b = p/(p+q)."""
    estados = ["a", "b"]
    M = {"a": {"a": 1 - p, "b": p}, "b": {"a": q, "b": 1 - q}}
    return estados, M


class TestPowerIteration(unittest.TestCase):
    def test_converge_a_la_solucion_exacta_de_dos_estados(self):
        estados, M = cadena_dos_estados(p=0.3, q=0.6)
        pi, iteraciones, historial = power_iteration(estados, M, tol=1e-12)
        self.assertAlmostEqual(pi["a"], 0.6 / 0.9, places=8)
        self.assertAlmostEqual(pi["b"], 0.3 / 0.9, places=8)
        self.assertGreater(iteraciones, 0)
        self.assertEqual(len(historial), iteraciones)
        # la distancia entre iteraciones sucesivas debe ir bajando (chequeo laxo: el
        # promedio de la segunda mitad es menor que el de la primera)
        mitad = len(historial) // 2
        if mitad > 0:
            self.assertLess(sum(historial[mitad:]), sum(historial[:mitad]) + 1e-9)

    def test_no_depende_del_punto_de_partida(self):
        estados, M = cadena_dos_estados()
        pi1, _, _ = power_iteration(estados, M, tol=1e-12, pi0={"a": 1.0, "b": 0.0})
        pi2, _, _ = power_iteration(estados, M, tol=1e-12, pi0={"a": 0.0, "b": 1.0})
        self.assertAlmostEqual(pi1["a"], pi2["a"], places=6)

    def test_estado_absorbente_sin_salida(self):
        # "b" no tiene fila -> se trata como absorbente: toda la masa termina en b
        estados = ["a", "b"]
        M = {"a": {"b": 1.0}, "b": {}}
        pi, _, _ = power_iteration(estados, M, tol=1e-12, max_iter=50)
        self.assertAlmostEqual(pi["b"], 1.0, places=8)
        self.assertAlmostEqual(pi["a"], 0.0, places=8)

    def test_distribucion_uniforme_es_punto_fijo(self):
        # matriz doblemente estocástica de 3 estados con simetría circular: la uniforme
        # es un punto fijo exacto, así que converge en 1 iteración
        estados = ["x", "y", "z"]
        M = {s: {t: 1 / 3 for t in estados} for s in estados}
        pi, iteraciones, _ = power_iteration(estados, M, tol=1e-12)
        for s in estados:
            self.assertAlmostEqual(pi[s], 1 / 3, places=10)
        self.assertEqual(iteraciones, 1)


class TestVerificarEstacionaria(unittest.TestCase):
    def test_pi_exacta_tiene_residuo_nulo(self):
        estados, M = cadena_dos_estados(p=0.3, q=0.6)
        pi = {"a": 0.6 / 0.9, "b": 0.3 / 0.9}
        residuo, ok = verificar_estacionaria(pi, M)
        self.assertTrue(ok)
        self.assertLess(residuo, 1e-9)

    def test_pi_incorrecta_falla(self):
        estados, M = cadena_dos_estados(p=0.3, q=0.6)
        residuo, ok = verificar_estacionaria({"a": 0.5, "b": 0.5}, M)
        self.assertFalse(ok)
        self.assertGreater(residuo, 1e-3)


class TestComparaciones(unittest.TestCase):
    def test_spearman_orden_identico(self):
        self.assertAlmostEqual(_spearman(["a", "b", "c"], ["a", "b", "c"]), 1.0)

    def test_spearman_orden_invertido(self):
        self.assertAlmostEqual(_spearman(["a", "b", "c"], ["c", "b", "a"]), -1.0)

    def test_comparar_rankings_coincidencia_total(self):
        pi = {"a": 0.5, "b": 0.3, "c": 0.2}
        referencia = {"a": 100, "b": 50, "c": 10}
        r = comparar_rankings(pi, referencia, top_n=2)
        self.assertEqual(r["coincidencias_top_n"], 2)
        self.assertAlmostEqual(r["spearman"], 1.0)


class TestDatasetReal(unittest.TestCase):
    def test_distribucion_estacionaria_es_valida(self):
        pi, estados, iteraciones, historial, estados_sin_salida = distribucion_estacionaria()
        self.assertEqual(set(pi), set(estados))
        self.assertAlmostEqual(sum(pi.values()), 1.0, places=8)
        self.assertTrue(all(v >= 0 for v in pi.values()))
        self.assertGreater(len(estados), 8)  # la tarea pide >= 8 estados

    def test_coincide_bastante_con_el_conteo_crudo(self):
        pi, estados, *_ = distribucion_estacionaria()
        crudo = conteo_scans_por_bin()
        r = comparar_rankings(pi, crudo, top_n=10)
        self.assertGreaterEqual(r["coincidencias_top_n"], 7)
        self.assertGreater(r["spearman"], 0.8)


if __name__ == "__main__":
    unittest.main()
