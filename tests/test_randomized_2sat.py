import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit3.Randomized2SAT import (  # noqa: E402
    REGLAS,
    clausulas_insatisfechas,
    construir_formula,
    describir_clausula,
    escalamiento,
    experimento_pasos,
    formula_aleatoria_satisfacible,
    interpretar,
    literal_verdadero,
    papadimitriou,
    resolver,
    sat_exacto,
    satisface,
    variante_insatisfacible,
)


def fuerza_bruta(clausulas, n):
    for bits in itertools.product([False, True], repeat=n):
        if satisface(clausulas, list(bits)):
            return True
    return False


class TestEvaluacion(unittest.TestCase):
    def test_literales(self):
        asign = [True, False]
        self.assertTrue(literal_verdadero(1, asign))
        self.assertFalse(literal_verdadero(-1, asign))
        self.assertTrue(literal_verdadero(-2, asign))
        self.assertFalse(literal_verdadero(2, asign))

    def test_clausulas_insatisfechas(self):
        clausulas = [(1, 2), (-1, 2), (-2, -2)]
        self.assertEqual(clausulas_insatisfechas(clausulas, [True, False]), [(-1, 2)])
        self.assertEqual(clausulas_insatisfechas(clausulas, [False, False]), [(1, 2)])
        self.assertFalse(satisface(clausulas, [True, True]))


class TestSolverExacto(unittest.TestCase):
    def test_coincide_con_fuerza_bruta_en_formulas_pequenas(self):
        rng = random.Random(1)
        for _ in range(300):
            n = rng.randint(1, 6)
            m = rng.randint(1, 4 * n)
            clausulas = [
                (rng.choice([-1, 1]) * rng.randint(1, n), rng.choice([-1, 1]) * rng.randint(1, n))
                for _ in range(m)
            ]
            sat, asign = sat_exacto(clausulas, n)
            self.assertEqual(sat, fuerza_bruta(clausulas, n), clausulas)
            if sat:
                self.assertTrue(satisface(clausulas, asign), clausulas)
            else:
                self.assertIsNone(asign)

    def test_contradiccion_directa(self):
        self.assertFalse(sat_exacto([(1, 1), (-1, -1)], 1)[0])

    def test_sin_clausulas_es_satisfacible(self):
        self.assertTrue(sat_exacto([], 3)[0])


class TestPapadimitriou(unittest.TestCase):
    def test_encuentra_solucion_en_formula_satisfacible(self):
        clausulas = [(1, 2), (-1, 3), (-2, -3), (2, 3)]
        self.assertTrue(sat_exacto(clausulas, 3)[0])
        for semilla in range(50):
            asign, pasos = papadimitriou(clausulas, 3, random.Random(semilla))
            self.assertIsNotNone(asign)
            self.assertTrue(satisface(clausulas, asign))
            self.assertGreaterEqual(pasos, 0)

    def test_sin_clausulas_no_necesita_inversiones(self):
        asign, pasos = papadimitriou([], 4, random.Random(0))
        self.assertEqual(pasos, 0)
        self.assertEqual(len(asign), 4)

    def test_en_formula_insatisfacible_se_rinde(self):
        asign, pasos = papadimitriou([(1, 1), (-1, -1)], 1, random.Random(0), max_pasos=50)
        self.assertIsNone(asign)
        self.assertEqual(pasos, 50)

    def test_resolver_es_reproducible(self):
        clausulas = [(1, 2), (-1, 3), (-2, -3), (2, 3)]
        self.assertEqual(resolver(clausulas, 3, semilla=5), resolver(clausulas, 3, semilla=5))

    def test_resolver_insatisfacible_devuelve_none(self):
        asign, total, rondas = resolver([(1, 1), (-1, -1)], 1, rondas=5)
        self.assertIsNone(asign)
        self.assertEqual(rondas, 5)
        self.assertEqual(total, 5 * 2)  # 5 rondas de 2n^2 = 2 inversiones

    def test_formulas_aleatorias_satisfacibles_se_resuelven(self):
        rng = random.Random(2)
        for n in (8, 16, 24):
            clausulas = formula_aleatoria_satisfacible(n, 3 * n, rng)
            self.assertTrue(sat_exacto(clausulas, n)[0])
            asign, _, _ = resolver(clausulas, n, rondas=30)
            self.assertIsNotNone(asign)
            self.assertTrue(satisface(clausulas, asign))


class TestFormulaDelProyecto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = construir_formula()

    def test_tiene_al_menos_8_variables_y_las_5_reglas(self):
        f = self.f
        self.assertGreaterEqual(f["n"], 8)
        self.assertEqual(len(f["nombres"]), f["n"])
        self.assertEqual(set(f["reglas"]), set(REGLAS))
        self.assertEqual(len(f["reglas"]), len(f["clausulas"]))

    def test_clausulas_validas(self):
        f = self.f
        for a, b in f["clausulas"]:
            for lit in (a, b):
                self.assertTrue(1 <= abs(lit) <= f["n"])

    def test_urgentes_cumplen_la_condicion_de_los_datos(self):
        f = self.f
        n_bins = 100
        self.assertTrue(f["urgentes"])
        for b in f["urgentes"]:
            self.assertIn(b, f["candidatos"])
            self.assertGreaterEqual(f["pi"][b], 2.0 / n_bins)
            self.assertGreaterEqual(f["bajo_stock"][b], f["tasa_global"])

    def test_regla_1_genera_una_clausula_por_bin_urgente(self):
        f = self.f
        r1 = [c for c, r in zip(f["clausulas"], f["reglas"]) if r == REGLAS[0]]
        self.assertEqual(len(r1), len(f["urgentes"]))

    def test_es_satisfacible_y_papadimitriou_encuentra_una_solucion(self):
        f = self.f
        sat, exacta = sat_exacto(f["clausulas"], f["n"])
        self.assertTrue(sat)
        self.assertTrue(satisface(f["clausulas"], exacta))
        asign, _, rondas = resolver(f["clausulas"], f["n"])
        self.assertIsNotNone(asign)
        self.assertTrue(satisface(f["clausulas"], asign))
        desviados, priorizados = interpretar(f, asign)
        self.assertLessEqual(len(desviados), 1)   # regla 3: un solo desvío por pasillo (todos en el mismo)
        for b in f["urgentes"]:
            self.assertTrue(b in desviados or b in priorizados)
        for b in desviados:
            self.assertNotIn(b, priorizados)

    def test_variante_insatisfacible(self):
        f = self.f
        clausulas, n = variante_insatisfacible(f)
        self.assertFalse(sat_exacto(clausulas, n)[0])
        asign, total, rondas = resolver(clausulas, n, rondas=3)
        self.assertIsNone(asign)
        self.assertEqual(total, 3 * 2 * n * n)

    def test_describir_clausula(self):
        texto = describir_clausula(self.f, self.f["clausulas"][0])
        self.assertIn("∨", texto)
        self.assertEqual(describir_clausula(self.f, (1, 1)), self.f["nombres"][0])


class TestExperimentos(unittest.TestCase):
    def test_las_rondas_terminan_dentro_de_la_cota(self):
        f = construir_formula()
        r = experimento_pasos(f["clausulas"], f["n"], repeticiones=200)
        self.assertLessEqual(r["media"], r["cota_esperanza"])
        self.assertGreaterEqual(r["dentro_de_2n2"], 0.5)

    def test_escalamiento_por_debajo_de_n_cuadrado(self):
        for fila in escalamiento(ns=(8, 16), repeticiones=30):
            self.assertLess(fila["media"], fila["cota_esperanza"])


if __name__ == "__main__":
    unittest.main()
