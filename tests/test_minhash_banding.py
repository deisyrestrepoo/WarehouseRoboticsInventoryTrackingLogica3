import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit5.MinHashBanding import (
    PRIMO,
    candidatos_secuencial,
    coeficientes,
    elegir_br,
    ejecutar_pipeline,
    evaluar,
    firma_minhash,
    generar_chunks,
    jaccard,
    jaccard_exacto,
    prob_candidato,
    shingles,
    tabla_opciones,
)


class TestMinHashBanding(unittest.TestCase):
    def test_shingles_y_jaccard(self):
        self.assertEqual(shingles(["a", "b", "c", "d"], 3), {"a|b|c", "b|c|d"})
        self.assertEqual(jaccard({1, 2, 3}, {2, 3, 4}), 0.5)
        self.assertEqual(jaccard(set(), {1}), 0.0)

    def test_firma_tiene_100_valores_y_es_determinista(self):
        coefs = coeficientes()
        conjunto = shingles([f"bin_{i}" for i in range(1, 12)])
        f1, f2 = firma_minhash(conjunto, coefs), firma_minhash(conjunto, coefs)
        self.assertEqual(len(f1), 100)
        self.assertEqual(f1, f2)

    def test_conjunto_vacio_da_firma_centinela(self):
        self.assertEqual(firma_minhash(set(), coeficientes(5)), [PRIMO] * 5)

    def test_conjuntos_iguales_tienen_firma_igual(self):
        coefs = coeficientes()
        a = shingles(["bin_1", "bin_2", "bin_3", "bin_4", "bin_5"])
        self.assertEqual(firma_minhash(a, coefs), firma_minhash(set(a), coefs))

    def test_firma_estima_el_jaccard(self):
        coefs = coeficientes()
        a = shingles([f"bin_{i}" for i in range(1, 31)])
        b = shingles([f"bin_{i}" for i in range(1, 26)] + ["bin_90", "bin_91", "bin_92", "bin_93", "bin_94"])
        fa, fb = firma_minhash(a, coefs), firma_minhash(b, coefs)
        estimado = sum(x == y for x, y in zip(fa, fb)) / len(fa)
        self.assertAlmostEqual(estimado, jaccard(a, b), delta=0.15)

    def test_prob_candidato(self):
        self.assertAlmostEqual(prob_candidato(0.8, 25, 4), 1 - (1 - 0.8 ** 4) ** 25)
        self.assertEqual(prob_candidato(0.0, 25, 4), 0.0)
        self.assertEqual(prob_candidato(1.0, 25, 4), 1.0)

    def test_opciones_usan_las_100_filas(self):
        for f in tabla_opciones():
            self.assertEqual(f["b"] * f["r"], 100)

    def test_eleccion_cumple_los_dos_criterios(self):
        e = elegir_br()
        self.assertLessEqual(e["fn_08"], 0.01)
        self.assertGreaterEqual(e["p_06"], 0.95)
        self.assertEqual((e["b"], e["r"]), (25, 4))

    def test_pipeline_con_pool_igual_al_secuencial(self):
        chunks = generar_chunks(n_chunks=120)
        res = ejecutar_pipeline(chunks, 25, 4, workers=2)
        self.assertEqual(res["candidatos"], candidatos_secuencial(res["firmas"], 25, 4))
        self.assertEqual(res["tuplas_map"], 120 * 25)

    def test_pares_con_conjuntos_identicos_son_candidatos(self):
        chunks = generar_chunks(n_chunks=50)
        chunks[7]["bin_sequence"] = list(chunks[3]["bin_sequence"])
        res = ejecutar_pipeline(chunks, 25, 4, workers=2)
        self.assertIn((3, 7), res["candidatos"])

    def test_evaluar_con_caso_chico(self):
        exacto = {(0, 1): 0.9, (0, 2): 0.7, (1, 2): 0.3}
        ev = evaluar({(0, 1), (1, 2)}, exacto, 3)
        self.assertEqual(ev["total_pares"], 3)
        self.assertEqual(ev["pares_verdaderos"], 2)
        self.assertEqual(ev["tp"], 1)
        self.assertAlmostEqual(ev["precision"], 0.5)
        self.assertAlmostEqual(ev["recall"], 0.5)
        self.assertAlmostEqual(ev["fn_empirico_08"], 0.0)

    def test_jaccard_exacto_solo_guarda_pares_con_overlap(self):
        chunks = [{"bin_sequence": ["a", "b", "c", "d"]}, {"bin_sequence": ["a", "b", "c", "d"]},
                  {"bin_sequence": ["x", "y", "z", "w"]}]
        self.assertEqual(jaccard_exacto(chunks), {(0, 1): 1.0})


if __name__ == "__main__":
    unittest.main()