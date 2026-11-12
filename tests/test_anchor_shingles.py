import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit5.AnchorShingles import (
    generar_chunks,
    jaccard,
    obtener_anclas,
    separacion,
    shingles_anclados,
    shingles_consecutivos,
)


class TestAnchorShingles(unittest.TestCase):
    def test_consecutivos_toman_todas_las_ventanas(self):
        self.assertEqual(shingles_consecutivos(["a", "b", "c", "d"], 3), {"a|b|c", "b|c|d"})

    def test_anclados_solo_empiezan_en_un_ancla(self):
        self.assertEqual(shingles_anclados(["a", "b", "c", "d"], 2, {"b"}), {"b|c"})

    def test_ancla_al_final_no_completa_shingle(self):
        self.assertEqual(shingles_anclados(["a", "b", "c"], 2, {"c"}), set())

    def test_anclado_es_subconjunto_del_consecutivo(self):
        for c in generar_chunks(n_chunks=50):
            for k in (2, 3, 4):
                self.assertLessEqual(shingles_anclados(c["bin_sequence"], k, {"bin_1", "bin_2"}),
                                     shingles_consecutivos(c["bin_sequence"], k))

    def test_jaccard(self):
        self.assertEqual(jaccard({1, 2, 3}, {2, 3, 4}), 0.5)
        self.assertEqual(jaccard(set(), {1}), 0.0)

    def test_generador_es_reproducible_y_balanceado(self):
        a, b = generar_chunks(n_chunks=100), generar_chunks(n_chunks=100)
        self.assertEqual([c["bin_sequence"] for c in a], [c["bin_sequence"] for c in b])
        self.assertEqual(sum(1 for c in a if c["cluster"] == 0), 10)

    def test_separacion_perfecta(self):
        conjuntos = [{1, 2}, {1, 2}, {8, 9}, {8, 9}]
        r = separacion(conjuntos, [0, 0, 1, 1], top=1)
        self.assertEqual(r["auc"], 1.0)
        self.assertEqual(r["precision_at_5"], 1.0)

    def test_anclas_son_cinco_bins_distintos(self):
        anclas, pr = obtener_anclas()
        self.assertEqual(len(set(anclas)), 5)
        self.assertAlmostEqual(sum(pr.values()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()