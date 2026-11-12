import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit5.DistanceMeasures import (  # noqa: E402
    asignar_metricas,
    auc,
    auc_de_medida,
    concentracion,
    construir_chunks_plantados,
    construir_representaciones,
    correlaciones,
    distancia_coseno,
    distancia_l1,
    distancia_l2,
    ejecutar,
    graficar_distribuciones,
    guardar_chunks,
    longitud_lcs,
    nombres_bins,
    pares_al_azar,
    pearson,
    rangos,
    similitud_lcs,
    spearman,
    vecinos_extremos,
)

try:
    import matplotlib  # noqa: F401
    HAY_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAY_MATPLOTLIB = False


def chunk(secuencia, cluster, cantidad=10.0, bajo=0.1):
    return {"bin_sequence": secuencia, "seq_length": len(secuencia), "cluster": cluster,
            "robot_id": "robot_1", "avg_quantity": cantidad, "low_stock_frac": bajo}


class TestMedidas(unittest.TestCase):
    def test_l1_y_l2_a_mano(self):
        self.assertEqual(distancia_l1((0, 0), (3, 4)), 7)
        self.assertEqual(distancia_l2((0, 0), (3, 4)), 5)

    def test_distancias_simetricas_y_cero_consigo_mismo(self):
        a, b = (1.0, -2.0, 0.5), (0.0, 4.0, 2.5)
        for f in (distancia_l1, distancia_l2, distancia_coseno):
            self.assertAlmostEqual(f(a, b), f(b, a))
            self.assertAlmostEqual(f(a, a), 0.0)

    def test_l2_cumple_la_desigualdad_triangular(self):
        a, b, c = (0, 0), (3, 1), (1, 5)
        self.assertLessEqual(distancia_l2(a, c), distancia_l2(a, b) + distancia_l2(b, c))

    def test_distancia_coseno(self):
        self.assertAlmostEqual(distancia_coseno((1, 0), (2, 0)), 0.0)
        self.assertAlmostEqual(distancia_coseno((1, 0), (0, 3)), 1.0)
        self.assertAlmostEqual(distancia_coseno((1, 0), (-1, 0)), 2.0)

    def test_coseno_con_vector_cero(self):
        self.assertEqual(distancia_coseno((0, 0), (1, 1)), 1.0)

    def test_lcs_clasico(self):
        self.assertEqual(longitud_lcs(list("AGGTAB"), list("GXTXAYB")), 4)
        self.assertEqual(longitud_lcs(list("abcde"), list("ace")), 3)

    def test_lcs_casos_borde(self):
        self.assertEqual(longitud_lcs([], list("abc")), 0)
        self.assertEqual(longitud_lcs(list("abc"), list("abc")), 3)
        self.assertEqual(longitud_lcs(list("abc"), list("xyz")), 0)

    def test_lcs_simetrico(self):
        a, b = list("bin_1 bin_5 bin_2 bin_9".split()), list("bin_5 bin_9 bin_3".split())
        self.assertEqual(longitud_lcs(a, b), longitud_lcs(b, a))

    def test_lcs_respeta_el_orden(self):
        self.assertEqual(longitud_lcs(["a", "b", "c"], ["c", "b", "a"]), 1)

    def test_similitud_lcs(self):
        self.assertEqual(similitud_lcs(list("abc"), list("abc")), 1.0)
        self.assertAlmostEqual(similitud_lcs(list("ace"), list("abcde")), 3 / 5)
        self.assertEqual(similitud_lcs([], []), 0.0)
        self.assertEqual(similitud_lcs(list("abc"), list("xyz")), 0.0)


class TestPares(unittest.TestCase):
    def test_pares_distintos_ordenados_y_reproducibles(self):
        pares = pares_al_azar(50, 200, semilla=3)
        self.assertEqual(len(pares), 200)
        self.assertEqual(len(set(pares)), 200)
        self.assertTrue(all(0 <= i < j < 50 for i, j in pares))
        self.assertEqual(pares, pares_al_azar(50, 200, semilla=3))
        self.assertNotEqual(pares, pares_al_azar(50, 200, semilla=4))

    def test_todos_los_pares(self):
        self.assertEqual(len(pares_al_azar(5, 10)), 10)

    def test_demasiados_pares(self):
        with self.assertRaises(ValueError):
            pares_al_azar(5, 11)


class TestEstadistica(unittest.TestCase):
    def test_rangos_con_empates(self):
        self.assertEqual(rangos([10, 20, 20, 30]), [1, 2.5, 2.5, 4])
        self.assertEqual(rangos([3, 1, 2]), [3, 1, 2])

    def test_pearson(self):
        self.assertAlmostEqual(pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)
        self.assertAlmostEqual(pearson([1, 2, 3, 4], [8, 6, 4, 2]), -1.0)
        self.assertIsNone(pearson([1, 1, 1], [1, 2, 3]))

    def test_spearman_mide_monotonia(self):
        x = [1, 2, 3, 4, 5]
        y = [v ** 3 for v in x]
        self.assertAlmostEqual(spearman(x, y), 1.0)
        self.assertLess(pearson(x, y), 1.0)

    def test_auc(self):
        self.assertEqual(auc([5, 6, 7], [1, 2, 3]), 1.0)
        self.assertEqual(auc([1, 2, 3], [5, 6, 7]), 0.0)
        self.assertEqual(auc([1, 2], [1, 2]), 0.5)
        self.assertIsNone(auc([], [1]))

    def test_auc_de_medida_respeta_la_direccion(self):
        filas = [
            {"mismo_cluster": True, "jaccard": 0.8, "l2": 0.1},
            {"mismo_cluster": True, "jaccard": 0.7, "l2": 0.2},
            {"mismo_cluster": False, "jaccard": 0.1, "l2": 0.9},
            {"mismo_cluster": False, "jaccard": 0.2, "l2": 0.8},
        ]
        self.assertEqual(auc_de_medida(filas, "jaccard"), 1.0)  # similitud: más alto es más parecido
        self.assertEqual(auc_de_medida(filas, "l2"), 1.0)  # distancia: más bajo es más parecido

    def test_correlaciones(self):
        filas = [{"jaccard": x / 10, "lcs": 2 * x / 10 + 0.1} for x in range(10)]
        c = correlaciones(filas, pares=(("jaccard", "lcs"),))
        self.assertAlmostEqual(c[("jaccard", "lcs")]["pearson"], 1.0)
        self.assertAlmostEqual(c[("jaccard", "lcs")]["spearman"], 1.0)

    def test_concentracion(self):
        c = concentracion([2.0, 2.0, 2.0])
        self.assertEqual(c["cv"], 0.0)
        self.assertEqual(c["contraste"], 0.0)
        c = concentracion([1.0, 3.0])
        self.assertAlmostEqual(c["cv"], 0.5)  # desviación 1 / media 2
        self.assertAlmostEqual(c["contraste"], 2.0)  # (3 - 1) / 1

    def test_concentracion_con_minimo_cero(self):
        self.assertIsNone(concentracion([0.0, 1.0])["contraste"])


class TestRepresentaciones(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            chunk(["bin_1", "bin_2", "bin_3", "bin_4"], 0, cantidad=10.0, bajo=0.0),
            chunk(["bin_1", "bin_2", "bin_3", "bin_9"], 0, cantidad=20.0, bajo=0.5),
            chunk(["bin_7", "bin_7", "bin_8", "bin_8"], 1, cantidad=30.0, bajo=1.0),
        ]
        self.rep = construir_representaciones(self.chunks, k=3)

    def test_participacion_suma_uno_y_cuenta_por_bin(self):
        bins = nombres_bins()
        for c in self.rep:
            self.assertAlmostEqual(sum(c["vector_bins"]), 1.0)
        self.assertAlmostEqual(self.rep[2]["vector_bins"][bins.index("bin_7")], 0.5)
        self.assertAlmostEqual(self.rep[0]["vector_bins"][bins.index("bin_1")], 0.25)
        self.assertEqual(self.rep[0]["vector_bins"][bins.index("bin_50")], 0)

    def test_dimensiones(self):
        self.assertEqual(len(self.rep[0]["vector_bins"]), 100)
        self.assertEqual(len(self.rep[0]["occupancy_vector"]), 102)
        self.assertEqual(self.rep[0]["occupancy_vector"][:100], self.rep[0]["vector_bins"])

    def test_metricas_en_z_score(self):
        for pos in (100, 101):
            valores = [c["occupancy_vector"][pos] for c in self.rep]
            self.assertAlmostEqual(sum(valores) / 3, 0.0)
            self.assertAlmostEqual(sum(v * v for v in valores) / 3, 1.0)

    def test_shingles_de_k_bins_consecutivos(self):
        self.assertEqual(self.rep[0]["shingles"], {"bin_1|bin_2|bin_3", "bin_2|bin_3|bin_4"})
        self.assertEqual(self.rep[0]["shingle_count"], 2)

    def test_no_modifica_los_chunks_originales(self):
        self.assertNotIn("occupancy_vector", self.chunks[0])

    def test_secuencia_de_largo_variable(self):
        c = construir_representaciones([chunk(["bin_1", "bin_2"], 0), chunk(["bin_1"] * 5, 1, 12.0, 0.2)])
        self.assertAlmostEqual(c[0]["vector_bins"][0], 0.5)
        self.assertAlmostEqual(c[1]["vector_bins"][0], 1.0)

    def test_metricas_constantes_dan_cero(self):
        c = construir_representaciones([chunk(["bin_1", "bin_2"], 0), chunk(["bin_3", "bin_4"], 1)])
        self.assertEqual(c[0]["occupancy_vector"][100:], (0.0, 0.0))


class TestMetricasDesdeScansReales(unittest.TestCase):
    def scans(self):
        return ([{"bin_location": "bin_1", "quantity": 10, "is_low_stock": False}] * 3
                + [{"bin_location": "bin_2", "quantity": 2, "is_low_stock": True}] * 3)

    def test_promedios_de_scans_del_mismo_bin(self):
        chunks = asignar_metricas([chunk(["bin_1", "bin_1"], 0), chunk(["bin_2", "bin_2"], 0)],
                                  self.scans(), semilla=1)
        self.assertEqual((chunks[0]["avg_quantity"], chunks[0]["low_stock_frac"]), (10.0, 0.0))
        self.assertEqual((chunks[1]["avg_quantity"], chunks[1]["low_stock_frac"]), (2.0, 1.0))

    def test_mezcla_de_bins(self):
        c = asignar_metricas([chunk(["bin_1", "bin_2"], 0)], self.scans(), semilla=1)[0]
        self.assertEqual(c["avg_quantity"], 6.0)
        self.assertEqual(c["low_stock_frac"], 0.5)

    def test_reproducible(self):
        a = asignar_metricas([chunk(["bin_1", "bin_2"] * 5, 0)], self.scans(), semilla=7)
        b = asignar_metricas([chunk(["bin_1", "bin_2"] * 5, 0)], self.scans(), semilla=7)
        self.assertEqual(a, b)


class TestVecinosExtremos(unittest.TestCase):
    def test_vecino_mas_cercano_del_mismo_cluster(self):
        chunks = []
        for cluster, base in ((0, 0.0), (1, 10.0)):
            for d in (0.0, 0.1, 0.2, 0.3):
                v = (base + d, base - d)
                chunks.append({"cluster": cluster, "occupancy_vector": v, "vector_bins": v})
        r = vecinos_extremos(chunks, n_consultas=8, semilla=1)
        self.assertEqual(r["l2"]["nn_mismo_cluster"], 1.0)
        self.assertLess(r["l2"]["razon_min_max"], 0.1)


class TestSobreDatasetReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = ejecutar(n_pares=1000, n_pares_correlacion=200, n_consultas=10)

    def test_chunks_y_clusters(self):
        chunks = self.r["chunks"]
        self.assertEqual(len(chunks), 1000)
        conteo = {}
        for c in chunks:
            conteo[c["cluster"]] = conteo.get(c["cluster"], 0) + 1
        self.assertEqual(sorted(conteo.values()), [100] * 10)
        self.assertTrue(all(len(c["occupancy_vector"]) == 102 for c in chunks))

    def test_reproducible(self):
        a = construir_chunks_plantados(semilla=5)
        b = construir_chunks_plantados(semilla=5)
        self.assertEqual(a, b)

    def test_pares(self):
        self.assertEqual(len(self.r["filas"]), 1000)
        self.assertEqual(len({(f["i"], f["j"]) for f in self.r["filas"]}), 1000)
        self.assertEqual(self.r["pares_correlacion"]["n"], 200)

    def test_jaccard_y_lcs_separan_los_clusters(self):
        for clave in ("jaccard", "lcs"):
            self.assertGreater(self.r["resumen"][clave]["auc"], 0.99)
            self.assertGreater(self.r["resumen"][clave]["media_mismo"], self.r["resumen"][clave]["media_otro"])

    def test_sin_las_metricas_todas_las_medidas_separan(self):
        for clave in ("coseno_bins", "l1_bins", "l2_bins"):
            self.assertGreater(self.r["resumen"][clave]["auc"], 0.99)

    def test_las_metricas_z_score_dominan_el_coseno_y_l2(self):
        self.assertGreater(self.r["normas"]["metricas"], 3 * self.r["normas"]["bins"])
        self.assertLess(self.r["resumen"]["coseno"]["auc"], 0.7)
        self.assertLess(self.r["resumen"]["l2"]["auc"], 0.7)

    def test_valores_en_rango(self):
        for f in self.r["filas"]:
            self.assertTrue(0 <= f["jaccard"] <= 1)
            self.assertTrue(0 <= f["lcs"] <= 1)
            self.assertTrue(-1 - 1e-9 <= f["coseno"] <= 1 + 1e-9)
            self.assertTrue(f["l1"] >= 0 and f["l2"] >= 0)
            self.assertTrue(f["l2"] <= f["l1"] + 1e-9)  # L2 <= L1 siempre

    def test_correlacion_jaccard_lcs_alta_y_con_el_coseno_completo_baja(self):
        c = self.r["correlacion_al_azar"]
        self.assertGreater(c[("jaccard", "lcs")]["pearson"], 0.8)
        self.assertLess(abs(c[("jaccard", "coseno")]["pearson"]), 0.3)
        self.assertGreater(c[("jaccard", "coseno_bins")]["pearson"], 0.6)

    def test_concentracion_y_vecinos(self):
        for clave, c in self.r["concentracion_pares"].items():
            self.assertGreater(c["cv"], 0)
        for clave, v in self.r["vecinos"].items():
            self.assertTrue(0 < v["razon_min_max"] < 1)
            self.assertTrue(0 <= v["nn_mismo_cluster"] <= 1)

    def test_guardar_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "chunks.json")
            guardar_chunks(self.r["chunks"], ruta)
            with open(ruta, encoding="utf-8") as f:
                datos = json.load(f)
        self.assertEqual(len(datos), 1000)
        self.assertEqual(set(datos[0]), {"chunk_id", "robot_id", "cluster", "bin_sequence", "seq_length",
                                         "avg_quantity", "low_stock_frac", "shingle_count",
                                         "occupancy_vector"})
        self.assertEqual(len(datos[0]["occupancy_vector"]), 102)
        self.assertEqual(datos[0]["shingle_count"], self.r["chunks"][0]["shingle_count"])

    @unittest.skipUnless(HAY_MATPLOTLIB, "matplotlib no está instalado")
    def test_grafico(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = graficar_distribuciones(self.r, os.path.join(d, "g.png"))
            self.assertTrue(os.path.getsize(ruta) > 10_000)


if __name__ == "__main__":
    unittest.main()
