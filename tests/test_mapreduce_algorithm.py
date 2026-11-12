import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit4.MapReduceAlgorithm import (  # noqa: E402
    agregar_por_sku,
    calcular_estandarizacion,
    caracterizar_clusters,
    ejecutar,
    etiquetar_clusters,
    kmeans,
    mapper_asignar,
    reducer_promediar,
    vector_estandarizado,
)


def scans_sinteticos():
    return [
        {"sku": "a", "quantity": 10, "is_low_stock": False},
        {"sku": "a", "quantity": 20, "is_low_stock": False},
        {"sku": "b", "quantity": 0, "is_low_stock": True},
        {"sku": "b", "quantity": 2, "is_low_stock": True},
        {"sku": "b", "quantity": 0, "is_low_stock": True},
    ]


class TestJob1AgregarPorSku(unittest.TestCase):
    def test_promedios_y_conteo(self):
        stats = agregar_por_sku(scans_sinteticos())
        self.assertEqual(set(stats), {"a", "b"})
        self.assertAlmostEqual(stats["a"]["avg_quantity"], 15.0)
        self.assertAlmostEqual(stats["a"]["low_stock_frac"], 0.0)
        self.assertEqual(stats["a"]["scan_count"], 2)
        self.assertAlmostEqual(stats["b"]["avg_quantity"], 2 / 3)
        self.assertAlmostEqual(stats["b"]["low_stock_frac"], 1.0)
        self.assertEqual(stats["b"]["scan_count"], 3)

    def test_no_depende_del_orden_del_stream(self):
        scans = scans_sinteticos()
        s1 = agregar_por_sku(scans)
        s2 = agregar_por_sku(list(reversed(scans)))
        self.assertEqual(s1, s2)


class TestEstandarizacion(unittest.TestCase):
    def test_vector_estandarizado_tiene_media_cero(self):
        stats = {
            "a": {"avg_quantity": 10, "low_stock_frac": 0.0, "scan_count": 2},
            "b": {"avg_quantity": 20, "low_stock_frac": 1.0, "scan_count": 4},
        }
        parametros = calcular_estandarizacion(stats)
        vectores = [vector_estandarizado(stats[s], parametros) for s in ("a", "b")]
        # con 2 puntos, el estandarizado queda en +1/-1 por coordenada
        for dim in range(3):
            self.assertAlmostEqual(vectores[0][dim] + vectores[1][dim], 0.0)

    def test_desviacion_cero_no_divide_por_cero(self):
        stats = {"a": {"avg_quantity": 5, "low_stock_frac": 0.0, "scan_count": 1},
                  "b": {"avg_quantity": 5, "low_stock_frac": 0.0, "scan_count": 1}}
        parametros = calcular_estandarizacion(stats)
        # no debe lanzar ZeroDivisionError
        vector_estandarizado(stats["a"], parametros)


class TestKMeans(unittest.TestCase):
    def test_mapper_asigna_al_mas_cercano(self):
        centroides = [(0.0, 0.0), (10.0, 10.0)]
        self.assertEqual(mapper_asignar((1.0, 1.0), centroides), 0)
        self.assertEqual(mapper_asignar((9.0, 9.0), centroides), 1)

    def test_reducer_promedia_por_coordenada(self):
        self.assertEqual(reducer_promediar([(0.0, 2.0), (2.0, 4.0)]), (1.0, 3.0))

    def test_converge_a_clusters_bien_separados(self):
        # tres nubes de puntos muy separadas: K-Means con k=3 las debe encontrar exacto
        rng_puntos = []
        centros_reales = [(-10.0, -10.0), (10.0, 10.0), (10.0, -10.0)]
        import random
        rng = random.Random(0)
        for cx, cy in centros_reales:
            for _ in range(20):
                rng_puntos.append((cx + rng.uniform(-0.5, 0.5), cy + rng.uniform(-0.5, 0.5)))

        centroides, asignaciones, iteraciones, wcss = kmeans(rng_puntos, k=3, semilla=1, reinicios=5)
        # cada grupo de 20 puntos consecutivos debe quedar junto en un solo cluster
        for i in range(3):
            etiquetas_grupo = set(asignaciones[i * 20:(i + 1) * 20])
            self.assertEqual(len(etiquetas_grupo), 1)
        self.assertLess(wcss, 50.0)  # deberia ser chico: los puntos casi no se mueven de su centro

    def test_mas_reinicios_nunca_empeora_la_inercia(self):
        import random
        rng = random.Random(2)
        puntos = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(60)]
        _, _, _, wcss_1 = kmeans(puntos, k=3, semilla=5, reinicios=1)
        _, _, _, wcss_10 = kmeans(puntos, k=3, semilla=5, reinicios=10)
        self.assertLessEqual(wcss_10, wcss_1 + 1e-9)


class TestCaracterizarYEtiquetar(unittest.TestCase):
    def test_etiqueta_critico_es_el_de_mayor_low_stock_frac(self):
        resumen = {
            0: {"n_skus": 10, "avg_quantity": 30, "low_stock_frac": 0.01, "scan_count": 5},
            1: {"n_skus": 5, "avg_quantity": 3, "low_stock_frac": 0.9, "scan_count": 6},
            2: {"n_skus": 2, "avg_quantity": 20, "low_stock_frac": 0.1, "scan_count": 80},
        }
        etiquetas = etiquetar_clusters(resumen)
        self.assertEqual(etiquetas[1], "critico")
        self.assertEqual(etiquetas[2], "rotacion rapida")
        self.assertEqual(etiquetas[0], "stock estable")

    def test_caracterizar_cuenta_bien_los_skus(self):
        stats = {"a": {"avg_quantity": 1, "low_stock_frac": 0, "scan_count": 1},
                  "b": {"avg_quantity": 3, "low_stock_frac": 0, "scan_count": 1}}
        resumen = caracterizar_clusters(["a", "b"], stats, [0, 0])
        self.assertEqual(resumen[0]["n_skus"], 2)
        self.assertAlmostEqual(resumen[0]["avg_quantity"], 2.0)


class TestDatasetReal(unittest.TestCase):
    def test_ejecutar_sobre_el_dataset_completo(self):
        r = ejecutar(reinicios=3)
        self.assertEqual(len(r["skus"]), len(r["stats_por_sku"]))
        self.assertEqual(sum(v["n_skus"] for v in r["resumen"].values()), len(r["skus"]))
        self.assertEqual(set(r["etiquetas"].values()), {"critico", "rotacion rapida", "stock estable"})
        # el cluster critico debe tener, en promedio, mas low_stock_frac que el estable
        criticos = [c for c, e in r["etiquetas"].items() if e == "critico"][0]
        estables = [c for c, e in r["etiquetas"].items() if e == "stock estable"][0]
        self.assertGreater(
            r["resumen"][criticos]["low_stock_frac"], r["resumen"][estables]["low_stock_frac"]
        )

    def test_inercia_no_negativa(self):
        r = ejecutar(reinicios=2)
        self.assertGreaterEqual(r["wcss"], 0.0)


if __name__ == "__main__":
    unittest.main()
