import json
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit4.CostAnalysis import (  # noqa: E402
    agregacion_sku,
    ajuste_lineal,
    analisis_kmeans,
    bytes_registro,
    clasificar_crecimiento,
    combinar_registros,
    datasets_por_factor,
    ejecutar,
    extrapolar,
    kmeans_con_costo,
    partir,
    tamano_shuffle,
)
from unit4.MapReduceAlgorithm import agregar_por_sku, kmeans_una_corrida  # noqa: E402

RUTA_DATASET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "dataset.json")


def scan(sku, cantidad):
    return {"sku": sku, "quantity": cantidad, "is_low_stock": cantidad <= 5}


def scans_sinteticos():
    # fragmentos de 3: [1,1,2] y [1,1,2] -> sku_1 y sku_2 aparecen en los dos
    return [scan("sku_1", 10), scan("sku_1", 4), scan("sku_2", 8),
            scan("sku_1", 6), scan("sku_1", 2), scan("sku_2", 3)]


def vectores_tres_nubes(n_por_nube=20, semilla=1):
    rng = random.Random(semilla)
    centros = [(0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (-10.0, 5.0, 0.0)]
    return [tuple(c + rng.gauss(0, 1) for c in centro)
            for centro in centros for _ in range(n_por_nube)]


class TestUtilidades(unittest.TestCase):
    def test_partir_por_tamano(self):
        self.assertEqual(partir(list(range(10)), tam=4), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])

    def test_partir_en_n_trozos_casi_iguales(self):
        trozos = partir(list(range(23)), n=8)
        self.assertEqual(len(trozos), 8)
        self.assertEqual([x for t in trozos for x in t], list(range(23)))
        self.assertLessEqual(max(map(len, trozos)) - min(map(len, trozos)), 1)

    def test_partir_pide_exactamente_un_criterio(self):
        with self.assertRaises(ValueError):
            partir([1, 2], tam=1, n=2)
        with self.assertRaises(ValueError):
            partir([1, 2])
        with self.assertRaises(ValueError):
            partir([1, 2], tam=0)
        with self.assertRaises(ValueError):
            partir([1, 2], n=0)

    def test_bytes_registro_json_compacto(self):
        self.assertEqual(bytes_registro("sku_1", (14, 0, 1)), len('["sku_1",[14,0,1]]'))
        self.assertEqual(bytes_registro("sku_1", (14, 0, 1)), 18)

    def test_tamano_shuffle(self):
        t = tamano_shuffle([("a", (1, 0, 1)), ("bb", (22, 1, 1))])
        self.assertEqual(t["registros"], 2)
        self.assertEqual(t["bytes"], bytes_registro("a", (1, 0, 1)) + bytes_registro("bb", (22, 1, 1)))

    def test_combinar_suma_por_clave_y_conserva_orden(self):
        out = combinar_registros([("a", (1, 0, 1)), ("b", (5, 1, 1)), ("a", (3, 1, 1))])
        self.assertEqual(out, [("a", (4, 1, 2)), ("b", (5, 1, 1))])

    def test_combinar_vacio(self):
        self.assertEqual(combinar_registros([]), [])

    def test_combinar_valores_de_cuatro_coordenadas(self):
        out = combinar_registros([(0, (1.0, 2.0, 3.0, 1)), (0, (0.5, 0.5, 0.5, 1))])
        self.assertEqual(out, [(0, (1.5, 2.5, 3.5, 2))])


class TestAgregacionSku(unittest.TestCase):
    def test_sin_combiner_viaja_un_registro_por_scan(self):
        r = agregacion_sku(scans_sinteticos(), con_combiner=False, tam_fragmento=3)
        self.assertEqual(r["mappers"], 2)
        self.assertEqual(r["registros_map"], 6)
        self.assertEqual(r["registros"], 6)

    def test_con_combiner_viaja_un_registro_por_sku_y_map_task(self):
        r = agregacion_sku(scans_sinteticos(), con_combiner=True, tam_fragmento=3)
        self.assertEqual(r["registros_map"], 6)
        self.assertEqual(r["registros"], 4)  # 2 SKUs en cada uno de los 2 fragmentos

    def test_mismo_resultado_con_y_sin_combiner(self):
        scans = scans_sinteticos()
        sin = agregacion_sku(scans, False, tam_fragmento=3)["stats"]
        con = agregacion_sku(scans, True, tam_fragmento=3)["stats"]
        self.assertEqual(sin, con)
        self.assertEqual(sin, agregar_por_sku(scans))

    def test_resultado_a_mano(self):
        stats = agregacion_sku(scans_sinteticos(), True, tam_fragmento=3)["stats"]
        self.assertEqual(stats["sku_1"], {"avg_quantity": 5.5, "low_stock_frac": 0.5, "scan_count": 4})
        self.assertEqual(stats["sku_2"]["scan_count"], 2)

    def test_numero_fijo_de_fragmentos(self):
        r = agregacion_sku(scans_sinteticos(), True, n_fragmentos=1)
        self.assertEqual(r["mappers"], 1)
        self.assertEqual(r["registros"], 2)  # con un solo map task solo importan los SKUs distintos

    def test_el_combiner_nunca_manda_mas(self):
        scans = scans_sinteticos() * 5
        sin = agregacion_sku(scans, False, tam_fragmento=4)
        con = agregacion_sku(scans, True, tam_fragmento=4)
        self.assertLessEqual(con["registros"], sin["registros"])
        self.assertLessEqual(con["bytes"], sin["bytes"])


class TestKMeansConCosto(unittest.TestCase):
    def test_identico_a_kmeans_una_corrida_de_la_task_2(self):
        v = vectores_tres_nubes()
        r = kmeans_con_costo(v, k=3, semilla=7, n_splits=4)
        centroides, asignaciones, iteraciones = kmeans_una_corrida(v, 3, 20, 1e-3, 7)
        self.assertEqual(r["asignaciones"], asignaciones)
        self.assertEqual(r["iteraciones"], iteraciones)
        self.assertEqual(r["centroides"], centroides)

    def test_sin_combiner_un_registro_por_punto_y_iteracion(self):
        v = vectores_tres_nubes()
        r = kmeans_con_costo(v, k=3, semilla=7, n_splits=4)
        for it in r["detalle"]:
            self.assertEqual(it["sin_combiner"]["registros"], len(v))
        self.assertEqual(r["total_sin"]["registros"], len(v) * r["iteraciones"])

    def test_con_combiner_a_lo_sumo_k_registros_por_map_task(self):
        v = vectores_tres_nubes()
        r = kmeans_con_costo(v, k=3, semilla=7, n_splits=4)
        for it in r["detalle"]:
            self.assertLessEqual(it["con_combiner"]["registros"], 3 * 4)
            self.assertGreaterEqual(it["con_combiner"]["registros"], 3)
            self.assertLess(it["con_combiner"]["bytes"], it["sin_combiner"]["bytes"])

    def test_el_combiner_no_cambia_los_centroides(self):
        r = kmeans_con_costo(vectores_tres_nubes(), k=3, semilla=7, n_splits=4)
        self.assertLess(r["dif_combiner"], 1e-9)

    def test_max_iter(self):
        r = kmeans_con_costo(vectores_tres_nubes(), k=3, max_iter=1, semilla=7, n_splits=4)
        self.assertEqual(r["iteraciones"], 1)
        self.assertEqual(len(r["detalle"]), 1)

    def test_analisis_sobre_stats_de_la_task_2(self):
        stats = agregar_por_sku([scan(f"sku_{i}", (i * 7) % 40) for i in range(1, 60)] * 3)
        r = analisis_kmeans(stats, n_splits=4)
        self.assertTrue(r["igual_a_task2"])
        self.assertEqual(r["puntos"], 59)


class TestCrecimiento(unittest.TestCase):
    def test_ajuste_de_una_recta_exacta(self):
        a = ajuste_lineal([1, 2, 4], [10, 20, 40])
        self.assertAlmostEqual(a["pendiente"], 10)
        self.assertAlmostEqual(a["intercepto"], 0, places=9)
        self.assertAlmostEqual(a["r2"], 1.0)
        self.assertEqual(a["razones"], [1, 2, 4])
        self.assertAlmostEqual(a["desviacion_max"], 0)

    def test_ajuste_de_valores_constantes(self):
        a = ajuste_lineal([1, 2, 4], [5, 5, 5])
        self.assertAlmostEqual(a["pendiente"], 0)
        self.assertIsNone(a["r2"])

    def test_extrapolar(self):
        self.assertAlmostEqual(extrapolar(ajuste_lineal([1, 2, 4], [10, 20, 40]), 10), 100)
        self.assertAlmostEqual(extrapolar(ajuste_lineal([1, 2, 4], [5, 15, 35]), 10), 95)

    def test_clasificacion(self):
        f = [1, 2, 4]
        self.assertEqual(clasificar_crecimiento(f, [10, 20, 40]), "lineal")
        self.assertEqual(clasificar_crecimiento(f, [10, 20.4, 39.1]), "lineal")
        self.assertEqual(clasificar_crecimiento(f, [10, 10.5, 10.8]), "constante")
        self.assertEqual(clasificar_crecimiento(f, [10, 15, 25]), "sublineal")
        self.assertEqual(clasificar_crecimiento(f, [10, 30, 100]), "superlineal")

    def test_una_recta_con_intercepto_se_extrapola_bien(self):
        a = ajuste_lineal([1, 2, 4], [105, 205, 405])
        self.assertAlmostEqual(extrapolar(a, 10), 1005)


class TestSobreDatasetReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = ejecutar(factores=(1, 2), factor_validacion=4)

    def test_el_factor_1x_es_el_dataset_real(self):
        with open(RUTA_DATASET, encoding="utf-8") as f:
            real = json.load(f)
        self.assertEqual(datasets_por_factor((1, 2))[1], real)

    def test_cada_tamano_contiene_al_anterior(self):
        d = datasets_por_factor((1, 2, 4))
        self.assertEqual(d[2][:10000], d[1])
        self.assertEqual(d[4][:20000], d[2])

    def test_tamanos_de_los_datasets(self):
        self.assertEqual({f: d["scans"] for f, d in self.r["por_factor"].items()},
                         {1: 10000, 2: 20000, 4: 40000})

    def test_los_resultados_no_cambian_con_el_combiner_ni_con_el_k_means_instrumentado(self):
        for d in self.r["por_factor"].values():
            self.assertTrue(d["resultados_iguales"])
            self.assertTrue(d["kmeans"]["igual_a_task2"])
            self.assertLess(d["kmeans"]["dif_combiner"], 1e-9)

    def test_sin_combiner_el_shuffle_es_un_registro_por_scan(self):
        for d in self.r["por_factor"].values():
            self.assertEqual(d["agregacion"]["sin"]["registros"], d["scans"])
            self.assertEqual(d["agregacion"]["sin"]["registros_map"], d["scans"])

    def test_el_combiner_reduce_el_shuffle(self):
        for d in self.r["por_factor"].values():
            self.assertLess(d["agregacion"]["con"]["registros"], d["agregacion"]["sin"]["registros"])
            self.assertLess(d["agregacion"]["con"]["bytes"], d["agregacion"]["sin"]["bytes"])
            self.assertLess(d["kmeans"]["total_con"]["registros"], d["kmeans"]["total_sin"]["registros"])

    def test_numero_de_map_tasks(self):
        self.assertEqual({f: d["agregacion"]["sin"]["mappers"] for f, d in self.r["por_factor"].items()},
                         {1: 8, 2: 16, 4: 32})

    def test_el_shuffle_de_la_agregacion_crece_linealmente(self):
        lin = self.r["linealidad"]
        for nombre in ("Agregación por SKU, registros, sin combiner",
                       "Agregación por SKU, registros, con combiner"):
            self.assertEqual(lin[nombre]["crecimiento"], "lineal")
            self.assertLess(abs(lin[nombre]["error_relativo"]), 0.02)

    def test_el_shuffle_de_k_means_no_depende_de_los_scans(self):
        km = self.r["linealidad"]["K-Means, registros por iteración, con combiner"]
        self.assertEqual(km["crecimiento"], "constante")
        self.assertTrue(all(v <= 24 for v in km["valores"]))

    def test_k_means_sin_combiner_manda_un_registro_por_sku(self):
        for d in self.r["por_factor"].values():
            km = d["kmeans"]
            self.assertEqual(km["total_sin"]["registros"], km["puntos"] * km["iteraciones"])
            self.assertEqual(km["puntos"], d["skus"])

    def test_sin_validacion_no_hay_prediccion(self):
        r = ejecutar(factores=(1, 2), factor_validacion=None)
        for e in r["linealidad"].values():
            self.assertNotIn("prediccion", e)


if __name__ == "__main__":
    unittest.main()
