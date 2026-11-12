import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit5.CosineLSH import (  # noqa: E402
    angulo_grados,
    candidatos,
    cargar_tramos_y_vectores,
    construir_chunks,
    construir_indice_bandas,
    construir_vectores,
    experimento_recall,
    firma,
    fraccion_bits_coincidentes,
    generar_hiperplanos,
    similitud_coseno,
    top5_fuerza_bruta,
    verificar_formula,
)


def scans_sinteticos(n_robots=3, scans_por_robot=16):
    scans = []
    for r in range(n_robots):
        for i in range(scans_por_robot):
            scans.append({
                "robot_id": f"robot_{r}",
                "stream_seq": i,
                "bin_location": f"bin_{1 + (i + r) % 4}",
                "quantity": 10 + i,
                "is_low_stock": i % 5 == 0,
            })
    return scans


class TestConstruirChunks(unittest.TestCase):
    def test_longitud_fija_y_descarta_el_resto(self):
        scans = scans_sinteticos(n_robots=1, scans_por_robot=10)
        chunks = construir_chunks(scans, longitud=4)
        self.assertEqual(len(chunks), 2)  # 10 // 4 = 2, sobran 2 scans sueltos
        self.assertEqual(len(chunks[0]["bins"]), 4)

    def test_no_mezcla_robots_distintos(self):
        scans = scans_sinteticos(n_robots=2, scans_por_robot=8)
        chunks = construir_chunks(scans, longitud=4)
        for c in chunks:
            self.assertTrue(c["chunk_id"].startswith(c["robot_id"]))

    def test_promedios_del_chunk(self):
        scans = [
            {"robot_id": "r", "stream_seq": 0, "bin_location": "bin_1", "quantity": 0, "is_low_stock": True},
            {"robot_id": "r", "stream_seq": 1, "bin_location": "bin_1", "quantity": 10, "is_low_stock": False},
        ]
        chunks = construir_chunks(scans, longitud=2)
        self.assertAlmostEqual(chunks[0]["avg_quantity"], 5.0)
        self.assertAlmostEqual(chunks[0]["low_stock_frac"], 0.5)


class TestConstruirVectores(unittest.TestCase):
    def test_participacion_suma_uno(self):
        chunks = construir_chunks(scans_sinteticos(n_robots=2, scans_por_robot=8), longitud=4)
        bins_ordenados = sorted({b for c in chunks for b in c["bins"]})
        vectores = construir_vectores(chunks, bins_ordenados)
        for v in vectores.values():
            participacion = v[: len(bins_ordenados)]
            self.assertAlmostEqual(sum(participacion), 1.0)

    def test_extra_queda_estandarizado_con_media_cero(self):
        chunks = construir_chunks(scans_sinteticos(n_robots=3, scans_por_robot=16), longitud=4)
        bins_ordenados = sorted({b for c in chunks for b in c["bins"]})
        vectores = construir_vectores(chunks, bins_ordenados)
        extras = [v[len(bins_ordenados):] for v in vectores.values()]
        media_avg_q = sum(e[0] for e in extras) / len(extras)
        self.assertAlmostEqual(media_avg_q, 0.0, places=6)


class TestSimilitudYAngulo(unittest.TestCase):
    def test_vectores_identicos_similitud_uno_angulo_cero(self):
        v = (1.0, 2.0, 3.0)
        self.assertAlmostEqual(similitud_coseno(v, v), 1.0)
        self.assertAlmostEqual(angulo_grados(v, v), 0.0, places=5)

    def test_vectores_ortogonales_similitud_cero_angulo_90(self):
        self.assertAlmostEqual(similitud_coseno((1.0, 0.0), (0.0, 1.0)), 0.0)
        self.assertAlmostEqual(angulo_grados((1.0, 0.0), (0.0, 1.0)), 90.0, places=5)

    def test_vectores_opuestos_angulo_180(self):
        self.assertAlmostEqual(angulo_grados((1.0, 0.0), (-1.0, 0.0)), 180.0, places=5)

    def test_vector_cero_no_lanza_error(self):
        self.assertEqual(similitud_coseno((0.0, 0.0), (1.0, 1.0)), 0.0)


class TestFirmaHiperplanos(unittest.TestCase):
    def test_firma_es_determinista_para_los_mismos_hiperplanos(self):
        hp = generar_hiperplanos(dim=5, n=20, semilla=1)
        v = (0.1, -0.2, 0.3, 0.4, -0.5)
        self.assertEqual(firma(v, hp), firma(v, hp))

    def test_fraccion_coincidentes_vectores_identicos_es_uno(self):
        hp = generar_hiperplanos(dim=4, n=50, semilla=2)
        v = (1.0, 2.0, -1.0, 0.5)
        self.assertEqual(fraccion_bits_coincidentes(firma(v, hp), firma(v, hp)), 1.0)

    def test_probabilidad_de_bit_distinto_se_acerca_a_angulo_sobre_180(self):
        # dos vectores ortogonales (angulo=90) -> con muchos hiperplanos, ~50% de bits distintos
        rng = random.Random(3)
        a = (1.0, 0.0)
        b = (0.0, 1.0)
        hp = generar_hiperplanos(dim=2, n=2000, semilla=4)
        frac = fraccion_bits_coincidentes(firma(a, hp), firma(b, hp))
        self.assertAlmostEqual(frac, 0.5, delta=0.05)


class TestVerificarFormula(unittest.TestCase):
    def test_error_baja_al_subir_n(self):
        chunks, vectores = cargar_tramos_y_vectores(scans_sinteticos(n_robots=4, scans_por_robot=40), longitud=4)
        resultados = verificar_formula(vectores, valores_n=(8, 128), n_pares=200, semilla=7)
        self.assertLess(resultados[128]["error_promedio"], resultados[8]["error_promedio"])


class TestBandingYRecall(unittest.TestCase):
    def test_candidatos_incluye_vectores_identicos(self):
        hp = generar_hiperplanos(dim=3, n=16, semilla=5)
        firmas = {
            "a": firma((1.0, 0.0, 0.0), hp),
            "b": firma((1.0, 0.0, 0.0), hp),  # identico a "a"
            "c": firma((-1.0, 0.0, 0.0), hp),  # opuesto
        }
        indice = construir_indice_bandas(firmas, bandas=4, filas=4)
        self.assertIn("b", candidatos("a", firmas, indice, bandas=4, filas=4))

    def test_top5_fuerza_bruta_devuelve_los_mas_cercanos(self):
        vectores = {
            "q": (1.0, 0.0),
            "cerca": (0.99, 0.01),
            "lejos1": (-1.0, 0.0),
            "lejos2": (0.0, 1.0),
        }
        top = top5_fuerza_bruta("q", vectores)
        self.assertEqual(top[0], "cerca")

    def test_recall_con_dataset_sintetico_mas_grande(self):
        chunks, vectores = cargar_tramos_y_vectores(
            scans_sinteticos(n_robots=6, scans_por_robot=80), longitud=4
        )
        self.assertGreaterEqual(len(chunks), 100)
        r = experimento_recall(vectores, n_consultas=10, semilla=1)
        self.assertGreaterEqual(r["recall_at_5_promedio"], 0.0)
        self.assertLessEqual(r["recall_at_5_promedio"], 1.0)
        self.assertGreaterEqual(r["comparaciones_ahorradas_promedio"], 0.0)


class TestDatasetReal(unittest.TestCase):
    def test_carga_sobre_el_dataset_completo(self):
        chunks, vectores = cargar_tramos_y_vectores()
        self.assertGreaterEqual(len(chunks), 1000)  # la tarea pide >= 1.000 tramos
        self.assertEqual(len(chunks), len(vectores))

    def test_recall_razonable_en_el_dataset_real(self):
        _, vectores = cargar_tramos_y_vectores()
        r = experimento_recall(vectores, n_consultas=15, semilla=9)
        self.assertGreater(r["recall_at_5_promedio"], 0.5)
        self.assertGreater(r["comparaciones_ahorradas_promedio"], 0.0)


if __name__ == "__main__":
    unittest.main()
