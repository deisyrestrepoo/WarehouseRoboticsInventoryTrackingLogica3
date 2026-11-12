import os
import random
import statistics
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit2.DistinctCount import (  # noqa: E402
    BITS,
    NUM_HASHES,
    PHI,
    ContadorFlajoletMartin,
    ConsultaSKUsDistintosFM,
    FuncionHashFM,
    cargar_stream,
    ceros_finales,
    evolucion,
    experimento,
    primer_cero,
)


class TestBits(unittest.TestCase):
    def test_ceros_finales(self):
        self.assertEqual(ceros_finales(1), 0)
        self.assertEqual(ceros_finales(0b1000), 3)
        self.assertEqual(ceros_finales(0b101000), 3)
        self.assertEqual(ceros_finales(12), 2)
        self.assertEqual(ceros_finales(0), BITS)

    def test_primer_cero(self):
        self.assertEqual(primer_cero(0), 0)
        self.assertEqual(primer_cero(0b1), 1)
        self.assertEqual(primer_cero(0b111), 3)
        self.assertEqual(primer_cero(0b1011), 2)
        self.assertEqual(primer_cero(0b110), 0)

    def test_ceros_finales_se_distribuyen_de_forma_geometrica(self):
        h = FuncionHashFM(random.Random(1))
        valores = [ceros_finales(h(x)) for x in range(1, 40001)]
        for r in range(4):
            esperado = 40000 * 2 ** -(r + 1)
            self.assertLess(abs(valores.count(r) - esperado), 0.05 * 40000)


class TestContador(unittest.TestCase):
    def test_vacio_vale_cero(self):
        c = ContadorFlajoletMartin()
        self.assertEqual(c.estimar(), 0.0)
        self.assertEqual(c.estimar_mediana(), 0.0)

    def test_repetidos_no_cambian_el_estado(self):
        c = ContadorFlajoletMartin(10, 7)
        for sku in ("a", "b", "c"):
            c.agregar(sku)
        antes = list(c.bitmaps)
        for _ in range(100):
            for sku in ("a", "b", "c"):
                c.agregar(sku)
        self.assertEqual(c.bitmaps, antes)

    def test_es_deterministico_con_la_misma_semilla(self):
        a, b = ContadorFlajoletMartin(10, 5), ContadorFlajoletMartin(10, 5)
        for i in range(500):
            a.agregar(f"sku_{i}")
            b.agregar(f"sku_{i}")
        self.assertEqual(a.bitmaps, b.bitmaps)
        self.assertEqual(a.estimar(), b.estimar())

    def test_la_estimacion_es_2_a_la_r_promedio_sobre_phi(self):
        c = ContadorFlajoletMartin(10, 3)
        for i in range(300):
            c.agregar(f"sku_{i}")
        r = c.posiciones()
        self.assertAlmostEqual(c.estimar(), 2 ** (sum(r) / 10) / PHI)

    def test_no_decrece_al_agregar_elementos(self):
        c = ContadorFlajoletMartin(10, 11)
        anterior = 0.0
        for i in range(2000):
            c.agregar(f"sku_{i}")
            actual = c.estimar()
            self.assertGreaterEqual(actual, anterior)
            anterior = actual

    def test_memoria_constante(self):
        c = ContadorFlajoletMartin(10)
        self.assertEqual(c.bytes_de_estado(), 80)
        self.assertEqual(len(c.bitmaps), 10)

    def test_num_hashes_invalido(self):
        with self.assertRaises(ValueError):
            ContadorFlajoletMartin(0)

    def test_precision_con_muchas_funciones_de_hash(self):
        n = 2000
        errores = []
        for semilla in range(10):
            c = ContadorFlajoletMartin(50, semilla)
            for i in range(n):
                c.agregar(f"sku_{i}")
            errores.append((c.estimar() - n) / n)
        self.assertLess(abs(statistics.mean(errores)), 0.15)
        self.assertLess(statistics.pstdev(errores), 0.25)


class TestConsultaPorDia(unittest.TestCase):
    def test_reinicia_al_cambiar_el_dia(self):
        q = ConsultaSKUsDistintosFM(10, 2)
        for i in range(300):
            q.procesar({"timestamp": "2026-01-01T10:00:00", "sku": f"sku_{i}"})
        dia1 = q.estimar()
        q.procesar({"timestamp": "2026-01-02T00:00:01", "sku": "sku_0"})
        self.assertGreater(dia1, 100)
        self.assertLess(q.estimar(), 5)  # un solo SKU en el día nuevo
        self.assertEqual(q.dia_actual, "2026-01-02")


class TestConDatasetReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scans = cargar_stream()

    def test_evolucion_termina_con_el_conteo_exacto(self):
        puntos = evolucion(self.scans, cada=2500)
        self.assertEqual(puntos[-1]["scans_procesados"], len(self.scans))
        self.assertEqual(puntos[-1]["real"], len({s["sku"] for s in self.scans}))
        reales = [p["real"] for p in puntos]
        self.assertEqual(reales, sorted(reales))
        self.assertTrue(all(p["estimado"] > 0 for p in puntos))

    def test_con_10_funciones_el_error_tipico_es_del_orden_de_0_78_sobre_raiz_m(self):
        r = experimento(self.scans, hashes=(10, 50), repeticiones=40)
        f10, f50 = r["filas"]
        self.assertEqual(f10["m"], NUM_HASHES)
        self.assertLess(f10["desviacion"], 1.6 * f10["desviacion_teorica"])
        self.assertLess(f50["desviacion"], f10["desviacion"])
        self.assertLess(abs(f10["sesgo"]), 0.3)


if __name__ == "__main__":
    unittest.main()
