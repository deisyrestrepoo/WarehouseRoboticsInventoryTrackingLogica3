import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit2.StreamSampling import (  # noqa: E402
    HashSKU,
    MuestraReservorioScans,
    MuestraSKUsPorHash,
    cargar_stream,
    error_estandar,
    evolucion,
    experimento,
    fraccion_real,
)


def scan(seq, sku, bajo):
    return {"stream_seq": seq, "sku": sku, "is_low_stock": bajo}


def stream_sintetico(n_skus=200, n_scans=2000, semilla=5):
    rng = random.Random(semilla)
    return [scan(i, f"sku_{rng.randrange(n_skus)}", rng.random() < 0.2) for i in range(n_scans)]


class TestFraccionReal(unittest.TestCase):
    def test_usa_el_ultimo_scan_de_cada_sku(self):
        scans = [scan(0, "a", True), scan(1, "b", False), scan(2, "a", False), scan(3, "c", True)]
        p, n = fraccion_real(scans)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(p, 1 / 3)  # solo c está en bajo stock

    def test_vacio(self):
        self.assertEqual(fraccion_real([]), (0.0, 0))


class TestReservorio(unittest.TestCase):
    def test_nunca_pasa_de_k_y_guarda_todo_si_cabe(self):
        scans = stream_sintetico(n_scans=500)
        m = MuestraReservorioScans(50)
        for s in scans:
            m.procesar(s)
        self.assertEqual(len(m.reservorio), 50)
        self.assertEqual(m.vistos, 500)
        chico = MuestraReservorioScans(1000)
        for s in scans:
            chico.procesar(s)
        self.assertEqual(len(chico.reservorio), 500)

    def test_es_reproducible_con_la_misma_semilla(self):
        scans = stream_sintetico()
        a, b = MuestraReservorioScans(30, 9), MuestraReservorioScans(30, 9)
        for s in scans:
            a.procesar(s)
            b.procesar(s)
        self.assertEqual(a.reservorio, b.reservorio)

    def test_cada_scan_entra_con_probabilidad_k_sobre_n(self):
        n, k, reps = 40, 10, 4000
        veces = [0] * n
        for r in range(reps):
            m = MuestraReservorioScans(k, semilla=r)
            for i in range(n):
                m.procesar(scan(i, f"s{i}", False))
            for s in m.reservorio:
                veces[s["stream_seq"]] += 1
        esperado = reps * k / n
        sigma = math.sqrt(reps * (k / n) * (1 - k / n))
        self.assertTrue(all(abs(v - esperado) < 5 * sigma for v in veces))

    def test_estimacion_con_el_ultimo_scan_muestreado_de_cada_sku(self):
        m = MuestraReservorioScans(10)
        for s in (scan(0, "a", True), scan(1, "a", False), scan(2, "b", True)):
            m.procesar(s)
        self.assertEqual(m.skus_en_muestra(), 2)
        self.assertAlmostEqual(m.estimar(), 0.5)  # a quedó en False, b en True

    def test_k_invalido(self):
        with self.assertRaises(ValueError):
            MuestraReservorioScans(0)


class TestMuestraSKUsPorHash(unittest.TestCase):
    def test_conserva_los_k_skus_de_menor_hash(self):
        scans = stream_sintetico()
        m = MuestraSKUsPorHash(25, semilla=3)
        for s in scans:
            m.procesar(s)
        h = HashSKU(random.Random(3))
        todos = sorted({s["sku"] for s in scans}, key=lambda sku: (h(sku), sku))
        self.assertEqual(set(m.estado), set(todos[:25]))
        self.assertEqual(m.tamano(), 25)

    def test_el_estado_de_cada_sku_muestreado_es_el_del_ultimo_scan(self):
        scans = stream_sintetico()
        m = MuestraSKUsPorHash(40, semilla=4)
        ultimo = {}
        for s in scans:
            m.procesar(s)
            ultimo[s["sku"]] = s["is_low_stock"]
        for sku, bajo in m.estado.items():
            self.assertEqual(bajo, ultimo[sku])

    def test_es_exacto_si_k_cubre_todos_los_skus(self):
        scans = stream_sintetico(n_skus=50)
        m = MuestraSKUsPorHash(1000)
        for s in scans:
            m.procesar(s)
        self.assertAlmostEqual(m.estimar(), fraccion_real(scans)[0])

    def test_no_depende_del_orden_de_llegada_de_los_skus(self):
        scans = stream_sintetico()
        a, b = MuestraSKUsPorHash(30, 8), MuestraSKUsPorHash(30, 8)
        for s in scans:
            a.procesar(s)
        for s in reversed(scans):
            b.procesar(s)
        self.assertEqual(set(a.estado), set(b.estado))

    def test_memoria_acotada(self):
        m = MuestraSKUsPorHash(20)
        for s in stream_sintetico(n_skus=500, n_scans=3000):
            m.procesar(s)
            self.assertLessEqual(len(m.estado), 20)
            self.assertEqual(len(m.heap), len(m.estado))

    def test_k_invalido(self):
        with self.assertRaises(ValueError):
            MuestraSKUsPorHash(0)


class TestErrorEstandar(unittest.TestCase):
    def test_formula_con_poblacion_finita(self):
        self.assertAlmostEqual(
            error_estandar(0.2, 100, 1000), math.sqrt(0.2 * 0.8 / 100 * 900 / 999)
        )

    def test_cero_si_se_muestrea_toda_la_poblacion(self):
        self.assertEqual(error_estandar(0.3, 500, 500), 0.0)
        self.assertEqual(error_estandar(0.3, 900, 500), 0.0)

    def test_baja_al_crecer_k(self):
        self.assertGreater(error_estandar(0.1, 100, 2000), error_estandar(0.1, 400, 2000))


class TestConDatasetReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scans = cargar_stream()

    def test_evolucion_termina_con_la_fraccion_real(self):
        puntos = evolucion(self.scans, 300, cada=2500)
        self.assertEqual(puntos[-1]["scans_procesados"], len(self.scans))
        self.assertAlmostEqual(puntos[-1]["real"], fraccion_real(self.scans)[0])
        self.assertTrue(all(0 <= p["real"] <= 1 for p in puntos))

    def test_experimento_el_error_baja_con_k_y_el_ic_cubre(self):
        r = experimento(self.scans, tamanos=(100, 900), repeticiones=40)
        chico, grande = r["filas"]
        self.assertLess(grande["muestreo_skus"]["desviacion"], chico["muestreo_skus"]["desviacion"])
        for f in r["filas"]:
            self.assertGreaterEqual(f["cobertura_ic95"], 0.8)
            self.assertLess(abs(f["muestreo_skus"]["sesgo"]), 3 * f["error_estandar_teorico"])


if __name__ == "__main__":
    unittest.main()
