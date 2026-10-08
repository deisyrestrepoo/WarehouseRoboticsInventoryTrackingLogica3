import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit2.FrequencyMoments import (  # noqa: E402
    calcular_momentos,
    cargar_stream,
    numero_efectivo,
    top_bins,
    trafico_por_pasillo,
)


def scans_sinteticos(conteo_por_bin):
    """conteo_por_bin: {"bin_1": 3, "bin_2": 1, ...} -> lista de scans con esos conteos."""
    scans = []
    i = 0
    for bin_, cuenta in conteo_por_bin.items():
        for _ in range(cuenta):
            scans.append({"bin_location": bin_, "aisle": f"aisle_{bin_}", "stream_seq": i})
            i += 1
    return scans


class TestCalcularMomentos(unittest.TestCase):
    def test_un_solo_bin(self):
        scans = scans_sinteticos({"bin_1": 5})
        m = calcular_momentos(scans)
        self.assertEqual(m["F0"], 1)
        self.assertEqual(m["F1"], 5)
        self.assertEqual(m["F2"], 25)

    def test_conteos_parejos(self):
        # 4 bins, 3 scans cada uno: F0=4, F1=12, F2=4*9=36
        scans = scans_sinteticos({"bin_1": 3, "bin_2": 3, "bin_3": 3, "bin_4": 3})
        m = calcular_momentos(scans)
        self.assertEqual(m["F0"], 4)
        self.assertEqual(m["F1"], 12)
        self.assertEqual(m["F2"], 36)

    def test_conteos_desparejos_dan_f2_mayor(self):
        # mismo F1 (12) repartido parejo vs. concentrado: F2 debe ser mayor al concentrar
        parejo = calcular_momentos(scans_sinteticos({"bin_1": 3, "bin_2": 3, "bin_3": 3, "bin_4": 3}))
        concentrado = calcular_momentos(scans_sinteticos({"bin_1": 9, "bin_2": 1, "bin_3": 1, "bin_4": 1}))
        self.assertEqual(parejo["F1"], concentrado["F1"])
        self.assertGreater(concentrado["F2"], parejo["F2"])

    def test_no_depende_del_orden_del_stream(self):
        scans = scans_sinteticos({"bin_1": 5, "bin_2": 2})
        m1 = calcular_momentos(scans)
        m2 = calcular_momentos(list(reversed(scans)))
        self.assertEqual(m1["F0"], m2["F0"])
        self.assertEqual(m1["F1"], m2["F1"])
        self.assertEqual(m1["F2"], m2["F2"])

    def test_clave_distinta(self):
        scans = [{"sku": "a"}, {"sku": "a"}, {"sku": "b"}]
        m = calcular_momentos(scans, clave="sku")
        self.assertEqual(m["F0"], 2)
        self.assertEqual(m["F1"], 3)
        self.assertEqual(m["F2"], 4 + 1)


class TestNumeroEfectivo(unittest.TestCase):
    def test_parejo_da_f0(self):
        # F1=12, F2=36 (4 bins parejos) -> numero efectivo = 144/36 = 4 = F0
        self.assertAlmostEqual(numero_efectivo(4, 12, 36), 4.0)

    def test_concentrado_da_menos_que_f0(self):
        # F1=12, F2 de {9,1,1,1} = 81+1+1+1=84 -> 144/84 ≈ 1.71, bastante menos que F0=4
        n_efectivo = numero_efectivo(4, 12, 84)
        self.assertLess(n_efectivo, 4)
        self.assertGreater(n_efectivo, 1)

    def test_f2_cero_no_divide_por_cero(self):
        self.assertEqual(numero_efectivo(0, 0, 0), 0.0)


class TestTopBinsYPasillo(unittest.TestCase):
    def test_top_bins_ordenado_descendente(self):
        scans = scans_sinteticos({"bin_1": 1, "bin_2": 5, "bin_3": 3})
        m = calcular_momentos(scans)
        self.assertEqual(top_bins(m["conteos"], top_n=2), [("bin_2", 5), ("bin_3", 3)])

    def test_trafico_por_pasillo(self):
        scans = [
            {"aisle": "aisle_1"}, {"aisle": "aisle_1"}, {"aisle": "aisle_2"},
        ]
        c = trafico_por_pasillo(scans)
        self.assertEqual(c["aisle_1"], 2)
        self.assertEqual(c["aisle_2"], 1)


class TestDatasetReal(unittest.TestCase):
    def test_momentos_sobre_el_dataset_completo(self):
        scans = cargar_stream()
        m = calcular_momentos(scans, clave="bin_location")
        self.assertEqual(m["F1"], len(scans))
        self.assertGreaterEqual(m["F1"], 10_000)
        self.assertGreaterEqual(m["F0"], 8)  # la tarea pide >= 8 bins/estados
        # F2 nunca puede ser menor que el minimo parejo F1^2/F0 (desigualdad de Cauchy-Schwarz)
        self.assertGreaterEqual(m["F2"], m["F1"] ** 2 / m["F0"])

    def test_numero_efectivo_esta_entre_1_y_f0(self):
        scans = cargar_stream()
        m = calcular_momentos(scans, clave="bin_location")
        n_efectivo = numero_efectivo(m["F0"], m["F1"], m["F2"])
        self.assertGreaterEqual(n_efectivo, 1.0)
        self.assertLessEqual(n_efectivo, m["F0"])


if __name__ == "__main__":
    unittest.main()
