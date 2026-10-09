import json
import os
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit1.BigDataContext import (  # noqa: E402
    GB,
    SEGUNDOS_ANIO,
    SEGUNDOS_DIA,
    analisis,
    bytes_por_registro,
    cargar_dataset,
    proyectar,
    tasa_scans_por_segundo,
)

BASE = datetime(2026, 1, 1, 0, 0, 0)


def scans_sinteticos(n, paso_seg=1, robots=4):
    return [
        {
            "scan_id": f"scan_{i}",
            "timestamp": (BASE + timedelta(seconds=i * paso_seg)).isoformat(),
            "robot_id": f"robot_{i % robots}",
            "sku": "sku_1",
            "bin_location": "bin_1",
            "quantity": 10,
            "is_low_stock": False,
        }
        for i in range(n)
    ]


class TestTasaScansPorSegundo(unittest.TestCase):
    def test_un_scan_por_segundo(self):
        self.assertAlmostEqual(tasa_scans_por_segundo(scans_sinteticos(101, 1)), 1.0)

    def test_un_scan_cada_dos_segundos(self):
        self.assertAlmostEqual(tasa_scans_por_segundo(scans_sinteticos(101, 2)), 0.5)

    def test_no_depende_del_orden(self):
        datos = scans_sinteticos(50)
        self.assertAlmostEqual(
            tasa_scans_por_segundo(datos), tasa_scans_por_segundo(list(reversed(datos)))
        )

    def test_intervalo_nulo_lanza_error(self):
        with self.assertRaises(ValueError):
            tasa_scans_por_segundo(scans_sinteticos(5, paso_seg=0))


class TestBytesPorRegistro(unittest.TestCase):
    def test_coincide_con_json_compacto(self):
        registro = scans_sinteticos(1)[0]
        esperado = len(json.dumps(registro, separators=(",", ":")).encode("utf-8"))
        self.assertEqual(bytes_por_registro([registro]), esperado)

    def test_es_el_promedio(self):
        corto = {"sku": "a"}
        largo = {"sku": "a" * 11}
        bc = len(json.dumps(corto, separators=(",", ":")).encode("utf-8"))
        bl = len(json.dumps(largo, separators=(",", ":")).encode("utf-8"))
        self.assertAlmostEqual(bytes_por_registro([corto, largo]), (bc + bl) / 2)


class TestProyectar(unittest.TestCase):
    def test_conversion_a_dia_y_anio(self):
        p = proyectar(2.0, 100)
        self.assertEqual(p["scans_por_seg"], 2.0)
        self.assertEqual(p["scans_por_dia"], 2.0 * SEGUNDOS_DIA)
        self.assertEqual(p["scans_por_anio"], 2.0 * SEGUNDOS_ANIO)
        self.assertAlmostEqual(p["gb_por_dia"], 2.0 * SEGUNDOS_DIA * 100 / GB)
        self.assertAlmostEqual(p["gb_por_anio"], 2.0 * SEGUNDOS_ANIO * 100 / GB)

    def test_el_factor_escala_todo_linealmente(self):
        base = proyectar(1.5, 200, 1)
        grande = proyectar(1.5, 200, 10)
        self.assertEqual(grande["factor"], 10)
        for clave in ("scans_por_seg", "scans_por_dia", "scans_por_anio",
                      "gb_por_dia", "gb_por_anio"):
            self.assertAlmostEqual(grande[clave], 10 * base[clave])


class TestAnalisis(unittest.TestCase):
    def test_dataset_sintetico(self):
        r = analisis(scans_sinteticos(101, 1, robots=4), factor_crecimiento=10)
        self.assertEqual(r["n_scans"], 101)
        self.assertEqual(r["robots"], 4)
        self.assertEqual(r["robots_con_crecimiento"], 40)
        self.assertAlmostEqual(r["actual"]["scans_por_seg"], 1.0)
        self.assertAlmostEqual(r["scans_por_seg_por_robot"], 0.25)
        self.assertAlmostEqual(r["crecimiento"]["scans_por_seg"], 10.0)

    def test_dataset_real(self):
        dataset = cargar_dataset()
        r = analisis(dataset)
        self.assertEqual(r["n_scans"], len(dataset))
        self.assertGreaterEqual(len(dataset), 10_000)
        self.assertEqual(r["robots_con_crecimiento"], 10 * r["robots"])
        self.assertAlmostEqual(
            r["crecimiento"]["gb_por_anio"], 10 * r["actual"]["gb_por_anio"]
        )
        self.assertGreater(r["bytes_por_registro"], 0)


if __name__ == "__main__":
    unittest.main()
