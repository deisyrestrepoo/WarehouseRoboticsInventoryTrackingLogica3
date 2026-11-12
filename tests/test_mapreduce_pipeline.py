import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit3.DTMCConstruction import cargar_scans, construir_aristas  # noqa: E402
from unit4.MapReducePipeline import (  # noqa: E402
    combinar_parciales,
    ejecutar_pipeline,
    escribir_fragmentos,
    guardar_nodos,
    hash_clave,
    job2,
    mapper_job1,
    mapper_job2,
    reduce_particion,
    reducer_job1,
    registros_de_aristas,
    shuffle_job1,
    validar_contra_unit3,
)


def leer_json(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def scan(robot, seq, bin_, cantidad):
    return {
        "scan_id": f"s{seq}",
        "robot_id": robot,
        "stream_seq": seq,
        "bin_location": bin_,
        "quantity": cantidad,
        "is_low_stock": cantidad <= 5,
    }


def scans_sinteticos():
    """
    r1: bin_1 -> bin_2 (llega con 4, bajo stock) -> bin_1 (llega con 6)
    r2: bin_2 -> bin_2 (llega con 2, bajo stock) -> bin_3 (llega con 20)
    Los dos robots se intercalan en el stream, como en el dataset real.
    """
    return [
        scan("r1", 0, "bin_1", 10),
        scan("r2", 1, "bin_2", 8),
        scan("r1", 2, "bin_2", 4),
        scan("r2", 3, "bin_2", 2),
        scan("r1", 4, "bin_1", 6),
        scan("r2", 5, "bin_3", 20),
    ]


class TestFragmentos(unittest.TestCase):
    def test_reparte_todos_los_scans_en_orden(self):
        scans = [scan("r1", i, "bin_1", 10) for i in range(23)]
        with tempfile.TemporaryDirectory() as d:
            rutas = escribir_fragmentos(scans, d, 8)
            partes = [leer_json(r) for r in rutas]
        self.assertEqual(len(rutas), 8)
        self.assertEqual([s for p in partes for s in p], scans)

    def test_tamanos_casi_iguales(self):
        scans = [scan("r1", i, "bin_1", 10) for i in range(23)]
        with tempfile.TemporaryDirectory() as d:
            rutas = escribir_fragmentos(scans, d, 8)
            tamanos = [len(leer_json(r)) for r in rutas]
        self.assertLessEqual(max(tamanos) - min(tamanos), 1)

    def test_mas_fragmentos_que_scans(self):
        scans = [scan("r1", i, "bin_1", 10) for i in range(3)]
        with tempfile.TemporaryDirectory() as d:
            rutas = escribir_fragmentos(scans, d, 5)
            tamanos = [len(leer_json(r)) for r in rutas]
        self.assertEqual(sorted(tamanos), [0, 0, 1, 1, 1])

    def test_cero_fragmentos_falla(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                escribir_fragmentos([], d, 0)


class TestJob1(unittest.TestCase):
    def test_mapper_emite_clave_robot_y_valor_completo(self):
        clave, valor = mapper_job1(scan("r7", 3, "bin_9", 4))
        self.assertEqual(clave, "r7")
        self.assertEqual(valor, (3, "bin_9", 4, 1))

    def test_hash_es_determinista_y_esta_en_rango(self):
        for r in (1, 2, 4, 8):
            self.assertEqual(hash_clave("robot_3", r), hash_clave("robot_3", r))
            self.assertTrue(0 <= hash_clave("robot_3", r) < r)

    def test_shuffle_manda_cada_robot_a_un_solo_reducer(self):
        salidas = [[mapper_job1(s)] for s in scans_sinteticos()]
        particiones = shuffle_job1(salidas, 3)
        for robot in ("r1", "r2"):
            donde = [i for i, p in enumerate(particiones) if robot in p]
            self.assertEqual(len(donde), 1)
        self.assertEqual(sum(len(v) for p in particiones for v in p.values()), 6)

    def test_reducer_ordena_por_stream_seq(self):
        # valores desordenados, como llegan de fragmentos distintos
        valores = [(4, "bin_1", 6, 0), (0, "bin_1", 10, 0), (2, "bin_2", 4, 1)]
        self.assertEqual(
            reducer_job1(valores),
            [(("bin_1", "bin_2"), (1, 4, 1)), (("bin_2", "bin_1"), (1, 6, 0))],
        )

    def test_robot_con_un_solo_scan_no_emite_aristas(self):
        self.assertEqual(reducer_job1([(0, "bin_1", 10, 0)]), [])

    def test_self_loop(self):
        valores = [(0, "bin_5", 10, 0), (1, "bin_5", 3, 1)]
        self.assertEqual(reducer_job1(valores), [(("bin_5", "bin_5"), (1, 3, 1))])

    def test_combinar_parciales_suma_por_arista(self):
        a = {("bin_1", "bin_2"): [1, 4, 1]}
        b = {("bin_1", "bin_2"): [2, 10, 0], ("bin_2", "bin_1"): [1, 6, 0]}
        self.assertEqual(
            combinar_parciales([a, b]),
            {("bin_1", "bin_2"): [3, 14, 1], ("bin_2", "bin_1"): [1, 6, 0]},
        )

    def test_reduce_particion_suma_robots_de_la_misma_particion(self):
        particion = {
            "r1": [(0, "bin_1", 10, 0), (1, "bin_2", 4, 1)],
            "r2": [(0, "bin_1", 9, 0), (1, "bin_2", 6, 0)],
        }
        self.assertEqual(reduce_particion(particion), {("bin_1", "bin_2"): [2, 10, 1]})


class TestJob2(unittest.TestCase):
    def test_map_emite_dos_tuplas_por_arista(self):
        salida = mapper_job2(("bin_1", "bin_2"), [3, 14, 1])
        self.assertEqual(len(salida), 2)
        self.assertEqual(salida[0], ("bin_1", ("out", 3, 0, 0)))
        self.assertEqual(salida[1], ("bin_2", ("in", 3, 14, 1)))

    def test_nodos_a_mano(self):
        acumulado = {
            ("bin_1", "bin_2"): [1, 4, 1],
            ("bin_2", "bin_1"): [1, 6, 0],
            ("bin_2", "bin_2"): [1, 2, 1],
            ("bin_2", "bin_3"): [1, 20, 0],
        }
        nodos, est = job2(acumulado)
        por_bin = {n["bin_location"]: n for n in nodos}
        self.assertEqual(est["tuplas_map"], 8)
        self.assertEqual(est["nodos"], 3)
        self.assertEqual(por_bin["bin_1"], {
            "bin_location": "bin_1", "out_move_count": 1, "in_move_count": 1,
            "net_flow": 0, "avg_quantity": 6.0, "low_stock_frac": 0.0,
        })
        self.assertEqual(por_bin["bin_2"], {
            "bin_location": "bin_2", "out_move_count": 3, "in_move_count": 2,
            "net_flow": -1, "avg_quantity": 3.0, "low_stock_frac": 1.0,
        })
        self.assertEqual(por_bin["bin_3"], {
            "bin_location": "bin_3", "out_move_count": 0, "in_move_count": 1,
            "net_flow": 1, "avg_quantity": 20.0, "low_stock_frac": 0.0,
        })

    def test_nodos_ordenados_numericamente(self):
        acumulado = {("bin_10", "bin_2"): [1, 5, 1], ("bin_2", "bin_10"): [1, 7, 0]}
        nodos, _ = job2(acumulado)
        self.assertEqual([n["bin_location"] for n in nodos], ["bin_2", "bin_10"])

    def test_bin_sin_entradas_tiene_promedios_en_cero(self):
        nodos, _ = job2({("bin_1", "bin_2"): [1, 4, 1]})
        origen = [n for n in nodos if n["bin_location"] == "bin_1"][0]
        self.assertEqual(origen["in_move_count"], 0)
        self.assertEqual(origen["avg_quantity"], 0.0)
        self.assertEqual(origen["low_stock_frac"], 0.0)


class TestPipelineSintetico(unittest.TestCase):
    def test_aristas_a_mano(self):
        r = ejecutar_pipeline(scans_sinteticos(), n_fragmentos=3, workers=1)
        self.assertEqual(r["aristas"], [
            {"edge_id": "e001", "from_bin": "bin_1", "to_bin": "bin_2",
             "move_count": 1, "avg_quantity": 4.0, "low_stock_frac": 1.0},
            {"edge_id": "e002", "from_bin": "bin_2", "to_bin": "bin_1",
             "move_count": 1, "avg_quantity": 6.0, "low_stock_frac": 0.0},
            {"edge_id": "e003", "from_bin": "bin_2", "to_bin": "bin_2",
             "move_count": 1, "avg_quantity": 2.0, "low_stock_frac": 1.0},
            {"edge_id": "e004", "from_bin": "bin_2", "to_bin": "bin_3",
             "move_count": 1, "avg_quantity": 20.0, "low_stock_frac": 0.0},
        ])

    def test_mismo_resultado_con_cualquier_workers_y_fragmentos(self):
        base = ejecutar_pipeline(scans_sinteticos(), n_fragmentos=1, workers=1)
        for fragmentos, workers in ((2, 2), (6, 2), (8, 3)):
            otro = ejecutar_pipeline(scans_sinteticos(), n_fragmentos=fragmentos, workers=workers)
            self.assertEqual(otro["aristas"], base["aristas"])
            self.assertEqual(otro["nodos"], base["nodos"])

    def test_el_orden_de_los_scans_en_el_archivo_no_importa(self):
        desordenado = list(reversed(scans_sinteticos()))
        a = ejecutar_pipeline(scans_sinteticos(), n_fragmentos=2, workers=1)
        b = ejecutar_pipeline(desordenado, n_fragmentos=2, workers=1)
        self.assertEqual(a["aristas"], b["aristas"])

    def test_estadisticas(self):
        est = ejecutar_pipeline(scans_sinteticos(), n_fragmentos=3, workers=2)["stats"]
        self.assertEqual(est["job1"]["tuplas_map"], 6)
        self.assertEqual(est["job1"]["movimientos"], 4)
        self.assertEqual(est["job1"]["aristas"], 4)
        self.assertEqual(est["job1"]["scans_por_fragmento"], [2, 2, 2])
        self.assertEqual(est["job2"]["nodos"], 3)

    def test_registros_de_aristas_redondea_como_la_unit3(self):
        aristas = registros_de_aristas({("bin_1", "bin_2"): [3, 10, 1]})
        self.assertEqual(aristas[0]["avg_quantity"], 3.33)
        self.assertEqual(aristas[0]["low_stock_frac"], 0.3333)

    def test_directorio_dado_se_conserva(self):
        with tempfile.TemporaryDirectory() as base:
            destino = os.path.join(base, "fragmentos")
            ejecutar_pipeline(scans_sinteticos(), n_fragmentos=3, workers=1, directorio=destino)
            self.assertEqual(len(os.listdir(destino)), 3)

    def test_workers_invalido_falla(self):
        with self.assertRaises(ValueError):
            ejecutar_pipeline(scans_sinteticos(), workers=0)


class TestPipelineSobreDatasetReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scans = cargar_scans()
        cls.paralelo = ejecutar_pipeline(cls.scans, n_fragmentos=8, workers=2)

    def test_aristas_identicas_a_las_de_la_unit3(self):
        self.assertEqual(self.paralelo["aristas"], construir_aristas(self.scans))

    def test_validacion_completa_contra_la_unit3(self):
        checks = validar_contra_unit3(self.scans, self.paralelo["aristas"], self.paralelo["nodos"])
        fallos = [nombre for nombre, ok, _ in checks if not ok]
        self.assertEqual(fallos, [])

    def test_la_validacion_detecta_un_nodo_alterado(self):
        nodos = [dict(n) for n in self.paralelo["nodos"]]
        nodos[0]["in_move_count"] += 1
        checks = validar_contra_unit3(self.scans, self.paralelo["aristas"], nodos)
        self.assertTrue(any(not ok for _, ok, _ in checks))

    def test_independiente_de_workers_y_fragmentos(self):
        serial = ejecutar_pipeline(self.scans, n_fragmentos=3, workers=1)
        self.assertEqual(serial["aristas"], self.paralelo["aristas"])
        self.assertEqual(serial["nodos"], self.paralelo["nodos"])

    def test_totales(self):
        robots = len({s["robot_id"] for s in self.scans})
        nodos = self.paralelo["nodos"]
        self.assertEqual(len(nodos), 100)
        self.assertEqual(sum(n["in_move_count"] for n in nodos), len(self.scans) - robots)
        self.assertEqual(sum(n["out_move_count"] for n in nodos), len(self.scans) - robots)
        self.assertEqual(sum(n["net_flow"] for n in nodos), 0)

    def test_campos_del_resumen_por_nodo(self):
        campos = {"bin_location", "out_move_count", "in_move_count", "net_flow",
                  "avg_quantity", "low_stock_frac"}
        for n in self.paralelo["nodos"]:
            self.assertEqual(set(n), campos)

    def test_guardar_nodos_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "nodes.json")
            guardar_nodos(self.paralelo["nodos"], ruta)
            with open(ruta, encoding="utf-8") as f:
                self.assertEqual(json.load(f), self.paralelo["nodos"])


if __name__ == "__main__":
    unittest.main()
