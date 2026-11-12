import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit4.ReducerSizeReplication import (
    experimento,
    job1_agrupar,
    job1_carga_por_reducer,
    job1_hot_key,
    job1_map,
    job1_reduce,
    job1_replication_rate,
    job2_hot_key,
    job2_map,
    job2_reduce,
    job2_replication_rate,
)


class TestReducerSizeReplication(unittest.TestCase):
    def setUp(self):
        # Mini-dataset de juguete: 2 robots, con un movimiento claro cada uno.
        self.scans = [
            {"robot_id": "robot_A", "stream_seq": 0, "bin_location": "bin_1", "quantity": 10, "is_low_stock": False},
            {"robot_id": "robot_A", "stream_seq": 1, "bin_location": "bin_2", "quantity": 4, "is_low_stock": True},
            {"robot_id": "robot_A", "stream_seq": 2, "bin_location": "bin_3", "quantity": 8, "is_low_stock": False},
            {"robot_id": "robot_B", "stream_seq": 0, "bin_location": "bin_2", "quantity": 2, "is_low_stock": True},
        ]

    def test_job1_replication_rate_es_uno(self):
        salida = job1_map(self.scans)
        self.assertEqual(len(salida), len(self.scans))
        self.assertEqual(job1_replication_rate(self.scans, salida), 1.0)

    def test_job1_hot_key_es_el_robot_con_mas_scans(self):
        grupos = job1_agrupar(job1_map(self.scans))
        robot, q = job1_hot_key(grupos)
        self.assertEqual(robot, "robot_A")
        self.assertEqual(q, 3)

    def test_job1_reduce_cuenta_aristas_consecutivas(self):
        grupos = job1_agrupar(job1_map(self.scans))
        aristas = job1_reduce(grupos)
        # robot_A: bin_1->bin_2, bin_2->bin_3 (2 aristas); robot_B no genera ninguna (1 solo scan)
        self.assertEqual(aristas[("bin_1", "bin_2")]["move_count"], 1)
        self.assertEqual(aristas[("bin_2", "bin_3")]["move_count"], 1)
        self.assertEqual(sum(e["move_count"] for e in aristas.values()), 2)

    def test_job1_carga_por_reducer_con_r_1_es_el_total(self):
        grupos = job1_agrupar(job1_map(self.scans))
        carga = job1_carga_por_reducer(grupos, 1)
        self.assertEqual(carga, [len(self.scans)])

    def test_job1_carga_maxima_nunca_baja_del_hot_key(self):
        grupos = job1_agrupar(job1_map(self.scans))
        _, q = job1_hot_key(grupos)
        for r in (1, 2, 4):
            carga_max = max(job1_carga_por_reducer(grupos, r))
            self.assertGreaterEqual(carga_max, q)

    def test_job2_replication_rate_es_dos(self):
        aristas = job1_reduce(job1_agrupar(job1_map(self.scans)))
        salida2 = job2_map(aristas)
        self.assertEqual(len(salida2), 2 * len(aristas))
        self.assertEqual(job2_replication_rate(aristas, salida2), 2.0)

    def test_job2_resumen_bin_2_recibe_dos_llegadas(self):
        aristas = job1_reduce(job1_agrupar(job1_map(self.scans)))
        grupos2 = {}
        for clave, valor in job2_map(aristas):
            grupos2.setdefault(clave, []).append(valor)
        resumen = job2_reduce(grupos2)
        self.assertEqual(resumen["bin_2"]["in_move_count"], 1)  # solo bin_1->bin_2 cuenta como arista
        self.assertEqual(resumen["bin_1"]["out_move_count"], 1)

    def test_job2_hot_key_tiene_mas_registros_que_cualquier_otro(self):
        aristas = job1_reduce(job1_agrupar(job1_map(self.scans)))
        grupos2 = {}
        for clave, valor in job2_map(aristas):
            grupos2.setdefault(clave, []).append(valor)
        bin_, q = job2_hot_key(grupos2)
        self.assertEqual(q, max(len(v) for v in grupos2.values()))
        self.assertIn(bin_, grupos2)

    def test_experimento_end_to_end_sobre_el_dataset_real(self):
        resultado = experimento()
        self.assertEqual(resultado["n_scans"], 10000)
        self.assertEqual(resultado["job1"]["r"], 1.0)
        self.assertEqual(resultado["job2"]["r"], 2.0)
        # La carga máxima con R=1 debe ser el total de scans.
        self.assertEqual(max(resultado["job1"]["cargas"][1]), resultado["n_scans"])
        # Con cualquier R, la carga máxima nunca puede bajar del hot key q.
        q1 = resultado["job1"]["q"]
        for r, cargas in resultado["job1"]["cargas"].items():
            self.assertGreaterEqual(max(cargas), q1)


if __name__ == "__main__":
    unittest.main()