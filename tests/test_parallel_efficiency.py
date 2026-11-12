import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit4.ParallelEfficiency import (  # noqa: E402
    escalar_dataset,
    escribir_shards,
    experimento,
    mapper_shards,
    reduce_combinar,
    repartir_shards,
)

SCRATCH_TESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_shards")


def scans_sinteticos():
    # 2 robots, 3 scans cada uno, ya en orden de stream_seq
    return [
        {"scan_id": "s0", "robot_id": "r1", "stream_seq": 0, "bin_location": "bin_1"},
        {"scan_id": "s1", "robot_id": "r1", "stream_seq": 1, "bin_location": "bin_2"},
        {"scan_id": "s2", "robot_id": "r1", "stream_seq": 2, "bin_location": "bin_1"},
        {"scan_id": "s3", "robot_id": "r2", "stream_seq": 0, "bin_location": "bin_3"},
        {"scan_id": "s4", "robot_id": "r2", "stream_seq": 1, "bin_location": "bin_3"},
    ]


class TestEscalarDataset(unittest.TestCase):
    def test_factor_1_devuelve_copia_igual(self):
        scans = scans_sinteticos()
        escalado = escalar_dataset(scans, 1)
        self.assertEqual(escalado, scans)
        self.assertIsNot(escalado, scans)  # es una copia, no el mismo objeto

    def test_factor_n_multiplica_el_tamano(self):
        scans = scans_sinteticos()
        escalado = escalar_dataset(scans, 3)
        self.assertEqual(len(escalado), 3 * len(scans))

    def test_stream_seq_sigue_creciendo_por_robot(self):
        escalado = escalar_dataset(scans_sinteticos(), 2)
        por_robot = {}
        for s in escalado:
            por_robot.setdefault(s["robot_id"], []).append(s["stream_seq"])
        for secuencias in por_robot.values():
            self.assertEqual(secuencias, sorted(secuencias))
            self.assertEqual(len(secuencias), len(set(secuencias)))  # sin duplicados


class TestShardsYMapReduce(unittest.TestCase):
    def setUp(self):
        os.makedirs(SCRATCH_TESTS, exist_ok=True)
        self.prefijo = os.path.join(SCRATCH_TESTS, "test")

    def tearDown(self):
        for f in os.listdir(SCRATCH_TESTS):
            os.remove(os.path.join(SCRATCH_TESTS, f))

    def test_mapper_cuenta_movimientos_consecutivos(self):
        rutas = escribir_shards(scans_sinteticos(), self.prefijo, n_shards=1)
        conteo = mapper_shards(rutas)
        # r1: bin_1 -> bin_2 -> bin_1 da dos movimientos; r2: bin_3 -> bin_3 da uno (self-loop)
        self.assertEqual(conteo[("bin_1", "bin_2")], 1)
        self.assertEqual(conteo[("bin_2", "bin_1")], 1)
        self.assertEqual(conteo[("bin_3", "bin_3")], 1)
        self.assertEqual(sum(conteo.values()), 3)

    def test_sharding_no_pierde_ni_duplica_robots(self):
        scans = scans_sinteticos()
        rutas = escribir_shards(scans, self.prefijo, n_shards=4)
        self.assertEqual(len(rutas), 4)
        total_leido = 0
        for r in rutas:
            import json
            with open(r, encoding="utf-8") as f:
                total_leido += len(json.load(f))
        self.assertEqual(total_leido, len(scans))

    def test_mismo_resultado_sin_importar_cuantos_shards(self):
        scans = scans_sinteticos()
        r1 = mapper_shards(escribir_shards(scans, self.prefijo + "_a", n_shards=1))
        rutas_b = escribir_shards(scans, self.prefijo + "_b", n_shards=2)
        parcial = reduce_combinar([mapper_shards([r]) for r in rutas_b])
        self.assertEqual(r1, parcial)

    def test_repartir_shards_reparte_parejo(self):
        rutas = [f"s{i}" for i in range(8)]
        grupos = repartir_shards(rutas, 4)
        self.assertEqual(len(grupos), 4)
        self.assertEqual(sorted(sum(grupos, [])), sorted(rutas))
        for g in grupos:
            self.assertEqual(len(g), 2)  # 8 shards / 4 workers = 2 cada uno


class TestExperimentoChico(unittest.TestCase):
    def setUp(self):
        os.makedirs(SCRATCH_TESTS, exist_ok=True)

    def tearDown(self):
        for f in os.listdir(SCRATCH_TESTS):
            os.remove(os.path.join(SCRATCH_TESTS, f))

    def test_experimento_con_datos_sinteticos(self):
        # dataset chico y pocas repeticiones, solo para que el test no tome mucho tiempo
        scans = escalar_dataset(scans_sinteticos(), 20)  # 100 scans
        filas = experimento(
            tamanos=(1,), workers=(1, 2), repeticiones=2,
            scans_base=scans, directorio_shards=SCRATCH_TESTS,
        )
        self.assertEqual(len(filas), 2)
        por_p = {f["p"]: f for f in filas}
        self.assertAlmostEqual(por_p[1]["speedup"], 1.0)
        self.assertAlmostEqual(por_p[1]["eficiencia"], 1.0)
        self.assertGreater(por_p[2]["speedup"], 0)
        self.assertAlmostEqual(por_p[2]["eficiencia"], por_p[2]["speedup"] / 2)
        # no deben quedar archivos de shards temporales despues de correr
        self.assertEqual(
            [f for f in os.listdir(SCRATCH_TESTS) if f.startswith("_shards_tmp")], []
        )


if __name__ == "__main__":
    unittest.main()
