import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit1.UniversalHashing import (  # noqa: E402
    P,
    FuncionHashUniversal,
    HashSumaCaracteres,
    IndiceHash,
    benchmark_busqueda,
    cargar_dataset,
    clave_a_entero,
    metricas_colision,
)


class TestFuncionHashUniversal(unittest.TestCase):
    def test_rango_y_determinismo(self):
        h = FuncionHashUniversal(100, random.Random(1))
        valores = [h(f"sku_{i}") for i in range(1000)]
        self.assertTrue(all(0 <= v < 100 for v in valores))
        self.assertEqual(valores, [h(f"sku_{i}") for i in range(1000)])

    def test_parametros_validos(self):
        h = FuncionHashUniversal(64, random.Random(2))
        self.assertTrue(1 <= h.a < P and 0 <= h.b < P)

    def test_num_buckets_invalido(self):
        with self.assertRaises(ValueError):
            FuncionHashUniversal(0)

    def test_probabilidad_de_colision_de_un_par_es_a_lo_sumo_uno_sobre_m(self):
        # Dos claves fijas y muchos miembros de la familia: Pr[colisión] <= 1/m
        m, pruebas = 64, 20_000
        rng = random.Random(3)
        colisiones = 0
        for _ in range(pruebas):
            h = FuncionHashUniversal(m, rng)
            colisiones += h("sku_1") == h("sku_2")
        self.assertLess(colisiones / pruebas, 1.5 / m)

    def test_clave_a_entero_es_inyectiva_en_el_dataset(self):
        skus = {s["sku"] for s in cargar_dataset()}
        self.assertEqual(len({clave_a_entero(s) for s in skus}), len(skus))


class TestMetricas(unittest.TestCase):
    def test_metricas_colision(self):
        m = metricas_colision([0, 1, 3, 2, 0])
        self.assertEqual(m["skus"], 6)
        self.assertEqual(m["pares_en_colision"], 3 + 1)  # C(3,2) + C(2,2)
        self.assertEqual(m["skus_en_colision"], 5)
        self.assertEqual(m["buckets_vacios"], 2)
        self.assertEqual(m["carga_max"], 3)

    def test_suma_de_caracteres_colisiona_con_anagramas(self):
        h = HashSumaCaracteres(1000)
        self.assertEqual(h("ab"), h("ba"))


class TestIndiceHash(unittest.TestCase):
    def test_buscar_coincide_con_busqueda_lineal(self):
        rng = random.Random(4)
        skus = [f"sku_{rng.randint(1, 200)}" for _ in range(2000)]
        indice = IndiceHash(FuncionHashUniversal(7, rng))  # tabla chica: fuerza cadenas largas
        for posicion, sku in enumerate(skus):
            indice.insertar(sku, posicion)
        for sku in set(skus):
            self.assertEqual(indice.buscar(sku)[0], [i for i, s in enumerate(skus) if s == sku])
        self.assertEqual(indice.buscar("sku_inexistente")[0], [])
        self.assertEqual(sum(indice.cargas()), len(set(skus)))

    def test_benchmark_sobre_dataset_real(self):
        b = benchmark_busqueda(cargar_dataset(), consultas_indice=200, consultas_lineal=20)
        self.assertTrue(b["coincide_con_busqueda_lineal"])
        self.assertGreater(b["aceleracion"], 1)


if __name__ == "__main__":
    unittest.main()
