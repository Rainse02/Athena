"""Check Note S6's governing equation and force-free end conditions."""

import unittest

import numpy as np

from axial_edge_theory import axial_edge_solution


class AxialEdgeTheoryTests(unittest.TestCase):
    def test_governing_equation_and_end_conditions(self):
        height = 20.0
        transfer = 2.0
        mismatch = 0.001
        y = np.linspace(0, height, 10001)
        delta, gradient = axial_edge_solution(y, height, transfer, mismatch)
        np.testing.assert_allclose(gradient[[0, -1]], mismatch, rtol=1e-14)
        self.assertEqual(delta[len(y) // 2], 0.0)
        np.testing.assert_allclose(delta[0], -delta[-1], rtol=1e-14)
        step = y[1] - y[0]
        second = (delta[2:] - 2 * delta[1:-1] + delta[:-2]) / step**2
        np.testing.assert_allclose(second, delta[1:-1] / transfer**2, atol=5e-11)

    def test_stable_form_matches_hyperbolic_expression(self):
        y = np.linspace(0, 3, 21)
        delta, gradient = axial_edge_solution(y, 3, 10, -0.002)
        expected = -0.002 * 10 * np.sinh((y - 1.5) / 10) / np.cosh(1.5 / 10)
        np.testing.assert_allclose(delta, expected, atol=1e-17)
        self.assertTrue(np.all(np.isfinite(gradient)))


if __name__ == "__main__":
    unittest.main()
