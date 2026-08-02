"""Small checks for the fixed ETF geometry and public module import."""

import unittest

import torch

from projects.buzzspot.co_dino_fixed_etf_head import build_simplex_etf


class ETFHeadTest(unittest.TestCase):
    def test_simplex_geometry(self) -> None:
        targets = build_simplex_etf(256, 4)
        gram = targets.T @ targets
        expected = torch.full((4, 4), -1 / 3)
        expected.fill_diagonal_(1)
        self.assertTrue(torch.allclose(gram, expected, atol=1e-5))
        self.assertTrue(torch.equal(targets, build_simplex_etf(256, 4)))


if __name__ == "__main__":
    unittest.main()
