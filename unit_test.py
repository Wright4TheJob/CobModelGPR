"""Unit and integration tests for cob analysis."""

import unittest
import analysis


class TestExtremaFunc(unittest.TestCase):
    """Test selection of most extreme points in n-d space."""

    def test_insufficient_data(self):
        """Raise error when no or insufficient data is provided."""
        bad_points = [[], [1], [[1, 2, 3, 4, 5]]]
        for bad_input in bad_points:
            with self.assertRaises(ValueError):
                analysis.extrema(bad_input)

    def test_1d_data(self):
        """Return correct values for simple 1D vectors."""
        inputs = [
            [[[0], [1], [2]], ([0], [2])],
            [[[-5], [10], [0], [3]], ([-5], [10])]
            ]
        for points, expected in inputs:
            self.assertEqual(analysis.extrema(points), expected)

    def test_2d_data(self):
        """Return correct max vector length for 2D vectors."""
        inputs = [
            [[[0, 0], [3, 4], [1, 1]], 5],
            [[[65, 72], [0, 0], [5, 10]], 97]
            ]
        for points, expected in inputs:
            extremes = analysis.extrema(points)
            self.assertEqual(analysis.vlen(extremes[0], extremes[1]), expected)


class TestVLenFunc(unittest.TestCase):
    """Test vector length in n-d space function."""

    def test_1d(self):
        """Return correct values for simple 1D vectors."""
        touples = [
            [[1], [0], 1],
            [[-1], [1], 2]
            ]
        for pointa, pointb, expected in touples:
            self.assertEqual(analysis.vlen(pointa, pointb), expected)

    def test_2d(self):
        """Return correct values for 2D vectors."""
        points = [
            [[0, 0], [3, 4], 5],
            [[65, 72], [0, 0], 97]
            ]
        for pointa, pointb, expected in points:
            self.assertEqual(analysis.vlen(pointa, pointb), expected)


if __name__ == '__main__':
    unittest.main()
