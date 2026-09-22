import unittest

import numpy as np

from deepseek_pipeline.saliency import gaussian_smooth


class SmoothingTests(unittest.TestCase):
    def test_short_sequence_stays_centered_and_keeps_length(self):
        actual = gaussian_smooth([0, 1, 0], sigma=2)
        offsets = np.arange(-6, 7)
        kernel = np.exp(-0.5 * (offsets / 2) ** 2)
        kernel /= kernel.sum()
        np.testing.assert_allclose(actual, kernel[5:8])
        self.assertEqual(actual.shape, (3,))
        self.assertEqual(int(actual.argmax()), 1)

    def test_single_value_uses_kernel_center(self):
        kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2) ** 2)
        actual = gaussian_smooth([1], sigma=2)
        np.testing.assert_allclose(actual, [1 / kernel.sum()])

    def test_long_sequence_matches_previous_centered_convolution(self):
        scores = np.arange(21, dtype=float)
        kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2) ** 2)
        kernel /= kernel.sum()
        np.testing.assert_allclose(gaussian_smooth(scores), np.convolve(scores, kernel, mode="same"))

    def test_empty_and_zero_sigma(self):
        self.assertEqual(gaussian_smooth([]).shape, (0,))
        original = np.array([1.0, 3.0])
        actual = gaussian_smooth(original, sigma=0)
        np.testing.assert_array_equal(actual, original)
        self.assertIsNot(actual, original)

    def test_invalid_arguments(self):
        for scores, sigma in (([[1]], 2), ([float("nan")], 2), ([1], -1), ([1], float("inf"))):
            with self.subTest(scores=scores, sigma=sigma), self.assertRaises(ValueError):
                gaussian_smooth(scores, sigma)


if __name__ == "__main__":
    unittest.main()
